import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.general_utils import rotation_to_quaternion, quaternion_multiply, quaternion_to_rotation_matrix
from utils.sh_utils import RGB2SH, eval_sh
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from arguments import OptimizationParams
from tqdm import tqdm
from mesh import Mesh, safe_normalize, dot
from mesh_utils import decimate_mesh, clean_mesh, poisson_mesh_reconstruction
import mcubes
import trimesh
import nvdiffrast.torch as dr
from torchvision.utils import save_image, make_grid

def gaussian_3d_coeff(xyzs, covs):
    # xyzs: [N, 3]
    # covs: [N, 6]
    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]
    a, b, c, d, e, f = covs[:, 0], covs[:, 1], covs[:, 2], covs[:, 3], covs[:, 4], covs[:, 5]

    # eps must be small enough !!!
    inv_det = 1 / (a * d * f + 2 * e * c * b - e**2 * a - c**2 * d - b**2 * f + 1e-24)
    inv_a = (d * f - e**2) * inv_det
    inv_b = (e * c - b * f) * inv_det
    inv_c = (e * b - c * d) * inv_det
    inv_d = (a * f - c**2) * inv_det
    inv_e = (b * c - e * a) * inv_det
    inv_f = (a * d - b**2) * inv_det

    power = -0.5 * (x**2 * inv_a + y**2 * inv_d + z**2 * inv_f) - x * y * inv_b - x * z * inv_c - y * z * inv_e

    power[power > 0] = -1e10 # abnormal values... make weights 0
        
    return torch.exp(power)


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.normal_activation = lambda x: torch.nn.functional.normalize(x, dim=-1, eps=1e-3)
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        if self.use_pbr:
            self.base_color_activation = torch.sigmoid
            self.roughness_activation = torch.sigmoid
            self.metallic_activation = torch.sigmoid

    def __init__(self, sh_degree: int, render_type='render'):
        self.render_type = render_type
        self.use_pbr = render_type in ['neilf']
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._normal = torch.empty(0)  # normal
        self._shs_dc = torch.empty(0)  # output radiance
        self._shs_rest = torch.empty(0)  # output radiance
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.normal_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.setup_functions()
        self.transform = {}
        if self.use_pbr:
            self._base_color = torch.empty(0)
            self._roughness = torch.empty(0)
            self._metallic = torch.empty(0)
            self._incidents_dc = torch.empty(0)
            self._incidents_rest = torch.empty(0)
            self._visibility_dc = torch.empty(0)
            self._visibility_rest = torch.empty(0)

        # load mesh.ply
        if self.render_type in ["normal", "bind"]:
            # if self.opt.gui:
            #     self.glctx = dr.RasterizeCudaContext() # support at most 2048 resolution.
            # else:
            #     self.glctx = dr.RasterizeGLContext(output_db=False) # will crash if using GUI...
            self.glctx = dr.RasterizeCudaContext() # support at most 2048 resolution.
            self.vertices = torch.empty(0)
            self.triangles = torch.empty(0)

            self._vertices_offsets = torch.empty(0)
        
        self.use_offset = False
        self.use_anchor_field = False
        self.fix_gaus = False
        self.use_precompute_global = False
        if self.render_type in ["bind"]:
            self.triangles_points = torch.empty(0)
            
    
    def adaptive_bind(self, 
                        use_offset=False, 
                        fix_gaus=False,
                        anchor_field=False,
                        bary_field=False,
                        use_ckpt=False,
                        HP=10,
                        n_gaussians_per_surface_triangle=3
                    ):
        # set use offset, if True, regrad self._xyz as offset params
        self.use_offset = use_offset
        self.fix_gaus = fix_gaus
        self.use_anchor_field = anchor_field
        self.use_bary_field = bary_field

        # initialize per triangle gaussian
        n_points = self.triangles.shape[0] * n_gaussians_per_surface_triangle

        if n_gaussians_per_surface_triangle == 1:
                self.surface_triangle_circle_radius = 1. / 2. / np.sqrt(3.)
                self.surface_triangle_bary_coords = torch.tensor(
                    [[1/3, 1/3, 1/3]],
                    dtype=torch.float32,
                    device=self.vertices.device,
                )[..., None]

        if n_gaussians_per_surface_triangle == 3:
            self.surface_triangle_circle_radius = 1. / 2. / (np.sqrt(3.) + 1.)
            self.surface_triangle_bary_coords = torch.tensor(
                [[1/2, 1/4, 1/4],
                [1/4, 1/2, 1/4],
                [1/4, 1/4, 1/2]],
                dtype=torch.float32,
                device=self.vertices.device,
            )[..., None]
        
        if n_gaussians_per_surface_triangle == 4:
            self.surface_triangle_circle_radius = 1 / (4. * np.sqrt(3.))
            self.surface_triangle_bary_coords = torch.tensor(
                [[1/3, 1/3, 1/3],
                [2/3, 1/6, 1/6],
                [1/6, 2/3, 1/6],
                [1/6, 1/6, 2/3]],
                dtype=torch.float32,
                device=self.vertices.device,
            )[..., None]  # n_gaussians_per_face, 3, 1
            
        if n_gaussians_per_surface_triangle == 6:
            self.surface_triangle_circle_radius = 1 / (4. + 2.*np.sqrt(3.))
            self.surface_triangle_bary_coords = torch.tensor(
                [[2/3, 1/6, 1/6],
                [1/6, 2/3, 1/6],
                [1/6, 1/6, 2/3],
                [1/6, 5/12, 5/12],
                [5/12, 1/6, 5/12],
                [5/12, 5/12, 1/6]],
                dtype=torch.float32,
                device=self.vertices.device,
            )[..., None]
    
        # gaussian from triangles, compute the points using barycenter coordinates
        faces_verts = self.vertices[self.triangles.long()]  # n_faces, 3, n_coords
        points = faces_verts[:, None] * self.surface_triangle_bary_coords[None]  # n_faces, n_gaussians_per_face, 3, n_coords
        points = points.sum(dim=-2).reshape(n_points, 3)  # n_faces, n_gaussians_per_face, n_coords
        
        # # gaussian from vertices, compute the points using barycenter coordinates
        # points = self.vertices.clone()

        # 1. fix the number of gaussian, directly optimize the gaussian position
        # 2. remove and densify the number of gaussian, directly optimize the gaussian position
        # 3. fix the number of gaussian, optimize the gaussian position offset
        # 4. remove and densify  the number of gaussian, optimize the gaussian position offset

        # initialize gaussian attributes
        if self.use_offset:
            fused_point_cloud = torch.zeros_like(points).float().cuda()
        else:
            fused_point_cloud = points
        self.triangles_points = points.clone()
        # initialize gs attributes
        fused_normal = torch.zeros_like(points).float().cuda()
        fused_color = torch.zeros_like(points).float().cuda()
        shs = torch.zeros((points.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        shs[:, :3, 0] = RGB2SH(fused_color)
        shs[:, 3:, 1:] = 0.0
        dist2 = torch.clamp_min(distCUDA2(points.float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((points.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((points.shape[0], 1), dtype=torch.float, device="cuda"))
        
        if self.use_anchor_field:            
            # We compute quaternions to enforce face normals to be the first axis of gaussians
            mesh = Mesh(v=self.vertices, f=self.triangles, device='cuda')
            mesh.auto_normal()
            self.face_normals = mesh.face_normals
            R_0 = self.face_normals.clone()

            # we use the first side of every triangle as the second base axis
            base_R_1 = torch.nn.functional.normalize(faces_verts[:, 0] - faces_verts[:, 1], dim=-1)

            # we use the cross product for the last base axis
            base_R_2 = torch.nn.functional.normalize(torch.cross(R_0, base_R_1, dim=-1))
            
            # we concatenate the three vectors to get the rotation matrix
            R = torch.cat([R_0[..., None],
                        base_R_1[..., None],
                        base_R_2[..., None]],
                        dim=-1).view(-1, 3, 3)
            
            # set anchor rotation and 3D position atrributes
            self.anchor_rot = R # get_rotation
            self.anchor_pos = faces_verts.mean(dim=1) # get_xyz

            # set anchor scale atrributes using first edge and its perpendicular
            l_e1 = (faces_verts[:, 0] - faces_verts[:, 1]).norm(dim=-1)
            l_e2 = (faces_verts[:, 1] - faces_verts[:, 2]).norm(dim=-1)
            l_e3 = (faces_verts[:, 2] - faces_verts[:, 0]).norm(dim=-1)
            p = 0.5 * (l_e1 + l_e2 + l_e3)
            area = torch.sqrt(p * (p-l_e1) * (p-l_e2) * (p-l_e3))
            perpendicular = area / (l_e1 + 1e-10) * 2
            
            # # K != 1
            # self.anchor_scale = 0.5 * (perpendicular + l_e1)[..., None] # get_scaling
            
            # K = 1
            self.anchor_scale = torch.ones_like(perpendicular)[..., None]
            
            
            # K = 0.5 * (l_e2 + l_e3)
            perpendicular = 0.5 * (l_e2 + l_e3)

            # K is split
            self.anchor_scale = torch.zeros_like(self.anchor_pos)
            self.anchor_scale[:, 1] = l_e1
            self.anchor_scale[:, 2] = perpendicular
            self.anchor_scale[:, 0] = 0.5 * (perpendicular + l_e1)
            
            # hyper params
            # H_Params = 1.
            # H_Params = 10.
            H_Params = HP # 0.2 # HP
            print(f"[H_Params]: {H_Params}")
            print("")
            self.anchor_scale = self.anchor_scale * H_Params # 100 for materials, 10 for hotdog
            


            # # K adpats after edting
            # mesh = trimesh.load("post_poi_mesh.ply", force='mesh', skip_material=True, process=False)
            # vertices = mesh.vertices
            # vertices = torch.from_numpy(vertices).float().cuda()
            # triangles = mesh.faces
            # triangles = torch.from_numpy(triangles).float().cuda()
            # faces_verts_ori = vertices[triangles.long()]  # n_faces, 3, n_coords
            # l_e1_original = (faces_verts_ori[:, 0] - faces_verts_ori[:, 1]).norm(dim=-1)
            # l_e2 = (faces_verts_ori[:, 1] - faces_verts_ori[:, 2]).norm(dim=-1)
            # l_e3 = (faces_verts_ori[:, 2] - faces_verts_ori[:, 0]).norm(dim=-1)
            # p = 0.5 * (l_e1_original + l_e2 + l_e3)
            # area_ori = torch.sqrt(p * (p-l_e1_original) * (p-l_e2) * (p-l_e3))
            # perpendicular_original = area_ori / (l_e1_original + 1e-10) * 2
            # self.anchor_scale = 0.5 * (perpendicular / perpendicular_original  + l_e1 / l_e1_original)[..., None]

            # set triangle idx for each gs ball
            self.binded_idx = torch.arange(self.triangles.shape[0]).unsqueeze(-1)
            self.binded_idx = self.binded_idx.expand(-1, n_gaussians_per_surface_triangle).reshape(-1)
            self.binded_idx = nn.Parameter(self.binded_idx, requires_grad=False)

            # set the initial local position of each gs ball to zero
            # fused_point_cloud = torch.zeros_like(points).float().cuda()
            fused_point_cloud = fused_point_cloud - self.anchor_pos[self.binded_idx]

        # initialize gaussian attributes
        # use_ckpt = True
        if not use_ckpt:
            self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
            self._normal = nn.Parameter(fused_normal.requires_grad_(True))
            self._rotation = nn.Parameter(rots.requires_grad_(True))
            self._scaling = nn.Parameter(scales.requires_grad_(True))
            self._opacity = nn.Parameter(opacities.requires_grad_(True))
            self._shs_dc = nn.Parameter(shs[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._shs_rest = nn.Parameter(shs[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        

    def load_mesh(self, mesh_path):
        # load mesh
        mesh = trimesh.load(mesh_path, force='mesh', skip_material=True, process=False)
        
        # extract vertices and triangles
        vertices = mesh.vertices
        triangles = mesh.faces
        
        NeuMesh = False
        if NeuMesh:
            # def bounding_box(mesh):
            #     min_coords = mesh.min(axis=0)
            #     max_coords = mesh.max(axis=0)
            #     return min_coords, max_coords
            # def bounding_box_size(min_coords, max_coords):
            #     return max_coords - min_coords
            # min_coords1, max_coords1 = bounding_box(mesh1)
            # min_coords2, max_coords2 = bounding_box(mesh2)
            # size1 = bounding_box_size(min_coords1, max_coords1)
            # size2 = bounding_box_size(min_coords2, max_coords2)
            # scale = size2 / size1

            vertices = vertices * 2.035

        self.vertices = torch.from_numpy(vertices).float().cuda() # [N, 3]
        self.triangles = torch.from_numpy(triangles).int().cuda()
        
        # learnable offsets for mesh vertex
        self._vertices_offsets = nn.Parameter(torch.zeros_like(self.vertices, dtype=torch.float, device="cuda").requires_grad_(True))
    
    def load_original_mesh(self, mesh_path):
        # load mesh
        mesh = trimesh.load(mesh_path, force='mesh', skip_material=True, process=False)
        
        # extract vertices and triangles
        vertices = mesh.vertices
        triangles = mesh.faces
        self.vertices_original = torch.from_numpy(vertices).float().cuda() # [N, 3]
        self.triangles_original = torch.from_numpy(triangles).int().cuda()
        

    def get_mesh_normal(self, viewmatrix, projmatrix):
        
        if self.render_type in ["normal", "bind"]:
            # pass
            vertices = self.vertices + self._vertices_offsets
            # mvp = viewmatrix @ projmatrix
            # vertices_clip = torch.matmul(F.pad(vertices, pad=(0, 1), mode='constant', value=1.0), torch.transpose(mvp, 0, 1)).float().unsqueeze(0) # [1, N, 4]
            mvp = projmatrix
            vertices_clip = torch.matmul(F.pad(vertices, pad=(0, 1), mode='constant', value=1.0), mvp).float().unsqueeze(0) # [1, N, 4]

            h, w = 800, 800
            rast, _ = dr.rasterize(self.glctx, vertices_clip, self.triangles, (h, w))

            xyzs, _ = dr.interpolate(vertices.unsqueeze(0), rast, self.triangles) # [1, H, W, 3]
            mask, _ = dr.interpolate(torch.ones_like(vertices[:, :1]).unsqueeze(0), rast, self.triangles) # [1, H, W, 1]
            
            def auto_normal(f, v):
                i0, i1, i2 = f[:, 0].long(), f[:, 1].long(), f[:, 2].long()
                v0, v1, v2 = v[i0, :], v[i1, :], v[i2, :]

                face_normals = torch.cross(v1 - v0, v2 - v0)

                # Splat face normals to vertices
                vn = torch.zeros_like(v)
                vn.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
                vn.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
                vn.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

                # Normalize, replace zero (degenerated) normals with some default value
                vn = torch.where(
                    dot(vn, vn) > 1e-20,
                    vn,
                    torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vn.device),
                )
                vn = safe_normalize(vn)

                return vn

            mesh = Mesh(v=vertices, f=self.triangles, device='cuda')
            mesh.auto_normal()
            normals, _ = dr.interpolate(mesh.vn.unsqueeze(0), rast, self.triangles)
            # normals, _ = dr.interpolate(auto_normal(self.triangles, vertices).unsqueeze(0), rast, self.triangles)
            # view_xyz = torch.matmul(viewmatrix.transpose(0, 1)[None], torch.cat([xyzs, torch.ones_like(xyzs[..., 0:1])], dim=-1)[0].reshape(-1, 4)[..., None]).reshape(800, 800, -1)
            return normals[0].permute(2,0,1), mask[..., 0]

            if False:
                mesh = trimesh.Trimesh(vertices.detach().cpu().numpy(), self.triangles.detach().cpu().numpy(), process=False)
                mesh.export(os.path.join(save_path, f'mesh_0.ply'))
                save_image(xyzs[0].permute(2,0,1)[0:3], "./debug_rast_xyzs.png")
                save_image(normals[0].permute(2,0,1)[0:3], "./debug_rast_normals.png")
                # save_image(rast[0].permute(2,0,1)[2:3], "./debug_rast_depths.png")
        else:
            return None

    @torch.no_grad()
    def set_transform(self, rotation=None, center=None, scale=None, offset=None, transform=None):
        if transform is not None:
            scale = transform[:3, :3].norm(dim=-1)

            self._scaling.data = self.scaling_inverse_activation(self.get_scaling * scale)
            xyz_homo = torch.cat([self._xyz.data, torch.ones_like(self._xyz[:, :1])], dim=-1)
            self._xyz.data = (xyz_homo @ transform.T)[:, :3]
            rotation = transform[:3, :3] / scale[:, None]
            self._normal.data = self._normal.data @ rotation.T
            rotation_q = rotation_to_quaternion(rotation[None])
            self._rotation.data = quaternion_multiply(rotation_q, self._rotation.data)
            return

        if center is not None:
            self._xyz.data = self._xyz.data - center
        if rotation is not None:
            self._xyz.data = (self._xyz.data @ rotation.T)
            self._normal.data = self._normal.data @ rotation.T
            rotation_q = rotation_to_quaternion(rotation[None])
            self._rotation.data = quaternion_multiply(rotation_q, self._rotation.data)
        if scale is not None:
            self._xyz.data = self._xyz.data * scale
            self._scaling.data = self.scaling_inverse_activation(self.get_scaling * scale)
        if offset is not None:
            self._xyz.data = self._xyz.data + offset

    def capture(self):
        captured_list = [
            self.active_sh_degree,
            self._xyz,
            self._normal,
            self._shs_dc,
            self._shs_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.normal_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        ]
        if self.use_pbr:
            captured_list.extend([
                self._base_color,
                self._roughness,
                self._metallic,
                self._incidents_dc,
                self._incidents_rest,
                self._visibility_dc,
                self._visibility_rest,
            ])

        return captured_list

    def restore(self, model_args, training_args,
                is_training=False, restore_optimizer=True):
        (self.active_sh_degree,
         self._xyz,
         self._normal,
         self._shs_dc,
         self._shs_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         normal_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args[:14]
        if len(model_args) > 14 and self.use_pbr:
            (self._base_color,
             self._roughness,
             self._metallic,
             self._incidents_dc,
             self._incidents_rest,
             self._visibility_dc,
             self._visibility_rest) = model_args[14:]

        if is_training:
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.normal_gradient_accum = normal_gradient_accum
            self.denom = denom
            if restore_optimizer:
                # TODO automatically match the opt_dict
                try:
                    self.optimizer.load_state_dict(opt_dict)
                except:
                    pass

    @property
    def get_scaling(self):
        if self.use_anchor_field:
            if self.use_precompute_global:
                return self.g_scale
            # return self.anchor_scale * self.scaling_activation(self._scaling)
            return self.anchor_scale[self.binded_idx] * self.scaling_activation(self._scaling)
            # return self.scaling_activation(self.anchor_scale * self._scaling) !!! OOM due to the large scale
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        if self.use_anchor_field:
            if self.use_precompute_global:
                return self.g_rot
            return self.rotation_activation(
                        quaternion_multiply(
                            rotation_to_quaternion(self.anchor_rot)[self.binded_idx], 
                            # rotation_to_quaternion(self.anchor_rot), 
                            self.rotation_activation(self._rotation)
                        )
                    )
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        if self.use_offset:
            return self._xyz + self.triangles_points
        if self.fix_gaus:
            return self.triangles_points
        if self.use_anchor_field:
            # return self.anchor_scale * torch.matmul(self.anchor_rot, self._xyz.unsqueeze(-1))[..., 0] + self.anchor_pos
            if self.use_bary_field:
                # bary_coord = torch.nn.functional.softmax(self._xyz, dim=-1) # torch.Size([810582, 3])
                bary_coord = torch.sigmoid(self._xyz) / torch.sigmoid(self._xyz).sum(dim=-1)[:, None]
                tri_pos = (bary_coord[:, None] * self.vertices[self.triangles.long()][self.binded_idx]).sum(dim=1) # torch.Size([810582, 3])
                offset = torch.sigmoid(self._normal[:, 0:1]) * 2 - 1.
                # tri_pos = tri_pos + self.face_normals[self.binded_idx] * offset * 0.001# * self.anchor_scale[self.binded_idx][:, 0:1] # * 0.001
                tri_pos = tri_pos + self.face_normals[self.binded_idx] * offset * self.anchor_scale[self.binded_idx][:, 0:1] # * 0.001
                return tri_pos
            else:
                if self.use_precompute_global:
                    return self.g_xyz
                return (
                            self.anchor_scale[self.binded_idx] * \
                            torch.matmul(self.anchor_rot[self.binded_idx], self._xyz.unsqueeze(-1))[..., 0] \
                            + self.anchor_pos[self.binded_idx]
                        )
            
            # bary_coord = torch.nn.functional.softmax(self._normal, dim=-1) # torch.Size([810582, 3])
            # tri_pos = (bary_coord[:, None] * self.vertices[self.triangles.long()][self.binded_idx]).sum(dim=1) # torch.Size([810582, 3])
            # local_tri_pos = tri_pos -  self.anchor_pos[self.binded_idx]
            # g_pos = tri_pos + self.face_normals[self.binded_idx] * self._xyz[:, 0:1]
            # return g_pos


        return self._xyz

    def set_precompute_global(self):
        self.use_precompute_global = True
        self.g_xyz = (
                            self.anchor_scale[self.binded_idx] * \
                            torch.matmul(self.anchor_rot[self.binded_idx], self._xyz.unsqueeze(-1))[..., 0] \
                            + self.anchor_pos[self.binded_idx]
                        )
        self.g_rot = self.rotation_activation(
                        quaternion_multiply(
                            rotation_to_quaternion(self.anchor_rot)[self.binded_idx], 
                            # rotation_to_quaternion(self.anchor_rot), 
                            self.rotation_activation(self._rotation)
                        )
                    )
        self.g_scale = self.anchor_scale[self.binded_idx] * self.scaling_activation(self._scaling)
    
    def unset_precompute_global(self):
        self.use_precompute_global = False
    
    @property
    def get_normal(self):
        return self.normal_activation(self._normal)

    @property
    def get_shs(self):
        """SH"""
        shs_dc = self._shs_dc
        shs_rest = self._shs_rest
        return torch.cat((shs_dc, shs_rest), dim=1)

    @property
    def get_incidents(self):
        """SH"""
        incidents_dc = self._incidents_dc
        incidents_rest = self._incidents_rest
        return torch.cat((incidents_dc, incidents_rest), dim=1)

    @property
    def get_visibility(self):
        """SH"""
        visibility_dc = self._visibility_dc
        visibility_rest = self._visibility_rest
        return torch.cat((visibility_dc, visibility_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_base_color(self):
        return self.base_color_activation(self._base_color)

    @property
    def get_roughness(self):
        return self.roughness_activation(self._roughness)

    @property
    def get_metallic(self):
        return self.metallic_activation(self._metallic)

    @property
    def get_brdf(self):
        return torch.cat([self.get_base_color, self.get_roughness, self.get_metallic], dim=-1)

    def get_by_names(self, names):
        if len(names) == 0:
            return None
        fs = []
        for name in names:
            fs.append(getattr(self, "get_" + name))
        return torch.cat(fs, dim=1)

    def split_by_names(self, features, names):
        results = {}
        last_idx = 0
        for name in names:
            current_shape = getattr(self, "_" + name).shape[1]
            results[name] = features[last_idx:last_idx + current_shape]
            last_idx += getattr(self, "_" + name).shape[1]
        return results

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling,
                                          scaling_modifier,
                                          self.get_rotation)

    def get_inverse_covariance(self, scaling_modifier=1):
        return self.covariance_activation(1 / self.get_scaling,
                                          1 / scaling_modifier,
                                          self.get_rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    @property
    def attribute_names(self):
        attribute_names = ['xyz', 'normal', 'shs_dc', 'shs_rest', 'scaling', 'rotation', 'opacity']
        if self.use_pbr:
            attribute_names.extend(['base_color', 'roughness', 'metallic',
                                    'incidents_dc', 'incidents_rest',
                                    'visibility_dc', 'visibility_rest'])
        return attribute_names

    
    @classmethod
    def create_from_gaussians(cls, gaussians_list, dataset):
        assert len(gaussians_list) > 0
        sh_degree = max(g.max_sh_degree for g in gaussians_list)
        gaussians = GaussianModel(sh_degree=sh_degree,
                                  render_type=gaussians_list[0].render_type)
        attribute_names = gaussians.attribute_names
        for attribute_name in attribute_names:
            setattr(gaussians, "_" + attribute_name,
                    nn.Parameter(torch.cat([getattr(g, "_" + attribute_name).data for g in gaussians_list],
                                           dim=0).requires_grad_(True)))

        return gaussians

    def create_from_ckpt(self, checkpoint_path, restore_optimizer=False):
        (model_args, first_iter) = torch.load(checkpoint_path)

        (self.active_sh_degree,
         self._xyz,
         self._normal,
         self._shs_dc,
         self._shs_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         normal_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args[:14]

        self.xyz_gradient_accum = xyz_gradient_accum
        self.normal_gradient_accum = normal_gradient_accum
        self.denom = denom

        if self.use_pbr:
            if len(model_args) > 14:
                (self._base_color,
                 self._roughness,
                 self._metallic,
                 self._incidents_dc,
                 self._incidents_rest,
                 self._visibility_dc,
                 self._visibility_rest) = model_args[14:]
            else:
                self._base_color = nn.Parameter(torch.zeros_like(self._xyz).requires_grad_(True))
                self._roughness = nn.Parameter(torch.zeros_like(self._xyz[..., :1]).requires_grad_(True))
                self._metallic = nn.Parameter(torch.zeros_like(self._xyz[..., :1]).requires_grad_(True))
                incidents = torch.zeros((self._xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()

                self._incidents_dc = nn.Parameter(
                    incidents[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
                self._incidents_rest = nn.Parameter(
                    incidents[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))

                visibility = torch.zeros((self._xyz.shape[0], 1, 4 ** 2)).float().cuda()
                self._visibility_dc = nn.Parameter(
                    visibility[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
                self._visibility_rest = nn.Parameter(
                    visibility[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))

        if restore_optimizer:
            # TODO automatically match the opt_dict
            try:
                self.optimizer.load_state_dict(opt_dict)
            except:
                print("Not loading optimizer state_dict!")

        return first_iter

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_normal = torch.tensor(np.asarray(pcd.normals)).float().cuda()
        fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        shs = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        shs[:, :3, 0] = RGB2SH(fused_color)
        shs[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._normal = nn.Parameter(fused_normal.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._shs_dc = nn.Parameter(shs[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._shs_rest = nn.Parameter(shs[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if self.use_pbr:
            base_color = torch.zeros_like(fused_point_cloud)
            roughness = torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
            metallic = torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")

            self._base_color = nn.Parameter(base_color.requires_grad_(True))
            self._roughness = nn.Parameter(roughness.requires_grad_(True))
            self._metallic = nn.Parameter(metallic.requires_grad_(True))

            incidents = torch.zeros((self._xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            self._incidents_dc = nn.Parameter(incidents[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._incidents_rest = nn.Parameter(incidents[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))

            visibility = torch.zeros((self._xyz.shape[0], 1, 4 ** 2)).float().cuda()
            self._visibility_dc = nn.Parameter(visibility[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._visibility_rest = nn.Parameter(visibility[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))

    def training_setup(self, training_args: OptimizationParams):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.normal_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._normal], 'lr': training_args.normal_lr, "name": "normal"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._shs_dc], 'lr': training_args.sh_lr, "name": "f_dc"},
            {'params': [self._shs_rest], 'lr': training_args.sh_lr / 20.0, "name": "f_rest"}
        ]

        if self.use_pbr:
            if training_args.light_rest_lr < 0:
                training_args.light_rest_lr = training_args.light_lr / 20.0
            if training_args.visibility_rest_lr < 0:
                training_args.visibility_rest_lr = training_args.visibility_lr / 20.0

            l.extend([
                {'params': [self._base_color], 'lr': training_args.base_color_lr, "name": "base_color"},
                {'params': [self._roughness], 'lr': training_args.roughness_lr, "name": "roughness"},
                {'params': [self._metallic], 'lr': training_args.metallic_lr, "name": "metallic"},
                {'params': [self._incidents_dc], 'lr': training_args.light_lr, "name": "incidents_dc"},
                {'params': [self._incidents_rest], 'lr': training_args.light_rest_lr, "name": "incidents_rest"},
                {'params': [self._visibility_dc], 'lr': training_args.visibility_lr, "name": "visibility_dc"},
                {'params': [self._visibility_rest], 'lr': training_args.visibility_rest_lr, "name": "visibility_rest"},
            ])
        if self.render_type in ["normal"]:
            l.extend([
                {'params': [self._vertices_offsets], 'lr': training_args.mesh_verts_lr, 'name': "vertices_offsets"}
            ])
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def update_learning_rate(self, iteration):
        """ Learning rate scheduling per step """
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._shs_dc.shape[1] * self._shs_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._shs_rest.shape[1] * self._shs_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        if self.use_pbr:
            for i in range(self._base_color.shape[1]):
                l.append('base_color_{}'.format(i))
            l.append('roughness')
            l.append('metallic')
            for i in range(self._incidents_dc.shape[1] * self._incidents_dc.shape[2]):
                l.append('incidents_dc_{}'.format(i))
            for i in range(self._incidents_rest.shape[1] * self._incidents_rest.shape[2]):
                l.append('incidents_rest_{}'.format(i))
            for i in range(self._visibility_dc.shape[1] * self._visibility_dc.shape[2]):
                l.append('visibility_dc_{}'.format(i))
            for i in range(self._visibility_rest.shape[1] * self._visibility_rest.shape[2]):
                l.append('visibility_rest_{}'.format(i))

        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        # if self.use_offset:
        #     xyz = self._xyz.detach().cpu().numpy() + self.triangles_points.detach().cpu().numpy()
        normal = self._normal.detach().cpu().numpy()
        sh_dc = self._shs_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        sh_rest = self._shs_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        attributes_list = [xyz, normal, sh_dc, sh_rest, opacities, scale, rotation]
        if self.use_pbr:
            attributes_list.extend([
                self._base_color.detach().cpu().numpy(),
                self._roughness.detach().cpu().numpy(),
                self._metallic.detach().cpu().numpy(),
                self._incidents_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy(),
                self._incidents_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy(),
                self._visibility_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy(),
                self._visibility_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy(),
            ])

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(attributes_list, axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        normal = np.stack((np.asarray(plydata.elements[0]["nx"]),
                           np.asarray(plydata.elements[0]["ny"]),
                           np.asarray(plydata.elements[0]["nz"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        shs_dc = np.zeros((xyz.shape[0], 3, 1))
        shs_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        shs_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        shs_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        shs_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            shs_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        shs_extra = shs_extra.reshape((shs_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._normal = nn.Parameter(torch.tensor(normal, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._shs_dc = nn.Parameter(torch.tensor(
            shs_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._shs_rest = nn.Parameter(torch.tensor(
            shs_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

        if self.use_pbr:
            base_color_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("base_color")]
            base_color_names = sorted(base_color_names, key=lambda x: int(x.split('_')[-1]))
            base_color = np.zeros((xyz.shape[0], len(base_color_names)))
            for idx, attr_name in enumerate(base_color_names):
                base_color[:, idx] = np.asarray(plydata.elements[0][attr_name])

            roughness = np.asarray(plydata.elements[0]["roughness"])[..., np.newaxis]
            metallic = np.asarray(plydata.elements[0]["metallic"])[..., np.newaxis]

            self._base_color = nn.Parameter(
                torch.tensor(base_color, dtype=torch.float, device="cuda").requires_grad_(True))
            self._roughness = nn.Parameter(
                torch.tensor(roughness, dtype=torch.float, device="cuda").requires_grad_(True))
            self._metallic = nn.Parameter(torch.tensor(metallic, dtype=torch.float, device="cuda").requires_grad_(True))

            incidents_dc = np.zeros((xyz.shape[0], 3, 1))
            incidents_dc[:, 0, 0] = np.asarray(plydata.elements[0]["incidents_dc_0"])
            incidents_dc[:, 1, 0] = np.asarray(plydata.elements[0]["incidents_dc_1"])
            incidents_dc[:, 2, 0] = np.asarray(plydata.elements[0]["incidents_dc_2"])
            extra_incidents_names = [p.name for p in plydata.elements[0].properties if
                                     p.name.startswith("incidents_rest_")]
            extra_incidents_names = sorted(extra_incidents_names, key=lambda x: int(x.split('_')[-1]))
            assert len(extra_incidents_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
            incidents_extra = np.zeros((xyz.shape[0], len(extra_incidents_names)))
            for idx, attr_name in enumerate(extra_incidents_names):
                incidents_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            incidents_extra = incidents_extra.reshape((incidents_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
            self._incidents_dc = nn.Parameter(torch.tensor(incidents_dc, dtype=torch.float, device="cuda").transpose(
                1, 2).contiguous().requires_grad_(True))
            self._incidents_rest = nn.Parameter(
                torch.tensor(incidents_extra, dtype=torch.float, device="cuda").transpose(
                    1, 2).contiguous().requires_grad_(True))

            visibility_dc = np.zeros((xyz.shape[0], 1, 1))
            visibility_dc[:, 0, 0] = np.asarray(plydata.elements[0]["visibility_dc_0"])
            extra_visibility_names = [p.name for p in plydata.elements[0].properties if
                                      p.name.startswith("visibility_rest_")]
            extra_visibility_names = sorted(extra_visibility_names, key=lambda x: int(x.split('_')[-1]))
            assert len(extra_visibility_names) == 4 ** 2 - 1
            visibility_extra = np.zeros((xyz.shape[0], len(extra_visibility_names)))
            for idx, attr_name in enumerate(extra_visibility_names):
                visibility_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            visibility_extra = visibility_extra.reshape((visibility_extra.shape[0], 1, 4 ** 2 - 1))
            self._visibility_dc = nn.Parameter(torch.tensor(visibility_dc, dtype=torch.float, device="cuda").transpose(
                1, 2).contiguous().requires_grad_(True))
            self._visibility_rest = nn.Parameter(
                torch.tensor(visibility_extra, dtype=torch.float, device="cuda").transpose(
                    1, 2).contiguous().requires_grad_(True))

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._normal = optimizable_tensors["normal"]
        self._shs_dc = optimizable_tensors["f_dc"]
        self._shs_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.normal_gradient_accum = self.normal_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        if self.use_anchor_field:
            self.binded_idx = self.binded_idx[valid_points_mask]
            
        if self.use_pbr:
            self._base_color = optimizable_tensors["base_color"]
            self._roughness = optimizable_tensors["roughness"]
            self._metallic = optimizable_tensors["metallic"]
            self._incidents_dc = optimizable_tensors["incidents_dc"]
            self._incidents_rest = optimizable_tensors["incidents_rest"]
            self._visibility_dc = optimizable_tensors["visibility_dc"]
            self._visibility_rest = optimizable_tensors["visibility_rest"]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]

                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_normal, new_shs_dc, new_shs_rest, new_opacities, new_scaling,
                              new_rotation, new_base_color=None, new_roughness=None,
                              new_metallic=None, new_incidents_dc=None, new_incidents_rest=None,
                              new_visibility_dc=None, new_visibility_rest=None, new_binded_idx=None):
        d = {"xyz": new_xyz,
             "normal": new_normal,
             "rotation": new_rotation,
             "scaling": new_scaling,
             "opacity": new_opacities,
             "f_dc": new_shs_dc,
             "f_rest": new_shs_rest}

        if self.use_pbr:
            d.update({
                "base_color": new_base_color,
                "roughness": new_roughness,
                "metallic": new_metallic,
                "incidents_dc": new_incidents_dc,
                "incidents_rest": new_incidents_rest,
                "visibility_dc": new_visibility_dc,
                "visibility_rest": new_visibility_rest,
            })

        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        self._xyz = optimizable_tensors["xyz"]
        self._normal = optimizable_tensors["normal"]
        self._rotation = optimizable_tensors["rotation"]
        self._scaling = optimizable_tensors["scaling"]
        self._opacity = optimizable_tensors["opacity"]
        self._shs_dc = optimizable_tensors["f_dc"]
        self._shs_rest = optimizable_tensors["f_rest"]

        if self.use_anchor_field:
            self.binded_idx = torch.cat((self.binded_idx, new_binded_idx), dim=0)
        
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.normal_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if self.use_pbr:
            self._base_color = optimizable_tensors["base_color"]
            self._roughness = optimizable_tensors["roughness"]
            self._metallic = optimizable_tensors["metallic"]
            self._incidents_dc = optimizable_tensors["incidents_dc"]
            self._incidents_rest = optimizable_tensors["incidents_rest"]
            self._visibility_dc = optimizable_tensors["visibility_dc"]
            self._visibility_rest = optimizable_tensors["visibility_rest"]

    def densify_and_split(self, grads, grad_threshold, scene_extent, grads_normal, grad_normal_threshold, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        padded_grad_normal = torch.zeros((n_init_points), device="cuda")
        padded_grad_normal[:grads_normal.shape[0]] = grads_normal.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask_normal = torch.where(padded_grad_normal >= grad_normal_threshold, True, False)
        # print("densify_and_split_normal:", selected_pts_mask_normal.sum().item(), "/", self.get_xyz.shape[0])

        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_normal)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask, torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)
        # print("densify_and_split:", selected_pts_mask.sum().item(), "/", self.get_xyz.shape[0])

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)  # (N, 3)
        means = torch.zeros((stds.size(0), 3), device="cuda")  # (N, 3)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)

        new_normal = self._normal[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_shs_dc = self._shs_dc[selected_pts_mask].repeat(N, 1, 1)
        new_shs_rest = self._shs_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        args = [new_xyz, new_normal, new_shs_dc, new_shs_rest, new_opacity, new_scaling, new_rotation]
        
        if self.use_anchor_field:
            new_binded_idx = self.binded_idx[selected_pts_mask].repeat(N)
            kwargs = {
                "new_binded_idx": new_binded_idx
            }

        if self.use_pbr:
            new_base_color = self._base_color[selected_pts_mask].repeat(N, 1)
            new_roughness = self._roughness[selected_pts_mask].repeat(N, 1)
            new_metallic = self._metallic[selected_pts_mask].repeat(N, 1)
            new_incidents_dc = self._incidents_dc[selected_pts_mask].repeat(N, 1, 1)
            new_incidents_rest = self._incidents_rest[selected_pts_mask].repeat(N, 1, 1)
            new_visibility_dc = self._visibility_dc[selected_pts_mask].repeat(N, 1, 1)
            new_visibility_rest = self._visibility_rest[selected_pts_mask].repeat(N, 1, 1)
            args.extend([
                new_base_color,
                new_roughness,
                new_metallic,
                new_incidents_dc,
                new_incidents_rest,
                new_visibility_dc,
                new_visibility_rest,
            ])

        if self.use_anchor_field:
            self.densification_postfix(*args, **kwargs)
        else:
            self.densification_postfix(*args)
        # self.densification_postfix(*args)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, grads_normal, grad_normal_threshold):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask_normal = torch.where(torch.norm(grads_normal, dim=-1) >= grad_normal_threshold, True, False)
        # print("densify_and_clone_normal:", selected_pts_mask_normal.sum().item(), "/", self.get_xyz.shape[0])
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_normal)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)
        # print("densify_and_clone:", selected_pts_mask.sum().item(), "/", self.get_xyz.shape[0])

        new_xyz = self._xyz[selected_pts_mask]
        new_normal = self._normal[selected_pts_mask]
        new_shs_dc = self._shs_dc[selected_pts_mask]
        new_shs_rest = self._shs_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        args = [new_xyz, new_normal, new_shs_dc, new_shs_rest, new_opacities,
                new_scaling, new_rotation]
        
        if self.use_anchor_field:
            new_binded_idx = self.binded_idx[selected_pts_mask]
            kwargs = {
                "new_binded_idx": new_binded_idx
            }
        
        if self.use_pbr:
            new_base_color = self._base_color[selected_pts_mask]
            new_roughness = self._roughness[selected_pts_mask]
            new_metallic = self._metallic[selected_pts_mask]
            new_incidents_dc = self._incidents_dc[selected_pts_mask]
            new_incidents_rest = self._incidents_rest[selected_pts_mask]
            new_visibility_dc = self._visibility_dc[selected_pts_mask]
            new_visibility_rest = self._visibility_rest[selected_pts_mask]

            args.extend([
                new_base_color,
                new_roughness,
                new_metallic,
                new_incidents_dc,
                new_incidents_rest,
                new_visibility_dc,
                new_visibility_rest,
            ])
        if self.use_anchor_field:
            self.densification_postfix(*args, **kwargs)
        else:
            self.densification_postfix(*args)
        # self.densification_postfix(*args)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, max_grad_normal):
        # print(self.xyz_gradient_accum.shape)
        grads = self.xyz_gradient_accum / self.denom
        grads_normal = self.normal_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        grads_normal[grads_normal.isnan()] = 0.0

        # if self._xyz.shape[0] < 1000000:
        self.densify_and_clone(grads, max_grad, extent, grads_normal, max_grad_normal)
        self.densify_and_split(grads, max_grad, extent, grads_normal, max_grad_normal)
        # self.densify_and_compact()

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def prune(self, min_opacity, extent, max_screen_size):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        self.normal_gradient_accum[update_filter] += torch.norm(
            self.normal_activation(self._normal.grad)[update_filter], dim=-1,
            keepdim=True)
        self.denom[update_filter] += 1
    
    @torch.no_grad()
    def extract_fields(self, resolution=128, num_blocks=16, relax_ratio=1.5):
        # resolution: resolution of field
        
        block_size = 2 / num_blocks

        assert resolution % block_size == 0
        split_size = resolution // num_blocks

        opacities = self.get_opacity

        # pre-filter low opacity gaussians to save computation
        mask = (opacities > 0.005).squeeze(1)

        opacities = opacities[mask]
        xyzs = self.get_xyz[mask]
        stds = self.get_scaling[mask]
        
        # normalize to ~ [-1, 1]
        mn, mx = xyzs.amin(0), xyzs.amax(0)
        self.center = (mn + mx) / 2
        # self.scale = 1.
        self.scale = 1.8 / (mx - mn).amax().item()
        # self.scale = 2. / (mx - mn).amax().item()

        xyzs = (xyzs - self.center) * self.scale
        stds = stds * self.scale

        covs = self.covariance_activation(stds, 1, self._rotation[mask])

        # tile
        device = opacities.device
        occ = torch.zeros([resolution] * 3, dtype=torch.float32, device=device)

        X = torch.linspace(-1, 1, resolution).split(split_size)
        Y = torch.linspace(-1, 1, resolution).split(split_size)
        Z = torch.linspace(-1, 1, resolution).split(split_size)


        # loop blocks (assume max size of gaussian is small than relax_ratio * block_size !!!)
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    # sample points [M, 3]
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(device)
                    # in-tile gaussians mask
                    vmin, vmax = pts.amin(0), pts.amax(0)
                    vmin -= block_size * relax_ratio
                    vmax += block_size * relax_ratio
                    mask = (xyzs < vmax).all(-1) & (xyzs > vmin).all(-1)
                    # if hit no gaussian, continue to next block
                    if not mask.any():
                        continue
                    mask_xyzs = xyzs[mask] # [L, 3]
                    mask_covs = covs[mask] # [L, 6]
                    mask_opas = opacities[mask].view(1, -1) # [L, 1] --> [1, L]

                    # query per point-gaussian pair.
                    g_pts = pts.unsqueeze(1).repeat(1, mask_covs.shape[0], 1) - mask_xyzs.unsqueeze(0) # [M, L, 3]
                    g_covs = mask_covs.unsqueeze(0).repeat(pts.shape[0], 1, 1) # [M, L, 6]

                    # batch on gaussian to avoid OOM
                    batch_g = 1024 * 4 * 4
                    val = 0
                    for start in range(0, g_covs.shape[1], batch_g):
                        end = min(start + batch_g, g_covs.shape[1])
                        w = gaussian_3d_coeff(g_pts[:, start:end].reshape(-1, 3), g_covs[:, start:end].reshape(-1, 6)).reshape(pts.shape[0], -1) # [M, l]
                        
                        # # all sum
                        # val += (mask_opas[:, start:end] * w).sum(-1)

                        # ball sum
                        # dist_thresh = 0.025 # 0.008 # 0.025 for other # 0.5 for ficus, ship
                        dist_thresh = 0.01 # 0.0075 # 0.025 for other # 0.5 for ficus, ship # 0.01 for drum, ficus, ship
                        dist_mask = g_pts[:, start:end].norm(dim=-1) < dist_thresh
                        # if w.max() < 1e-10 or dist_mask.sum()==0:
                        #     continue
                        # val = (mask_opas[:, start:end] * w * dist_mask).amax(dim=-1)
                        # val = (torch.ones_like(mask_opas[:, start:end]) * w * dist_mask).amax(dim=-1)
                        # val += (mask_opas[:, start:end] * w * dist_mask).sum(-1)
                        # val += (mask_opas[:, start:end] * dist_mask).amax(dim=-1)
                        val += (torch.ones_like(mask_opas[:, start:end] * w ) * dist_mask).sum(dim=-1)

                    # kiui.lo(val, mask_opas, w)
                
                    occ[xi * split_size: xi * split_size + len(xs), 
                        yi * split_size: yi * split_size + len(ys), 
                        zi * split_size: zi * split_size + len(zs)] = val.reshape(len(xs), len(ys), len(zs)) 
        
        # kiui.lo(occ, verbose=1)

        return occ
    
    def extract_mesh(self, path, density_thresh=1, resolution=128, relax_ratio=1.5, num_blocks=16, decimate_target=1e5):

        os.makedirs(os.path.dirname(path), exist_ok=True)

        occ = self.extract_fields(resolution, num_blocks=num_blocks, relax_ratio=relax_ratio).detach().cpu().numpy()

        vertices, triangles = mcubes.marching_cubes(occ, density_thresh)
        vertices = vertices / (resolution - 1.0) * 2 - 1

        # transform back to the original space
        vertices = vertices / self.scale + self.center.detach().cpu().numpy()

        vertices, triangles = clean_mesh(vertices, triangles, remesh=True, remesh_size=0.015)
        # if decimate_target > 0 and triangles.shape[0] > decimate_target:
        #     vertices, triangles = decimate_mesh(vertices, triangles, decimate_target)

        v = torch.from_numpy(vertices.astype(np.float32)).contiguous().cuda()
        f = torch.from_numpy(triangles.astype(np.int32)).contiguous().cuda()

        print(
            f"[INFO] marching cubes result: {v.shape} ({v.min().item()}-{v.max().item()}), {f.shape}"
        )

        mesh = Mesh(v=v, f=f, device='cuda')

        return mesh
