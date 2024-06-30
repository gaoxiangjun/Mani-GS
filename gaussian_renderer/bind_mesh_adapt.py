
# This is script for 3D Gaussian Splatting rendering

import math
import torch
import torch.nn.functional as F
from arguments import OptimizationParams
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.loss_utils import ssim
from utils.image_utils import psnr
from .r3dg_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def chamfer_distance(pts0, pts1):
    """
    计算两个点云之间的Chamfer距离
    pts0 is mesh verts
    pts1 is gs points
    """
    
    # pts1 = pc.get_xyz.detach().clone()
    # pts0 = (pc.vertices + pc._vertices_offsets)
    bs = 4096 * 4
    sampled_idx = torch.randint(0, pts0.shape[0], [bs])
    pts0 = pts0[sampled_idx]

    dist_x_y = torch.cdist(pts0[None], pts1[None])
    dist0 = torch.min(dist_x_y, dim=2)[0]
    
    # def nearest_dist(pts0, pts1, batch_size=512):
    #     pn0, pn1 = pts0.shape[0], pts1.shape[0]
    #     dists = []
    #     for i in range(0, pn0, batch_size):
    #         dist = torch.norm(pts0[i:i+batch_size,None,:] - pts1[None,:,:], dim=-1)
    #         dists.append(torch.min(dist,1)[0])
    #     dists = torch.cat(dists,0)
    #     return dists
    # dist0 = nearest_dist(pts_pr, pts_gt, batch_size=4096)

    # chamfer = (torch.mean(dist0) + torch.mean(dist1)) / 2
    chamfer = torch.mean(dist0)


def render_view(camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, 
                scaling_modifier, override_color, computer_pseudo_normal=True):
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means

    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(camera.FoVx * 0.5)
    tanfovy = math.tan(camera.FoVy * 0.5)
    intrinsic = camera.intrinsics
    raster_settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height),
        image_width=int(camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=float(intrinsic[0, 2]),
        cy=float(intrinsic[1, 2]),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=camera.camera_center,
        prefiltered=False,
        backward_geometry=True,
        computer_pseudo_normal=computer_pseudo_normal,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.compute_SHs_python:
            shs_view = pc.get_shs.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - camera.camera_center.repeat(pc.get_shs.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_shs
    else:
        colors_precomp = override_color

    features = pc.get_normal
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    (num_rendered, num_contrib, rendered_image, rendered_opacity, rendered_depth,
     rendered_feature, rendered_pseudo_normal, rendered_surface_xyz, radii) = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        features=features,
    )
    rendered_normal = rendered_feature

    # mesh_normal = pc.get_mesh_normal(camera.world_view_transform, camera.full_proj_transform)

    # from torchvision.utils import save_image
    # save_image(rendered_image, "./debug_rendered_images.png")
    results = {"render": rendered_image,
               "opacity": rendered_opacity,
               "depth": rendered_depth,
               "normal": rendered_normal,
               "pseudo_normal": rendered_pseudo_normal,
               "surface_xyz": rendered_surface_xyz,
               "viewspace_points": screenspace_points,
               "visibility_filter": radii > 0,
               "radii": radii,
               "num_rendered": num_rendered,
            #    "mesh_normal": mesh_normal,
               "num_contrib": num_contrib}
    
    return results

def calculate_loss(viewpoint_camera, pc, render_pkg, opt):
    tb_dict = {
        "num_points": pc.get_xyz.shape[0],
    }
    
    rendered_image = render_pkg["render"]
    rendered_opacity = render_pkg["opacity"]
    rendered_depth = render_pkg["depth"]
    rendered_normal = render_pkg["normal"]
    gt_image = viewpoint_camera.original_image.cuda()
    image_mask = viewpoint_camera.image_mask.cuda()

    Ll1 = F.l1_loss(rendered_image, gt_image)
    ssim_val = ssim(rendered_image, gt_image)
    tb_dict["loss_l1"] = Ll1.item()
    tb_dict["psnr"] = psnr(rendered_image, gt_image).mean().item()
    tb_dict["ssim"] = ssim_val.item()
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)

    if opt.lambda_depth > 0:
        gt_depth = viewpoint_camera.depth.cuda()
        depth_mask = gt_depth > 0
        sur_mask = torch.logical_xor(image_mask.bool(), depth_mask)

        loss_depth = F.l1_loss(rendered_depth[~sur_mask], gt_depth[~sur_mask])
        tb_dict["loss_depth"] = loss_depth.item()
        loss = loss + opt.lambda_depth * loss_depth

    if opt.lambda_mask_entropy > 0:
        o = rendered_opacity.clamp(1e-6, 1 - 1e-6)
        loss_mask_entropy = -(image_mask * torch.log(o) + (1-image_mask) * torch.log(1 - o)).mean()
        tb_dict["loss_mask_entropy"] = loss_mask_entropy.item()
        loss = loss + opt.lambda_mask_entropy * loss_mask_entropy

    if opt.lambda_normal_render_depth > 0:
        normal_pseudo = render_pkg['pseudo_normal']
        loss_normal_render_depth = F.mse_loss(
            rendered_normal * image_mask, normal_pseudo.detach() * image_mask)
        tb_dict["loss_normal_render_depth"] = loss_normal_render_depth.item()
        loss = loss + opt.lambda_normal_render_depth * loss_normal_render_depth

    if opt.lambda_normal_mvs_depth > 0:
        gt_depth = viewpoint_camera.depth.cuda()
        depth_mask = (gt_depth > 0).float()
        mvs_normal = viewpoint_camera.normal.cuda()
        loss_normal_mvs_depth = F.mse_loss(
            rendered_normal * depth_mask, mvs_normal * depth_mask)
        tb_dict["loss_normal_mvs_depth"] = loss_normal_mvs_depth.item()
        loss = loss + opt.lambda_normal_mvs_depth * loss_normal_mvs_depth

    if opt.lambda_mesh_normal > 0:
        mesh_normal = render_pkg['mesh_normal']
        loss_mesh_normal = F.mse_loss(
            mesh_normal, rendered_normal.detach()
            # mesh_normal * image_mask, rendered_normal.detach() * image_mask
            # mesh_normal * image_mask, normal_pseudo.detach() * image_mask
        )
        tb_dict["loss_mesh_normal"] = loss_mesh_normal.item()
        loss = loss + opt.lambda_mesh_normal * loss_mesh_normal 
        
        mesh_mask = render_pkg['mesh_mask']
        o = rendered_opacity.clamp(1e-6, 1 - 1e-6).detach()
        loss_mesh_mask_entropy = -(mesh_mask * torch.log(o) + (1-mesh_mask) * torch.log(1 - o)).mean()
        tb_dict["loss_mesh_mask_entropy"] = loss_mesh_mask_entropy.item()
        loss = loss + opt.lambda_mask_entropy * loss_mesh_mask_entropy * 10

        from mesh import Mesh
        mesh = Mesh(v=(pc.vertices + pc._vertices_offsets), f=pc.triangles, device='cuda')
        mesh.auto_normal()
        lap_loss = mesh.laplacian()
        consis_loss = mesh.normal_consistency()
        
        loss = loss + (pc._vertices_offsets.norm(dim=-1) ** 2).mean() * 0.1 # 0.1 * F.mse_loss(pc._vertices_offsets, torch.zeros_like(pc._vertices_offsets))

        # valid_mask = (pc._vertices_offsets.norm(dim=-1)) > 0.02
        # loss = loss + (pc._vertices_offsets.norm(dim=-1) ** 2)[valid_mask].mean() # 0.1 * F.mse_loss(pc._vertices_offsets, torch.zeros_like(pc._vertices_offsets))
        loss = loss + 0.0001 * consis_loss
        loss = loss + 0.0005 * lap_loss

        # # mesh.write_ply("./mesh.ply")
        chf_dist = chamfer_distance((pc.vertices + pc._vertices_offsets), pc.get_xyz.detach().clone())
        loss = loss + 0.1 * chf_dist

    eps_pos = 0.005 # 0.01
    local_pos = pc.get_xyz - pc.anchor_pos[pc.binded_idx]
    valid_mask = (local_pos.abs() > eps_pos)
    loss_local_pos = (local_pos.abs()[valid_mask] - eps_pos).mean()
    tb_dict["loss_local_pos"] = loss_local_pos.item()
    loss = loss + 0.0 * loss_local_pos # 0.05

    eps_scale = 0.01 # 0.01 # triangle length
    valid_mask = (pc.get_scaling.amax(dim=-1) > eps_scale)
    # valid_mask = (pc.scaling_activation(pc._scaling) > eps_scale)
    loss_local_scale = (pc.get_scaling.amax(dim=-1)[valid_mask] - eps_scale).mean()
    # loss_local_scale = (pc.scaling_activation(pc._scaling)[valid_mask] - eps_scale).mean()
    tb_dict["loss_local_scale"] = loss_local_scale.item()
    loss = loss + 0.0 * loss_local_scale # 0.01
    
    tb_dict["loss"] = loss.item()
    
    return loss, tb_dict

def render_bind_mesh_adapt(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, 
           scaling_modifier=1.0,override_color=None, opt: OptimizationParams = None, 
           is_training=False, dict_params=None):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """
    results = render_view(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color,
                          computer_pseudo_normal=True if opt is not None and opt.lambda_normal_render_depth>0 else False)

    results["hdr"] = viewpoint_camera.hdr

    if is_training:
        loss, tb_dict = calculate_loss(viewpoint_camera, pc, results, opt)
        results["tb_dict"] = tb_dict
        results["loss"] = loss
    
    return results
