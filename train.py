import os
import torch
import torch.nn.functional as F
import torchvision
from collections import defaultdict
from random import randint
from gaussian_renderer import render_fn_dict
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.loss_utils import ssim
from utils.image_utils import psnr, visualize_depth
from lpipsPyTorch import lpips
from utils.system_utils import prepare_output_and_logger
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.derect_light_sh import DirectLightEnv
from scene.gamma_trans import LearningGammaTransform
from utils.graphics_utils import hdr2ldr
from torchvision.utils import save_image, make_grid
from mesh_utils import decimate_mesh, clean_mesh, poisson_mesh_reconstruction
import time


def training(dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams, is_pbr=False):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    """
    Setup Gaussians
    """
    gaussians = GaussianModel(dataset.sh_degree, render_type=args.type)
    scene = Scene(dataset, gaussians)
    if args.checkpoint:
        print("Create Gaussians from checkpoint {}".format(args.checkpoint))
        first_iter = gaussians.create_from_ckpt(args.checkpoint, restore_optimizer=True)

    elif scene.loaded_iter:
        gaussians.load_ply(os.path.join(dataset.model_path,
                                        "point_cloud",
                                        "iteration_" + str(scene.loaded_iter),
                                        "point_cloud.ply"))
    else:
        gaussians.create_from_pcd(scene.scene_info.point_cloud, scene.cameras_extent)

    # load mesh
    os.makedirs(os.path.join(args.model_path, 'eval'), exist_ok=True)
    
    if args.type in ['normal']:
        mesh_path = os.path.join(os.path.dirname(args.model_path), '3dgs', 'eval', 'mesh_mc_30K.ply')
        gaussians.load_mesh(mesh_path)
    
    if args.type in ['bind', 'bind_mesh_adapt']:
        mesh_type_dict = {
            "neus": 'mesh_neus_decimate.ply',
            "poi": 'mesh_poi_clean.ply',
            "mcube": 'mesh_mc_30K.ply',
            "softbody": 'mesh_35K.ply'
        }
        mesh_path = os.path.join(os.path.dirname(args.model_path), mesh_type_dict[args.mesh_type])
        # mesh_path = os.path.join(os.path.dirname(args.model_path), 'mesh_neus_decimate.ply')
        # mesh_path = os.path.join(os.path.dirname(args.model_path), 'mesh_poi_clean.ply')
        # mesh_path = os.path.join(os.path.dirname(args.model_path), 'mesh_mc.ply')
        # mesh_path = os.path.join(os.path.dirname(args.model_path), 'mesh_mc_30K.ply')
        # mesh_path = os.path.join(os.path.dirname(args.model_path), 'mesh_screened_poisson.ply')
        # mesh_path = os.path.join(os.path.dirname(args.model_path), 'mesh_35K.ply')
        # mesh_path = os.path.join(os.path.dirname(args.model_path), 'mesh_sugar.ply')
        
        gaussians.load_mesh(mesh_path)

        densification_bind = False

        n_gaussians_per_surface_triangle = args.N_tri # 1 # 3
        print("args.bary_field", args.bary_field)
        gaussians.adaptive_bind(
                # use_offset=True, # False, free points or (trangle points + free offset)
                use_offset=False, # False, free points or (trangle points + free offset)
                # fix_gaus=True, # False, fixed points, i.e. trangle points
                fix_gaus=False, # False, fixed points, i.e. trangle points
                anchor_field=True, # True, 
                # anchor_field=False, # True, 
                # bary_field=True,
                bary_field=args.bary_field, # False, 
                HP=args.HP,
                use_ckpt=args.checkpoint,
                n_gaussians_per_surface_triangle=n_gaussians_per_surface_triangle
            )

    gaussians.training_setup(opt)

    """
    Setup PBR components
    """
    pbr_kwargs = dict()
    
    """ Prepare render function and bg"""
    render_fn = render_fn_dict[args.type]
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    """ Training """
    viewpoint_stack = None
    ema_dict_for_log = defaultdict(int)
    progress_bar = tqdm(range(first_iter + 1, opt.iterations + 1), desc="Training progress",
                        initial=first_iter, total=opt.iterations)

    torch.cuda.synchronize() 
    all_training_time_start = time.time()
    for iteration in progress_bar:
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        loss = 0
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == args.debug_from:
            pipe.debug = True

        pbr_kwargs["iteration"] = iteration - first_iter
        render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background,
                               opt=opt, is_training=True, dict_params=pbr_kwargs)

        viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        tb_dict = render_pkg["tb_dict"]
        loss += render_pkg["loss"]
        loss.backward()

        with torch.no_grad():
            if pipe.save_training_vis:
                save_training_vis(viewpoint_cam, gaussians, background, render_fn,
                                  pipe, opt, first_iter, iteration, pbr_kwargs)
            # Progress bar
            pbar_dict = {"num": gaussians.get_xyz.shape[0]}
            for k in tb_dict:
                if k in ["psnr", "psnr_pbr"]:
                    ema_dict_for_log[k] = 0.4 * tb_dict[k] + 0.6 * ema_dict_for_log[k]
                    pbar_dict[k] = f"{ema_dict_for_log[k]:.{7}f}"
            # if iteration % 10 == 0:
            progress_bar.set_postfix(pbar_dict)

            # Log and save
            # training_report(tb_writer, iteration, tb_dict,
            #                 scene, render_fn, pipe=pipe,
            #                 bg_color=background, dict_params=pbr_kwargs)
            training_report_fps(tb_writer, iteration, tb_dict,
                            scene, render_fn, pipe=pipe,
                            bg_color=background, dict_params=pbr_kwargs)

            # densification
            # if iteration < opt.densify_until_iter:
            if (iteration < opt.densify_until_iter) and densification_bind:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    min_opacity = 0.005 # 0.005
                    gaussians.densify_and_prune(opt.densify_grad_threshold, min_opacity, scene.cameras_extent, size_threshold,
                                                opt.densify_grad_normal_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            gaussians.step()
            for component in pbr_kwargs.values():
                try:
                    component.step()
                except:
                    pass

            # save checkpoints, point cloud .ply
            if iteration % args.save_interval == 0 or iteration == args.iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration % args.checkpoint_interval == 0 or iteration == args.iterations:
                
                torch.save((gaussians.capture(), iteration),
                           os.path.join(scene.model_path, "chkpnt" + str(iteration) + ".pth"))

                for com_name, component in pbr_kwargs.items():
                    try:
                        torch.save((component.capture(), iteration),
                                   os.path.join(scene.model_path, f"{com_name}_chkpnt" + str(iteration) + ".pth"))
                        print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    except:
                        pass

                    print("[ITER {}] Saving {} Checkpoint".format(iteration, com_name))
    
    torch.cuda.synchronize() 
    all_training_time_end = time.time()
    time_cost = all_training_time_end - all_training_time_start
    print("\n[Training Cost {} mins {:.4f} secs] ".format(time_cost // 60, time_cost % 60))

    if args.type in ["render"]:
        print("Beginning M-Cube mesh extraction! ")
        mesh_path = os.path.join(os.path.dirname(args.model_path), 'mesh_mc_30K.ply')
        # marching cube surface extraction
        # mesh = gaussians.extract_mesh(mesh_path, density_thresh=0.1, resolution=128, num_blocks=16, relax_ratio=1.5)
        mesh = gaussians.extract_mesh(mesh_path, density_thresh=0.0001, resolution=256, num_blocks=16, relax_ratio=.5) # 0.005 works, 0.0001 for ficus
        # mesh = gaussians.extract_mesh(mesh_path, density_thresh=0.05, resolution=128, num_blocks=16, relax_ratio=1.5)
        mesh.write_ply(mesh_path)
        print("Finished M-Cube mesh extraction! ")
        
        def remove_outside_triangles(vertices, triangles, box):
            box_min = box[0]
            box_max = box[1]
            
            triangle_centers = (vertices[triangles[:, 0]] + vertices[triangles[:, 1]] + vertices[triangles[:, 2]]) / 3
            inside_triangles = torch.logical_and(
                torch.all(triangle_centers >= box_min, dim=1),
                torch.all(triangle_centers <= box_max, dim=1)
            )

            inside_triangle_indices = torch.nonzero(inside_triangles).squeeze()
            inside_triangles = triangles[inside_triangle_indices]

            return inside_triangles


        print("Beginning screened_poisson mesh extraction! ")
        import pymeshlab
        valid_mask = (gaussians.get_opacity > 0.6)[..., 0]
        verts = gaussians.get_xyz[valid_mask].detach().cpu().numpy()
        normals = gaussians.get_normal[valid_mask].detach().cpu().numpy()
        # verts = gaussians.get_xyz.detach().cpu().numpy()
        # normals = gaussians.get_normal.detach().cpu().numpy()
        m = pymeshlab.Mesh(verts, v_normals_matrix=normals)
        ms = pymeshlab.MeshSet()
        ms.add_mesh(m, "cube_mesh")
        ms.generate_surface_reconstruction_screened_poisson(depth=8, threads=16)
        mesh_path = os.path.join(os.path.dirname(args.model_path), 'mesh_screened_poisson.ply')
        # ms.save_current_mesh("screened_poisson.ply")
        ms.save_current_mesh(mesh_path)
        print("Finished screened_poisson mesh extraction! ")

        import trimesh
        dilation = 0.0
        box = torch.cat([gaussians.get_xyz.amin(dim=0)[None]-dilation, gaussians.get_xyz.amax(dim=0)[None]+dilation], dim=0)
        mesh = trimesh.load(mesh_path, force='mesh', skip_material=True, process=False)
        
        DTU = True if "DTU" in args.model_path else False
        if DTU:
            vertices, triangles = clean_mesh(mesh.vertices, mesh.faces, remesh=True, remesh_size=0.015)
            mesh.vertices = vertices
            mesh.faces = triangles

        vertices = torch.tensor(mesh.vertices).cuda()
        triangles = torch.tensor(mesh.faces).cuda()
        
        # from mesh import Mesh
        # inside_triangles = remove_outside_triangles(vertices, triangles, box)
        # mesh_poi_clean = Mesh(v=vertices, f=inside_triangles, device='cuda')
        # mesh_path = os.path.join(os.path.dirname(args.model_path), 'mesh_poi_clean.ply')
        # mesh_poi_clean.write_ply(mesh_path)

        from knn_cuda import KNN
        knn = KNN(k=1, transpose_mode=True)
        ref = gaussians.get_xyz[None]
        triangle_centers = (vertices[triangles[:, 0]] + vertices[triangles[:, 1]] + vertices[triangles[:, 2]]) / 3
        query = triangle_centers[None]
        dists = []
        idxs = []
        batch_size = 1024 * 8
        for i in range(0, query.shape[1], batch_size):
            dist, idx = knn(ref, query[:, i:i+batch_size])
            dists.append(dist[0])
            idxs.append(idx[0])
        dists = torch.cat(dists,0)
        idxs = torch.cat(idxs,0)

        thresh = 0.05 if not DTU else 0.2 # 0.05 for nerf synthetic, 0.2 for DTU
        valid_tri = triangles[(dists < thresh)[:, 0]]

        from mesh import Mesh
        mesh_poi_clean = Mesh(v=vertices, f=valid_tri, device='cuda')
        mesh_path = os.path.join(os.path.dirname(args.model_path), 'mesh_poi_clean.ply')
        mesh_poi_clean.write_ply(mesh_path)

    if args.type in ["normal"]:
        # mesh_path = os.path.join(args.model_path, "eval", "mesh_normal_refine_2.ply")
        mesh_path = os.path.join(os.path.dirname(args.model_path), "mesh_normal_refine.ply")
        from mesh import Mesh
        mesh = Mesh(v=(gaussians.vertices + gaussians._vertices_offsets), f=gaussians.triangles, device='cuda')
        mesh.write_ply(mesh_path)

    # if dataset.eval:
    if dataset.eval and (args.type not in ["normal"]):
        if args.eval_dynamic:
            eval_dynamic_render(scene, gaussians, render_fn, pipe, background, opt, pbr_kwargs, args.dyn_mesh_dir)
        else:
            eval_render(scene, gaussians, render_fn, pipe, background, opt, pbr_kwargs)
    
    if "tnt" in dataset.model_path: # when tnt, eval is False
        if args.eval_dynamic:
            eval_dynamic_render(scene, gaussians, render_fn, pipe, background, opt, pbr_kwargs, args.dyn_mesh_dir)
        else:
            eval_render(scene, gaussians, render_fn, pipe, background, opt, pbr_kwargs)


def training_report_fps(tb_writer, iteration, tb_dict, scene: Scene, renderFunc, pipe,
                    bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None,
                    opt: OptimizationParams = None, is_training=False, **kwargs):

    # Report test and samples of training set
    if iteration % args.test_interval == 0:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': scene.getTrainCameras()})

        for config in validation_configs:
            
            scene.gaussians.set_precompute_global()
            torch.cuda.synchronize() 
            time_start = time.time()

            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                psnr_pbr_test = 0.0
                for idx, viewpoint in enumerate(
                        tqdm(config['cameras'], desc="Evaluating " + config['name'], leave=False)):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, pipe, bg_color,
                                            scaling_modifier, override_color, opt, is_training,
                                            **kwargs)

                    image = render_pkg["render"]
                    gt_image = viewpoint.original_image.cuda()
                    l1_test += F.l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                psnr_pbr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                
                torch.cuda.synchronize() 
                time_end = time.time() - time_start
                fps = len(config['cameras']) / time_end
                scene.gaussians.unset_precompute_global()

                
                # print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} FPS {}".format(iteration, config['name'], l1_test, psnr_test, fps))
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} FPS {} Total_Num {} ".format(iteration, config['name'], l1_test,
                                                                                    psnr_test, fps, scene.gaussians.get_xyz.shape[0]))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr_pbr', psnr_pbr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - rendering FPS', fps, iteration)
                if iteration == args.iterations:
                    with open(os.path.join(args.model_path, config['name'] + "_loss.txt"), 'w') as f:
                        f.write("L1 {} PSNR {} PSNR_PBR {}".format(l1_test, psnr_test, psnr_pbr_test))

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


def training_report(tb_writer, iteration, tb_dict, scene: Scene, renderFunc, pipe,
                    bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None,
                    opt: OptimizationParams = None, is_training=False, **kwargs):
    if tb_writer:
        for key in tb_dict:
            tb_writer.add_scalar(f'train_loss_patches/{key}', tb_dict[key], iteration)

    # Report test and samples of training set
    if iteration % args.test_interval == 0:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                psnr_pbr_test = 0.0
                for idx, viewpoint in enumerate(
                        tqdm(config['cameras'], desc="Evaluating " + config['name'], leave=False)):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, pipe, bg_color,
                                            scaling_modifier, override_color, opt, is_training,
                                            **kwargs)

                    image = render_pkg["render"]
                    gt_image = viewpoint.original_image.cuda()

                    opacity = torch.clamp(render_pkg["opacity"], 0.0, 1.0)
                    depth = render_pkg["depth"]
                    depth = (depth - depth.min()) / (depth.max() - depth.min())
                    normal = torch.clamp(
                        render_pkg.get("normal", torch.zeros_like(image)) / 2 + 0.5 * opacity, 0.0, 1.0)

                    # BRDF
                    base_color = torch.clamp(render_pkg.get("base_color", torch.zeros_like(image)), 0.0, 1.0)
                    roughness = torch.clamp(render_pkg.get("roughness", torch.zeros_like(depth)), 0.0, 1.0)
                    metallic = torch.clamp(render_pkg.get("metallic", torch.zeros_like(depth)), 0.0, 1.0)
                    image_pbr = render_pkg.get("pbr", torch.zeros_like(image))

                    # For HDR images
                    if render_pkg["hdr"]:
                        # print("HDR detected!")
                        image = hdr2ldr(image)
                        image_pbr = hdr2ldr(image_pbr)
                        gt_image = hdr2ldr(gt_image)
                    else:
                        image = torch.clamp(image, 0.0, 1.0)
                        image_pbr = torch.clamp(image_pbr, 0.0, 1.0)
                        gt_image = torch.clamp(gt_image, 0.0, 1.0)

                    grid = torchvision.utils.make_grid(
                        torch.stack([image, image_pbr, gt_image,
                                     opacity.repeat(3, 1, 1), depth.repeat(3, 1, 1), normal,
                                     base_color, roughness.repeat(3, 1, 1), metallic.repeat(3, 1, 1)], dim=0), nrow=3)

                    if tb_writer and (idx < 2):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             grid[None], global_step=iteration)

                    l1_test += F.l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    psnr_pbr_test += psnr(image_pbr, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                psnr_pbr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} PSNR_PBR {}".format(iteration, config['name'], l1_test,
                                                                                    psnr_test, psnr_pbr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr_pbr', psnr_pbr_test, iteration)
                if iteration == args.iterations:
                    with open(os.path.join(args.model_path, config['name'] + "_loss.txt"), 'w') as f:
                        f.write("L1 {} PSNR {} PSNR_PBR {}".format(l1_test, psnr_test, psnr_pbr_test))

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


def save_training_vis(viewpoint_cam, gaussians, background, render_fn, pipe, opt, first_iter, iteration, pbr_kwargs):
    os.makedirs(os.path.join(args.model_path, "visualize"), exist_ok=True)
    with torch.no_grad():
        if iteration % pipe.save_training_vis_iteration == 0 or iteration == first_iter + 1:
            render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background,
                                   opt=opt, is_training=False, dict_params=pbr_kwargs)

            visualization_list = [
                render_pkg["render"],
                visualize_depth(render_pkg["depth"]),
                render_pkg["opacity"].repeat(3, 1, 1),
                render_pkg["normal"] * 0.5 + 0.5,
                viewpoint_cam.original_image.cuda(),
                visualize_depth(viewpoint_cam.depth.cuda()),
                viewpoint_cam.normal.cuda() * 0.5 + 0.5,
                render_pkg["pseudo_normal"] * 0.5 + 0.5,
            ]

            grid = torch.stack(visualization_list, dim=0)
            grid = make_grid(grid, nrow=4)
            save_image(grid, os.path.join(args.model_path, "visualize", f"{iteration:06d}.png"))


def eval_render(scene, gaussians, render_fn, pipe, background, opt, pbr_kwargs):
    psnr_test = 0.0
    ssim_test = 0.0
    lpips_test = 0.0
    test_cameras = scene.getTestCameras()
    # test_cameras = scene.getTrainCameras()
    os.makedirs(os.path.join(args.model_path, 'eval', 'render'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'eval', 'gt'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'eval', 'normal'), exist_ok=True)


    progress_bar = tqdm(range(0, len(test_cameras)), desc="Evaluating",
                        initial=0, total=len(test_cameras))

    with torch.no_grad():
        # gaussians.set_precompute_global()
        for idx in progress_bar:
            viewpoint = test_cameras[idx]
            results = render_fn(viewpoint, gaussians, pipe, background, opt=opt, is_training=False,
                                dict_params=pbr_kwargs)
            if gaussians.use_pbr:
                image = results["pbr"]
            else:
                image = results["render"]

            image = torch.clamp(image, 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            psnr_test += psnr(image, gt_image).mean().double()
            ssim_test += ssim(image, gt_image).mean().double()
            lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()

            image = torch.cat([image, results['opacity']], dim=0)

            save_image(image, os.path.join(args.model_path, 'eval', "render", f"{viewpoint.image_name}.png"))
            # save_image(gt_image, os.path.join(args.model_path, 'eval', "gt", f"{viewpoint.image_name}.png"))
            # save_image(results["normal"] * 0.5 + 0.5, os.path.join(args.model_path, 'eval', "normal", f"{viewpoint.image_name}.png"))
            
        # gaussians.unset_precompute_global()
    psnr_test /= len(test_cameras)
    ssim_test /= len(test_cameras)
    lpips_test /= len(test_cameras)
    
    # with open(os.path.join(args.model_path, 'eval', "eval_no_lpips.txt"), "w") as f:
    with open(os.path.join(args.model_path, 'eval', "eval.txt"), "w") as f:
        f.write(f"psnr: {psnr_test}\n")
        f.write(f"ssim: {ssim_test}\n")
        f.write(f"lpips: {lpips_test}\n")
    print("\n[ITER {}] Evaluating {}: PSNR {} SSIM {} LPIPS {}".format(args.iterations, "test", psnr_test, ssim_test,
                                                                       lpips_test))


def eval_dynamic_render(scene, gaussians, render_fn, pipe, background, opt, pbr_kwargs, dyn_mesh_dir=None):
    test_cameras = scene.getTestCameras()
    # test_cameras = scene.getTrainCameras()
    
    # CAM_IDX_list = [30] # 0, 30, 6
    # CAM_IDX_list = list(range(len(test_cameras)))[0:200:10]
    CAM_IDX_list = [0, 6, 30, 50, 90, 125] # 0, 30, 6
    dynamic_render_dir = 'dynamic_render'

    if dyn_mesh_dir==None:
        mesh_folder = "output/NeRF_Syn/lego/final_our_deform_stretch"
    else:
        mesh_folder = dyn_mesh_dir
    # mesh_folder = "output/NeRF_Syn/lego/final_our_deform_bend"
    mesh_files = os.listdir(mesh_folder)
    mesh_files = [x for x in mesh_files if (".obj" in x) or (".ply" in x)]

    # mesh_files = [x for x in mesh_files if ("r_6" in x)]

    # save_dir = os.path.join(args.model_path, 'eval', dynamic_render_dir)
    save_dir = os.path.join(args.model_path, 'eval', os.path.basename(mesh_folder))
    os.makedirs(save_dir, exist_ok=True)

    print("Dynamic Rendering in Progress. ")
    with torch.no_grad():
        for mesh_idx, mesh_name in enumerate(mesh_files):
            mesh_path = os.path.join(mesh_folder, mesh_name)
            gaussians.load_mesh(mesh_path)
            # print("args.bary_field", args.bary_field)
            n_gaussians_per_surface_triangle = args.N_tri # 1 # 3
            gaussians.adaptive_bind(
                    # use_offset=True, # False, free points or (trangle points + free offset)
                    use_offset=False, # False, free points or (trangle points + free offset)
                    # fix_gaus=True, # False, fixed points, i.e. trangle points
                    fix_gaus=False, # False, fixed points, i.e. trangle points
                    anchor_field=True, # True, 
                    # anchor_field=False, # True, 
                    bary_field=args.bary_field,
                    HP=args.HP,
                    use_ckpt=True,
                    n_gaussians_per_surface_triangle=n_gaussians_per_surface_triangle
                )
            for CAM_IDX in CAM_IDX_list:
                print(f"OBJ_Name: {mesh_name}, CAM_IDX: {CAM_IDX}")
                viewpoint = test_cameras[CAM_IDX]
                results = render_fn(viewpoint, gaussians, pipe, background, opt=opt, is_training=False,
                                    dict_params=pbr_kwargs)
                image = results["render"]
                image = torch.clamp(image, 0.0, 1.0)
                image = torch.cat([image, results['opacity']], dim=0)
                
                mesh_name = mesh_name.split(".")[0].split("_")[-1]

                os.makedirs(save_dir, exist_ok=True)
                save_image(image, os.path.join(save_dir, f"{viewpoint.image_name}_{int(mesh_name):04d}.png"))

                # os.makedirs(save_dir+"/normal_bg", exist_ok=True)
                # save_image(results["mesh_normal"]* 0.5 + 0.5, os.path.join(save_dir, "normal_bg", f"{viewpoint.image_name}_{int(mesh_name):04d}.png"))

    print("Dynamic Rendering Done. ")

    
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--gui', action='store_true', default=False, help="use gui")
    
    parser.add_argument('-t', '--type', choices=['render', 'normal', 'neilf', 'bind', 'bind_mesh_adapt'], default='render')
    parser.add_argument('--eval_dynamic', action='store_true', default=False)
    parser.add_argument('--dyn_mesh_dir', type=str, default=None, help="the dynamic mesh sequence folder")
    parser.add_argument('-mt', '--mesh_type', choices=['neus', 'poi', 'mcube', 'softbody'], default='neus')
    parser.add_argument('--bary_field', action='store_true', default=False)
    parser.add_argument("--HP", type=int, default=10, help="hyper params")
    parser.add_argument("--N_tri", type=int, default=3, help="the number of 3DGS attached on each triangle")

    # parser.add_argument('--bary_field', type=bool, default=False)
    
    parser.add_argument("--test_interval", type=int, default=2500)
    parser.add_argument("--save_interval", type=int, default=10000)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_interval", type=int, default=10000)
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    print(f"Current model path: {args.model_path}")
    print(f"Current rendering type:  {args.type}")
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    is_pbr = args.type in ['neilf']
    training(lp.extract(args), op.extract(args), pp.extract(args), is_pbr=is_pbr)

    # All done
    print("\nTraining complete.")
