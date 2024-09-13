import torch
import numpy as np

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from gs_utils.visualize.viser_renderer import render_point_cloud

if __name__ == '__main__':
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 600

    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    # load_images can take a list of images or a directory
    images = load_images(['/home/jiahuih/workspace/yiflu-workspace/SCube/splat_output_waymo_wds/wdb:nvidia-toronto_xcube-scene-scube_waymo_wds_benchmark_pure3dunet_res1024_view3_rgb_pano_full_w_perceptual/test_starting_at_100_gt_sep/0/108_0_gt.jpg', 
                          '/home/jiahuih/workspace/yiflu-workspace/SCube/splat_output_waymo_wds/wdb:nvidia-toronto_xcube-scene-scube_waymo_wds_benchmark_pure3dunet_res1024_view3_rgb_pano_full_w_perceptual/test_starting_at_100_gt_sep/0/115_0_gt.jpg',
                          '/home/jiahuih/workspace/yiflu-workspace/SCube/splat_output_waymo_wds/wdb:nvidia-toronto_xcube-scene-scube_waymo_wds_benchmark_pure3dunet_res1024_view3_rgb_pano_full_w_perceptual/test_starting_at_100_gt_sep/0/123_0_gt.jpg',
                          '/home/jiahuih/workspace/yiflu-workspace/SCube/splat_output_waymo_wds/wdb:nvidia-toronto_xcube-scene-scube_waymo_wds_benchmark_pure3dunet_res1024_view3_rgb_pano_full_w_perceptual/test_starting_at_100_gt_sep/0/108_1_gt.jpg', 
                          '/home/jiahuih/workspace/yiflu-workspace/SCube/splat_output_waymo_wds/wdb:nvidia-toronto_xcube-scene-scube_waymo_wds_benchmark_pure3dunet_res1024_view3_rgb_pano_full_w_perceptual/test_starting_at_100_gt_sep/0/115_1_gt.jpg',
                          '/home/jiahuih/workspace/yiflu-workspace/SCube/splat_output_waymo_wds/wdb:nvidia-toronto_xcube-scene-scube_waymo_wds_benchmark_pure3dunet_res1024_view3_rgb_pano_full_w_perceptual/test_starting_at_100_gt_sep/0/123_1_gt.jpg',
                          '/home/jiahuih/workspace/yiflu-workspace/SCube/splat_output_waymo_wds/wdb:nvidia-toronto_xcube-scene-scube_waymo_wds_benchmark_pure3dunet_res1024_view3_rgb_pano_full_w_perceptual/test_starting_at_100_gt_sep/0/108_2_gt.jpg', 
                          '/home/jiahuih/workspace/yiflu-workspace/SCube/splat_output_waymo_wds/wdb:nvidia-toronto_xcube-scene-scube_waymo_wds_benchmark_pure3dunet_res1024_view3_rgb_pano_full_w_perceptual/test_starting_at_100_gt_sep/0/115_2_gt.jpg',
                          '/home/jiahuih/workspace/yiflu-workspace/SCube/splat_output_waymo_wds/wdb:nvidia-toronto_xcube-scene-scube_waymo_wds_benchmark_pure3dunet_res1024_view3_rgb_pano_full_w_perceptual/test_starting_at_100_gt_sep/0/123_2_gt.jpg',
                          ], size=512)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    # here, view1, pred1, view2, pred2 are dicts of lists of len(2)
    #  -> because we symmetrize we have (im1, im2) and (im2, im1) pairs
    # in each view you have:
    # an integer image identifier: view1['idx'] and view2['idx']
    # the img: view1['img'] and view2['img']
    # the image shape: view1['true_shape'] and view2['true_shape']
    # an instance string output by the dataloader: view1['instance'] and view2['instance']
    # pred1 and pred2 contains the confidence values: pred1['conf'] and pred2['conf']
    # pred1 contains 3D points for view1['img'] in view1['img'] space: pred1['pts3d']
    # pred2 contains 3D points for view2['img'] in view1['img'] space: pred2['pts3d_in_other_view']

    # next we'll use the global_aligner to align the predictions
    # depending on your task, you may be fine with the raw output and not need it
    # with only two input images, you could use GlobalAlignerMode.PairViewer: it would just convert the output
    # if using GlobalAlignerMode.PairViewer, no need to run compute_global_alignment
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    preset_focal = True
    if preset_focal:
        intrinsic = [2055.55614936, 2055.55614936,  939.65746989,  641.07218219, 1920, 1280] # waymo front camera
        fx, fy, cx, cy, W, H = intrinsic
        resized_W = 512
        resized_fx = fx * resized_W / W
        resized_fy = fy * resized_W / W
        resized_cx = cx * resized_W / W
        resized_cy = cy * resized_W / W

        scene.preset_focal([resized_fx]*len(images))
        scene.preset_principal_point([[resized_cx, resized_cy]]*len(images))

    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

    # retrieve useful values from scene:
    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d() # len is 3
    confidence_masks = scene.get_masks()

    print('focal lengths:', focals)
    breakpoint()

    # visualize reconstruction
    # scene.show()
    pts3d_xyz_merged = torch.cat([pts.flatten(0,1) for pts in pts3d], dim=0).detach().cpu().numpy() * 100
    pts3d_color_merged = np.concatenate([img.reshape(-1,3) for img in imgs], axis=0)
    confidence_masks_merged = torch.cat(confidence_masks, dim=0).detach().cpu().numpy().flatten()
    render_point_cloud(pts3d_xyz_merged[confidence_masks_merged], pts3d_color_merged[confidence_masks_merged])
