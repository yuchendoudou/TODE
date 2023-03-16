import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from inference import Inferencer
import matplotlib.pyplot as plt
import os
from utils.data_preparation import process_data, exr_loader


DILATION_KERNEL = np.array([[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0]]).astype(np.uint8)


def dropout_random_ellipses_4corruptmask(mask, noise_params):
    """ Randomly drop a few ellipses in the image for robustness.
        This is adapted from the DexNet 2.0 codebase.
        Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py

        @param depth_img: a [H x W] set of depth z values
    """
    dropout_mask = mask.copy()

    # Sample number of ellipses to dropout
    num_ellipses_to_dropout = np.random.poisson(noise_params['ellipse_dropout_mean'])

    # Sample ellipse centers
    zero_pixel_indices = np.array(np.where(dropout_mask == 0)).T # Shape: [#nonzero_pixels x 2]
    dropout_centers_indices = np.random.choice(zero_pixel_indices.shape[0], size=num_ellipses_to_dropout)
    dropout_centers = zero_pixel_indices[dropout_centers_indices, :] # Shape: [num_ellipses_to_dropout x 2]

    # Sample ellipse radii and angles
    x_radii = np.random.gamma(noise_params['ellipse_gamma_shape'], noise_params['ellipse_gamma_scale'], size=num_ellipses_to_dropout)
    y_radii = np.random.gamma(noise_params['ellipse_gamma_shape'], noise_params['ellipse_gamma_scale'], size=num_ellipses_to_dropout)
    angles = np.random.randint(0, 360, size=num_ellipses_to_dropout)

    # Dropout ellipses
    for i in range(num_ellipses_to_dropout):
        center = dropout_centers[i, :]
        x_radius = np.round(x_radii[i]).astype(int)
        y_radius = np.round(y_radii[i]).astype(int)
        angle = angles[i]

        # get ellipse mask
        tmp_mask = np.zeros_like(dropout_mask, dtype=np.uint8)
        tmp_mask = cv2.ellipse(tmp_mask, tuple(center[::-1]), (x_radius, y_radius), angle=angle, startAngle=0, endAngle=360, color=1, thickness=-1)
        # update depth and corrupt mask
        dropout_mask[tmp_mask == 1] = 1

    return dropout_mask


def handle_depth(depth, depth_gt, depth_gt_mask):
    depth[depth_gt_mask==1] = 0
    depth_gt_mask_uint8 = np.where(depth < 0.000000001, 255, 0).astype(np.uint8)
    depth_gt_mask_uint8[depth_gt_mask_uint8 != 0] = 1
    depth_uint8 = depth.copy() / depth.max() * 255
    depth_uint8 = np.array(depth_uint8, dtype=np.uint8)
    depth_uint8 = cv2.inpaint(depth_uint8, depth_gt_mask_uint8, 5, cv2.INPAINT_NS)
    depth_uint8 = np.array(depth_uint8, dtype=np.float32) / 255 * depth.max()


    # depth_noise = np.random.randn(*(depth.shape))
    # depth_noise = depth_noise / np.abs(depth_noise).max() * depth.max() / 1000.0


    # depth[depth_gt_mask==1] = depth_uint8[depth_gt_mask == 1] + depth_noise[depth_gt_mask == 1]

    
    mask_pixel_indices1 = np.array(np.where(depth_gt_mask == 1)).T # Shape: [#nonzero_pixels x 2]
    dropout_size = int(mask_pixel_indices1.shape[0] * 0.003)
    dropout_centers_indices = np.random.choice(mask_pixel_indices1.shape[0], size=dropout_size)
    dropout_centers = mask_pixel_indices1[dropout_centers_indices, :] 
    x_radii = np.random.gamma(3.0, 2.0, size=dropout_size)
    y_radii = np.random.gamma(3.0, 2.0, size=dropout_size)
    angles = np.random.randint(0, 360, size=dropout_size)

    result_mask = np.zeros_like(depth_gt_mask, dtype=np.uint8)

    for i in range(dropout_size // 2):
        center = dropout_centers[i, :]
        x_radius = np.round(x_radii[i]).astype(int)
        y_radius = np.round(y_radii[i]).astype(int)
        angle = angles[i]

        # get ellipse mask
        tmp_mask = np.zeros_like(depth_gt_mask, dtype=np.uint8)
        tmp_mask = cv2.ellipse(tmp_mask, tuple(center[::-1]), (x_radius, y_radius), angle=angle, startAngle=0, endAngle=360, color=1, thickness=-1)
        # update depth and corrupt mask
        result_mask[tmp_mask == 1] = 1

    mask = np.logical_and(result_mask, depth_gt_mask_uint8)
    depth[mask==1] = depth_uint8[mask == 1]


    result_mask = np.zeros_like(depth_gt_mask, dtype=np.uint8)

    for i in range(dropout_size - dropout_size // 2):
        center = dropout_centers[i + dropout_size // 2, :]
        x_radius = np.round(x_radii[i + dropout_size // 2]).astype(int)
        y_radius = np.round(y_radii[i + dropout_size // 2]).astype(int)
        angle = angles[i + dropout_size // 2]

        # get ellipse mask
        tmp_mask = np.zeros_like(depth_gt_mask, dtype=np.uint8)
        tmp_mask = cv2.ellipse(tmp_mask, tuple(center[::-1]), (x_radius, y_radius), angle=angle, startAngle=0, endAngle=360, color=1, thickness=-1)
        # update depth and corrupt mask
        result_mask[tmp_mask == 1] = 1

    mask = np.logical_and(result_mask, depth_gt_mask_uint8)
    depth[mask==1] = depth_gt[mask==1]

    return depth



# def handle_depth(depth, depth_gt_mask):
#     depth[depth_gt_mask==1] = 0
#     depth_gt_mask_uint8 = np.where(depth < 0.000000001, 255, 0).astype(np.uint8)
#     depth_gt_mask_uint8[depth_gt_mask_uint8 != 0] = 1
#     depth_uint8 = depth.copy() / depth.max() * 255
#     depth_uint8 = np.array(depth_uint8, dtype=np.uint8)
#     depth_uint8 = cv2.inpaint(depth_uint8, depth_gt_mask_uint8, 3, cv2.INPAINT_NS)
#     depth_uint8 = np.array(depth_uint8, dtype=np.float32) / 255 * depth.max()

#     depth[depth_gt_mask==1] = depth_uint8[depth_gt_mask == 1]

    

    # mask_pixel_indices = np.array(np.where(depth_gt_mask == 1)).T # Shape: [#nonzero_pixels x 2]
    # dropout_size = int(mask_pixel_indices.shape[0] * 0.005)
    # dropout_centers_indices = np.random.choice(mask_pixel_indices.shape[0], size=dropout_size)
    # dropout_centers = mask_pixel_indices[dropout_centers_indices, :] 
    # x_radii = np.random.gamma(3.0, 2.0, size=dropout_size)
    # y_radii = np.random.gamma(3.0, 2.0, size=dropout_size)
    # angles = np.random.randint(0, 360, size=dropout_size)

    # result_mask = np.zeros_like(depth_gt_mask, dtype=np.uint8)

    # for i in range(dropout_size):
    #     center = dropout_centers[i, :]
    #     x_radius = np.round(x_radii[i]).astype(int)
    #     y_radius = np.round(y_radii[i]).astype(int)
    #     angle = angles[i]

    #     # get ellipse mask
    #     tmp_mask = np.zeros_like(depth_gt_mask, dtype=np.uint8)
    #     tmp_mask = cv2.ellipse(tmp_mask, tuple(center[::-1]), (x_radius, y_radius), angle=angle, startAngle=0, endAngle=360, color=1, thickness=-1)
    #     # update depth and corrupt mask
    #     result_mask[tmp_mask == 1] = 1
    
    # mask = np.logical_and(np.logical_not(result_mask), depth_gt_mask_uint8)
    # depth[mask==1] = 0
    return depth

def handle_depth2(depth, depth_gt_mask):
    rgb_mask = np.where(depth < 0.000000001, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))

    neg_zero_mask_dilated = cv2.dilate(rgb_mask, kernel = kernel)
    depth[neg_zero_mask_dilated == 255] = 0.0
    contours,hierarch=cv2.findContours(neg_zero_mask_dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i]) #计算轮廓所占面积
        if area < 320 * 240 // 16:                         #将area小于阈值区域填充背景色，由于OpenCV读出的是BGR值
            cv2.drawContours(depth,[contours[i]],-1, (0), thickness=-1)     #原始图片背景BGR值(84,1,68)
    
    depth[neg_zero_mask_dilated == 255] = 0.0
    return depth

# def convertPNG(pngfile,outdir):
#     # READ THE DEPTH
#     im_depth = cv2.imread(pngfile)
#     #apply colormap on deoth image(image must be converted to 8-bit per pixel first)
#     im_color=cv2.applyColorMap(cv2.convertScaleAbs(im_depth,alpha=15),cv2.COLORMAP_JET)
#     #convert to mat png
#     im=Image.fromarray(im_color)
#     #save image
#     im.save(os.path.join(outdir,os.path.basename(pngfile)))


def draw_point_cloud(color, depth, camera_intrinsics, use_mask = False, use_inpainting = True, scale = 1000.0, inpainting_radius = 5, fault_depth_limit = 0.2, epsilon = 0.01):
    """
    Given the depth image, return the point cloud in open3d format.
    The code is adapted from [graspnet.py] in the [graspnetAPI] repository.
    """
    d = depth.copy()
    c = color.copy() / 255.0
    
    if use_inpainting:
        fault_mask = (d < fault_depth_limit * scale)
        d[fault_mask] = 0
        inpainting_mask = (np.abs(d) < epsilon * scale).astype(np.uint8)  
        d = cv2.inpaint(d, inpainting_mask, inpainting_radius, cv2.INPAINT_NS)

    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

    xmap, ymap = np.arange(d.shape[1]), np.arange(d.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)

    points_z = d / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z
    points = np.stack([points_x, points_y, points_z], axis = -1)

    if use_mask:
        mask = (points_z > 0)
        points = points[mask]
        c = c[mask]
    else:
        points = points.reshape((-1, 3))
        c = c.reshape((-1, 3))
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(c)
    return cloud

inferencer = Inferencer()

# rgb_mask = np.array(Image.open('/home/apollo/TransCG/cleargrasp/cleargrasp-dataset-train/cup-with-waves-train/segmentation-masks/000000000-segmentation-mask.png'), dtype = np.float32)
# rgb = np.array(Image.open('/home/apollo/TransCG/cleargrasp/cleargrasp-dataset-train/cup-with-waves-train/rgb-imgs/000000000-rgb.jpg'), dtype = np.float32)
# depth = exr_loader('/home/apollo/TransCG/cleargrasp/cleargrasp-dataset-train/cup-with-waves-train/depth-imgs-rectified/000000000-depth-rectified.exr', ndim = 1, ndim_representation = ['R'])
# depth_gt = exr_loader('/home/apollo/TransCG/cleargrasp/cleargrasp-dataset-train/cup-with-waves-train/depth-imgs-rectified/000000000-depth-rectified.exr', ndim = 1, ndim_representation = ['R'])

# rgb_mask = np.array(Image.open('/home/apollo/TransCG/cleargrasp/cleargrasp-dataset-test-val/real-test/d415/000000082-mask.png'), dtype = np.float32)
# rgb = np.array(Image.open('/home/apollo/TransCG/cleargrasp/cleargrasp-dataset-test-val/real-test/d415/000000082-transparent-rgb-img.jpg'), dtype = np.float32)
# depth = exr_loader('/home/apollo/TransCG/cleargrasp/cleargrasp-dataset-test-val/real-test/d415/000000082-transparent-depth-img.exr', ndim = 1, ndim_representation = ['R'])
# depth_gt = exr_loader('/home/apollo/TransCG/cleargrasp/cleargrasp-dataset-test-val/real-test/d415/000000082-opaque-depth-img.exr', ndim = 1, ndim_representation = ['R'])
rgb = np.array(Image.open('transcg_data/scene3/1/rgb1.png'), dtype = np.float32)
rgb_mask = np.array(Image.open('transcg_data/scene3/1/depth1-gt-mask.png'), dtype = np.float32)
depth = np.array(Image.open('transcg_data/scene3/1/depth1.png'), dtype = np.float32)
depth_gt = np.array(Image.open('transcg_data/scene3/1/depth1-gt.png'), dtype = np.float32)

depth = depth / 1000
depth_gt = depth_gt / 1000
rgbcopy = rgb.copy()
depth_copy = depth.copy()
rgb_mask_copy = rgb_mask.copy()

res = inferencer.inference(rgb, depth)

cam_intrinsics = np.load('transcg_data/camera_intrinsics/1-camIntrinsics-D435.npy')
depth_gt[np.isnan(depth_gt)] = 0.0
# depth_gt[rgb_mask != 0] = 0
rgb_mask = np.where(depth_gt < 0.000000001, 255, 0).astype(np.uint8)
depth = cv2.resize(depth, (320, 240), interpolation = cv2.INTER_NEAREST)
depth_copy = cv2.resize(depth_copy, (320, 240), interpolation = cv2.INTER_NEAREST)
depth_gt = cv2.resize(depth_gt, (320, 240), interpolation = cv2.INTER_NEAREST)
res = cv2.resize(res, (320, 240), interpolation = cv2.INTER_NEAREST)
rgb_mask = cv2.resize(rgb_mask, (320, 240), interpolation = cv2.INTER_NEAREST).astype(np.uint8)
rgb_mask_copy = cv2.resize(rgb_mask_copy, (320, 240), interpolation = cv2.INTER_NEAREST).astype(np.uint8)
rgb_mask_copy[rgb_mask_copy != 0] = 1
rgb_mask[rgb_mask != 0] = 1

depth_mask = np.where(depth < 0.000000001, 255, 0).astype(np.uint8)
depth_mask[depth_mask != 0] = 1
depth_gt_new = handle_depth(depth_gt.copy(), depth_gt.copy(), rgb_mask_copy)
# depth = handle_depth(depth.copy(), depth_mask)
# depth_gt_mask = dropout_random_ellipses_4corruptmask(rgb_mask_copy, {"ellipse_dropout_mean": 20, "ellipse_gamma_shape": 10.0, "ellipse_gamma_scale": 1.0})
# depth[depth_gt_mask==1] = 0

# depth_gt[rgb_mask==255] = 0
# depth_gt[np.isnan(depth_gt)] = 0.0
# rgb_mask = np.where(depth_gt < 0.000000001, 255, 0).astype(np.uint8)
# rgb_mask[rgb_mask != 0] = 1
# depth_gt = depth_gt / depth.max() * 255
# depth_gt = np.array(depth_gt, dtype=np.uint8)
# depth_gt = cv2.inpaint(depth_gt, rgb_mask, 5, cv2.INPAINT_NS)
# depth_gt = np.array(depth_gt, dtype=np.float32) / 255 * depth.max()

# depth_gt = np.where(depth_gt < 0.3, 0, depth_gt)
# depth_gt = np.where(depth_gt > 1.5, 0, depth_gt)
neg_zero_mask = np.where(depth_gt < 0.0000001)
# zero_mask = np.logical_not(neg_zero_mask)
res[neg_zero_mask] = 0
depth_gt[neg_zero_mask] = 0
depth[neg_zero_mask] = 0
neg_zero_mask = np.where(depth_gt > 5)
res[neg_zero_mask] = 0
depth_gt[neg_zero_mask] = 0
depth[neg_zero_mask] = 0
# depth_gt[neg_zero_mask] = 0

# res = np.clip(res, 0.3, 1.5)
# depth = np.clip(depth, 0.3, 1.5)
# depth_gt = np.clip(depth_gt, 0.3, 1.5)


fig, axs = plt.subplots(2, 2)
tt="hsv"
rgb_1=rgbcopy.astype(np.int8)
axs.flat[0].imshow(rgb_1,cmap=tt)
axs.flat[0].set_title("rgb")
axs.flat[1].imshow(rgb_mask,cmap=tt)
axs.flat[1].set_title("original")
axs.flat[2].imshow(depth,cmap=tt)
axs.flat[2].set_title("model output")
axs.flat[3].imshow(depth_gt,cmap=tt)
axs.flat[3].set_title("groud truth")
plt.show()



# cloud = draw_point_cloud(rgb, res, cam_intrinsics, scale = 1.0)
# cloud_gt = draw_point_cloud(rgb, depth_gt, cam_intrinsics, scale = 1.0)

# frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
# sphere = o3d.geometry.TriangleMesh.create_sphere(0.002,20).translate([0,0,0.490])
# o3d.visualization.draw_geometries([cloud, cloud_gt, frame, sphere])

