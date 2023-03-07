import numpy as np
import matplotlib.pyplot as plt
from numpy.matlib import repmat
from skimage import io
import os
import cv2
import json
import torch
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
import math
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    SfMPerspectiveCameras,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)


def read_rgb(rgb_file):
    rgb = cv2.imread(rgb_file)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    return rgb


def read_depth(depth_file, index):
    depth = np.load(depth_file)
    depth = depth.astype(np.float32)
    road_number = int(index[1])
    if road_number <= 3 or road_number == 8:
        median = 0.020230255103424497
    else:
        median = 0.01544308428127076       
    depth = depth /33
    depth = depth *median/np.median(depth)
    return depth


def point_cloud_to_image(points, color ,K, width=128, height=128, transformation = None):
    points = np.transpose(points, (1,0))
    if transformation is not None:
        tmp = np.ones((4,points.shape[1]))
        tmp[:3,:] = points
        tmp = transformation @ tmp
    else:
        tmp = points        
    tmp = K @ tmp
    tmp1 = tmp/tmp[2,:]
    u_cord = np.clip(np.round(tmp1[0,:]),0,width - 1).astype(np.int64)
    v_cord = np.clip(np.round(tmp1[1,:]),0,height - 1).astype(np.int64)
    imtmp = np.zeros((height,width,3)).astype(np.int64)
    imtmp[v_cord, u_cord,:]= (color).astype(np.int64)


def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotationMatrixToEulerAngles(R) :
    assert(isRotationMatrix(R))    
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])   
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])*180/np.pi


def eulerAnglesToRotationMatrix(theta) : 
    theta *= np.pi / 180   
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])                    
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])                
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])                   
    R = np.dot(R_x, np.dot( R_y, R_z ))
    return R


def get_rendered_image(depth, img, rel_pose12, K):    
    rel_pose12 = rel_pose12.copy()
    rel_pose12[0:2,3] *= -1
    angles = rotationMatrixToEulerAngles(rel_pose12[:3,:3])
    rel_pose12[:3,:3] = eulerAnglesToRotationMatrix(angles * np.array([1.0,1.0, -1.0]))
    verts, color = depth_to_local_point_cloud(depth, color=img,k = K)    
    verts = torch.Tensor(verts).to(device)
    rgb = torch.Tensor(color).to(device)
    point_cloud = Pointclouds(points=[verts], features=[rgb])    
    R = torch.from_numpy(rel_pose12[:3,:3].astype(np.float32)).unsqueeze(0)
    T = torch.FloatTensor(1*rel_pose12[:3,3].astype(np.float32)).unsqueeze(0)
    cameras = SfMPerspectiveCameras(device=device, R=R, T=T,
                                        focal_length = torch.FloatTensor([[1,1]]),
                                       principal_point = torch.FloatTensor([[0,0]]))
    raster_settings = PointsRasterizationSettings(
            image_size=(img.shape[0], img.shape[1]), 
            radius = 0.01,
            points_per_pixel = 100
        )
    renderer = PointsRenderer(
            rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
            compositor=AlphaCompositor()
        )
    images = renderer(point_cloud)    
    return images[0, ..., :3].cpu().numpy()


def depth_to_local_point_cloud(image, color=None, k = np.eye(3),max_depth=1.1):
    """
    Convert an image containing CARLA encoded depth-map to a 2D array containing
    the 3D position (relative to the camera) of each pixel and its corresponding
    RGB color of an array.
    "max_depth" is used to omit the points that are far enough.
    Reference: 
    https://github.com/carla-simulator/driving-benchmarks/blob/master/version084/carla/image_converter.py
    """
    far = 1000.0  # max depth in meters.
    normalized_depth = image
    height, width = image.shape
    # 2d pixel coordinates
    pixel_length = width * height
    u_coord = repmat(np.r_[width-1:-1:-1],
                     height, 1).reshape(pixel_length)
    v_coord = repmat(np.c_[height-1:-1:-1],
                     1, width).reshape(pixel_length)
    if color is not None:
        color = color.reshape(pixel_length, 3)
    normalized_depth = np.reshape(normalized_depth, pixel_length)
    # Search for pixels where the depth is greater than max_depth to delete them
    max_depth_indexes = np.where(normalized_depth > max_depth)
    normalized_depth = np.delete(normalized_depth, max_depth_indexes)
    u_coord = np.delete(u_coord, max_depth_indexes)
    v_coord = np.delete(v_coord, max_depth_indexes)
    if color is not None:
        color = np.delete(color, max_depth_indexes, axis=0)
    p2d = np.array([u_coord, v_coord, np.ones_like(u_coord)])
    p3d = np.dot(np.linalg.inv(k), p2d)
    p3d *= normalized_depth * far
    p3d = np.transpose(p3d, (1,0))
    return p3d, color / 255.0

def compute_noise(j, N_sample):   
    x_bias = np.random.normal(0, 2/3)
    y_bias = np.random.uniform(-0.05, 0.05) + (-N_sample/10 + j*0.2)
    yaw_bias = np.random.normal(0, 0.05)
    print('random transfer',x_bias,y_bias, yaw_bias)
    return x_bias, y_bias, yaw_bias

def compute_TRMatrix(x,y,yaw):
    Tr = np.array([[np.cos(yaw),    -np.sin(yaw),     0,         x],
                    [np.sin(yaw),    np.cos(yaw),     0,         y],
                    [0,              0,               1,         0],
                    [0,              0,               0,         1]
                    ])
    return Tr

def compute_delta(Tr):
    delta_x = Tr[0,3]
    delta_y = Tr[1,3]
    delta_yaw = np.arcsin(Tr[1,0])
    return delta_x, delta_y, delta_yaw

def compute_Transformation(x,y,yaw):
    Tr = np.array([[np.cos(yaw), 0, np.sin(yaw), y],
                  [0,            1, 0,           0], 
                  [-np.sin(yaw), 0, np.cos(yaw), x],
                  [0,            0, 0,           1]])
    return Tr

def center_resize(img, w, h):
    y = img.shape[1]/2 - h/2
    crop_img = img[:int(w), int(y):int(y+h),:]
    crop_img = np.array(crop_img, dtype='uint8')
    resize_img = cv2.resize(crop_img, (128, 128), interpolation=cv2.INTER_AREA)
    return crop_img, resize_img


def save_image(Folder, img, index, j):
    bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    status = cv2.imwrite(os.path.join(Folder, index+'_%02d.png'%j),bgr_img)
    return status

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='Data augmentation via image rendering.'
    )
    parser.add_argument('--rgb-path', type=str, help='Path to the folder containing on-trajectory RGB images collected in the CARLA simulator.')
    parser.add_argument('--depth-path', type=str, help='Path to the folder containing depth maps for on-trajectory RGB images collected in the CARLA simulator.')
    parser.add_argument('--out-path', type=str, help='Path to the output folder.')
    parser.add_argument('--txt-path', type=str, help='Path to the txt file generated by data collection in CARLA simulator, including')
    args = parser.parse_args()

    # if the output folder not exited, create it
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    # compute the delta imformation from the original txt file saved by collection with the CARLA simulator, save the txt file to the same folder as the original txt file
    delta_path = os.path.join(os.path.dirname(args.txt_path), 'delta_data.txt')
    out = open(delta_path,'a+')
    delta_data = np.empty((0,6))
    with open(args.txt_path, 'r') as infile:
        for line in infile:
            line = line.rstrip()
            words = line.split()
            ref_x = float(words[1])
            ref_y = float(words[2])
            ref_yaw = float(words[3])
            ref_v = float(words[4])
            ego_x = float(words[5])
            ego_y = float(words[6])
            ego_yaw = float(words[7])
            ego_v = float(words[8])
            ref_Tr = compute_TRMatrix(ref_x, ref_y, ref_yaw)
            ego_Tr = compute_TRMatrix(ego_x, ego_y, ego_yaw)
            TrET = np.linalg.inv(ego_Tr) @ (ref_Tr)
            delta_x, delta_y, delta_yaw = compute_delta(TrET)        
            print(words[0],ref_v, ego_v, delta_x, delta_y, delta_yaw, file = out)
    out.close()
        
    # read the delta information
    fh = open(delta_path, 'r')
    de_target_state = np.empty([0, 4], dtype=float)
    de_ego_state = np.empty([0, 4], dtype=float) 
    de_labels = []
    for line in fh:                
        line = line.rstrip()       
        words = line.split()   
        index = (words[0].rsplit('.', 1))[0]
        de_labels.append([index, float(words[1]), float(words[2]), float(words[3]), float(words[4]), float(words[5])])

    # read the camera intrinsic matrix 
    intrinsics_file = os.path.join("./camera_intrinsic.json")
    with open(intrinsics_file) as f:
        K = json.load(f)
    K = np.array(K) 

    for i in range(len(de_labels)):
    
        de_index, de_vt, de_v1, de_delta_x, de_delta_y, de_delta_yaw = de_labels[i]
        rgb_file   = os.path.join(args.rgb_path, de_index+".png")
        depth_file = os.path.join(args.depth_path, de_index+".npy")
        de_poseT2 = compute_TRMatrix(de_delta_x, de_delta_y, de_delta_yaw)   #target vehicle extrinsic matrix in ego vehicle coordinates
        
        if de_vt > 6:
            N_sample = 32
        else:
            N_sample = 22

        for j in range(N_sample):
            print(f'{i}-th image, {j}-th sample')
            #set random transfer            
            if j == N_sample-1:
                noise_x, noise_y, noise_yaw = 0, 0, 0
            else:
                noise_x, noise_y, noise_yaw = compute_noise(j, N_sample)
                        
            #compute the new delta information with augmentation
            poseE2 = compute_TRMatrix(noise_x, noise_y, noise_yaw)  
            de_TrET2 = np.linalg.inv(poseE2) @ (de_poseT2)  
            de_newdelta_x, de_newdelta_y, de_newdelta_yaw = compute_delta(de_TrET2)

            # compute the relative transformation matrix between original and transformed ego vehicle extrinsic matrix
            pose1 = compute_Transformation(0,0,0)   #original ego vehicle extrinsic matrix
            pose2 = compute_Transformation(noise_x, noise_y, noise_yaw)   #transformed extrinsic matrix
            rel_pose12 = np.linalg.inv(pose2) @ (pose1)

            # read the RGB image and the depth map
            rgb = read_rgb(rgb_file)
            # Note that the depth values below are normalized to between 0 and 1.  
            depth = read_depth(depth_file, de_index) 
            height, width = depth.shape

            pixel_length = width * height
            u_coord = repmat(np.r_[width-1:-1:-1],
                                height, 1).reshape(pixel_length)
            v_coord = repmat(np.c_[height-1:-1:-1],
                                1, width).reshape(pixel_length)

            depths = depth[v_coord, u_coord] * 1000
            color = rgb[v_coord, u_coord]

            pixel_length = width * height
            u_coord = repmat(np.r_[width-1:-1:-1],
                                height, 1).reshape(pixel_length)
            v_coord = repmat(np.c_[height-1:-1:-1],
                                1, width).reshape(pixel_length)

            # Remove points greater than 1000
            max_depth_indexes = np.where(depths > 1000)
            depths[max_depth_indexes] = 0

            homogenous = np.vstack((u_coord, v_coord, np.ones_like(u_coord)))
            p3d = np.linalg.inv(K) @ (homogenous*depths)

            rendered_image = (get_rendered_image(depth, rgb, rel_pose12, K) * 255).astype(np.int64)

            # center crop and resize the img
            crop_img, resize_img = center_resize(rendered_image, 350, 600)

            # save image to disk
            status = save_image(args.out_path, resize_img, de_index, j)

            # save the augmentation txt file to the same folder as the original txt file
            de_data = open(os.path.join(os.path.dirname(args.txt_path), 'augmentation.txt'),'a+')
            print(de_index+'_%02d.png'%j, de_vt, de_v1,
                de_newdelta_x, de_newdelta_y, de_newdelta_yaw, file = de_data)
            de_data.close()
            