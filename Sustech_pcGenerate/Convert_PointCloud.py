import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PIL_Image
from typing import List
import open3d as o3d
import mujoco

def cammat2o3d(cam_mat, width, height):
    cx = cam_mat[0,2]
    fx = cam_mat[0,0]
    cy = cam_mat[1,2]
    fy = cam_mat[1,1]

    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


class PointCloudGenerator(object):
    """
    initialization function

    @param sim:       MuJoCo simulation object
    @param min_bound: If not None, list len(3) containing smallest x, y, and z
        values that will not be cropped
    @param max_bound: If not None, list len(3) containing largest x, y, and z
        values that will not be cropped
    """
    def __init__(self, sim=None, cam_names=None, img_size=480):
        self.sim = sim
        self.img_width = img_size
        self.img_height = img_size
        self.cam_mat = np.array([
            [385.9849853515625, 0, 318.4468688964844],
            [0, 617.5814819335938, 245.88998413085938],
            [0, 0, 1]
          ]) 

        t = np.array([0.33916, 1.21842, 0.05350])  # in meters
        R = np.array([[0.7542, 0.0152, 0.6564],
                      [-0.6149, 0.3671, 0.6980],
                      [-0.2304, -0.9300, 0.2862]])        

        # 构建 4x4 变换矩阵
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = R
        extrinsic_matrix[:3, 3] = t
        self.extrinsic_matrix = extrinsic_matrix
    
    def generateCroppedPointCloud(self, depth_data, cam_name, save_img_dir=None, device_id=0):
        #color_img, depth = self.captureImage(cam_name, capture_depth=True, device_id=device_id)

        #if save_img_dir is not None:
         #   self.saveImg(depth, save_img_dir, f"depth_test_{cam_name}")
         #   self.saveImg(color_img, save_img_dir, f"color_test_{cam_name}")
        
        od_cammat = cammat2o3d(self.cam_mat, self.img_width, self.img_height)
        od_depth = o3d.geometry.Image(depth_data)
        o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(od_depth, od_cammat)
        
        # 计算相机到世界的变换矩阵
        c2w = self.extrinsic_matrix
        transformed_cloud = o3d_cloud.transform(c2w)

        return np.asarray(transformed_cloud.points), depth_data.squeeze()