import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PIL_Image
from typing import List
import open3d as o3d

def cammat2o3d(cam_mat, width, height):
    cx = cam_mat[0,2]
    fx = cam_mat[0,0]
    cy = cam_mat[1,2]
    fy = cam_mat[1,1]

    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


class PointCloudGenerator(object):
    """
    initialization function

    @param min_bound: If not None, list len(3) containing smallest x, y, and z
        values that will not be cropped
    @param max_bound: If not None, list len(3) containing largest x, y, and z
        values that will not be cropped
    """
    def __init__(self,  img_size=480):
        self.img_width = img_size
        self.img_height = img_size
        # self.cam_mat = np.array([
        #     [390.64122761, 0, 450.30159532],
        #     [0, 392.57988967, 200.70517741],
        #     [0, 0, 1]
        #   ]) 
        # self.cam_mat = np.array([
        # [110.85125168,   0.,          64.,        ],
        # [  0.,         110.85125168,  64. ,       ],
        # [  0.    ,       0.     ,      1.        ]
        # ])
        #push-wall
        self.cam_mat = np.array([
            [101.39696962,   0.,          42.        ],
            [  0.,         101.39696962,  42.        ],
            [  0.,           0.,          1.        ]
        ])

        # t = np.array([0.33916, 1.21842, 0.05350])  # in meters
        # R = np.array([[0.7542, 0.0152, 0.6564],
        #               [-0.6149, 0.3671, 0.6980],
        #               [-0.2304, -0.9300, 0.2862]])  
        extrinsic_matrix = np.array([
            [1.,  0.,  0.,  0.],
            [0., -1.,  0.,  0.],
            [0.,  0., -1.,  0.],
            [0.,  0.,  0.,  1.]
        ])

        # 构建 4x4 变换矩阵
        # extrinsic_matrix = np.eye(4)
        # extrinsic_matrix[:3, :3] = R
        # extrinsic_matrix[:3, 3] = t
        self.extrinsic_matrix = extrinsic_matrix
    
    def generateCroppedPointCloud(self, depth_data, save_img_dir=None, device_id=0):
        #color_img, depth = self.captureImage(cam_name, capture_depth=True, device_id=device_id)

        #if save_img_dir is not None:
         #   self.saveImg(depth, save_img_dir, f"depth_test_{cam_name}")
         #   self.saveImg(color_img, save_img_dir, f"color_test_{cam_name}")
        print("转换前连续性:", depth_data.flags['C_CONTIGUOUS'])  # the result is true ,which means it is continuous
        depth_data = (depth_data * 1000).astype(np.uint16)
        assert depth_data.dtype in [np.uint16, np.float32], "深度图数据类型需为 uint16 或 float32"
        od_cammat = cammat2o3d(self.cam_mat, self.img_width, self.img_height)
        print("alive still")
        od_depth = o3d.geometry.Image(depth_data)
        print("alive still1")
        o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(od_depth, od_cammat)
        
        # 计算相机到世界的变换矩阵
        c2w = self.extrinsic_matrix
        transformed_cloud = o3d_cloud.transform(c2w)

        return np.asarray(transformed_cloud.points), depth_data.squeeze()