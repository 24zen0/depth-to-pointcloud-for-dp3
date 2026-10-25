"""Pinhole projection with one explicit depth scale and camera-to-world transform."""
import numpy as np


class PointCloudGenerator:
    def __init__(self, sim=None, cam_names=None, img_size=480, *, camera_matrix=None,
                 camera_to_world=None, depth_scale=None, width=None, height=None):
        if camera_matrix is None or camera_to_world is None or depth_scale is None:
            raise ValueError("Provide camera_matrix, camera_to_world and depth_scale from calibration")
        self.cam_mat = np.asarray(camera_matrix, dtype=float)
        self.extrinsic_matrix = np.asarray(camera_to_world, dtype=float)
        self.img_width, self.img_height = width, height
        self.depth_scale = float(depth_scale)
        if self.cam_mat.shape != (3, 3) or self.extrinsic_matrix.shape != (4, 4):
            raise ValueError("Expected 3x3 intrinsics and 4x4 camera-to-world transform")
        if not np.isfinite(self.cam_mat).all() or not np.isfinite(self.extrinsic_matrix).all():
            raise ValueError("Calibration must be finite")
        if not np.isfinite(self.depth_scale) or self.depth_scale <= 0 or self.cam_mat[0, 0] <= 0 or self.cam_mat[1, 1] <= 0:
            raise ValueError("Depth scale and focal lengths must be positive")
        if not np.allclose(self.extrinsic_matrix[3], [0, 0, 0, 1]):
            raise ValueError("camera_to_world must be a homogeneous transform")

    def generateCroppedPointCloud(self, depth_data, cam_name=None, save_img_dir=None, device_id=0):
        """Project and transform a frame. Workspace cropping is performed downstream."""
        depth = np.asarray(depth_data, dtype=float)
        if depth.ndim != 2:
            raise ValueError("Expected one (H, W) depth frame")
        h, w = depth.shape
        if (self.img_width is not None and w != self.img_width) or (self.img_height is not None and h != self.img_height):
            raise ValueError("Depth dimensions do not match calibration")
        v, u = np.indices(depth.shape)
        z = depth / self.depth_scale
        mask = np.isfinite(z) & (z > 0)
        x = (u[mask] - self.cam_mat[0, 2]) * z[mask] / self.cam_mat[0, 0]
        y = (v[mask] - self.cam_mat[1, 2]) * z[mask] / self.cam_mat[1, 1]
        camera = np.column_stack((x, y, z[mask]))
        world = camera @ self.extrinsic_matrix[:3, :3].T + self.extrinsic_matrix[:3, 3]
        return world.astype(np.float32), depth
