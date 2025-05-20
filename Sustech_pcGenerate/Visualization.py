import open3d as o3d
import numpy as np
import zarr

# 加载Zarr数据
zarr_path = "/home/slam/3D-Diffusion-Policy/3D-Diffusion-Policy/data/5_18_simple_2.zarr/data/processed_point_clouds.zarr"
zarr_root = zarr.open(zarr_path, mode='r')
point_clouds = zarr_root['pointcloud']  # 形状 [N, 1014, 6]

# 选择帧（例如第1200帧）
frame_idx = 5
pc_data = point_clouds[frame_idx]

# 仅提取坐标（忽略颜色信息）
points = pc_data[:, :3]  # 只取xyz坐标

# 创建Open3D点云（无颜色）
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# 坐标系（红色-X，绿色-Y，蓝色-Z）
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

# 创建可视化器
vis = o3d.visualization.Visualizer()
vis.create_window()

# 添加几何体
vis.add_geometry(pcd)
vis.add_geometry(coord_frame)

# 设置灰色背景（RGB值0-1）
render_opt = vis.get_render_option()
render_opt.background_color = np.array([0.5, 0.5, 0.5])  # 中灰色
render_opt.point_size = 3.0  # 可选：调整点大小

# 运行可视化器
vis.run()
vis.destroy_window()