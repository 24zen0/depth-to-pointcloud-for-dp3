import open3d as o3d
import numpy as np
import zarr
from Cloud_Process import boundary

# 加载Zarr数据
zarr_path = "/home/slam/3D-Diffusion-Policy/3D-Diffusion-Policy/data/metaworld_push-wall_expert.zarr/data"
zarr_root = zarr.open(zarr_path, mode='r')
point_clouds = zarr_root['point_cloud']  # 形状 [N, 1024, 6]

# 选择帧（例如第0帧）
frame_idx = 0
pc_data = point_clouds[frame_idx]

# 仅提取坐标（忽略颜色信息）
points = pc_data[:, :3]  # 只取xyz坐标

# 1. 缩放点云坐标（修正缩放方式）
scaling_factor = 1000  # 放大1000倍（假设原始数据是米，转为毫米级显示）
points_scaled = points * scaling_factor

# 2. 确保数据类型正确（Open3D需要float64）
points_scaled = points_scaled.astype(np.float64)

# 3. 检查数据有效性
print("点云坐标范围（缩放后）：")
print("X: min={:.4f}, max={:.4f}".format(np.min(points_scaled[:,0]), np.max(points_scaled[:,0])))
print("Y: min={:.4f}, max={:.4f}".format(np.min(points_scaled[:,1]), np.max(points_scaled[:,1])))
print("Z: min={:.4f}, max={:.4f}".format(np.min(points_scaled[:,2]), np.max(points_scaled[:,2])))

# 创建Open3D点云
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_scaled)

# 4. 调整WORK_SPACE范围（与缩放后的坐标匹配）
WORK_SPACE_SCALED = [
    [-300, 600],    # X范围（毫米）
    [-750, -450],    # Y范围
    [-850, -70]    # Z范围
]
min_bound, max_bound = boundary(WORK_SPACE_SCALED)
custom_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
custom_bbox.color = (0, 1, 0)  # 绿色

# 坐标系（按比例缩放大小）
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=200,  # 坐标系大小（毫米级）
    origin=[0, 0, 0]  # 坐标系原点
)

# 创建可视化器
vis = o3d.visualization.Visualizer()
vis.create_window()

# 添加几何体
vis.add_geometry(pcd)
vis.add_geometry(custom_bbox)
vis.add_geometry(coord_frame)

# 可视化设置
render_opt = vis.get_render_option()
render_opt.background_color = np.array([0.5, 0.5, 0.5])  # 灰色背景
render_opt.point_size = 3.0  # 点大小

# 调整视角
ctr = vis.get_view_control()
ctr.set_zoom(1)  # 初始缩放系数

# 运行可视化
vis.run()
vis.destroy_window()