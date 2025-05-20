import os
import zarr
import numpy as np
import open3d as o3d
from Convert_PointCloud import PointCloudGenerator
from Cloud_Process import preprocess_point_cloud

def process_depth_to_pointcloud(source_zarr_path):
    """
    处理Zarr中的深度图像并生成点云
    
    参数:
        source_zarr_path: 源Zarr文件路径
    """
    # 打开源Zarr文件(只读模式)
    source_root = zarr.open(source_zarr_path, mode='r+')  # 使用r+模式以便添加新数据
    
    # 检查是否已存在Point_cloud组，如果不存在则创建
    if 'Point_cloud' not in source_root:
        source_root.create_group('Point_cloud')
    point_cloud_group = source_root['Point_cloud']
    
    # 模拟MuJoCo仿真对象(实际不需要真实仿真)
    class DummySim:
        pass
    
    # 初始化点云生成器
    sim = DummySim()
    pc_generator = PointCloudGenerator(sim, cam_names=["dummy_cam"])
    
    # 获取深度数据组
    depth_group = source_root['data/depth']
    
    # 准备存储处理后的点云
    processed_points = []
    
    # 遍历所有深度图像
    for i in range(len(depth_group)):
        depth_img = depth_group[i]
        
        # 生成原始点云
        points, _ = pc_generator.generateCroppedPointCloud("dummy_cam")
        
        # 预处理点云(裁剪、采样)
        processed_point = preprocess_point_cloud(points)
        
        # 添加到结果列表
        processed_points.append(processed_point)
    
    # 转换为numpy数组 [N, 1024, 6]
    processed_points = np.array(processed_points)
    
    # 压缩配置(与原始数据一致)
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    
    # 保存处理后的点云
    if 'processed' not in point_cloud_group:
        point_cloud_group.create_dataset(
            name='processed',
            data=processed_points,
            shape=processed_points.shape,
            chunks=(100, 1024, 6),  # 分块大小优化读取效率
            dtype='float32',        # 使用float32节省空间
            compressor=compressor
        )
    else:
        point_cloud_group['processed'][:] = processed_points
    
    # ------------------------------
    # 可视化验证(可选)
    # ------------------------------
    sample_idx = 0  # 查看第0帧
    sample_pcd = o3d.geometry.PointCloud()
    sample_pcd.points = o3d.utility.Vector3dVector(processed_points[sample_idx, :, :3])
    sample_pcd.colors = o3d.utility.Vector3dVector(processed_points[sample_idx, :, 3:] / 255.0)
    o3d.visualization.draw_geometries([sample_pcd], window_name="Sample Point Cloud")
    
    return processed_points

if __name__ == "__main__":
    source_zarr_path = "/home/slam/3D-Diffusion-Policy/3D-Diffusion-Policy/data/5_18_simple_2.zarr"  # 替换为你的Zarr文件路径
    processed_points = process_depth_to_pointcloud(source_zarr_path)
    print(f"处理完成! 共生成 {len(processed_points)} 帧点云数据")