import os
import numpy as np
from Convert_PointCloud import PointCloudGenerator
from Cloud_Process import preprocess_point_cloud, farthest_point_sampling
from imputing_zarr import read_in_depth, generate_pcd_zarr

# 1. 初始化参数
zarr_path = "/home/slam/3D-Diffusion-Policy/3D-Diffusion-Policy/data/5_18_simple_2.zarr/data"
output_zarr_path = "/home/slam/3D-Diffusion-Policy/3D-Diffusion-Policy/data/5_18_simple_2.zarr/data/processed_point_clouds.zarr"
cam_name = "Realsense D435i"

# 2. 读取深度数据（添加详细检查）
print("正在加载深度数据...")
depth_from_robot = read_in_depth(zarr_path)
print(f"深度数据形状: {depth_from_robot.shape}, 数据类型: {depth_from_robot.dtype}")

# 3. 初始化点云生成器（添加调试信息）
pc_generator = PointCloudGenerator(
    sim=None,
    img_size=depth_from_robot.shape[1]  # 使用深度图高度
)
print("点云生成器初始化完成")

# 4. 处理流程优化
all_processed_points = []
valid_frames = 0

for i in range(min(10, depth_from_robot.shape[0])):  # 先只处理前10帧用于调试
    print(f"\n正在处理第 {i} 帧...")
    current_depth = depth_from_robot[i]
    
    # 检查深度图有效性
    print(f"深度图范围: {np.min(current_depth)} - {np.max(current_depth)}")
    if np.all(current_depth == 0):
        print("警告: 深度图全为零值")
        all_processed_points.append(np.zeros((1024, 6)))
        continue
    
    try:
        # 生成点云（添加详细日志）
        print("正在生成点云...")
        points, _ = pc_generator.generateCroppedPointCloud(
            depth_data=current_depth,
            cam_name=cam_name
        )
        print(f"生成点云形状: {points.shape if isinstance(points, np.ndarray) else '无效'}")
        
        if not isinstance(points, np.ndarray) or points.size == 0:
            raise ValueError("生成的点云为空")
            
        # 最远点采样
        print("正在进行最远点采样...")
        sampled_points, _ = farthest_point_sampling(points)
        print(f"采样后点云形状: {sampled_points.shape}")
        
        # 预处理
        print("正在预处理点云...")
        processed_points = preprocess_point_cloud(sampled_points)
        print(f"处理后点云形状: {processed_points.shape}")
        
        # 验证最终输出
        if sampled_points.shape == (1024, 3):
            all_processed_points.append(sampled_points)
            valid_frames += 1
        else:
            print(f"警告: 无效的输出形状 {sampled_points.shape}")
            all_processed_points.append(np.zeros((1024, 6)))
            
    except Exception as e:
        print(f"处理第 {i} 帧时出错: {str(e)}")
        all_processed_points.append(np.zeros((1024, 6)))

# 5. 结果统计和保存
print(f"\n处理完成,有效帧数: {valid_frames}/{depth_from_robot.shape[0]}")

if valid_frames > 0:
    try:
        all_processed_points = np.stack(all_processed_points, axis=0)
        print(f"最终点云数组形状: {all_processed_points.shape}")
        
        # 保存结果
        success = generate_pcd_zarr(all_processed_points, output_zarr_path)
        if success:
            print(f"点云已保存至: {output_zarr_path}")
    except Exception as e:
        print(f"保存结果时出错: {str(e)}")
else:
    print("警告: 没有有效帧被处理")