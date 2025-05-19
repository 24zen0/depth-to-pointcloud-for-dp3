import zarr
import numpy as np
from typing import Optional


def read_in_depth(zarr_path: str) -> np.ndarray:
    """
    从 Zarr 文件读取深度数据（固定从 "depth" 数据集读取）
    
    Args:
        zarr_path: Zarr 文件或目录的路径
        
    Returns:
        depth_array: 深度数据 NumPy 数组 (形状为 [H, W])
        
    Raises:
        FileNotFoundError: 如果 Zarr 路径不存在
        KeyError: 如果 "depth" 数据集不存在
        ValueError: 如果数据不是 2D 数组
    """
    try:
        # 打开 Zarr 文件
        zarr_group = zarr.open(zarr_path, mode='r')
        
        # 检查 depth 数据集是否存在
        if "depth" not in zarr_group:
            raise KeyError("Zarr 文件中必须包含 'depth' 数据集")
            
        # 获取深度数据
        depth_data = zarr_group["depth"]
        
        # 验证数据形状
        if depth_data.ndim != 2:
            raise ValueError(f"深度数据应为 2D 数组，但获取到 {depth_data.ndim}D 数据")
            
        return np.array(depth_data)  # 将数据加载到内存
        
    except zarr.errors.PathNotFoundError:
        raise FileNotFoundError(f"Zarr 路径不存在: {zarr_path}")

def generate_pcd_zarr(
    pcd_array: np.ndarray,
    output_zarr_path: str,
    chunk_shape: tuple = (1024, 3),
    compressor: Optional[zarr.Compressor] = None
) -> None:
    """
    将(N,3)形状的点云保存为Zarr
    
    Args:
        pcd_array: 输入点云数组
        output_zarr_path: 输出路径(.zarr目录)
        chunk_shape: 存储分块大小
        compressor: 压缩器实例
    """
    if pcd_array.ndim != 2 or pcd_array.shape[1] != 3:
        raise ValueError("点云必须是(N,3)形状的数组")

    # 设置默认压缩（zstd算法，平衡模式）
    if compressor is None:
        compressor = zarr.Blosc(cname='zstd', clevel=5, shuffle=2)

    zarr.save_array(
        store=output_zarr_path,
        data=pcd_array,
        chunks=chunk_shape,
        compressor=compressor
    )