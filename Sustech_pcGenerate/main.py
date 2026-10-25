"""Convert a calibrated depth sequence to fixed-size policy point clouds."""
import argparse
import json
from pathlib import Path
import numpy as np
from Convert_PointCloud import PointCloudGenerator
from Cloud_Process import preprocess_point_cloud
from Inputing_zarr import read_in_depth, generate_pcd_zarr


def convert_frames(depth, calibration, num_points=1024):
    depth = np.asarray(depth)
    if depth.ndim != 3:
        raise ValueError("Expected depth sequence (T, H, W)")
    if num_points <= 0 or len(depth) == 0:
        raise ValueError("Use a nonempty sequence and positive point count")
    generator = PointCloudGenerator(camera_matrix=calibration['intrinsics'],
        camera_to_world=calibration['camera_to_world'], depth_scale=calibration['depth_scale'],
        width=calibration['width'], height=calibration['height'])
    clouds = np.zeros((len(depth), num_points, 3), dtype=np.float32)
    valid = np.zeros(len(depth), dtype=bool)
    for i, frame in enumerate(depth):
        points, _ = generator.generateCroppedPointCloud(frame)
        if len(points) == 0:
            continue
        bounds = np.asarray(calibration['workspace'])
        keep = (points > bounds[:, 0]).all(axis=1) & (points < bounds[:, 1]).all(axis=1)
        if not keep.any():
            continue
        clouds[i] = preprocess_point_cloud(points, workspace=bounds, num_points=num_points)
        valid[i] = True
    return clouds, valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='.npy sequence or Zarr group containing depth')
    parser.add_argument('output', help='New Zarr output directory')
    parser.add_argument('--calibration', required=True, help='Calibration JSON; see examples/')
    parser.add_argument('--num-points', type=int, default=1024)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error('Output already exists; choose a new directory')
    calibration = json.loads(Path(args.calibration).read_text())
    depth = np.load(args.input, allow_pickle=False) if args.input.endswith('.npy') else read_in_depth(args.input)
    clouds, valid = convert_frames(depth, calibration, args.num_points)
    generate_pcd_zarr(clouds, str(output), chunk_shape=(1, args.num_points, 3))
    import zarr
    group = zarr.open_group(str(output), mode='a')
    group.array('valid_frames', valid)
    group.attrs['calibration'] = calibration
    print(json.dumps({'shape': list(clouds.shape), 'valid_frames': int(valid.sum()), 'total_frames': len(valid)}))


if __name__ == '__main__':
    main()
