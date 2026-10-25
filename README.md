# Depth to Point Cloud for 3D Diffusion Policy

Depth-image conversion, workspace filtering, and fixed-size point sampling for a real-robot desktop manipulation project using [3D Diffusion Policy](https://arxiv.org/abs/2403.03954).

[Technical report](https://24zen0.github.io/files/3D_Diffusion_Policy_Technical_Report.pdf) · [Project and results](https://github.com/24zen0/Desktop-Cleaning-via-3D-Diffusion-Policy) · [Robot demo](https://24zen0.github.io/#diffusion-policy)

## Role in the project

The policy consumes a compact representation of the scene rather than a full depth image. This component explores how to project depth into 3D, retain the manipulation workspace, and sample 1,024 points for the policy input.

Yitao Zeng contributed point-cloud preprocessing within the team project and participated in the integrated calibration, demonstration, training, and deployment pipeline. Policy training and teleoperation are maintained in the [component repositories](https://github.com/24zen0/Desktop-Cleaning-via-3D-Diffusion-Policy#code-map).

## Processing workflow

```text
Depth frames in Zarr
    → pinhole back-projection using camera intrinsics
    → calibrated coordinate transformation
    → workspace filtering
    → farthest-point sampling
    → point-cloud observations for the policy
```

The files contain camera-specific development scripts and reusable functions. The workflow above describes the project pipeline; the scripts provide separate conversion, preprocessing, and visualization paths.

| Workspace cropping | Plane-filtering experiment |
| --- | --- |
| ![Workspace-cropped point cloud](assets/workspace-cloud.png) | ![Plane-filtering diagnostic](assets/plane-filtering.png) |

These project diagnostics also appear in the technical report. The two panels illustrate different preprocessing approaches, rather than the same frame before and after processing. The project adopted workspace cropping and fixed-size sampling; RANSAC plane filtering was explored as an alternative.

## Code map

All Python modules are under [`Sustech_pcGenerate/`](Sustech_pcGenerate/).

| File | Purpose |
| --- | --- |
| [`Inputing_zarr.py`](Sustech_pcGenerate/Inputing_zarr.py) | Read the `depth` array; write a point-cloud array to a Zarr group |
| [`Convert_PointCloud.py`](Sustech_pcGenerate/Convert_PointCloud.py) | Open3D pinhole back-projection and a camera-to-world transform |
| [`Cloud_Process.py`](Sustech_pcGenerate/Cloud_Process.py) | Workspace bounds, point preprocessing, and PyTorch3D farthest-point sampling |
| [`FilteringPlane.py`](Sustech_pcGenerate/FilteringPlane.py) | RANSAC plane detection, removal, and visualization helpers |
| [`RealSenseCamera.py`](Sustech_pcGenerate/RealSenseCamera.py) | Live RealSense depth acquisition and point-cloud visualization |
| [`Visualization.py`](Sustech_pcGenerate/Visualization.py) | Inspect saved point clouds with a workspace bounding box |
| [`VisDepth.py`](Sustech_pcGenerate/VisDepth.py) | Inspect depth data |
| [`main.py`](Sustech_pcGenerate/main.py) | Original camera-specific debug driver, processing up to ten frames |

## Data interface

- **Input:** a Zarr group containing `depth`, with a frame sequence shaped `(T, H, W)`. If the recording has an outer `data` group, pass the path ending in `.zarr/data`.
- **Projection:** a depth frame plus camera intrinsics produces XYZ points. The Open3D path applies the extrinsic transform stored in `PointCloudGenerator`.
- **Sampling:** `farthest_point_sampling(points, num_points=1024, use_cuda=True)` takes an `(N, 3)` NumPy array and returns sampled XYZ points and their indices. The default uses CUDA; `use_cuda=False` selects CPU execution.
- **Storage:** `generate_pcd_zarr` writes a dataset named `pointcloud` by default. Its default chunk shape is for a two-dimensional array; pass a matching three-dimensional chunk shape when saving a frame sequence.

## Inspect a recording

Clone this repository and install the dependencies for Zarr inspection:

```bash
git clone https://github.com/24zen0/depth-to-pointcloud-for-dp3.git
cd depth-to-pointcloud-for-dp3
python -m pip install numpy 'zarr<3' numcodecs
```

From the repository root, replace `path/to/recording.zarr/data` with your recording's depth-group path:

```bash
python - path/to/recording.zarr/data <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path('Sustech_pcGenerate').resolve()))
from Inputing_zarr import read_in_depth

depth = read_in_depth(sys.argv[1])
print('Depth shape:', depth.shape)
print('Depth dtype:', depth.dtype)
print('First-frame range:', depth[0].min(), depth[0].max())
PY
```

Geometry conversion additionally uses Open3D, MuJoCo, Matplotlib, and Pillow. The sampling/preprocessing module imports PyTorch, torchvision, PyTorch3D, tqdm, and termcolor. Install PyTorch3D for the PyTorch/CUDA combination used by your policy environment. Live acquisition also requires `pyrealsense2` and a RealSense camera.

## Adapting the camera-specific scripts

Use the modules with the calibration and data layout of your own setup:

1. Set the camera intrinsics, image dimensions, depth units, and extrinsic transform. The constants in `Convert_PointCloud.py` belong to the project camera; its constructor currently sets width and height from the same `img_size` argument.
2. Define the crop in the transformed frame. `Cloud_Process.py` stores the project bounds as `x: (-0.11, 0.318)`, `y: (-0.178, 0.078)`, and `z: (0.14, 0.55)`, with strict inequalities.
3. Keep one consistent scale and transform through the pipeline. `Convert_PointCloud.py` uses Open3D's transform convention; `preprocess_point_cloud` contains a separate scale and row-vector matrix operation. Adapt those paths together before composing them, so depth is not scaled or transformed twice.
4. Match output dimensions to the policy interface. Use XYZ-only `(T, 1024, 3)` or retain color channels only when the downstream policy expects them. For a three-dimensional sequence, pass `chunk_shape=(1, 1024, 3)` to the Zarr writer when using XYZ-only data.

For the original `main.py` driver, first change its `imputing_zarr` import to the actual module name `Inputing_zarr` and replace the absolute dataset paths. It is a ten-frame debug script: it currently computes a preprocessed cloud but appends the sampled cloud, and uses six-channel zero arrays for failed frames. Integrating the individual functions lets you define the output format and failed-frame handling for your dataset explicitly.

## References and credit

- [3D Diffusion Policy](https://github.com/YanjieZe/3D-Diffusion-Policy) provides the underlying policy framework.
- [Project materials, authors, instructor, and related code](https://github.com/24zen0/Desktop-Cleaning-via-3D-Diffusion-Policy).
- Code is distributed under this repository's [MIT license](LICENSE).
