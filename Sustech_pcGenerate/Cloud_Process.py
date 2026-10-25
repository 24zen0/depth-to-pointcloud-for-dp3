"""Workspace cropping and fixed-size sampling of metric XYZ point clouds."""
import numpy as np

DEFAULT_WORKSPACE = ((-0.11, 0.318), (-0.178, 0.078), (0.14, 0.55))


def farthest_point_sampling(points, num_points=1024, use_cuda=False):
    """Return sampled points and indices; repeat indices when N < num_points.

    CPU sampling starts at index zero for reproducibility. CUDA is opt-in and
    requires a compatible PyTorch/PyTorch3D installation.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("Expected a nonempty (N, 3) point array")
    if not np.isfinite(points).all() or num_points <= 0:
        raise ValueError("Points must be finite and num_points positive")
    count = min(num_points, len(points))
    if use_cuda:
        import torch
        from pytorch3d.ops import sample_farthest_points
        tensor = torch.as_tensor(points, device="cuda").unsqueeze(0)
        _, indices = sample_farthest_points(tensor, K=count)
        chosen = indices[0].cpu().numpy()
    else:
        chosen = np.empty(count, dtype=np.int64)
        distance = np.full(len(points), np.inf)
        index = 0
        for i in range(count):
            chosen[i] = index
            distance = np.minimum(distance, ((points - points[index]) ** 2).sum(axis=1))
            distance[chosen[:i + 1]] = -1
            index = int(distance.argmax())
    chosen = np.resize(chosen, num_points)
    return points[chosen], chosen


def preprocess_point_cloud(points, use_cuda=False, workspace=DEFAULT_WORKSPACE, num_points=1024):
    """Crop already transformed, metre-valued XYZ/XYZRGB; sample exactly once.

    No depth scaling or extrinsic transformation is applied here. Extra columns
    (e.g. RGB) follow the selected XYZ indices without changing their values.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("Expected (N, 3+) points")
    bounds = np.asarray(workspace, dtype=float)
    if bounds.shape != (3, 2) or not np.all(bounds[:, 0] < bounds[:, 1]):
        raise ValueError("workspace must contain three increasing [min, max] pairs")
    xyz = points[:, :3]
    mask = np.isfinite(xyz).all(axis=1) & (xyz > bounds[:, 0]).all(axis=1) & (xyz < bounds[:, 1]).all(axis=1)
    cropped = points[mask]
    _, indices = farthest_point_sampling(cropped[:, :3], num_points, use_cuda)
    return cropped[indices].astype(np.float32)


def boundary(workspace):
    bounds = np.asarray(workspace)
    return bounds[:, 0], bounds[:, 1]
