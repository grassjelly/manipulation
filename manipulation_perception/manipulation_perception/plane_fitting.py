"""
RANSAC plane fitting with SVD inlier refinement.
No ROS dependency.
"""
from __future__ import annotations

import numpy as np


def ransac_plane(
    points: np.ndarray,
    iterations: int = 200,
    threshold: float = 0.005,
    min_inliers: int = 8,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Fit a plane to points via RANSAC, refining the normal with SVD over
    the best inlier set.

    Parameters
    ----------
    points      : (N, 3) float array in any consistent coordinate frame.
    iterations  : number of RANSAC trials.
    threshold   : inlier distance threshold in metres.
    min_inliers : minimum required inliers for a valid fit.

    Returns
    -------
    (normal, centroid) – both (3,) float64, or (None, None) on failure.
    normal is unit-length but its sign is NOT canonicalized here.
    """
    n = len(points)
    best_normal: np.ndarray | None = None
    best_centroid: np.ndarray | None = None
    best_count = 0

    for _ in range(iterations):
        idx        = np.random.choice(n, 3, replace=False)
        p1, p2, p3 = points[idx]
        raw        = np.cross(p2 - p1, p3 - p1)
        norm       = np.linalg.norm(raw)
        if norm < 1e-8:
            continue
        normal = raw / norm

        dists   = np.abs((points - p1) @ normal)
        mask    = dists < threshold
        inliers = int(mask.sum())

        if inliers > best_count:
            best_count    = inliers
            inlier_pts    = points[mask]
            best_centroid = inlier_pts.mean(axis=0)
            # refine normal via SVD for accuracy
            _, _, Vt   = np.linalg.svd(inlier_pts - best_centroid)
            best_normal = Vt[-1]

    if best_count < min_inliers:
        return None, None

    return best_normal, best_centroid
