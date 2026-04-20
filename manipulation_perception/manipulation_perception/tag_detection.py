"""
AprilTag / ArUco detection and pixel-to-3D projection.
No ROS dependency.
"""
from __future__ import annotations

import cv2
import numpy as np

ARUCO_FAMILY_MAP: dict[str, int] = {
    'DICT_APRILTAG_36h11': cv2.aruco.DICT_APRILTAG_36h11,
    'DICT_APRILTAG_25h9':  cv2.aruco.DICT_APRILTAG_25h9,
    'DICT_APRILTAG_16h5':  cv2.aruco.DICT_APRILTAG_16h5,
}


class _LegacyDetector:
    """Thin wrapper around the pre-4.7 cv2.aruco.detectMarkers API."""

    def __init__(self, aruco_id: int) -> None:
        self._dict   = cv2.aruco.Dictionary_get(aruco_id)
        self._params = cv2.aruco.DetectorParameters_create()

    def detectMarkers(self, gray: np.ndarray):
        return cv2.aruco.detectMarkers(gray, self._dict, parameters=self._params)


def create_detector(family_str: str):
    """
    Return a detector object with a ``detectMarkers(gray)`` method.

    Uses ``cv2.aruco.ArucoDetector`` on OpenCV ≥ 4.7 to avoid segfaults
    caused by mixing the old ``detectMarkers`` free function with the new
    ``DetectorParameters`` / ``getPredefinedDictionary`` objects.
    Falls back to a legacy wrapper on older OpenCV builds.
    """
    aruco_id = ARUCO_FAMILY_MAP.get(family_str)
    if aruco_id is None:
        raise ValueError(
            f'Unknown tag family "{family_str}". '
            f'Valid values: {list(ARUCO_FAMILY_MAP)}'
        )

    if hasattr(cv2.aruco, 'ArucoDetector'):
        d = cv2.aruco.getPredefinedDictionary(aruco_id)
        p = cv2.aruco.DetectorParameters()
        return cv2.aruco.ArucoDetector(d, p)

    return _LegacyDetector(aruco_id)


def detect_target_tag(
    gray: np.ndarray,
    detector,
    target_id: int,
) -> np.ndarray | None:
    """
    Run the detector and return the (4, 2) corner array for *target_id*,
    or None if not found.
    Corners are ordered: top-left, top-right, bottom-right, bottom-left.
    """
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None
    for i, tag_id in enumerate(ids.flatten()):
        if int(tag_id) == target_id:
            return corners[i][0]   # shape (4, 2)
    return None


def _depth_to_metres(depth_img: np.ndarray) -> np.ndarray:
    """Convert a raw depth image to float64 metres regardless of encoding.

    16UC1 (uint16) → values are in millimetres → multiply by 1e-3.
    32FC1 (float32) → values are already in metres → cast only.
    """
    if depth_img.dtype == np.float32 or depth_img.dtype == np.float64:
        return depth_img.astype(np.float64)
    return depth_img.astype(np.float64) * 1e-3


def bbox_points_to_3d(
    tag_corners: np.ndarray,
    depth_img: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray | None:
    """
    Lift every valid depth pixel inside the tag bounding box to 3D.

    depth_img : aligned depth image (uint16 mm or float32 m).
    Returns (N, 3) float64 array in camera frame, or None if too few points.
    """
    H, W = depth_img.shape[:2]
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    u_min = int(np.clip(tag_corners[:, 0].min(), 0, W - 1))
    u_max = int(np.clip(tag_corners[:, 0].max(), 0, W - 1))
    v_min = int(np.clip(tag_corners[:, 1].min(), 0, H - 1))
    v_max = int(np.clip(tag_corners[:, 1].max(), 0, H - 1))

    patch  = _depth_to_metres(depth_img[v_min:v_max + 1, u_min:u_max + 1])
    vs, us = np.mgrid[v_min:v_max + 1, u_min:u_max + 1]
    valid  = patch > 0.0

    if valid.sum() < 10:
        return None

    d = patch[valid]
    X = (us[valid] - cx) * d / fx
    Y = (vs[valid] - cy) * d / fy
    return np.column_stack([X, Y, d])


def corners_to_3d(
    tag_corners: np.ndarray,
    depth_img: np.ndarray,
    camera_matrix: np.ndarray,
) -> list[np.ndarray] | None:
    """
    Project each of the 4 tag corner pixels to a 3D point using the depth
    image. Falls back to a 3×3 neighbourhood mean for zero-depth pixels.
    Returns a list of 4 (3,) arrays, or None if any corner has no depth.
    """
    H, W = depth_img.shape[:2]
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    result: list[np.ndarray] = []
    for u, v in tag_corners:
        ui = int(np.clip(round(u), 0, W - 1))
        vi = int(np.clip(round(v), 0, H - 1))
        d  = float(_depth_to_metres(depth_img[vi:vi+1, ui:ui+1])[0, 0])

        if d <= 0.0:
            u0, u1 = max(0, ui - 1), min(W - 1, ui + 1)
            v0, v1 = max(0, vi - 1), min(H - 1, vi + 1)
            patch  = _depth_to_metres(depth_img[v0:v1 + 1, u0:u1 + 1])
            valid  = patch[patch > 0.0]
            if len(valid) == 0:
                return None
            d = float(valid.mean())

        result.append(np.array([(u - cx) * d / fx, (v - cy) * d / fy, d]))
    return result
