"""
Rigid-body transform helpers for tag-to-camera calibration.
No ROS dependency.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def compute_tag_frame(
    ransac_normal: np.ndarray,
    corner_3d: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Derive orthonormal tag axes from the RANSAC plane normal and the
    3D corner positions.

    Tag convention  (matches AprilTag library)
    ------------------------------------------
    X : corner[0] → corner[1]  (right in image)
    Y : cross(Z, X)             (down in image)
    Z : RANSAC normal, INTO the tag / away from camera
        (positive camera-Z component, i.e. z_axis[2] > 0).

    Returns (x_axis, y_axis, z_axis) each (3,), or None on failure.
    """
    # Z points INTO the tag (away from camera), matching the AprilTag library convention.
    z_axis = np.array(ransac_normal, dtype=np.float64)
    if z_axis[2] < 0:
        z_axis = -z_axis

    x_axis = corner_3d[0] - corner_3d[1]
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-6:
        return None
    x_axis /= x_norm

    # Reorthogonalize against the authoritative RANSAC normal
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    return x_axis, y_axis, z_axis


def build_T_ref_cam(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    centroid_cam: np.ndarray,
    tag_x: float,
    tag_y: float,
    tag_z: float,
    tag_roll: float,
    tag_pitch: float,
    tag_yaw: float,
) -> np.ndarray:
    """
    Compute T_ref_cam (4×4) given the detected tag pose in camera frame
    and the known tag pose in the reference frame.

    T_ref_cam = T_ref_tag  @  inv(T_cam_tag)

    Parameters
    ----------
    x_axis, y_axis, z_axis : orthonormal tag axes expressed in camera frame.
    centroid_cam            : tag centroid in camera frame (3,).
    tag_x/y/z              : tag position in reference frame (metres).
    tag_roll/pitch/yaw     : tag orientation in reference frame (radians, RPY
                             applied in Z-Y-X / extrinsic XYZ order).

    Returns
    -------
    T_ref_cam : (4, 4) float64 homogeneous transform.
    """
    R_cam_tag         = np.column_stack([x_axis, y_axis, z_axis])

    T_cam_tag         = np.eye(4)
    T_cam_tag[:3, :3] = R_cam_tag
    T_cam_tag[:3, 3]  = centroid_cam

    R_ref_tag         = Rotation.from_euler('xyz', [tag_roll, tag_pitch, tag_yaw]).as_matrix()
    T_ref_tag         = np.eye(4)
    T_ref_tag[:3, :3] = R_ref_tag
    T_ref_tag[:3, 3]  = [tag_x, tag_y, tag_z]

    R_inv             = R_cam_tag.T
    T_cam_tag_inv     = np.eye(4)
    T_cam_tag_inv[:3, :3] = R_inv
    T_cam_tag_inv[:3, 3]  = -R_inv @ centroid_cam

    return T_ref_tag @ T_cam_tag_inv


def rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Return unit quaternion [x, y, z, w] from a (3, 3) rotation matrix."""
    return Rotation.from_matrix(R).as_quat()
