"""
ObjectFinder — ROS-free library for prompt-driven 3-D object localisation.

Given an RGB image, aligned depth image, camera intrinsics, and a 4×4
camera-to-reference transform, it segments objects by text prompt and returns
their 3-D pose (position + orientation) expressed in the reference frame.

No ROS types are used here.  The caller is responsible for obtaining
``camera_matrix`` and ``T_ref_cam`` from whatever source is available
(ROS TF, calibration YAML, OpenCV solvePnP, …).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .prompt_to_segment import SegmentResult


@dataclass
class ObjectPose:
    """3-D pose of one detected object instance, expressed in the reference frame."""
    xyz: np.ndarray           # (3,)  translation [m]
    quaternion: np.ndarray    # (4,)  rotation as [x, y, z, w]
    centroid_px: tuple[int, int]  # (u, v) pixel centroid used for deprojection
    mask: np.ndarray          # bool (H, W) — the raw segmentation mask


@runtime_checkable
class SegmentorProtocol(Protocol):
    """
    Minimal interface that any segmentor must satisfy.
    Both Sam2Segmentor and Sam3Segmentor already implement this.
    """
    def segment(self, rgb: np.ndarray, prompt: str) -> list[SegmentResult]:
        ...


class ObjectFinder:
    """
    Localises objects described by a text prompt in 3-D space.

    Parameters
    ----------
    segmentor:
        Any object implementing ``SegmentorProtocol`` (Sam2Segmentor,
        Sam3Segmentor, or a mock for testing).  Created and owned by the
        caller so that expensive model loading is not repeated.
    camera_matrix:
        3×3 pinhole intrinsic matrix K:
            [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
        Units: pixels.  Obtain once from the camera calibration or a
        sensor_msgs/CameraInfo message.
    """

    def __init__(
        self,
        segmentor: SegmentorProtocol,
        camera_matrix: np.ndarray,
    ) -> None:
        if not isinstance(segmentor, SegmentorProtocol):
            raise TypeError(f'{type(segmentor)} does not implement SegmentorProtocol')
        if camera_matrix.shape != (3, 3):
            raise ValueError(f'camera_matrix must be (3,3), got {camera_matrix.shape}')

        self._segmentor = segmentor
        self._K = camera_matrix.astype(np.float64)

    def get_object_pose(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        prompt: str,
        T_ref_cam: np.ndarray,
    ) -> list[ObjectPose]:
        """
        Segment ``prompt`` in ``rgb`` and return one 3-D pose per instance.

        Parameters
        ----------
        rgb:
            uint8 (H, W, 3) image in RGB order.
        depth:
            Depth image aligned to ``rgb``.  Accepts:
              - uint16  → values are millimetres (RealSense default)
              - float32 → values are metres
        prompt:
            Free-text description of the object to find.
        T_ref_cam:
            4×4 homogeneous transform that maps points from the **camera**
            optical frame to the **reference** frame.

            Structure:
                T = | R  t |   R ∈ SO(3),  t ∈ ℝ³
                    | 0  1 |

            In a ROS system this comes from tf2; elsewhere build it from
            calibration data::

                T = np.eye(4)
                T[:3, :3] = rotation_matrix   # camera orientation in ref frame
                T[:3, 3]  = translation_m     # camera origin in ref frame [m]

        Returns
        -------
        list[ObjectPose]
            One entry per detected instance.  Empty when nothing is found or
            no instance has a valid depth reading at its centroid.
        """
        if T_ref_cam.shape != (4, 4):
            raise ValueError(f'T_ref_cam must be (4,4), got {T_ref_cam.shape}')

        instances = self._segmentor.segment(rgb, prompt)
        poses: list[ObjectPose] = []

        for seg in instances:
            xyz = _deproject(seg.centroid_px, depth, self._K, T_ref_cam)
            if xyz is None:
                continue

            yaw_cam = _mask_to_yaw(seg.mask)
            q = _yaw_cam_to_ref_quat(yaw_cam, T_ref_cam)

            poses.append(ObjectPose(
                xyz=xyz,
                quaternion=q,
                centroid_px=seg.centroid_px,
                mask=seg.mask,
            ))

        return poses


def _deproject(
    centroid_px: tuple[int, int],
    depth: np.ndarray,
    K: np.ndarray,
    T_ref_cam: np.ndarray,
) -> np.ndarray | None:
    """
    Back-project a pixel + depth reading to a 3-D point in the reference frame.

    Pinhole camera model (forward projection):
        u = fx * (X/Z) + cx
        v = fy * (Y/Z) + cy

    Inverting for X and Y given known Z:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

    The resulting point P_cam = [X, Y, Z, 1]ᵀ (homogeneous) is then brought
    into the reference frame by the 4×4 extrinsic transform:
        P_ref = T_ref_cam @ P_cam

    Parameters
    ----------
    centroid_px:
        (u, v) pixel coordinates (column, row).
    depth:
        Aligned depth image.  uint16 → treated as mm; float32 → treated as m.
    K:
        3×3 camera intrinsic matrix.
    T_ref_cam:
        4×4 camera-to-reference homogeneous transform.

    Returns
    -------
    np.ndarray (3,) in reference-frame metres, or None if depth is invalid.
    """
    u, v = centroid_px
    H, W = depth.shape[:2]

    if not (0 <= u < W and 0 <= v < H):
        return None

    raw = float(depth[v, u])
    if raw <= 0.0:
        return None

    # Convert raw reading to metres.
    # RealSense depth streams are uint16 in millimetres; float images are already metres.
    Z = raw * 1e-3 if depth.dtype == np.uint16 else raw

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Inverse pinhole: reconstruct 3-D ray in camera frame and scale by Z.
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    # Homogeneous point in camera frame; transform to reference frame.
    P_cam = np.array([X, Y, Z, 1.0])
    return (T_ref_cam @ P_cam)[:3]


def _mask_to_yaw(mask: np.ndarray) -> float:
    """
    Estimate the dominant in-plane orientation of a segmentation mask.

    Strategy: fit a minimum-area bounding rectangle (MBR) to the mask pixels
    and read off the angle of its long axis.

    OpenCV's ``minAreaRect`` returns:
        (center, (width, height), angle)
    where ``angle`` ∈ [-90°, 0°) is the angle of the **width** edge measured
    counter-clockwise from the positive X axis (i.e. from "rightward").

    Convention ambiguity: when the rectangle is narrow-end-first, the short
    dimension is called ``width`` and the angle refers to that short edge.
    To get the long-axis angle we add 90° when width < height:

        long_axis_angle = angle + 90°   if width < height
                        = angle          otherwise

    This ensures the returned yaw always describes the long axis of the object
    as seen by the camera, in the camera's image plane.

    Parameters
    ----------
    mask:
        Boolean (H, W) array — True where the object is present.

    Returns
    -------
    float
        Yaw angle in **radians** around the camera Z axis (optical axis).
        Returns 0.0 when the mask has fewer than 5 points (MBR is undefined).
    """
    ys, xs = np.where(mask)
    if len(xs) < 5:
        return 0.0

    # minAreaRect expects (N, 1, 2) or (N, 2) float32 points.
    pts = np.column_stack([xs, ys]).astype(np.float32)
    _, (w, h), angle_deg = cv2.minAreaRect(pts)

    # Normalise so the angle describes the long axis regardless of rect orientation.
    if w < h:
        angle_deg += 90.0

    return np.deg2rad(angle_deg)


def _yaw_cam_to_ref_quat(yaw_cam: float, T_ref_cam: np.ndarray) -> np.ndarray:
    """
    Express a camera-frame yaw rotation in the reference frame as a quaternion.

    The mask yaw is a rotation around the camera's Z axis (the optical axis,
    pointing toward the scene).  To express this orientation in the reference
    frame we compose two rotations:

        R_obj_in_ref = R_ref_cam  ⊗  R_z(yaw_cam)

    Read right-to-left: first rotate by ``yaw_cam`` around the camera Z axis,
    then apply the camera-to-reference rotation.  The result is the object's
    orientation expressed in reference-frame coordinates.

    Parameters
    ----------
    yaw_cam:
        In-plane yaw in radians (output of ``_mask_to_yaw``).
    T_ref_cam:
        4×4 camera-to-reference transform; only the upper-left 3×3 block
        (the rotation R_ref_cam) is used here.

    Returns
    -------
    np.ndarray (4,)
        Quaternion [x, y, z, w] representing the object orientation in the
        reference frame.
    """
    R_ref_cam = T_ref_cam[:3, :3]

    # Pure yaw around the camera optical axis (+Z in the optical convention).
    R_z = Rotation.from_euler('z', yaw_cam).as_matrix()

    # Compose: bring the yaw into the reference frame.
    R_obj = R_ref_cam @ R_z

    return Rotation.from_matrix(R_obj).as_quat()  # [x, y, z, w]
