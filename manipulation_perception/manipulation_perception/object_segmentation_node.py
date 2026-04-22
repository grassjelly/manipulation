"""
ROS2 node for open-vocabulary object segmentation via SAM3 + LLM.

Subscribes to a synchronised colour + aligned-depth stream, runs the
Sam3Segmentor at 1 Hz, deprojets the detected centroid to 3-D and
broadcasts a TF frame under *reference_frame*.

Optionally publishes:
  ~/segmentation_vis   (sensor_msgs/Image)   — colour-mask overlay
"""
from __future__ import annotations

import time

import cv2
import numpy as np
import rclpy
import rclpy.duration
import rclpy.time
import message_filters
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
import tf2_ros
from scipy.spatial.transform import Rotation

from .prompt_to_segment import LiteLLMClient, SegmentResult
from .sam2_segmentor import Sam2Segmentor
# from .sam3_segmentor import Sam3Segmentor

_INSTANCE_COLORS: list[tuple[int, int, int]] = [
    ( 60, 200,  60),  # green
    (220,  80,  80),  # red
    ( 80, 140, 220),  # blue
    (220, 180,  50),  # yellow
    (180,  60, 220),  # purple
    ( 50, 210, 200),  # cyan
    (230, 120,  50),  # orange
    (200,  50, 130),  # pink
]


class ObjectSegmentationNode(Node):
    def __init__(self) -> None:
        super().__init__('object_segmentation')

        # ── parameters ────────────────────────────────────────────────────
        self.declare_parameter('camera_topic',           '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic',            '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic',      '/camera/camera/color/camera_info')
        self.declare_parameter('prompt',                 'fork')
        self.declare_parameter('device',                 'cuda')
        self.declare_parameter('litellm_model',          'openai/gemini-3.1-flash-lite-preview')
        self.declare_parameter('litellm_host',           'http://localhost:4000')
        self.declare_parameter('litellm_api_key',        'sk-1234')
        self.declare_parameter('camera_frame',           'camera_color_optical_frame')
        self.declare_parameter('reference_frame',        'link_base')
        self.declare_parameter('enable_visualization',   True)

        def gp(name):
            return self.get_parameter(name).get_parameter_value()

        camera_topic      = gp('camera_topic').string_value
        depth_topic       = gp('depth_topic').string_value
        camera_info_topic = gp('camera_info_topic').string_value
        self._prompt      = gp('prompt').string_value
        device            = gp('device').string_value
        self._cam_frame   = gp('camera_frame').string_value
        self._ref_frame   = gp('reference_frame').string_value
        self._enable_vis  = gp('enable_visualization').bool_value

        llm_client = LiteLLMClient(
            model    = gp('litellm_model').string_value,
            api_base = gp('litellm_host').string_value,
            api_key  = gp('litellm_api_key').string_value,
        )

        # ── internal state ─────────────────────────────────────────────────
        self._bridge          = CvBridge()
        self._camera_matrix: np.ndarray | None = None
        self._latest_color:  Image | None = None
        self._latest_depth:  Image | None = None

        # ── subscriptions ──────────────────────────────────────────────────
        self._info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self._info_cb, 10
        )

        color_sub = message_filters.Subscriber(self, Image, camera_topic)
        depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=5, slop=0.05
        )
        self._sync.registerCallback(self._sync_cb)

        # ── TF ─────────────────────────────────────────────────────────────
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer      = tf2_ros.Buffer()
        self.tf_listener    = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── optional visualisation publishers ──────────────────────────────
        if self._enable_vis:
            self._vis_pub = self.create_publisher(Image, '~/segmentation_vis', 10)

        # ── load model (blocks until ready) ────────────────────────────────
        self.get_logger().info(f'Loading Sam3Segmentor (device={device})…')
        self._segmentor = Sam2Segmentor(
            llm_client=llm_client,
            device=device,
        )
        self.get_logger().info(
            f'Sam3Segmentor ready.  prompt="{self._prompt}" '
            f'ref="{self._ref_frame}" cam="{self._cam_frame}" '
            f'vis={self._enable_vis}'
        )
   
        self.create_timer(1.0, self._segment_timer_cb)

    # ── sensor callbacks ───────────────────────────────────────────────────

    def _info_cb(self, msg: CameraInfo) -> None:
        if self._camera_matrix is None:
            self._camera_matrix = np.array(msg.k).reshape(3, 3)
            self.get_logger().info(
                f'Camera intrinsics received (frame: "{msg.header.frame_id}").'
            )

    def _sync_cb(self, color_msg: Image, depth_msg: Image) -> None:
        self._latest_color = color_msg
        self._latest_depth = depth_msg

    # ── 1 Hz segmentation timer ────────────────────────────────────────────

    def _segment_timer_cb(self) -> None:
        if (
            self._camera_matrix is None
            or self._latest_color is None
            or self._latest_depth is None
        ):
            self.get_logger().warn(
                'Waiting for camera info and first synced RGBD frame…',
                throttle_duration_sec=5.0,
            )
            return

        color_msg = self._latest_color
        depth_msg = self._latest_depth

        bgr   = self._bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        try:
            _t0 = time.monotonic()
            instances = self._segmentor.segment(rgb, self._prompt)
            segment_ms = (time.monotonic() - _t0) * 1000.0
        except Exception as exc:
            self.get_logger().error(f'Segmentation failed: {exc}', throttle_duration_sec=5.0)
            import traceback
            self.get_logger().debug(traceback.format_exc())
            return

        if not instances:
            self.get_logger().info(
                f'No segment found for prompt "{self._prompt}".',
                throttle_duration_sec=5.0,
            )
            return

        try:
            tf_cam_to_ref = self.tf_buffer.lookup_transform(
                self._ref_frame,
                self._cam_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
            T_ref_cam = _tf_msg_to_matrix(tf_cam_to_ref)
        except Exception as exc:
            self.get_logger().warn(
                f'TF "{self._ref_frame}" ← "{self._cam_frame}" unavailable: {exc}',
                throttle_duration_sec=2.0,
            )
            return

        stamp      = self.get_clock().now()
        label_slug = self._prompt.replace(' ', '_').lower()

        for idx, seg in enumerate(instances):
            xyz_ref = _deproject(seg.centroid_px, depth, self._camera_matrix, T_ref_cam)
            if xyz_ref is None:
                self.get_logger().warn(
                    f'Instance {idx}: no valid depth at centroid; skipping TF.',
                    throttle_duration_sec=2.0,
                )
                continue
            yaw_cam = _mask_to_yaw(seg.mask)
            self._publish_object_tf(stamp, f'object_{label_slug}_{idx}', xyz_ref, yaw_cam, T_ref_cam)

        if self._enable_vis:
            overlay = _draw_overlay(rgb, instances, segment_ms, self._prompt)
            vis_msg = self._bridge.cv2_to_imgmsg(
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), encoding='bgr8'
            )
            vis_msg.header = color_msg.header
            self._vis_pub.publish(vis_msg)

    # ── TF publishing ──────────────────────────────────────────────────────

    def _publish_object_tf(
        self, stamp, child_frame: str, xyz: np.ndarray,
        yaw_cam: float = 0.0, T_ref_cam: np.ndarray | None = None,
    ) -> None:
        # Rotate the MBR yaw (around camera Z / optical axis) into reference frame.
        if T_ref_cam is not None:
            R_obj = T_ref_cam[:3, :3] @ Rotation.from_euler('z', yaw_cam).as_matrix()
            q = Rotation.from_matrix(R_obj).as_quat()  # [x, y, z, w]
        else:
            q = np.array([0.0, 0.0, 0.0, 1.0])

        t = TransformStamped()
        t.header.stamp    = stamp.to_msg()
        t.header.frame_id = self._ref_frame
        t.child_frame_id  = child_frame
        t.transform.translation.x = float(xyz[0])
        t.transform.translation.y = float(xyz[1])
        t.transform.translation.z = float(xyz[2])
        t.transform.rotation.x = float(q[0])
        t.transform.rotation.y = float(q[1])
        t.transform.rotation.z = float(q[2])
        t.transform.rotation.w = float(q[3])
        self.tf_broadcaster.sendTransform(t)


# ── module-level helpers ──────────────────────────────────────────────────────

def _mask_to_yaw(mask: np.ndarray) -> float:
    """Return yaw (radians) of the long axis of the mask's minimum bounding rectangle."""
    ys, xs = np.where(mask)
    if len(xs) < 5:
        return 0.0
    pts = np.column_stack([xs, ys]).astype(np.float32)
    _, (w, h), angle = cv2.minAreaRect(pts)
    # minAreaRect angle ∈ [-90, 0): normalise so angle follows the long axis
    if w < h:
        angle += 90.0
    return np.deg2rad(angle)


def _deproject(
    centroid_px: tuple[int, int],
    depth_img: np.ndarray,
    camera_matrix: np.ndarray,
    T_ref_cam: np.ndarray,
) -> np.ndarray | None:
    u, v = centroid_px
    H, W = depth_img.shape[:2]
    if not (0 <= u < W and 0 <= v < H):
        return None

    raw = float(depth_img[v, u])
    if raw <= 0.0:
        return None

    Z = raw * 1e-3 if depth_img.dtype == np.uint16 else raw

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    p_cam = np.array([(u - cx) * Z / fx, (v - cy) * Z / fy, Z, 1.0])
    return (T_ref_cam @ p_cam)[:3]


def _draw_overlay(
    rgb: np.ndarray,
    instances: list[SegmentResult],
    segment_ms: float,
    prompt: str,
) -> np.ndarray:
    canvas = rgb.copy()
    for idx, seg in enumerate(instances):
        color = _INSTANCE_COLORS[idx % len(_INSTANCE_COLORS)]
        layer = np.zeros_like(canvas)
        layer[seg.mask] = color
        canvas = cv2.addWeighted(canvas, 0.55, layer, 0.45, 0)

        # Minimum bounding rectangle + orientation arrow
        ys, xs = np.where(seg.mask)
        if len(xs) >= 5:
            pts = np.column_stack([xs, ys]).astype(np.float32)
            rect = cv2.minAreaRect(pts)
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.drawContours(canvas, [box], 0, color, 2)
            (rx, ry), (rw, rh), rangle = rect
            if rw < rh:
                rangle += 90.0
            arrow_len = max(rw, rh) * 0.45
            ex = int(rx + arrow_len * np.cos(np.deg2rad(rangle)))
            ey = int(ry + arrow_len * np.sin(np.deg2rad(rangle)))
            cv2.arrowedLine(canvas, (int(rx), int(ry)), (ex, ey),
                            (255, 255, 0), 2, tipLength=0.3)

        u, v = seg.centroid_px
        cv2.drawMarker(canvas, (u, v), color, cv2.MARKER_CROSS, 24, 2)
        cv2.putText(canvas, str(idx), (u + 14, v - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    H, W = canvas.shape[:2]
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    shadow = (0, 0, 0)

    time_text = f'{segment_ms:.0f} ms'
    tx, ty = 8, H - 8
    cv2.putText(canvas, time_text, (tx + 1, ty + 1), font, scale, shadow, thickness, cv2.LINE_AA)
    cv2.putText(canvas, time_text, (tx, ty),         font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    (pw, _), _ = cv2.getTextSize(prompt, font, scale, thickness)
    px, py = W - pw - 8, H - 8
    cv2.putText(canvas, prompt, (px + 1, py + 1), font, scale, shadow, thickness, cv2.LINE_AA)
    cv2.putText(canvas, prompt, (px, py),         font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return canvas


def _tf_msg_to_matrix(tf_stamped) -> np.ndarray:
    tr  = tf_stamped.transform.translation
    rot = tf_stamped.transform.rotation
    T   = np.eye(4)
    T[:3, :3] = Rotation.from_quat([rot.x, rot.y, rot.z, rot.w]).as_matrix()
    T[:3, 3]  = [tr.x, tr.y, tr.z]
    return T


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ObjectSegmentationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
