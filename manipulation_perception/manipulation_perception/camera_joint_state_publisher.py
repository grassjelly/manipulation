import cv2
import rclpy
import rclpy.duration
import rclpy.time
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
import numpy as np
import tf2_ros
from scipy.spatial.transform import Rotation

from .tag_detection import (
    ARUCO_FAMILY_MAP,
    create_detector,
    detect_target_tag,
    bbox_points_to_3d,
    corners_to_3d,
)
# create_detector returns a single detector object (ArucoDetector on OpenCV ≥ 4.7,
# _LegacyDetector on older builds) to avoid the segfault caused by mixing the new
# getPredefinedDictionary/DetectorParameters objects with the old detectMarkers free fn.
from .plane_fitting import ransac_plane
from .transform_math import compute_tag_frame, build_T_ref_cam, rotation_to_quaternion


class CameraTagTFNode(Node):
    def __init__(self):
        super().__init__('camera_tag_tf')

        self.declare_parameter('camera_topic',      '/camera/camera/color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('depth_topic',       '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('tag_family',        'DICT_APRILTAG_25h9')
        self.declare_parameter('tag_id',            0)
        self.declare_parameter('tag_size',          0.05)
        self.declare_parameter('tag_x',             0.45)
        self.declare_parameter('tag_y',             -0.245)
        self.declare_parameter('tag_z',             0.0)
        self.declare_parameter('tag_roll',           3.14159)  # π: Z into tag (library convention)
        self.declare_parameter('tag_pitch',         0.0)
        self.declare_parameter('tag_yaw',          -1.5708)  # −π/2: printed "up" = link_base +X
        self.declare_parameter('reference_frame',   'link_base')
        self.declare_parameter('camera_frame',      'camera_link')
        self.declare_parameter('tag_frame',         'april_tag')

        def gp(name):
            return self.get_parameter(name).get_parameter_value()

        camera_topic      = gp('camera_topic').string_value
        camera_info_topic = gp('camera_info_topic').string_value
        depth_topic       = gp('depth_topic').string_value
        tag_family_str    = gp('tag_family').string_value
        self.target_id    = gp('tag_id').integer_value
        self.tag_size     = gp('tag_size').double_value
        self.tag_x        = gp('tag_x').double_value
        self.tag_y        = gp('tag_y').double_value
        self.tag_z        = gp('tag_z').double_value
        self.tag_roll     = gp('tag_roll').double_value
        self.tag_pitch    = gp('tag_pitch').double_value
        self.tag_yaw      = gp('tag_yaw').double_value
        self.ref_frame    = gp('reference_frame').string_value
        self.cam_frame    = gp('camera_frame').string_value
        self.tag_frame    = gp('tag_frame').string_value

        if tag_family_str not in ARUCO_FAMILY_MAP:
            self.get_logger().error(
                f'Unknown tag_family "{tag_family_str}". '
                f'Valid values: {list(ARUCO_FAMILY_MAP)}. Defaulting to DICT_APRILTAG_25h9.'
            )
            tag_family_str = 'DICT_APRILTAG_25h9'

        self._detector = create_detector(tag_family_str)

        self._publish_tag_static_tf()

        self.bridge               = CvBridge()
        self.camera_matrix        = None
        self.camera_optical_frame = None   # read from CameraInfo header
        self.latest_depth         = None
        self._cached_tf           = None
        self._broadcasting        = False
        self._T_camlink_optical   = None   # cached fixed rotation, looked up once

        self.image_sub = self.create_subscription(Image,      camera_topic,      self._image_cb, 10)
        self.info_sub  = self.create_subscription(CameraInfo, camera_info_topic, self._info_cb,  10)
        self.depth_sub = self.create_subscription(Image,      depth_topic,       self._depth_cb, 10)

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer      = tf2_ros.Buffer()
        self.tf_listener    = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_timer       = None

        self.get_logger().info(
            f'camera_tag_tf started — id={self.target_id} family={tag_family_str} '
            f'size={self.tag_size * 1e3:.0f} mm '
            f'ref="{self.ref_frame}" cam="{self.cam_frame}"'
        )

    # ── static tag TF ─────────────────────────────────────────────────────

    def _publish_tag_static_tf(self) -> None:
        broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        t = TransformStamped()
        t.header.stamp    = self.get_clock().now().to_msg()
        t.header.frame_id = self.ref_frame
        t.child_frame_id  = self.tag_frame

        t.transform.translation.x = self.tag_x
        t.transform.translation.y = self.tag_y
        t.transform.translation.z = self.tag_z

        q = rotation_to_quaternion(
            Rotation.from_euler('xyz', [self.tag_roll, self.tag_pitch, self.tag_yaw]).as_matrix()
        )
        t.transform.rotation.x = float(q[0])
        t.transform.rotation.y = float(q[1])
        t.transform.rotation.z = float(q[2])
        t.transform.rotation.w = float(q[3])

        broadcaster.sendTransform(t)
        self.get_logger().info(
            f'Static TF published: "{self.ref_frame}" → "{self.tag_frame}" '
            f'xyz=({self.tag_x},{self.tag_y},{self.tag_z}) '
            f'rpy=({self.tag_roll:.3f},{self.tag_pitch:.3f},{self.tag_yaw:.3f})'
        )

    # ── sensor callbacks ──────────────────────────────────────────────────

    def _info_cb(self, msg: CameraInfo) -> None:
        if self.camera_matrix is None:
            self.camera_matrix        = np.array(msg.k).reshape(3, 3)
            self.camera_optical_frame = msg.header.frame_id
            self.get_logger().info(
                f'Camera intrinsics received. Optical frame: "{self.camera_optical_frame}"'
            )

    def _depth_cb(self, msg: Image) -> None:
        self.latest_depth = msg

    def _image_cb(self, msg: Image) -> None:
        if self.camera_matrix is None or self.latest_depth is None:
            self.get_logger().warn(
                'Waiting for camera info and first depth frame…',
                throttle_duration_sec=5.0,
            )
            return

        gray = cv2.cvtColor(
            self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8'),
            cv2.COLOR_BGR2GRAY,
        )
        tag_corners = detect_target_tag(gray, self._detector, self.target_id)
        if tag_corners is None:
            self.get_logger().warn(
                f'Tag id={self.target_id} not detected in frame.',
                throttle_duration_sec=5.0,
            )
            return

        depth_img = self.bridge.imgmsg_to_cv2(
            self.latest_depth, desired_encoding='passthrough')

        self._try_compute_tf(tag_corners, depth_img)

    # ── TF computation ────────────────────────────────────────────────────

    def _try_compute_tf(self, tag_corners: np.ndarray, depth_img: np.ndarray) -> None:
        # ── resolve optical→camera_link rotation (once, then cached) ─────
        if self._T_camlink_optical is None:
            if self.camera_optical_frame == self.cam_frame:
                # same frame — identity, no lookup needed
                self._T_camlink_optical = np.eye(4)
            else:
                try:
                    tf_msg = self.tf_buffer.lookup_transform(
                        self.cam_frame,
                        self.camera_optical_frame,
                        rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=1.0),
                    )
                    self._T_camlink_optical = self._tf_msg_to_matrix(tf_msg)
                    self.get_logger().info(
                        f'Cached TF "{self.cam_frame}" ← "{self.camera_optical_frame}"'
                    )
                except Exception as e:
                    self.get_logger().warn(
                        f'Waiting for TF "{self.cam_frame}"←"{self.camera_optical_frame}": {e}',
                        throttle_duration_sec=2.0,
                    )
                    return

        # ── 3D reconstruction ─────────────────────────────────────────────
        points_3d = bbox_points_to_3d(tag_corners, depth_img, self.camera_matrix)
        if points_3d is None:
            self.get_logger().warn('Too few valid depth pixels in tag bounding box.')
            return

        normal, centroid = ransac_plane(points_3d)
        if normal is None:
            self.get_logger().warn('RANSAC plane fit failed.')
            return

        corner_3d = corners_to_3d(tag_corners, depth_img, self.camera_matrix)
        if corner_3d is None:
            self.get_logger().warn('Failed to lift tag corners to 3D.')
            return

        axes = compute_tag_frame(normal, corner_3d)
        if axes is None:
            self.get_logger().warn('Degenerate tag corner geometry.')
            return

        # ── compose transforms ────────────────────────────────────────────
        x_axis, y_axis, z_axis = axes

        # T_ref_optical: pose of the color optical frame in reference_frame
        T_ref_optical = build_T_ref_cam(
            x_axis, y_axis, z_axis, centroid,
            self.tag_x, self.tag_y, self.tag_z,
            self.tag_roll, self.tag_pitch, self.tag_yaw,
        )

        # T_ref_camlink = T_ref_optical @ inv(T_camlink_optical)
        # because optical = camlink @ T_camlink_optical  →  camlink = ref @ inv(T_camlink_optical)
        self._cached_tf = T_ref_optical @ np.linalg.inv(self._T_camlink_optical)

        if not self._broadcasting:
            self._destroy_subs()
            self._broadcasting = True
            self.tf_timer = self.create_timer(0.05, self._broadcast_tf)
            self.get_logger().info(
                f'Tag {self.target_id} detected. '
                'Subscriptions destroyed; broadcasting TF at 20 Hz.'
            )

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _tf_msg_to_matrix(tf_stamped) -> np.ndarray:
        tr  = tf_stamped.transform.translation
        rot = tf_stamped.transform.rotation
        T   = np.eye(4)
        T[:3, :3] = Rotation.from_quat([rot.x, rot.y, rot.z, rot.w]).as_matrix()
        T[:3, 3]  = [tr.x, tr.y, tr.z]
        return T

    # ── TF broadcast ──────────────────────────────────────────────────────

    def _broadcast_tf(self) -> None:
        if self._cached_tf is None:
            return
        T = self._cached_tf
        t = TransformStamped()
        t.header.stamp    = self.get_clock().now().to_msg()
        t.header.frame_id = self.ref_frame
        t.child_frame_id  = self.cam_frame

        t.transform.translation.x = float(T[0, 3])
        t.transform.translation.y = float(T[1, 3])
        t.transform.translation.z = float(T[2, 3])

        q = rotation_to_quaternion(T[:3, :3])
        t.transform.rotation.x = float(q[0])
        t.transform.rotation.y = float(q[1])
        t.transform.rotation.z = float(q[2])
        t.transform.rotation.w = float(q[3])

        self.tf_broadcaster.sendTransform(t)

    # ── cleanup ───────────────────────────────────────────────────────────

    def _destroy_subs(self) -> None:
        self.destroy_subscription(self.image_sub)
        self.destroy_subscription(self.info_sub)
        self.destroy_subscription(self.depth_sub)
        self.image_sub = self.info_sub = self.depth_sub = None


def main(args=None):
    rclpy.init(args=args)
    node = CameraTagTFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
