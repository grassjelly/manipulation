#!/usr/bin/env python3
"""
ROS2 service server that exposes ObjectFinder over the find_objects service.

Subscribes to a synchronised colour + depth stream and, on each service call,
runs ObjectFinder against the latest cached frame, returning all detected poses.
"""
from __future__ import annotations

import cv2
import numpy as np
import rclpy
import rclpy.duration
import rclpy.time
import message_filters
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import tf2_ros
from scipy.spatial.transform import Rotation

from manipulation_perception.sam3_segmentor import Sam3Segmentor
from manipulation_perception.object_finder import ObjectFinder
from manipulation_interfaces.srv import FindObjects


class ObjectsFinderServer(Node):
    def __init__(self) -> None:
        super().__init__('objects_finder_server')

        self.declare_parameter('camera_topic',      '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic',       '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('device',            'cuda')
        self.declare_parameter('camera_frame',      'camera_color_optical_frame')
        self.declare_parameter('reference_frame',   'link_base')

        def gp(name):
            return self.get_parameter(name).get_parameter_value()

        camera_topic      = gp('camera_topic').string_value
        depth_topic       = gp('depth_topic').string_value
        camera_info_topic = gp('camera_info_topic').string_value
        device            = gp('device').string_value
        self._cam_frame   = gp('camera_frame').string_value
        self._ref_frame   = gp('reference_frame').string_value

        self._sensor_cb_group = MutuallyExclusiveCallbackGroup()
        self._exec_cb_group   = ReentrantCallbackGroup()

        self._bridge        = CvBridge()
        self._latest_color: Image | None = None
        self._latest_depth: Image | None = None
        self._finder: ObjectFinder | None = None

        self._info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self._info_cb, 10,
            callback_group=self._sensor_cb_group,
        )

        color_sub = message_filters.Subscriber(self, Image, camera_topic)
        depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=5, slop=0.05
        )
        self._sync.registerCallback(self._sync_cb)

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._service = self.create_service(
            FindObjects, 'find_objects', self._handle,
            callback_group=self._exec_cb_group,
        )

        self.get_logger().info(f'Loading Sam3Segmentor (device={device})…')
        self._segmentor = Sam3Segmentor(device=device)
        self.get_logger().info(
            f'ObjectsFinderServer ready. '
            f'ref="{self._ref_frame}" cam="{self._cam_frame}"'
        )

    # ── sensor callbacks ───────────────────────────────────────────────────

    def _info_cb(self, msg: CameraInfo) -> None:
        if self._finder is None:
            camera_matrix = np.array(msg.k).reshape(3, 3)
            self._finder = ObjectFinder(self._segmentor, camera_matrix)
            self.get_logger().info('Camera intrinsics received. ObjectFinder ready.')

    def _sync_cb(self, color_msg: Image, depth_msg: Image) -> None:
        self._latest_color = color_msg
        self._latest_depth = depth_msg

    # ── service handler ────────────────────────────────────────────────────

    def _handle(self, request: FindObjects.Request, response: FindObjects.Response):
        log = self.get_logger()
        prompt = request.object_prompt

        if self._finder is None or self._latest_color is None or self._latest_depth is None:
            response.found   = False
            response.message = 'Camera not ready'
            log.warn(response.message, throttle_duration_sec=5.0)
            return response

        bgr   = self._bridge.imgmsg_to_cv2(self._latest_color, desired_encoding='bgr8')
        rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = self._bridge.imgmsg_to_cv2(self._latest_depth, desired_encoding='passthrough')

        try:
            tf_cam_to_ref = self.tf_buffer.lookup_transform(
                self._ref_frame,
                self._cam_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5),
            )
            T_ref_cam = _tf_msg_to_matrix(tf_cam_to_ref)
        except Exception as exc:
            response.found   = False
            response.message = f'TF lookup failed: {exc}'
            log.error(response.message)
            return response

        try:
            poses = self._finder.get_object_pose(rgb, depth, prompt, T_ref_cam)
        except Exception as exc:
            response.found   = False
            response.message = f'ObjectFinder failed: {exc}'
            log.error(response.message)
            return response

        if not poses:
            response.found   = False
            response.message = f'Object not found: "{prompt}"'
            log.info(response.message)
            return response

        response.found   = True
        response.message = f'Found {len(poses)} object(s) matching "{prompt}"'
        response.x  = [float(p.xyz[0])        for p in poses]
        response.y  = [float(p.xyz[1])        for p in poses]
        response.z  = [float(p.xyz[2])        for p in poses]
        response.qx = [float(p.quaternion[0]) for p in poses]
        response.qy = [float(p.quaternion[1]) for p in poses]
        response.qz = [float(p.quaternion[2]) for p in poses]
        response.qw = [float(p.quaternion[3]) for p in poses]
        log.info(
            f'Found {len(poses)} "{prompt}" — '
            f'first at ({response.x[0]:.3f}, {response.y[0]:.3f}, {response.z[0]:.3f})'
        )
        return response


def _tf_msg_to_matrix(tf_stamped) -> np.ndarray:
    tr  = tf_stamped.transform.translation
    rot = tf_stamped.transform.rotation
    T   = np.eye(4)
    T[:3, :3] = Rotation.from_quat([rot.x, rot.y, rot.z, rot.w]).as_matrix()
    T[:3, 3]  = [tr.x, tr.y, tr.z]
    return T


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ObjectsFinderServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
