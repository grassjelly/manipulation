#!/usr/bin/env python3
"""
Action server for placing an object at a given (x, y) location.

Flow: move to (x, y, drop_height) → open gripper
"""
from __future__ import annotations

import rclpy
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from manipulation_apps.moveit import MoveIt
from manipulation_apps.gripper import Gripper
from manipulation_apps.utils import make_pose
from manipulation_interfaces.action import PlaceObject


class PlaceObjectServer(Node):
    def __init__(self) -> None:
        super().__init__('place_object_server')

        self.declare_parameter('reference_frame',    'link_base')
        self.declare_parameter('default_drop_height', 0.15)

        def gp(name):
            return self.get_parameter(name).get_parameter_value()

        self._ref_frame           = gp('reference_frame').string_value
        self._default_drop_height = gp('default_drop_height').double_value

        self._exec_cb_group = ReentrantCallbackGroup()
        self._action_server = ActionServer(
            self,
            PlaceObject,
            'place_object',
            self._execute_cb,
            callback_group=self._exec_cb_group,
        )

        self._robot   = MoveIt(use_sim_time=False)
        self._gripper = Gripper()
        self.get_logger().info('PlaceObjectServer ready.')

    # ── action execution ───────────────────────────────────────────────────

    def _execute_cb(self, goal_handle):
        log = self.get_logger()
        req = goal_handle.request

        def feedback(status: str) -> None:
            fb = PlaceObject.Feedback()
            fb.status = status
            goal_handle.publish_feedback(fb)

        def abort(message: str):
            log.error(message)
            goal_handle.abort()
            result = PlaceObject.Result()
            result.success = False
            result.message = message
            return result

        x, y = req.x, req.y
        qx, qy, qz, qw = req.qx, req.qy, req.qz, req.qw

        h = req.drop_height
        if h <= 0.0:
            h = self._default_drop_height

        feedback('Moving to drop position')
        self._robot.move_to(
            pose=make_pose(self._ref_frame, x, y, h, qx, qy, qz, qw)
        )
        self._gripper.open()

        log.info(f'Place complete at ({x:.3f}, {y:.3f}, z={h:.3f})')
        goal_handle.succeed()
        result = PlaceObject.Result()
        result.success = True
        result.message = 'Place complete'
        return result


def main(args=None) -> None:
    rclpy.init(args=args)
    server   = PlaceObjectServer()
    executor = MultiThreadedExecutor()
    executor.add_node(server)
    executor.add_node(server._gripper)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
