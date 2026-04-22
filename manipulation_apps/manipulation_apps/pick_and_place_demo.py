#!/usr/bin/env python3
"""
Demo combining pick and place action clients.

Calls find_object to resolve each prompt to a pose, then drives the
pick_object and place_object action servers with that pose.
"""
from __future__ import annotations

import threading
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from manipulation_interfaces.action import PickObject, PlaceObject
from manipulation_interfaces.srv import FindObject
from manipulation_apps.moveit import MoveIt

_PICK_PROMPT  = 'green block'
_PLACE_PROMPT = 'square container'


class PickAndPlaceDemo(Node):
    def __init__(self) -> None:
        super().__init__('pick_and_place_demo')

        self._sensor_cb_group = MutuallyExclusiveCallbackGroup()
        self._exec_cb_group   = ReentrantCallbackGroup()

        self._robot = MoveIt(use_sim_time=False)

        self._find_client  = self.create_client(
            FindObject, 'find_object', callback_group=self._sensor_cb_group
        )
        self._pick_client  = ActionClient(
            self, PickObject, 'pick_object', callback_group=self._exec_cb_group
        )
        self._place_client = ActionClient(
            self, PlaceObject, 'place_object', callback_group=self._exec_cb_group
        )

        self._timer = self.create_timer(1.0, self._run, callback_group=self._exec_cb_group)

    # ── helpers ────────────────────────────────────────────────────────────

    def _find(self, prompt: str) -> FindObject.Response | None:
        if not self._find_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('find_object service not available')
            return None
        req = FindObject.Request()
        req.object_prompt = prompt
        future = self._find_client.call_async(req)
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        done.wait()
        return future.result()

    def _send_action(self, client, goal, label: str):
        log = self.get_logger()
        future = client.send_goal_async(
            goal,
            feedback_callback=lambda fb: log.info(f'[{label}] {fb.feedback.status}'),
        )
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        done.wait()
        handle = future.result()

        if not handle.accepted:
            log.error(f'{label} goal rejected')
            return None

        result_future = handle.get_result_async()
        done2 = threading.Event()
        result_future.add_done_callback(lambda _: done2.set())
        done2.wait()
        return result_future.result().result

    # ── demo sequence ──────────────────────────────────────────────────────

    def _run(self) -> None:
        self._timer.destroy()
        log = self.get_logger()

        self._pick_client.wait_for_server()
        self._place_client.wait_for_server()

        # ── locate both objects before moving ─────────────────────────────
        log.info(f'Finding "{_PICK_PROMPT}"…')
        pick_pose = self._find(_PICK_PROMPT)
        if pick_pose is None or not pick_pose.found:
            log.error(f'Could not find "{_PICK_PROMPT}": {getattr(pick_pose, "message", "")}')
            return

        log.info(f'Finding "{_PLACE_PROMPT}"…')
        place_pose = self._find(_PLACE_PROMPT)
        if place_pose is None or not place_pose.found:
            log.error(f'Could not find "{_PLACE_PROMPT}": {getattr(place_pose, "message", "")}')
            return

        log.info(
            f'Both objects found — pick at ({pick_pose.x:.3f}, {pick_pose.y:.3f}, {pick_pose.z:.3f}), '
            f'place at ({place_pose.x:.3f}, {place_pose.y:.3f})'
        )

        # ── pick ──────────────────────────────────────────────────────────
        pick_goal = PickObject.Goal()
        pick_goal.x  = pick_pose.x
        pick_goal.y  = pick_pose.y
        pick_goal.z  = pick_pose.z
        pick_goal.qx = pick_pose.qx
        pick_goal.qy = pick_pose.qy
        pick_goal.qz = pick_pose.qz
        pick_goal.qw = pick_pose.qw
        pick_goal.pre_grasp_height = 0.0  # use server default

        log.info(f'Picking at ({pick_pose.x:.3f}, {pick_pose.y:.3f}, {pick_pose.z:.3f})')
        pick_result = self._send_action(self._pick_client, pick_goal, 'pick')
        if pick_result is None or not pick_result.success:
            log.error(f'Pick failed: {getattr(pick_result, "message", "goal rejected")}')
            return
        log.info('Pick succeeded')

        # ── place ─────────────────────────────────────────────────────────
        place_goal = PlaceObject.Goal()
        place_goal.x  = place_pose.x
        place_goal.y  = place_pose.y
        place_goal.qx = place_pose.qx
        place_goal.qy = place_pose.qy
        place_goal.qz = place_pose.qz
        place_goal.qw = place_pose.qw
        place_goal.drop_height = 0.0  # use server default

        log.info(f'Placing at ({place_pose.x:.3f}, {place_pose.y:.3f})')
        place_result = self._send_action(self._place_client, place_goal, 'place')
        if place_result is None or not place_result.success:
            log.error(f'Place failed: {getattr(place_result, "message", "goal rejected")}')
            return

        log.info('Returning home…')
        self._robot.move_to(configuration_name='home')
        log.info('Demo complete')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PickAndPlaceDemo()
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
