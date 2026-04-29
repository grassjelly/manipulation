#!/usr/bin/env python3
"""
Demo combining pick and place action clients.

Calls find_objects to resolve each prompt to a list of poses, picks the first
match, then drives the pick_object and place_object action servers with that pose.
"""
from __future__ import annotations

import threading
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from manipulation_interfaces.action import PickObject, PlaceObject
from manipulation_interfaces.srv import FindObjects
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
            FindObjects, 'find_objects', callback_group=self._sensor_cb_group
        )
        self._pick_client  = ActionClient(
            self, PickObject, 'pick_object', callback_group=self._exec_cb_group
        )
        self._place_client = ActionClient(
            self, PlaceObject, 'place_object', callback_group=self._exec_cb_group
        )

        self._timer = self.create_timer(1.0, self._run, callback_group=self._exec_cb_group)

    # ── helpers ────────────────────────────────────────────────────────────

    def _find(self, prompt: str) -> FindObjects.Response | None:
        if not self._find_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('find_objects service not available')
            return None
        req = FindObjects.Request()
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
        pick_resp = self._find(_PICK_PROMPT)
        if pick_resp is None or not pick_resp.found:
            log.error(f'Could not find "{_PICK_PROMPT}": {getattr(pick_resp, "message", "")}')
            return

        log.info(f'Finding "{_PLACE_PROMPT}"…')
        place_resp = self._find(_PLACE_PROMPT)
        if place_resp is None or not place_resp.found:
            log.error(f'Could not find "{_PLACE_PROMPT}": {getattr(place_resp, "message", "")}')
            return

        # use the first detected instance of each object
        px, py, pz       = pick_resp.x[0],  pick_resp.y[0],  pick_resp.z[0]
        pqx, pqy, pqz, pqw = pick_resp.qx[0], pick_resp.qy[0], pick_resp.qz[0], pick_resp.qw[0]
        lx, ly           = place_resp.x[0], place_resp.y[0]
        lqx, lqy, lqz, lqw = place_resp.qx[0], place_resp.qy[0], place_resp.qz[0], place_resp.qw[0]

        log.info(
            f'Both objects found — pick at ({px:.3f}, {py:.3f}, {pz:.3f}), '
            f'place at ({lx:.3f}, {ly:.3f})'
        )

        # ── pick ──────────────────────────────────────────────────────────
        pick_goal = PickObject.Goal()
        pick_goal.x  = px
        pick_goal.y  = py
        pick_goal.z  = pz
        pick_goal.qx = pqx
        pick_goal.qy = pqy
        pick_goal.qz = pqz
        pick_goal.qw = pqw
        pick_goal.pre_grasp_height = 0.0  # use server default

        log.info(f'Picking at ({px:.3f}, {py:.3f}, {pz:.3f})')
        pick_result = self._send_action(self._pick_client, pick_goal, 'pick')
        if pick_result is None or not pick_result.success:
            log.error(f'Pick failed: {getattr(pick_result, "message", "goal rejected")}')
            return
        log.info('Pick succeeded')

        # ── place ─────────────────────────────────────────────────────────
        place_goal = PlaceObject.Goal()
        place_goal.x  = lx
        place_goal.y  = ly
        place_goal.qx = lqx
        place_goal.qy = lqy
        place_goal.qz = lqz
        place_goal.qw = lqw
        place_goal.drop_height = 0.0  # use server default

        log.info(f'Placing at ({lx:.3f}, {ly:.3f})')
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
