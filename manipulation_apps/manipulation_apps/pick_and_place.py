#!/usr/bin/env python3
"""
Pick-and-place demo using the robot_arm_python library, wrapped in a ROS2 node.

MoveIt and Gripper both manage their own internal executors, so only this node
needs to be added to the MultiThreadedExecutor.
"""
import rclpy
import rclpy.logging
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from manipulation_apps.moveit import MoveIt
# from robot_arm_python.gripper import Gripper
from manipulation_apps.utils import make_pose


class RobotArmNode(Node):
    def __init__(self) -> None:
        super().__init__("robot_arm_node")

        self.robot = MoveIt(use_sim_time=False)
        # self.gripper = Gripper()

        # Defer motion until the executor is spinning.
        self._timer = self.create_timer(1.0, self._run)

    def _run(self) -> None:
        self._timer.destroy()

        log = self.get_logger()

        # ------------------------------------------------------------------
        # 1. Go home
        # ------------------------------------------------------------------
        log.info("Step 1: moving to 'home' configuration")
        self.robot.move_to(configuration_name="home")

        # ------------------------------------------------------------------
        # 2. Move to pre-grasp pose
        # ------------------------------------------------------------------
        log.info("Step 2: moving to pre-grasp pose")
        self.robot.move_to(
            pose=make_pose("link_base", x=0.287, y=0.0, z=0.25,
                           qx=1.0, qy=6.49523e-05, qz=2.89529e-05, qw=-1.87263e-06)
        )
        # self.gripper.open()

        # ------------------------------------------------------------------
        # 3. Descend to grasp pose and close gripper
        # ------------------------------------------------------------------
        log.info("Step 3: descending to grasp pose")
        self.robot.move_to(
            pose=make_pose("link_base", x=0.287, y=0.0, z=0.15358,
                           qx=1.0, qy=6.49523e-05, qz=2.89529e-05, qw=-1.87263e-06)
        )
        # self.gripper.close()

        # ------------------------------------------------------------------
        # 4. Move to pre-grasp pose
        # ------------------------------------------------------------------
        log.info("Step 4: moving to pre-grasp pose")
        self.robot.move_to(
            pose=make_pose("link_base", x=0.287, y=0.0, z=0.25,
                           qx=1.0, qy=6.49523e-05, qz=2.89529e-05, qw=-1.87263e-06)
        )
        # self.gripper.open()


        # ------------------------------------------------------------------
        # 5. Move to drop pose via joint-space goal
        # ------------------------------------------------------------------
        log.info("Step 4: moving to drop pose (joint-space)")
        self.robot.move_to(
            joint_positions={
                "joint1": 0.4862401783466339,
                "joint2": 0.756127119064331,
                "joint3": 1.144195318222046,
                "joint4": 0.022067390382289886,
                "joint5": 0.4046036899089813,
                "joint6": 0.44482606649398804,
            }
        )
        # self.gripper.open()

        # ------------------------------------------------------------------
        # 5. Return home
        # ------------------------------------------------------------------
        log.info("Step 5: returning home")
        self.robot.move_to(configuration_name="home")

        log.info("Demo complete")
        self.destroy_node()
        rclpy.shutdown()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RobotArmNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()