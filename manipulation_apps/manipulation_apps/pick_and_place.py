#!/usr/bin/env python3
"""
Pick-and-place demo using the robot_arm_python library, wrapped in a ROS2 node.

MoveIt and Gripper both manage their own internal executors, so only this node
needs to be added to the MultiThreadedExecutor.
"""
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from manipulation_apps.moveit import MoveIt
from manipulation_apps.gripper import Gripper
from manipulation_apps.utils import make_pose


class RobotArmNode(Node):
    def __init__(self) -> None:
        super().__init__("robot_arm_node")

        self.robot = MoveIt(use_sim_time=False)
        self.gripper = Gripper()

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
            pose=make_pose("link_base", x=0.287, y=0.0, z=0.15,
                           qx=1.0, qy=6.49523e-05, qz=2.89529e-05, qw=-1.87263e-06)
        )
        self.gripper.open()
        # ------------------------------------------------------------------
        # 3. Descend to grasp pose and close gripper
        # ------------------------------------------------------------------
        log.info("Step 3: ascending to grasp pose")
        self.robot.move_to(
            pose=make_pose("link_base", x=0.287, y=0.0, z=0.01,
                           qx=1.0, qy=6.49523e-05, qz=2.89529e-05, qw=-1.87263e-06)
        )
        self.gripper.close()

        # ------------------------------------------------------------------
        # 3. Ascent to pre-grasp pose
        # ------------------------------------------------------------------
        log.info("Step 4: moving to pre-grasp pose")
        self.robot.move_to(
            pose=make_pose("link_base", x=0.287, y=0.0, z=0.15,
                           qx=1.0, qy=6.49523e-05, qz=2.89529e-05, qw=-1.87263e-06)
        )
        self.gripper.open()

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