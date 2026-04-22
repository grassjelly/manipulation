import time

import rclpy
from rclpy.node import Node
from xarm_msgs.srv import Call


class Gripper(Node):
    def __init__(self) -> None:
        super().__init__("robot_arm_gripper")

        self._open_client = self.create_client(Call, "/ufactory/open_lite6_gripper")
        self._close_client = self.create_client(Call, "/ufactory/close_lite6_gripper")
        self._stop_client = self.create_client(Call, "/ufactory/stop_lite6_gripper")

        self.get_logger().info(f"Waiting for gripper services...")
        for client in (self._open_client, self._close_client, self._stop_client):
            client.wait_for_service()
        self.get_logger().info(f"Gripper services are available.")

    def open(self) -> bool:
        self._call(self._open_client)
        time.sleep(1)
        return self._call(self._stop_client)

    def close(self) -> bool:
        ret = self._call(self._close_client)
        time.sleep(1)
        return ret

    def stop(self) -> bool:
        return self._call(self._stop_client)

    def _call(self, client) -> bool:
        future = client.call_async(Call.Request())
        rclpy.spin_until_future_complete(self, future)
        ret = future.result().ret
        if ret != 0:
            self.get_logger().warning(f"Gripper service returned error code: {ret}")
        return ret == 0
