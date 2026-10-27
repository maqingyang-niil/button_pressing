#!/usr/bin/env python3
"""Command-level gripper backend for the simulation.

The Gazebo model's S4C2 joints are fixed closed, so this node only
acknowledges the close command; it does not animate a gripper joint.
"""
import rclpy
from rclpy.node import Node
from rm_ros_interfaces.msg import Gripperset
from std_msgs.msg import Bool


class SimGripper(Node):

    def __init__(self):
        super().__init__('sim_gripper')
        self.cmd_topic = str(self.declare_parameter(
            'ctag_cmd_topic', '/rm_driver/set_ctag_gripper_position_cmd').value)
        self.result_topic = str(self.declare_parameter(
            'ctag_result_topic',
            '/rm_driver/set_ctag_gripper_position_result').value)
        self.result_delay = float(self.declare_parameter(
            'result_delay', 0.5).value)

        self.result_pub = self.create_publisher(Bool, self.result_topic, 10)
        self.cmd_sub = self.create_subscription(
            Gripperset, self.cmd_topic, self.on_command, 10)
        self.pending = None
        self.create_timer(0.05, self.tick)
        self.get_logger().info(
            'sim gripper ready: %s -> %s; Gazebo gripper is fixed closed'
            % (self.cmd_topic, self.result_topic))

    def on_command(self, msg):
        if not 0 <= msg.position <= 1000:
            self.get_logger().warn(
                'invalid gripper position %d; ignoring' % msg.position)
            return
        self.pending = (self.get_clock().now(), int(msg.position))
        self.get_logger().info(
            'gripper command position=%d (sim model fixed closed)'
            % msg.position)

    def tick(self):
        if self.pending is None:
            return
        start, position = self.pending
        elapsed = (self.get_clock().now() - start).nanoseconds / 1e9
        if elapsed < self.result_delay:
            return
        result = Bool()
        result.data = True
        self.result_pub.publish(result)
        self.get_logger().info(
            'gripper command acknowledged position=%d' % position)
        self.pending = None


def main(args=None):
    rclpy.init(args=args)
    node = SimGripper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
