#!/usr/bin/env python3
"""Triggered fake button detection for the simulation."""
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool

from rm65_button_press.button_pose_io import load_button_pose_file


class SimVision(Node):

    def __init__(self):
        super().__init__('sim_vision')
        self.button_position = [float(v) for v in self.declare_parameter(
            'button_position', [-0.4737, -0.0035, 0.4538]).value]
        self.button_orientation = [float(v) for v in self.declare_parameter(
            'button_orientation', [0.5459, 0.4920, 0.5064, 0.4511]).value]
        self.base_frame = str(self.declare_parameter('base_frame', 'base_link').value)
        self.button_pose_file = str(self.declare_parameter(
            'button_pose_file', '').value)
        if self.button_pose_file:
            try:
                position, orientation, frame_id = load_button_pose_file(
                    self.button_pose_file)
            except (OSError, ValueError) as exc:
                raise ValueError('failed to load button_pose_file %s: %s' % (
                    self.button_pose_file, exc)) from exc
            if frame_id and frame_id != self.base_frame:
                self.get_logger().warn(
                    'button pose frame %s differs from base_frame %s; '
                    'interpreting it as %s'
                    % (frame_id, self.base_frame, self.base_frame))
            self.button_position = position
            self.button_orientation = orientation
        self.pose_source = self.button_pose_file or 'parameters'
        self.trigger_topic = str(self.declare_parameter(
            'trigger_topic', '/ibvs/capture_trigger').value)
        self.button_pose_topic = str(self.declare_parameter(
            'button_pose_topic', '/ibvs/button_pose').value)
        self.status_topic = str(self.declare_parameter(
            'status_topic', '/ibvs/capture_success').value)
        self.settle_duration = float(self.declare_parameter(
            'settle_duration', 0.5).value)

        output_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pose_pub = self.create_publisher(
            PoseStamped, self.button_pose_topic, output_qos)
        self.status_pub = self.create_publisher(Bool, self.status_topic, output_qos)
        self.trigger_sub = self.create_subscription(
            Bool, self.trigger_topic, self.on_trigger, 10)
        self.pending_time = None
        self.create_timer(0.05, self.tick)
        self.get_logger().info(
            'sim vision ready; waiting for %s, button %s from %s'
            % (self.trigger_topic, self.button_position, self.pose_source))

    def on_trigger(self, msg):
        if not msg.data:
            return
        if self.pending_time is not None:
            self.get_logger().warn('detection already pending; trigger ignored')
            return
        self.pending_time = self.get_clock().now()
        self.get_logger().info(
            'received capture trigger; settling %.2fs' % self.settle_duration)

    def tick(self):
        if self.pending_time is None:
            return
        elapsed = (self.get_clock().now() - self.pending_time).nanoseconds / 1e9
        if elapsed < self.settle_duration:
            return
        message = PoseStamped()
        message.header.frame_id = self.base_frame
        message.header.stamp = self.get_clock().now().to_msg()
        message.pose.position.x = self.button_position[0]
        message.pose.position.y = self.button_position[1]
        message.pose.position.z = self.button_position[2]
        message.pose.orientation.x = self.button_orientation[0]
        message.pose.orientation.y = self.button_orientation[1]
        message.pose.orientation.z = self.button_orientation[2]
        message.pose.orientation.w = self.button_orientation[3]
        self.pose_pub.publish(message)
        status = Bool()
        status.data = True
        self.status_pub.publish(status)
        self.pending_time = None
        self.get_logger().info(
            '6D button pose published: (%.4f, %.4f, %.4f) q=(%.4f, %.4f, %.4f, %.4f)'
            % (self.button_position[0], self.button_position[1],
               self.button_position[2], self.button_orientation[0],
               self.button_orientation[1], self.button_orientation[2],
               self.button_orientation[3]))


def main(args=None):
    rclpy.init(args=args)
    node = SimVision()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
