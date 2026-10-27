#!/usr/bin/env python3
"""Show the target button in Gazebo and RViz for the press simulation."""
import math

import numpy as np
import rclpy
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray

from rm65_button_press.button_pose_io import load_button_pose_file
from rm65_button_press.geometry import matrix_to_quaternion, quaternion_to_matrix

SDF = """<sdf version="1.6">
  <model name="button_panel">
    <static>true</static>
    <link name="link">
      <visual name="panel">
        <pose>0 0 -0.025 0 0 0</pose>
        <geometry><box><size>0.16 0.16 0.03</size></box></geometry>
        <material><ambient>0.35 0.35 0.35 1</ambient><diffuse>0.45 0.45 0.45 1</diffuse></material>
      </visual>
      <visual name="cap">
        <pose>0 0 0 0 0 0</pose>
        <geometry><cylinder><radius>0.028</radius><length>0.02</length></cylinder></geometry>
        <material><ambient>0.9 0.05 0.05 1</ambient><diffuse>0.9 0.05 0.05 1</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>"""


class SimScene(Node):

    def __init__(self):
        super().__init__('sim_scene')
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
        self.tool_offset = float(self.declare_parameter('tool_offset', 0.143).value)
        self.marker_topic = str(self.declare_parameter(
            'marker_topic', '/button_press_sim/button_markers').value)
        self.model_name = str(self.declare_parameter(
            'spawn_model_name', 'button_panel').value)
        self.spawn_service = str(self.declare_parameter(
            'spawn_service', '/spawn_entity').value)

        if len(self.button_position) != 3 or len(self.button_orientation) != 4:
            raise ValueError('button_position/button_orientation size is invalid')
        self.button_matrix = quaternion_to_matrix(self.button_orientation)
        self.button_normal = self.button_matrix[:, 2].copy()
        self.pressed = False

        marker_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, marker_qos)
        self.pose_sub = self.create_subscription(
            Pose, '/rm_driver/udp_arm_position', self.on_pose, 10)
        self.current_pose = None
        self.spawn_client = self.create_client(SpawnEntity, self.spawn_service)
        self.spawn_sent = False
        self.create_timer(0.2, self.tick)

        self.get_logger().info(
            'sim scene button at (%.4f, %.4f, %.4f) normal (%.3f, %.3f, %.3f) from %s'
            % (self.button_position[0], self.button_position[1],
               self.button_position[2], self.button_normal[0],
               self.button_normal[1], self.button_normal[2], self.pose_source))

    def on_pose(self, msg):
        self.current_pose = msg

    def button_pose(self):
        pose = Pose()
        pose.position.x = self.button_position[0]
        pose.position.y = self.button_position[1]
        pose.position.z = self.button_position[2]
        pose.orientation.x = self.button_orientation[0]
        pose.orientation.y = self.button_orientation[1]
        pose.orientation.z = self.button_orientation[2]
        pose.orientation.w = self.button_orientation[3]
        return pose

    def tick(self):
        if not self.spawn_sent:
            self.try_spawn()
        self.publish_markers()

    def try_spawn(self):
        if not self.spawn_client.service_is_ready():
            return
        self.spawn_sent = True
        request = SpawnEntity.Request()
        request.name = self.model_name
        request.xml = SDF
        request.initial_pose = self.button_pose()
        request.reference_frame = 'world'
        future = self.spawn_client.call_async(request)
        future.add_done_callback(self.on_spawn_done)

    def on_spawn_done(self, future):
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error('button model spawn failed: %s' % exc)
            return
        if result is not None and result.success:
            self.get_logger().info('button model spawned in Gazebo: %s' % self.model_name)
        else:
            message = result.status_message if result is not None else 'no result'
            self.get_logger().warn('button model spawn: %s' % message)

    def publish_markers(self):
        pressed = False
        if self.current_pose is not None:
            flange = np.array([self.current_pose.position.x,
                               self.current_pose.position.y,
                               self.current_pose.position.z])
            rotation = quaternion_to_matrix(
                (self.current_pose.orientation.x, self.current_pose.orientation.y,
                 self.current_pose.orientation.z, self.current_pose.orientation.w))
            tip = flange + rotation @ np.array([0.0, 0.0, self.tool_offset])
            distance = float(np.linalg.norm(tip - np.asarray(self.button_position)))
            pressed = distance < 0.012
        if pressed != self.pressed:
            self.get_logger().info('button %s' % ('PRESSED' if pressed else 'released'))
            self.pressed = pressed

        stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()

        cap = Marker()
        cap.header.frame_id = self.base_frame
        cap.header.stamp = stamp
        cap.ns = 'button'
        cap.id = 0
        cap.type = Marker.CYLINDER
        cap.action = Marker.ADD
        cap.pose = self.button_pose()
        cap.scale.x = 0.056
        cap.scale.y = 0.056
        cap.scale.z = 0.02
        cap.color.r = 0.1 if pressed else 0.9
        cap.color.g = 0.9 if pressed else 0.05
        cap.color.b = 0.1
        cap.color.a = 1.0
        markers.markers.append(cap)

        panel = Marker()
        panel.header = cap.header
        panel.ns = 'button'
        panel.id = 1
        panel.type = Marker.CUBE
        panel.action = Marker.ADD
        panel.pose = self.button_pose()
        panel.pose.position.x -= self.button_normal[0] * 0.025
        panel.pose.position.y -= self.button_normal[1] * 0.025
        panel.pose.position.z -= self.button_normal[2] * 0.025
        panel.scale.x = 0.16
        panel.scale.y = 0.16
        panel.scale.z = 0.03
        panel.color.r = 0.4
        panel.color.g = 0.4
        panel.color.b = 0.4
        panel.color.a = 1.0
        markers.markers.append(panel)

        arrow_matrix = np.column_stack(
            (-self.button_normal, self.button_matrix[:, 1], self.button_matrix[:, 0]))
        arrow_q = matrix_to_quaternion(arrow_matrix)
        arrow = Marker()
        arrow.header = cap.header
        arrow.ns = 'button'
        arrow.id = 2
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        arrow.pose.position.x = self.button_position[0] + self.button_normal[0] * 0.10
        arrow.pose.position.y = self.button_position[1] + self.button_normal[1] * 0.10
        arrow.pose.position.z = self.button_position[2] + self.button_normal[2] * 0.10
        arrow.pose.orientation.x = arrow_q[0]
        arrow.pose.orientation.y = arrow_q[1]
        arrow.pose.orientation.z = arrow_q[2]
        arrow.pose.orientation.w = arrow_q[3]
        arrow.scale.x = 0.10
        arrow.scale.y = 0.012
        arrow.scale.z = 0.02
        arrow.color.r = 0.1
        arrow.color.g = 0.4
        arrow.color.b = 1.0
        arrow.color.a = 0.9
        markers.markers.append(arrow)

        text = Marker()
        text.header = cap.header
        text.ns = 'button'
        text.id = 3
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = self.button_position[0] + self.button_normal[0] * 0.16
        text.pose.position.y = self.button_position[1] + self.button_normal[1] * 0.16
        text.pose.position.z = self.button_position[2] + self.button_normal[2] * 0.16 + 0.04
        text.scale.z = 0.03
        text.color.r = 0.1 if pressed else 1.0
        text.color.g = 1.0 if pressed else 0.3
        text.color.b = 0.1
        text.color.a = 1.0
        text.text = 'PRESSED' if pressed else 'BUTTON'
        markers.markers.append(text)

        self.marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = SimScene()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
