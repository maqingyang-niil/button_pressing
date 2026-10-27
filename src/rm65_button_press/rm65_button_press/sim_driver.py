#!/usr/bin/env python3
"""Simulated rm_driver Movej/Movel backend using MoveIt2 in Gazebo."""
import math
import threading
import time

import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from pymoveit2 import MoveIt2
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rm_ros_interfaces.msg import Movej, Movel
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener


class SimDriver(Node):

    def __init__(self):
        super().__init__('sim_driver')
        self.group_name = str(self.declare_parameter('group_name', 'rm_group').value)
        self.base_link = str(self.declare_parameter('base_link', 'base_link').value)
        self.ee_link = str(self.declare_parameter('ee_link', 'Link6').value)
        self.joint_names = list(self.declare_parameter(
            'joint_names',
            ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']).value)
        self.initial_joints_deg = [float(v) for v in self.declare_parameter(
            'initial_joints_deg', [0.0] * 6).value]
        self.initial_move_delay = float(self.declare_parameter(
            'initial_move_delay', 3.0).value)
        self.initial_max_attempts = int(self.declare_parameter(
            'initial_max_attempts', 10).value)
        self.initial_retry_period = float(self.declare_parameter(
            'initial_retry_period', 5.0).value)
        self.move_timeout = float(self.declare_parameter('move_timeout', 60.0).value)
        self.cartesian_max_step = float(self.declare_parameter(
            'cartesian_max_step', 0.005).value)
        self.cartesian_fraction_threshold = float(self.declare_parameter(
            'cartesian_fraction_threshold', 0.8).value)
        self.position_tolerance = float(self.declare_parameter(
            'position_tolerance', 0.003).value)
        self.orientation_tolerance_deg = float(self.declare_parameter(
            'orientation_tolerance_deg', 2.0).value)

        self.moveit2 = MoveIt2(
            node=self,
            joint_names=self.joint_names,
            base_link_name=self.base_link,
            end_effector_name=self.ee_link,
            group_name=self.group_name,
            ignore_new_calls_while_executing=True,
            use_move_group_action=True,
        )
        self.moveit2.planner_id = 'RRTConnect'
        self.moveit2.max_velocity = 0.3
        self.moveit2.max_acceleration = 0.2

        self.movel_sub = self.create_subscription(
            Movel, '/rm_driver/movel_cmd', self.on_movel, 10)
        self.movej_sub = self.create_subscription(
            Movej, '/rm_driver/movej_cmd', self.on_movej, 10)
        self.stop_sub = self.create_subscription(
            Bool, '/rm_driver/move_stop_cmd', self.on_stop, 10)
        self.movel_result_pub = self.create_publisher(
            Bool, '/rm_driver/movel_result', 10)
        self.movej_result_pub = self.create_publisher(
            Bool, '/rm_driver/movej_result', 10)
        self.pose_pub = self.create_publisher(
            Pose, '/rm_driver/udp_arm_position', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_timer(0.05, self.publish_pose)

        self._lock = threading.Lock()
        self._pending_movel = None
        self._pending_movej = None
        self._stop_requested = False
        self.initial_done = False
        self.get_logger().info(
            'sim driver ready: group=%s base=%s ee=%s initial=%s deg'
            % (self.group_name, self.base_link, self.ee_link,
               self.initial_joints_deg))

    def on_movel(self, msg):
        with self._lock:
            self._pending_movel = msg

    def on_movej(self, msg):
        with self._lock:
            self._pending_movej = msg

    def on_stop(self, msg):
        if msg.data:
            self._stop_requested = True

    def take_pending(self):
        with self._lock:
            if self._pending_movel is not None:
                message = self._pending_movel
                self._pending_movel = None
                return 'movel', message
            if self._pending_movej is not None:
                message = self._pending_movej
                self._pending_movej = None
                return 'movej', message
        return None, None

    def publish_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_link, self.ee_link, rclpy.time.Time())
        except Exception:
            return
        message = Pose()
        message.position.x = transform.transform.translation.x
        message.position.y = transform.transform.translation.y
        message.position.z = transform.transform.translation.z
        message.orientation.x = transform.transform.rotation.x
        message.orientation.y = transform.transform.rotation.y
        message.orientation.z = transform.transform.rotation.z
        message.orientation.w = transform.transform.rotation.w
        self.pose_pub.publish(message)

    def wait_for_move_action(self, timeout):
        client = getattr(self.moveit2, '_MoveIt2__move_action_client', None)
        if client is None:
            return False
        start = time.time()
        while rclpy.ok() and time.time() - start < timeout:
            if client.server_is_ready():
                return True
            time.sleep(0.2)
        return False

    def move_to_joints(self, joints, label):
        ok = self.moveit2.move_to_configuration(joints, timeout_sec=self.move_timeout)
        if ok:
            ok = bool(self.moveit2.wait_until_executed())
        self.get_logger().info('%s: %s' % (label, 'ok' if ok else 'failed'))
        return bool(ok)

    def move_to_pose(self, pose, label):
        stamped = PoseStamped()
        stamped.header.frame_id = self.base_link
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.pose = pose
        ok = self.moveit2.move_to_pose(
            pose=stamped,
            target_link=self.ee_link,
            tolerance_position=self.position_tolerance,
            tolerance_orientation=math.radians(self.orientation_tolerance_deg),
            cartesian=True,
            cartesian_max_step=self.cartesian_max_step,
            cartesian_fraction_threshold=self.cartesian_fraction_threshold,
            timeout_sec=self.move_timeout,
        )
        if ok:
            ok = bool(self.moveit2.wait_until_executed())
        self.get_logger().info('%s: %s' % (label, 'ok' if ok else 'failed'))
        return bool(ok)

    def run_initial_move(self):
        self.get_logger().info('waiting for MoveIt move_action server')
        if not self.wait_for_move_action(60.0):
            self.get_logger().error('MoveIt move_action server not ready')
            return False
        joints = [math.radians(value) for value in self.initial_joints_deg]
        return self.move_to_joints(joints, 'initial flipped detect pose')


def main(args=None):
    rclpy.init(args=args)
    node = SimDriver()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    start = time.time()
    next_initial_attempt = start + node.initial_move_delay
    initial_attempts = 0
    try:
        while rclpy.ok():
            now = time.time()
            if not node.initial_done and now >= next_initial_attempt:
                initial_attempts += 1
                if node.run_initial_move():
                    node.initial_done = True
                elif initial_attempts >= node.initial_max_attempts:
                    node.get_logger().error(
                        'initial detect pose failed after %d attempts; '
                        'continuing without the initial move' % initial_attempts)
                    node.initial_done = True
                else:
                    node.get_logger().warn(
                        'initial detect pose attempt %d/%d failed; retrying in %.1fs'
                        % (initial_attempts, node.initial_max_attempts,
                           node.initial_retry_period))
                    next_initial_attempt = now + node.initial_retry_period
            kind, message = node.take_pending()
            if kind == 'movel':
                ok = node.move_to_pose(message.pose, 'movel')
                result = Bool()
                result.data = ok
                node.movel_result_pub.publish(result)
            elif kind == 'movej':
                ok = node.move_to_joints([float(v) for v in message.joint], 'movej')
                result = Bool()
                result.data = ok
                node.movej_result_pub.publish(result)
            if node._stop_requested:
                node._stop_requested = False
                try:
                    node.moveit2.cancel_execution()
                except Exception:  # noqa: BLE001
                    pass
                node.get_logger().warn('move_stop received')
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
