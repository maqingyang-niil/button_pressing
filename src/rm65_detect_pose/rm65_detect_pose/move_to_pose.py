import math
import os
import time

import rclpy
from rclpy.node import Node
from rm_ros_interfaces.msg import Gripperset, Movej
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

JOINT_NAMES = [f'joint{i}' for i in range(1, 7)]


class MoveToPose(Node):

    def __init__(self):
        super().__init__('move_to_pose')

        self.declare_parameter('joint_degrees', [0.0] * 6)
        self.declare_parameter('speed', 10)
        self.declare_parameter('delay', 2.0)
        self.declare_parameter('tolerance_deg', 0.5)
        self.declare_parameter('timeout', 60.0)
        self.declare_parameter('driver_wait_timeout', 20.0)
        self.declare_parameter('trigger_topic', '/ibvs/capture_trigger')
        self.declare_parameter('trigger_enabled', False)
        self.declare_parameter('gripper_close_enabled', True)
        self.declare_parameter('gripper_close_position', 0)
        self.declare_parameter('gripper_close_timeout', 8.0)
        self.declare_parameter('gripper_settle_time', 1.0)
        self.declare_parameter(
            'ctag_cmd_topic', '/rm_driver/set_ctag_gripper_position_cmd')
        self.declare_parameter(
            'ctag_result_topic', '/rm_driver/set_ctag_gripper_position_result')

        self.target_deg = [float(v) for v in self.get_parameter('joint_degrees').value]
        self.speed = int(self.get_parameter('speed').value)
        self.delay = float(self.get_parameter('delay').value)
        self.tolerance_deg = float(self.get_parameter('tolerance_deg').value)
        self.timeout = float(self.get_parameter('timeout').value)
        self.driver_wait_timeout = float(
            self.get_parameter('driver_wait_timeout').value)
        self.trigger_topic = str(self.get_parameter('trigger_topic').value)
        self.trigger_enabled = bool(self.get_parameter('trigger_enabled').value)
        self.gripper_close_enabled = bool(
            self.get_parameter('gripper_close_enabled').value)
        self.gripper_close_position = int(
            self.get_parameter('gripper_close_position').value)
        self.gripper_close_timeout = float(
            self.get_parameter('gripper_close_timeout').value)
        self.gripper_settle_time = float(
            self.get_parameter('gripper_settle_time').value)
        self.ctag_cmd_topic = str(self.get_parameter('ctag_cmd_topic').value)
        self.ctag_result_topic = str(
            self.get_parameter('ctag_result_topic').value)

        if len(self.target_deg) != 6:
            raise ValueError(
                f'joint_degrees must contain 6 values, got {len(self.target_deg)}')
        if not 1 <= self.speed <= 100:
            raise ValueError(f'speed must be within 1..100, got {self.speed}')
        if not 0 <= self.gripper_close_position <= 1000:
            raise ValueError(
                'gripper_close_position must be within 0..1000, got '
                f'{self.gripper_close_position}')
        if self.gripper_close_timeout <= 0.0:
            raise ValueError('gripper_close_timeout must be > 0')
        if self.gripper_settle_time < 0.0:
            raise ValueError('gripper_settle_time must be >= 0')

        self.target_rad = [math.radians(d) for d in self.target_deg]
        self.latest_positions = None
        self.latest_names = []
        self.phase = 'wait_driver'
        self.phase_start = self.get_clock().now()
        self.last_log = None
        self.exit_code = None
        self.gripper_result = None
        self.gripper_result_time = None
        self.gripper_command_sent = False
        self.gripper_command_time = None

        self.publisher = self.create_publisher(Movej, '/rm_driver/movej_cmd', 10)
        self.trigger_publisher = self.create_publisher(
            Bool, self.trigger_topic, 10)
        self.gripper_publisher = self.create_publisher(
            Gripperset, self.ctag_cmd_topic, 10)
        self.subscription = self.create_subscription(
            JointState, '/joint_states', self.on_joint_state, 10)
        self.gripper_subscription = self.create_subscription(
            Bool, self.ctag_result_topic, self.on_gripper_result, 10)
        self.timer = self.create_timer(0.1, self.tick)

        self.get_logger().info(
            'target joints (deg): ' + ', '.join(f'{v:.3f}' for v in self.target_deg) +
            f'; speed={self.speed}%; tolerance={self.tolerance_deg} deg; '
            f'gripper_close={self.gripper_close_enabled} '
            f'position={self.gripper_close_position}')

    def on_joint_state(self, msg):
        self.latest_positions = list(msg.position)
        self.latest_names = list(msg.name)

    def on_gripper_result(self, msg):
        self.gripper_result = bool(msg.data)
        self.gripper_result_time = self.get_clock().now()

    def current_target_positions(self):
        if self.latest_positions is None:
            return None
        if all(name in self.latest_names for name in JOINT_NAMES):
            return [self.latest_positions[self.latest_names.index(name)]
                    for name in JOINT_NAMES]
        if len(self.latest_positions) >= 6:
            return self.latest_positions[:6]
        return None

    def errors_deg(self):
        current = self.current_target_positions()
        if current is None:
            return None
        return [abs(math.degrees(c - t)) for c, t in zip(current, self.target_rad)]

    def format_deg(self, values):
        return ', '.join(f'{v:.3f}' for v in values)

    def elapsed(self, start):
        return (self.get_clock().now() - start).nanoseconds / 1e9

    def tick(self):
        now = self.get_clock().now()

        if self.phase == 'wait_driver':
            if self.current_target_positions() is None:
                missing = '/joint_states'
            elif self.publisher.get_subscription_count() == 0:
                missing = 'subscriber on /rm_driver/movej_cmd'
            else:
                missing = None
            if missing is None:
                self.get_logger().info(
                    f'driver ready ({self.publisher.get_subscription_count()} subscriber); '
                    'current error (deg): ' + self.format_deg(self.errors_deg()))
                self.phase = 'settle'
                self.phase_start = now
                return
            if self.elapsed(self.phase_start) > self.driver_wait_timeout:
                self.get_logger().error(
                    f'no {missing} after {self.driver_wait_timeout:.0f}s; '
                    f'check that rm_driver is running and that ROS_DOMAIN_ID matches '
                    f'(this process: {os.environ.get("ROS_DOMAIN_ID", "0")})')
                self.exit_code = 1
            return

        if self.phase == 'settle':
            if self.elapsed(self.phase_start) < self.delay:
                return
            self.send_command()
            self.phase = 'moving'
            self.phase_start = now
            return

        if self.phase == 'moving':
            errors = self.errors_deg()
            if (self.last_log is None or
                    (now - self.last_log).nanoseconds / 1e9 >= 2.0):
                self.last_log = now
                self.get_logger().info(
                    'moving; error (deg): ' +
                    (self.format_deg(errors) if errors is not None
                     else 'no joint_states'))
            if errors is not None and max(errors) <= self.tolerance_deg:
                self.get_logger().info(
                    'reached target; final error (deg): ' + self.format_deg(errors))
                self.get_logger().info(
                    'final joints (deg): ' +
                    self.format_deg(math.degrees(v)
                                    for v in self.current_target_positions()))
                if self.gripper_close_enabled:
                    self.start_gripper_close(now)
                else:
                    self.finish()
                return
            if self.elapsed(self.phase_start) > self.timeout:
                self.get_logger().error(
                    f'timeout after {self.timeout:.1f}s; error (deg): ' +
                    self.format_deg(errors))
                self.exit_code = 1
            return

        if self.phase == 'close_gripper':
            self.tick_close_gripper(now)
            return

        if self.phase == 'gripper_settle':
            if self.elapsed(self.phase_start) < self.gripper_settle_time:
                return
            self.get_logger().info('gripper closed and settled')
            self.finish()

    def start_gripper_close(self, now):
        self.phase = 'close_gripper'
        self.phase_start = now
        self.gripper_result = None
        self.gripper_result_time = None
        self.gripper_command_sent = False
        self.gripper_command_time = None
        self.get_logger().info(
            'detect pose reached; closing gripper: position=%d'
            % self.gripper_close_position)

    def tick_close_gripper(self, now):
        if not self.gripper_command_sent:
            subscribers = self.gripper_publisher.get_subscription_count()
            if subscribers == 0:
                if self.elapsed(self.phase_start) > self.gripper_close_timeout:
                    self.get_logger().error(
                        f'no subscriber on {self.ctag_cmd_topic}; '
                        'gripper close command was not sent')
                    self.exit_code = 1
                return
            msg = Gripperset()
            msg.position = self.gripper_close_position
            msg.block = True
            self.gripper_publisher.publish(msg)
            self.gripper_command_sent = True
            self.gripper_command_time = now
            self.get_logger().info(
                'gripper close command sent: position=%d (%d subscriber)'
                % (self.gripper_close_position, subscribers))
            return

        if (self.gripper_result is not None and
                self.gripper_result_time is not None and
                self.gripper_result_time >= self.gripper_command_time):
            if self.gripper_result:
                self.get_logger().info('gripper closed')
                self.phase = 'gripper_settle'
                self.phase_start = self.gripper_result_time
            else:
                self.get_logger().error('gripper close returned false')
                self.exit_code = 1
            return

        if self.elapsed(self.gripper_command_time) > self.gripper_close_timeout:
            self.get_logger().error(
                f'gripper close timeout after {self.gripper_close_timeout:.1f}s')
            self.exit_code = 1

    def finish(self):
        self.publish_capture_trigger()
        self.exit_code = 0

    def publish_capture_trigger(self):
        if not self.trigger_enabled:
            self.get_logger().info(
                'trigger_enabled is false; vision capture not requested')
            return
        msg = Bool()
        msg.data = True
        self.trigger_publisher.publish(msg)
        # 节点随后立即退出，留出时间完成可靠传输
        time.sleep(0.3)
        subscriber_count = self.trigger_publisher.get_subscription_count()
        if subscriber_count == 0:
            self.get_logger().warn(
                f'capture trigger published on {self.trigger_topic} but no '
                'subscriber was found; check that the vision node is running '
                'and ROS_DOMAIN_ID matches on both hosts')
        else:
            self.get_logger().info(
                f'capture trigger published on {self.trigger_topic} '
                f'({subscriber_count} subscriber)')

    def send_command(self):
        msg = Movej()
        msg.joint = self.target_rad
        msg.speed = self.speed
        msg.block = True
        msg.trajectory_connect = 0
        msg.dof = 6
        self.publisher.publish(msg)
        self.get_logger().info(f'movej command sent (speed {self.speed}%)')


def main(args=None):
    rclpy.init(args=args)
    node = MoveToPose()
    try:
        while rclpy.ok() and node.exit_code is None:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    exit_code = node.exit_code if node.exit_code is not None else 1
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    return exit_code
