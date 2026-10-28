"""Press a cabinet button along the detected button normal with the RM65-BI."""
import copy
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rm_ros_interfaces.msg import Gripperset, Movel
from std_msgs.msg import Bool

from rm65_button_press.capture_handshake import validate_capture_pose_metadata
from rm65_button_press.geometry import (
    compute_press_targets,
    normalize_quaternion,
    quaternion_to_matrix,
    rotation_angle_deg,
    validate_press_targets,
)

PHASE_WAIT_DRIVER = 'WAIT_DRIVER'
PHASE_CLOSE_GRIPPER = 'CLOSE_GRIPPER'
PHASE_WAIT_POSE = 'WAIT_POSE'
PHASE_CAPTURE = 'CAPTURE'
PHASE_ALIGN = 'ALIGN'
PHASE_MOVE_STANDOFF = 'MOVE_STANDOFF'
PHASE_SETTLE = 'SETTLE'
PHASE_PRESS = 'PRESS'
PHASE_HOLD = 'HOLD'
PHASE_RETREAT_STANDOFF = 'RETREAT_STANDOFF'
PHASE_RETURN_POSITION = 'RETURN_POSITION'
PHASE_RESTORE_ORIENTATION = 'RESTORE_ORIENTATION'


class PressButton(Node):

    def __init__(self):
        super().__init__('press_button')

        self.button_pose_topic = str(self.declare_parameter(
            'button_pose_topic', '/ibvs/button_pose').value)
        self.current_pose_topic = str(self.declare_parameter(
            'current_pose_topic', '/rm_driver/udp_arm_position').value)
        self.movel_cmd_topic = str(self.declare_parameter(
            'movel_cmd_topic', '/rm_driver/movel_cmd').value)
        self.movel_result_topic = str(self.declare_parameter(
            'movel_result_topic', '/rm_driver/movel_result').value)
        self.move_stop_topic = str(self.declare_parameter(
            'move_stop_topic', '/rm_driver/move_stop_cmd').value)
        self.ctag_cmd_topic = str(self.declare_parameter(
            'ctag_cmd_topic', '/rm_driver/set_ctag_gripper_position_cmd').value)
        self.ctag_result_topic = str(self.declare_parameter(
            'ctag_result_topic', '/rm_driver/set_ctag_gripper_position_result').value)
        self.trigger_topic = str(self.declare_parameter(
            'trigger_topic', '/ibvs/capture_trigger').value)
        self.capture_success_topic = str(self.declare_parameter(
            'capture_success_topic', '/ibvs/capture_success').value)
        self.expected_button_frame = str(self.declare_parameter(
            'expected_button_frame', 'base_link').value)
        self.trigger_on_start = bool(
            self.declare_parameter('trigger_on_start', True).value)

        self.tool_offset = float(self.declare_parameter('tool_offset', 0.143).value)
        self.standoff_distance = float(
            self.declare_parameter('standoff_distance', 0.10).value)
        self.press_depth = float(self.declare_parameter('press_depth', 0.005).value)
        self.align_speed = int(self.declare_parameter('align_speed', 3).value)
        self.approach_speed = int(self.declare_parameter('approach_speed', 10).value)
        self.press_speed = int(self.declare_parameter('press_speed', 3).value)
        self.retreat_speed = int(self.declare_parameter('retreat_speed', 10).value)

        self.position_tolerance = float(
            self.declare_parameter('position_tolerance', 0.003).value)
        self.press_position_tolerance = float(
            self.declare_parameter('press_position_tolerance', 0.005).value)
        self.orientation_tolerance_deg = float(
            self.declare_parameter('orientation_tolerance_deg', 2.0).value)
        self.alignment_tolerance_deg = float(
            self.declare_parameter('alignment_tolerance_deg', 2.0).value)

        self.settle_time = float(self.declare_parameter('settle_time', 0.5).value)
        self.hold_time = float(self.declare_parameter('hold_time', 0.4).value)

        self.driver_wait_timeout = float(
            self.declare_parameter('driver_wait_timeout', 20.0).value)
        self.pose_wait_timeout = float(
            self.declare_parameter('pose_wait_timeout', 120.0).value)
        self.move_timeout = float(self.declare_parameter('move_timeout', 30.0).value)
        self.align_timeout = float(self.declare_parameter('align_timeout', 20.0).value)
        self.press_timeout = float(self.declare_parameter('press_timeout', 20.0).value)
        self.stationary_timeout = float(
            self.declare_parameter('stationary_timeout', 3.0).value)
        self.vision_wait_timeout = float(
            self.declare_parameter('vision_wait_timeout', 10.0).value)
        self.pose_stamp_tolerance = float(
            self.declare_parameter('pose_stamp_tolerance', 0.5).value)
        self.max_flange_radius = float(
            self.declare_parameter('max_flange_radius', 0.61).value)
        self.max_normal_tilt_deg = float(
            self.declare_parameter('max_normal_tilt_deg', 35.0).value)
        self.min_button_distance = float(
            self.declare_parameter('min_button_distance', 0.15).value)
        self.max_button_distance = float(
            self.declare_parameter('max_button_distance', 0.80).value)
        self.min_alignment_clearance = float(
            self.declare_parameter('min_alignment_clearance', 0.05).value)
        self.max_alignment_rotation_deg = float(
            self.declare_parameter('max_alignment_rotation_deg', 60.0).value)

        self.gripper_close_enabled = bool(
            self.declare_parameter('gripper_close_enabled', True).value)
        self.gripper_close_position = int(
            self.declare_parameter('gripper_close_position', 0).value)
        self.gripper_close_repeats = int(
            self.declare_parameter('gripper_close_repeats', 1).value)
        self.gripper_close_interval = float(
            self.declare_parameter('gripper_close_interval', 0.2).value)
        self.gripper_close_timeout = float(
            self.declare_parameter('gripper_close_timeout', 8.0).value)
        self.gripper_settle_time = float(
            self.declare_parameter('gripper_settle_time', 1.0).value)

        self.dry_run = bool(self.declare_parameter('dry_run', False).value)
        self.approach_only = bool(self.declare_parameter('approach_only', False).value)

        self.validate_parameters()

        pose_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.movel_pub = self.create_publisher(Movel, self.movel_cmd_topic, 10)
        self.move_stop_pub = self.create_publisher(Bool, self.move_stop_topic, 10)
        self.gripper_pub = self.create_publisher(Gripperset, self.ctag_cmd_topic, 10)
        self.trigger_pub = self.create_publisher(Bool, self.trigger_topic, 10)

        self.create_subscription(
            PoseStamped, self.button_pose_topic, self.on_button_pose, pose_qos)
        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            Bool, self.capture_success_topic, self.on_capture_success, status_qos)
        self.create_subscription(
            Pose, self.current_pose_topic, self.on_current_pose, 10)
        self.create_subscription(
            Bool, self.movel_result_topic, self.on_movel_result, 10)
        self.create_subscription(
            Bool, self.ctag_result_topic, self.on_gripper_result, 10)

        self.node_start_time = self.now()
        self.phase = PHASE_WAIT_DRIVER
        self.phase_start = self.node_start_time

        self.current_pose = None
        self.current_pose_time = None
        self.pending_pose = None
        self.pending_pose_time = None
        self.last_pose_stamp = None
        self.trigger_baseline_stamp = None
        self.trigger_stamp_ns = None
        self.trigger_sent = False
        self.capture_result = None
        self.capture_wait_start = None

        self.movel_result = None
        self.gripper_any_success = False
        self.gripper_attempts = 0
        self.gripper_last_publish = None

        self.targets = None
        self.tool_quaternion = None
        self.tool_matrix = None

        self.segment_target_position = None
        self.segment_target_matrix = None
        self.segment_position_tolerance = None
        self.segment_orientation_tolerance_deg = None
        self.segment_start = None
        self.motion_active = False
        self.stop_sent = False

        self.stationary_reference = None
        self.stationary_reference_time = None
        self.capture_start = None
        self.retreat_reason = None
        self.exit_code = None

        self.timer = self.create_timer(0.05, self.tick)
        self.get_logger().info(
            'press_button ready: dry_run=%s approach_only=%s '
            'standoff=%.3f m press_depth=%.3f m tool_offset=%.3f m '
            'align_speed=%d%%' % (
                self.dry_run, self.approach_only, self.standoff_distance,
                self.press_depth, self.tool_offset, self.align_speed))

    def validate_parameters(self):
        if self.tool_offset < 0.0:
            raise ValueError('tool_offset must be >= 0')
        if self.standoff_distance <= 0.0:
            raise ValueError('standoff_distance must be > 0')
        if self.press_depth < 0.0:
            raise ValueError('press_depth must be >= 0')
        for name, value in (('align_speed', self.align_speed),
                            ('approach_speed', self.approach_speed),
                            ('press_speed', self.press_speed),
                            ('retreat_speed', self.retreat_speed)):
            if not 1 <= value <= 100:
                raise ValueError('%s must be within 1..100, got %d' % (name, value))
        if not 0 <= self.gripper_close_position <= 1000:
            raise ValueError('gripper_close_position must be within 0..1000')
        if self.gripper_close_repeats < 1:
            raise ValueError('gripper_close_repeats must be >= 1')
        if self.gripper_close_interval < 0.0:
            raise ValueError('gripper_close_interval must be >= 0')
        if self.position_tolerance <= 0.0 or self.press_position_tolerance <= 0.0:
            raise ValueError('position tolerances must be > 0')
        if self.orientation_tolerance_deg <= 0.0:
            raise ValueError('orientation_tolerance_deg must be > 0')
        if self.alignment_tolerance_deg <= 0.0:
            raise ValueError('alignment_tolerance_deg must be > 0')
        if self.align_timeout <= 0.0:
            raise ValueError('align_timeout must be > 0')
        if self.min_alignment_clearance < 0.0:
            raise ValueError('min_alignment_clearance must be >= 0')
        if not 0.0 < self.max_alignment_rotation_deg <= 180.0:
            raise ValueError('max_alignment_rotation_deg must be within (0, 180]')
        if not self.capture_success_topic:
            raise ValueError('capture_success_topic must not be empty')
        if not self.expected_button_frame:
            raise ValueError('expected_button_frame must not be empty')
        if self.vision_wait_timeout <= 0.0:
            raise ValueError('vision_wait_timeout must be > 0')
        if self.pose_stamp_tolerance < 0.0:
            raise ValueError('pose_stamp_tolerance must be >= 0')

    def now(self):
        return self.get_clock().now()

    def elapsed(self, start):
        return (self.now() - start).nanoseconds / 1e9

    def on_current_pose(self, msg):
        values = [msg.position.x, msg.position.y, msg.position.z,
                  msg.orientation.x, msg.orientation.y, msg.orientation.z,
                  msg.orientation.w]
        if not all(math.isfinite(value) for value in values):
            return
        q_norm = math.sqrt(msg.orientation.x ** 2 + msg.orientation.y ** 2 +
                           msg.orientation.z ** 2 + msg.orientation.w ** 2)
        if q_norm < 1e-6:
            return
        self.current_pose = msg
        self.current_pose_time = self.now()

    def on_button_pose(self, msg):
        values = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z,
                  msg.pose.orientation.x, msg.pose.orientation.y,
                  msg.pose.orientation.z, msg.pose.orientation.w]
        if not all(math.isfinite(value) for value in values):
            return
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        self.last_pose_stamp = stamp
        if self.phase != PHASE_WAIT_POSE or not self.trigger_sent:
            return
        if (self.trigger_baseline_stamp is not None and
                stamp == self.trigger_baseline_stamp):
            self.get_logger().info(
                'ignored button pose from before the current trigger %d.%09d'
                % stamp)
            return
        try:
            validate_capture_pose_metadata(
                msg.header.frame_id, msg.header.stamp.sec,
                msg.header.stamp.nanosec, self.expected_button_frame,
                self.trigger_stamp_ns, self.pose_stamp_tolerance)
        except ValueError as exc:
            self.fail(1, 'invalid button pose: %s' % exc)
            return
        self.pending_pose = msg
        self.pending_pose_time = self.now()
        self.get_logger().info('received button pose %d.%09d' % stamp)

    def on_capture_success(self, msg):
        if self.phase != PHASE_WAIT_POSE or not self.trigger_sent:
            return
        self.capture_result = bool(msg.data)
        if not self.capture_result:
            self.fail(1, 'vision capture reported failure')
            return
        self.get_logger().info('vision capture reported success')

    def on_movel_result(self, msg):
        self.movel_result = bool(msg.data)

    def on_gripper_result(self, msg):
        if msg.data:
            self.gripper_any_success = True
            self.get_logger().info('gripper result: true')

    def tick(self):
        if self.exit_code is not None:
            return
        try:
            if self.phase == PHASE_WAIT_DRIVER:
                self.tick_wait_driver()
            elif self.phase == PHASE_CLOSE_GRIPPER:
                self.tick_close_gripper()
            elif self.phase == PHASE_WAIT_POSE:
                self.tick_wait_pose()
            elif self.phase == PHASE_CAPTURE:
                self.tick_capture()
            elif self.phase == PHASE_ALIGN:
                self.tick_segment(PHASE_ALIGN, self.align_timeout,
                                  'alignment', self.on_align_reached, 1)
            elif self.phase == PHASE_MOVE_STANDOFF:
                self.tick_segment(PHASE_MOVE_STANDOFF, self.move_timeout,
                                  'standoff', self.on_standoff_reached, 1)
            elif self.phase == PHASE_SETTLE:
                self.tick_settle()
            elif self.phase == PHASE_PRESS:
                self.tick_press()
            elif self.phase == PHASE_HOLD:
                self.tick_hold()
            elif self.phase == PHASE_RETREAT_STANDOFF:
                self.tick_segment(PHASE_RETREAT_STANDOFF, self.move_timeout,
                                  'retreat', self.on_retreat_reached, 2)
            elif self.phase == PHASE_RETURN_POSITION:
                self.tick_segment(PHASE_RETURN_POSITION, self.move_timeout,
                                  'return position',
                                  self.on_return_position_reached, 2)
            elif self.phase == PHASE_RESTORE_ORIENTATION:
                self.tick_segment(PHASE_RESTORE_ORIENTATION,
                                  self.align_timeout, 'restore orientation',
                                  self.on_return_reached, 2)
        except Exception as exc:  # noqa: BLE001 - keep the node from hanging on a bug
            self.get_logger().error('press_button exception: %s' % exc)
            self.fail(1, 'unhandled exception in state machine')

    def tick_wait_driver(self):
        if (self.current_pose is not None and
                self.movel_pub.get_subscription_count() > 0):
            self.get_logger().info(
                'driver ready (%d movel subscriber(s))'
                % self.movel_pub.get_subscription_count())
            if self.gripper_close_enabled and not self.dry_run:
                self.start_gripper_close()
            else:
                self.enter_wait_pose()
            return
        if self.elapsed(self.phase_start) > self.driver_wait_timeout:
            self.fail(1, 'driver not ready: no current pose or movel subscriber')

    def start_gripper_close(self):
        self.phase = PHASE_CLOSE_GRIPPER
        self.phase_start = self.now()
        self.gripper_attempts = 0
        self.gripper_any_success = False
        self.gripper_last_publish = None
        self.gripper_wait_logged = False
        self.get_logger().info(
            'closing gripper: position=%d repeats=%d interval=%.2fs'
            % (self.gripper_close_position, self.gripper_close_repeats,
               self.gripper_close_interval))

    def tick_close_gripper(self):
        now = self.now()
        if self.gripper_attempts < self.gripper_close_repeats:
            if (self.gripper_last_publish is None or
                    self.elapsed(self.gripper_last_publish) >=
                    self.gripper_close_interval):
                if self.gripper_pub.get_subscription_count() == 0:
                    if not self.gripper_wait_logged:
                        self.get_logger().warn(
                            'waiting for gripper command subscriber on %s'
                            % self.ctag_cmd_topic)
                        self.gripper_wait_logged = True
                    if self.elapsed(self.phase_start) > self.gripper_close_timeout:
                        self.fail(1, 'gripper command has no subscriber')
                    return
                msg = Gripperset()
                msg.position = self.gripper_close_position
                msg.block = True
                self.gripper_pub.publish(msg)
                self.gripper_attempts += 1
                self.gripper_last_publish = now
                self.get_logger().info(
                    'gripper close command %d/%d'
                    % (self.gripper_attempts, self.gripper_close_repeats))
            return

        if self.gripper_any_success:
            if self.elapsed(self.gripper_last_publish) >= self.gripper_settle_time:
                self.get_logger().info('gripper closed')
                self.enter_wait_pose()
            return

        if self.elapsed(self.phase_start) > self.gripper_close_timeout:
            self.fail(1, 'gripper close failed: no successful result')

    def enter_wait_pose(self):
        self.phase = PHASE_WAIT_POSE
        self.phase_start = self.now()
        self.pending_pose = None
        self.pending_pose_time = None
        self.trigger_baseline_stamp = None
        self.trigger_stamp_ns = None
        self.trigger_sent = False
        self.capture_result = None
        self.capture_wait_start = None
        if self.trigger_on_start:
            self.get_logger().info(
                'waiting for a vision subscriber on %s' % self.trigger_topic)
        else:
            self.start_capture_wait(False)
            self.get_logger().info(
                'waiting for an external capture trigger and fresh result')

    def start_capture_wait(self, publish_trigger):
        now = self.now()
        self.pending_pose = None
        self.pending_pose_time = None
        self.capture_result = None
        self.trigger_baseline_stamp = self.last_pose_stamp
        self.trigger_stamp_ns = now.nanoseconds
        self.capture_wait_start = now
        self.trigger_sent = True
        if publish_trigger:
            msg = Bool()
            msg.data = True
            self.trigger_pub.publish(msg)
            self.get_logger().info(
                'capture trigger published on %s (%d subscriber)'
                % (self.trigger_topic,
                   self.trigger_pub.get_subscription_count()))
        self.get_logger().info(
            'waiting for capture_success=true and a fresh button pose')

    def tick_wait_pose(self):
        if not self.trigger_sent:
            subscriber_count = self.trigger_pub.get_subscription_count()
            if subscriber_count > 0:
                self.start_capture_wait(True)
            elif self.elapsed(self.phase_start) > self.vision_wait_timeout:
                self.fail(
                    1,
                    'vision not ready: no subscriber on %s within %.1fs'
                    % (self.trigger_topic, self.vision_wait_timeout))
            return
        if self.capture_result is True and self.pending_pose is not None:
            self.capture_start = self.now()
            self.stationary_reference = None
            self.stationary_reference_time = None
            self.phase = PHASE_CAPTURE
            self.get_logger().info(
                'vision handshake complete; capturing flange pose')
            return
        if self.elapsed(self.capture_wait_start) > self.pose_wait_timeout:
            if self.capture_result is True:
                message = 'capture succeeded but no fresh button pose was received'
            else:
                message = 'no vision capture result was received'
            self.fail(1, '%s within %.1fs' % (message, self.pose_wait_timeout))

    def tick_capture(self):
        if (self.current_pose is None or self.current_pose_time is None or
                self.elapsed(self.current_pose_time) > 0.5):
            self.check_capture_timeout('no fresh current flange pose')
            return
        if self.stationary_reference is None:
            self.stationary_reference = copy.deepcopy(self.current_pose)
            self.stationary_reference_time = self.now()
            return
        if self.elapsed(self.stationary_reference_time) < 0.2:
            return

        current = self.current_pose
        reference = self.stationary_reference
        position_delta = float(np.linalg.norm(
            np.array([current.position.x, current.position.y, current.position.z]) -
            np.array([reference.position.x, reference.position.y,
                      reference.position.z])))
        rotation_delta = rotation_angle_deg(
            quaternion_to_matrix((current.orientation.x, current.orientation.y,
                                  current.orientation.z, current.orientation.w)),
            quaternion_to_matrix((reference.orientation.x, reference.orientation.y,
                                  reference.orientation.z, reference.orientation.w)))
        if position_delta > 0.001 or rotation_delta > 0.5:
            self.stationary_reference = copy.deepcopy(current)
            self.stationary_reference_time = self.now()
            self.check_capture_timeout('flange is still moving')
            return
        self.compute_and_start()

    def check_capture_timeout(self, message):
        if self.elapsed(self.capture_start) > self.stationary_timeout:
            self.fail(1, message)

    def compute_and_start(self):
        button = self.pending_pose.pose
        current = self.current_pose
        button_position = (button.position.x, button.position.y, button.position.z)
        button_orientation = (button.orientation.x, button.orientation.y,
                              button.orientation.z, button.orientation.w)
        flange_position = (current.position.x, current.position.y, current.position.z)
        flange_orientation = (current.orientation.x, current.orientation.y,
                              current.orientation.z, current.orientation.w)
        try:
            targets = compute_press_targets(
                button_position, button_orientation, flange_position,
                flange_orientation, self.tool_offset, self.standoff_distance,
                self.press_depth)
        except ValueError as exc:
            self.fail(1, 'geometry error: %s' % exc)
            return
        self.log_targets(targets)
        errors = validate_press_targets(
            targets, self.max_flange_radius, self.max_normal_tilt_deg,
            self.min_button_distance, self.max_button_distance,
            self.min_alignment_clearance, self.max_alignment_rotation_deg)
        if errors:
            self.fail(1, 'target validation failed: ' + '; '.join(errors))
            return

        self.targets = targets
        self.tool_quaternion = targets.aligned_orientation
        self.tool_matrix = quaternion_to_matrix(targets.aligned_orientation)
        if self.dry_run:
            self.get_logger().info('dry_run: targets computed, no motion')
            self.exit_code = 0
            return
        self.send_movel(targets.return_position, self.tool_quaternion,
                        self.align_speed, self.position_tolerance,
                        self.alignment_tolerance_deg)
        self.phase = PHASE_ALIGN

    def log_targets(self, targets):
        def fmt(vector):
            return '(%.4f, %.4f, %.4f)' % tuple(float(value) for value in vector)

        self.get_logger().info(
            'button %s normal %s tilt=%.1fdeg distance=%.4fm' % (
                fmt(targets.button_position), fmt(targets.button_normal),
                targets.normal_tilt_deg, targets.button_distance))
        self.get_logger().info(
            'tip_offset %s standoff %s press %s return %s' % (
                fmt(targets.tip_offset), fmt(targets.standoff_position),
                fmt(targets.press_position), fmt(targets.return_position)))
        self.get_logger().info(
            'standoff_radius=%.4fm press_radius=%.4fm travel=%.4fm' % (
                targets.standoff_radius, targets.press_radius,
                float(np.linalg.norm(targets.press_position -
                                     targets.standoff_position))))
        self.get_logger().info(
            'alignment rotation=%.2fdeg sweep_clearance=%.4fm' % (
                targets.alignment_angle_deg, targets.rotation_clearance))

    def send_movel(self, position, quaternion, speed, position_tolerance,
                   orientation_tolerance_deg=None):
        quaternion = normalize_quaternion(quaternion)
        msg = Movel()
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        msg.pose.orientation.x = float(quaternion[0])
        msg.pose.orientation.y = float(quaternion[1])
        msg.pose.orientation.z = float(quaternion[2])
        msg.pose.orientation.w = float(quaternion[3])
        msg.speed = int(speed)
        msg.trajectory_connect = 0
        msg.block = True
        self.movel_result = None
        self.segment_target_position = np.asarray(position, dtype=float)
        self.segment_target_matrix = quaternion_to_matrix(quaternion)
        self.segment_position_tolerance = float(position_tolerance)
        self.segment_orientation_tolerance_deg = float(
            self.orientation_tolerance_deg if orientation_tolerance_deg is None
            else orientation_tolerance_deg)
        self.segment_start = self.now()
        self.motion_active = True
        self.stop_sent = False
        self.movel_pub.publish(msg)
        self.get_logger().info(
            'Movel -> (%.4f, %.4f, %.4f) speed=%d%% tolerance=%.1fmm' % (
                position[0], position[1], position[2], speed,
                position_tolerance * 1000.0))

    def segment_status(self, timeout):
        if self.movel_result is False:
            return 'rejected'
        if (self.current_pose is None or self.current_pose_time is None or
                self.elapsed(self.current_pose_time) > 0.5):
            if self.elapsed(self.segment_start) > timeout:
                return 'timeout'
            return None
        current = self.current_pose
        position_error = float(np.linalg.norm(
            np.array([current.position.x, current.position.y, current.position.z]) -
            self.segment_target_position))
        rotation_error = rotation_angle_deg(
            quaternion_to_matrix((current.orientation.x, current.orientation.y,
                                  current.orientation.z, current.orientation.w)),
            self.segment_target_matrix)
        if (position_error <= self.segment_position_tolerance and
                rotation_error <= self.segment_orientation_tolerance_deg):
            self.get_logger().info(
                'reached target: position_error=%.1fmm orientation_error=%.2fdeg'
                % (position_error * 1000.0, rotation_error))
            return 'done'
        if self.elapsed(self.segment_start) > timeout:
            self.get_logger().error(
                'segment timeout: position_error=%.1fmm orientation_error=%.2fdeg'
                % (position_error * 1000.0, rotation_error))
            return 'timeout'
        return None

    def tick_segment(self, phase, timeout, label, on_done, failure_code):
        status = self.segment_status(timeout)
        if status == 'done':
            self.motion_active = False
            on_done()
        elif status in ('rejected', 'timeout'):
            self.fail(failure_code, '%s move %s' % (label, status))

    def on_standoff_reached(self):
        self.phase = PHASE_SETTLE
        self.phase_start = self.now()

    def on_align_reached(self):
        self.get_logger().info(
            'tool +Z aligned with button approach within %.2fdeg'
            % self.alignment_tolerance_deg)
        self.send_movel(self.targets.standoff_position, self.tool_quaternion,
                        self.approach_speed, self.position_tolerance)
        self.phase = PHASE_MOVE_STANDOFF

    def tick_settle(self):
        if self.elapsed(self.phase_start) < self.settle_time:
            return
        if self.approach_only:
            self.get_logger().info('approach_only: retreating without pressing')
            self.start_retreat('pilot')
            return
        self.send_movel(self.targets.press_position, self.tool_quaternion,
                        self.press_speed, self.press_position_tolerance)
        self.phase = PHASE_PRESS

    def tick_press(self):
        status = self.segment_status(self.press_timeout)
        if status == 'done':
            self.motion_active = False
            self.phase = PHASE_HOLD
            self.phase_start = self.now()
        elif status in ('rejected', 'timeout'):
            self.fail(
                1,
                'press move %s; motion stopped without automatic retreat'
                % status)

    def tick_hold(self):
        if self.elapsed(self.phase_start) >= self.hold_time:
            self.start_retreat('success')

    def start_retreat(self, reason):
        self.retreat_reason = reason
        self.send_movel(self.targets.standoff_position, self.tool_quaternion,
                        self.retreat_speed, self.position_tolerance)
        self.phase = PHASE_RETREAT_STANDOFF

    def on_retreat_reached(self):
        self.send_movel(self.targets.return_position,
                        self.tool_quaternion, self.retreat_speed,
                        self.position_tolerance)
        self.phase = PHASE_RETURN_POSITION

    def on_return_position_reached(self):
        self.send_movel(self.targets.return_position,
                        self.targets.return_orientation, self.align_speed,
                        self.position_tolerance,
                        self.alignment_tolerance_deg)
        self.phase = PHASE_RESTORE_ORIENTATION

    def on_return_reached(self):
        self.get_logger().info('returned to detect pose; exit 0')
        self.exit_code = 0

    def safe_stop(self, reason):
        if not self.motion_active or self.stop_sent:
            return False
        self.stop_sent = True
        msg = Bool()
        msg.data = True
        subscriber_count = self.move_stop_pub.get_subscription_count()
        try:
            self.move_stop_pub.publish(msg)
        except Exception as exc:  # noqa: BLE001 - best effort during shutdown
            self.get_logger().error('failed to publish move_stop: %s' % exc)
            return False
        if subscriber_count == 0:
            self.get_logger().error(
                'move_stop published but no driver subscriber was found: %s'
                % reason)
        else:
            self.get_logger().warn(
                'move_stop published (%d subscriber): %s'
                % (subscriber_count, reason))
        return True

    def fail(self, code, message):
        self.get_logger().error(message)
        if self.exit_code is None:
            self.safe_stop(message)
            self.exit_code = code


def main(args=None):
    rclpy.init(args=args)
    node = PressButton()
    try:
        while rclpy.ok() and node.exit_code is None:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        node.exit_code = 130
        node.safe_stop('keyboard interrupt')
    except Exception as exc:  # noqa: BLE001 - stop before leaving the process
        node.get_logger().error('executor exception: %s' % exc)
        node.exit_code = 1
        node.safe_stop('executor exception')
    if node.stop_sent and rclpy.ok():
        for _ in range(3):
            rclpy.spin_once(node, timeout_sec=0.05)
    exit_code = node.exit_code if node.exit_code is not None else 1
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    return exit_code
