"""Validation helpers for pairing a vision trigger with its pose result."""

NANOSECONDS_PER_SECOND = 1_000_000_000


def stamp_to_nanoseconds(sec, nanosec):
    """Convert a ROS time stamp to integer nanoseconds."""
    sec = int(sec)
    nanosec = int(nanosec)
    if sec < 0 or not 0 <= nanosec < NANOSECONDS_PER_SECOND:
        raise ValueError('invalid button pose timestamp')
    return sec * NANOSECONDS_PER_SECOND + nanosec


def validate_capture_pose_metadata(frame_id, sec, nanosec, expected_frame,
                                   trigger_stamp_ns, stamp_tolerance):
    """Reject poses from the wrong frame or from before the current trigger."""
    if frame_id != expected_frame:
        raise ValueError(
            "button pose frame_id '%s' does not match expected '%s'"
            % (frame_id, expected_frame))

    pose_stamp_ns = stamp_to_nanoseconds(sec, nanosec)
    if pose_stamp_ns == 0:
        raise ValueError('button pose timestamp is zero')

    tolerance_ns = int(float(stamp_tolerance) * NANOSECONDS_PER_SECOND)
    if pose_stamp_ns < int(trigger_stamp_ns) - tolerance_ns:
        raise ValueError('button pose timestamp is older than the current trigger')
    return pose_stamp_ns
