import pytest

from rm65_button_press.capture_handshake import (
    stamp_to_nanoseconds,
    validate_capture_pose_metadata,
)


def test_stamp_to_nanoseconds():
    assert stamp_to_nanoseconds(12, 34) == 12_000_000_034


def test_accepts_expected_frame_and_fresh_stamp():
    stamp = validate_capture_pose_metadata(
        'base_link', 11, 0, 'base_link', 10_000_000_000, 0.0)
    assert stamp == 11_000_000_000


def test_rejects_wrong_frame():
    with pytest.raises(ValueError, match='frame_id'):
        validate_capture_pose_metadata(
            'camera_link', 11, 0, 'base_link', 10_000_000_000, 0.0)


def test_rejects_pose_older_than_trigger():
    with pytest.raises(ValueError, match='older'):
        validate_capture_pose_metadata(
            'base_link', 9, 0, 'base_link', 10_000_000_000, 0.0)


def test_allows_configured_clock_tolerance():
    stamp = validate_capture_pose_metadata(
        'base_link', 9, 800_000_000, 'base_link', 10_000_000_000, 0.5)
    assert stamp == 9_800_000_000


def test_rejects_zero_stamp():
    with pytest.raises(ValueError, match='zero'):
        validate_capture_pose_metadata(
            'base_link', 0, 0, 'base_link', 0, 0.0)
