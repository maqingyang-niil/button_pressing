import math

import numpy as np
import pytest

from rm65_button_press.geometry import (
    compute_press_targets,
    matrix_to_quaternion,
    normalize_quaternion,
    quaternion_to_matrix,
    rotation_angle_deg,
    validate_press_targets,
)

BUTTON_MATRIX = np.array([[0.0, 0.0, -1.0],
                          [0.0, 1.0, 0.0],
                          [1.0, 0.0, 0.0]])
TOOL_MATRIX = np.array([[0.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0],
                        [-1.0, 0.0, 0.0]])


def horizontal_targets(button_position=(0.5, 0.0, 0.4),
                       flange_position=(0.1, 0.0, 0.4)):
    return compute_press_targets(
        button_position,
        matrix_to_quaternion(BUTTON_MATRIX),
        flange_position,
        matrix_to_quaternion(TOOL_MATRIX),
        0.143,
        0.1,
        0.005,
    )


def test_quaternion_matrix_roundtrip():
    quaternion = matrix_to_quaternion(BUTTON_MATRIX)
    assert np.isclose(np.linalg.norm(quaternion), 1.0)
    assert np.allclose(quaternion_to_matrix(quaternion), BUTTON_MATRIX, atol=1e-9)


def test_quaternion_rejects_zero_norm():
    with pytest.raises(ValueError):
        normalize_quaternion([0.0, 0.0, 0.0, 0.0])


def test_compute_targets_aligned_horizontal_case():
    targets = horizontal_targets()

    assert np.allclose(targets.button_normal, [-1.0, 0.0, 0.0])
    assert np.allclose(targets.approach, [1.0, 0.0, 0.0])
    assert np.allclose(targets.tip_offset, [0.143, 0.0, 0.0])
    assert np.allclose(targets.standoff_position, [0.257, 0.0, 0.4])
    assert np.allclose(targets.press_position, [0.362, 0.0, 0.4])
    assert np.allclose(targets.return_position, [0.1, 0.0, 0.4])
    assert np.allclose(targets.return_orientation,
                       matrix_to_quaternion(TOOL_MATRIX))
    assert math.isclose(targets.button_distance, 0.4, abs_tol=1e-9)
    assert math.isclose(targets.normal_tilt_deg, 0.0, abs_tol=1e-9)


def test_validate_accepts_horizontal_case():
    assert validate_press_targets(horizontal_targets(), 0.61, 35.0, 0.15, 0.80) == []


def test_validate_rejects_button_behind():
    targets = compute_press_targets(
        (-0.5, 0.0, 0.4),
        matrix_to_quaternion(BUTTON_MATRIX),
        (0.1, 0.0, 0.4),
        matrix_to_quaternion(TOOL_MATRIX),
        0.143,
        0.1,
        0.005,
    )
    errors = validate_press_targets(targets, 0.61, 35.0, 0.15, 0.80)
    assert any("ahead" in error or "toward the flange" in error for error in errors)


def test_validate_rejects_normal_tilt():
    identity = (0.0, 0.0, 0.0, 1.0)
    targets = compute_press_targets(
        (0.0, 0.0, 0.5), identity, (0.0, 0.0, 0.9), identity, 0.143, 0.1, 0.005
    )
    errors = validate_press_targets(targets, 0.61, 35.0, 0.15, 0.80)
    assert any("tilt" in error for error in errors)


def test_validate_rejects_reach():
    targets = horizontal_targets(button_position=(0.65, 0.0, 0.4))
    errors = validate_press_targets(targets, 0.61, 35.0, 0.15, 0.80)
    assert any("radius" in error for error in errors)


def test_rotation_angle_deg():
    assert math.isclose(rotation_angle_deg(np.eye(3), np.eye(3)), 0.0, abs_tol=1e-9)
    rotate_z_90 = np.array([[0.0, -1.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0]])
    assert math.isclose(rotation_angle_deg(np.eye(3), rotate_z_90), 90.0, abs_tol=1e-6)
