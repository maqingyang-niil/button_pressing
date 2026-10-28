"""Pure geometry for the RM65 button-press task."""
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PressTargets:
    button_position: np.ndarray
    button_normal: np.ndarray
    approach: np.ndarray
    aligned_orientation: np.ndarray
    tip_offset: np.ndarray
    standoff_position: np.ndarray
    press_position: np.ndarray
    return_position: np.ndarray
    return_orientation: np.ndarray
    button_distance: float
    normal_tilt_deg: float
    alignment_angle_deg: float
    rotation_clearance: float
    standoff_radius: float
    press_radius: float


def normalize_quaternion(quaternion: Sequence[float]) -> np.ndarray:
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,):
        raise ValueError(f"quaternion must have 4 elements, got {q.shape}")
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-9:
        raise ValueError("quaternion norm is invalid")
    return q / norm


def quaternion_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    x, y, z, w = normalize_quaternion(quaternion)
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=float)
    if m.shape != (3, 3):
        raise ValueError(f"rotation matrix must be 3x3, got {m.shape}")
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=float)
    q /= np.linalg.norm(q)
    if q[3] < 0.0:
        q = -q
    return q


def rotation_angle_deg(matrix_a: np.ndarray, matrix_b: np.ndarray) -> float:
    a = np.asarray(matrix_a, dtype=float)
    b = np.asarray(matrix_b, dtype=float)
    cosine = (float(np.trace(a.T @ b)) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def compute_aligned_orientation(button_orientation,
                                flange_orientation) -> np.ndarray:
    """Rotate tool +Z into the button using the shortest 3-D rotation."""
    button_matrix = quaternion_to_matrix(button_orientation)
    flange_matrix = quaternion_to_matrix(flange_orientation)
    z_target = -button_matrix[:, 2]
    z_current = flange_matrix[:, 2]
    cosine = float(np.clip(np.dot(z_current, z_target), -1.0, 1.0))
    cross = np.cross(z_current, z_target)
    sine = float(np.linalg.norm(cross))

    if sine < 1e-9:
        if cosine > 0.0:
            delta = np.eye(3)
        else:
            # The axes are opposite. Rotate 180 degrees around current tool X;
            # it is guaranteed to be perpendicular to current tool Z.
            axis = flange_matrix[:, 0]
            delta = 2.0 * np.outer(axis, axis) - np.eye(3)
    else:
        axis = cross / sine
        skew = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ])
        delta = np.eye(3) + sine * skew + (1.0 - cosine) * (skew @ skew)

    # Apply only the shortest alignment rotation. This preserves the current
    # roll instead of copying the visually noisy rotation about the normal.
    aligned_matrix = delta @ flange_matrix
    return matrix_to_quaternion(aligned_matrix)


def compute_press_targets(button_position, button_orientation, flange_position,
                          flange_orientation, tool_offset, standoff_distance,
                          press_depth) -> PressTargets:
    s = np.asarray(button_position, dtype=float)
    f = np.asarray(flange_position, dtype=float)
    if s.shape != (3,) or f.shape != (3,):
        raise ValueError("positions must have 3 elements")
    if tool_offset < 0.0 or standoff_distance <= 0.0 or press_depth < 0.0:
        raise ValueError("tool_offset/standoff_distance/press_depth are invalid")

    button_matrix = quaternion_to_matrix(button_orientation)
    button_normal = button_matrix[:, 2].copy()
    approach = -button_normal
    flange_matrix = quaternion_to_matrix(flange_orientation)
    aligned_orientation = compute_aligned_orientation(
        button_orientation, flange_orientation)
    aligned_matrix = quaternion_to_matrix(aligned_orientation)
    tip_offset = aligned_matrix @ np.array([0.0, 0.0, float(tool_offset)])

    standoff_position = s - float(standoff_distance) * approach - tip_offset
    press_position = s + float(press_depth) * approach - tip_offset
    normal_tilt_deg = math.degrees(
        math.asin(float(np.clip(abs(approach[2]), 0.0, 1.0))))
    button_distance = float(np.linalg.norm(s - f))
    alignment_angle_deg = rotation_angle_deg(flange_matrix, aligned_matrix)
    plane_distance = float(np.dot(button_normal, f - s))
    rotation_clearance = plane_distance - float(tool_offset)

    return PressTargets(
        button_position=s,
        button_normal=button_normal,
        approach=approach,
        aligned_orientation=aligned_orientation,
        tip_offset=tip_offset,
        standoff_position=standoff_position,
        press_position=press_position,
        return_position=f.copy(),
        return_orientation=normalize_quaternion(flange_orientation),
        button_distance=button_distance,
        normal_tilt_deg=normal_tilt_deg,
        alignment_angle_deg=alignment_angle_deg,
        rotation_clearance=rotation_clearance,
        standoff_radius=float(np.linalg.norm(standoff_position)),
        press_radius=float(np.linalg.norm(press_position)),
    )


def validate_press_targets(targets: PressTargets, max_flange_radius: float,
                           max_normal_tilt_deg: float, min_button_distance: float,
                           max_button_distance: float,
                           min_alignment_clearance: float = 0.05,
                           max_alignment_rotation_deg: float = 60.0) -> list[str]:
    errors: list[str] = []
    vectors = [targets.button_position, targets.button_normal, targets.approach,
               targets.aligned_orientation, targets.tip_offset,
               targets.standoff_position, targets.press_position,
               targets.return_position, targets.return_orientation]
    if not all(np.all(np.isfinite(vector)) for vector in vectors):
        return ["targets contain non-finite values"]

    outward = float(np.dot(targets.button_normal,
                           targets.return_position - targets.button_position))
    if outward <= 0.0:
        errors.append("button normal does not point toward the flange")

    ahead = float(np.dot(targets.approach,
                         targets.button_position - targets.return_position))
    if ahead <= 0.05:
        errors.append("button is not ahead of the flange by more than 0.05 m")

    if not min_button_distance <= targets.button_distance <= max_button_distance:
        errors.append(
            f"button distance {targets.button_distance:.3f} m outside "
            f"[{min_button_distance:.3f}, {max_button_distance:.3f}]")

    if targets.normal_tilt_deg > max_normal_tilt_deg:
        errors.append(
            f"button normal tilt {targets.normal_tilt_deg:.1f} deg exceeds "
            f"{max_normal_tilt_deg:.1f} deg")

    if targets.rotation_clearance < min_alignment_clearance:
        errors.append(
            f"alignment sweep clearance {targets.rotation_clearance:.3f} m "
            f"is below {min_alignment_clearance:.3f} m")

    if targets.alignment_angle_deg > max_alignment_rotation_deg:
        errors.append(
            f"alignment rotation {targets.alignment_angle_deg:.1f} deg exceeds "
            f"{max_alignment_rotation_deg:.1f} deg")

    if targets.standoff_radius > max_flange_radius:
        errors.append(
            f"standoff flange radius {targets.standoff_radius:.3f} m exceeds "
            f"{max_flange_radius:.3f} m")

    if targets.press_radius > max_flange_radius:
        errors.append(
            f"press flange radius {targets.press_radius:.3f} m exceeds "
            f"{max_flange_radius:.3f} m")

    return errors
