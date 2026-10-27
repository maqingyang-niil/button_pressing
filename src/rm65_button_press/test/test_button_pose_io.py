import json


def write_pose(tmp_path, data):
    path = tmp_path / 'button_pose.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    return str(path)


def test_loads_valid_pose_and_normalizes_quaternion(tmp_path):
    from rm65_button_press.button_pose_io import load_button_pose_file

    path = write_pose(tmp_path, {
        'frame_id': 'base_link',
        'position': [1.0, 2.0, 3.0],
        'orientation': [0.0, 0.0, 0.0, 2.0],
    })

    position, orientation, frame_id = load_button_pose_file(path)

    assert position == [1.0, 2.0, 3.0]
    assert orientation == [0.0, 0.0, 0.0, 1.0]
    assert frame_id == 'base_link'


def test_rejects_missing_position(tmp_path):
    import pytest
    from rm65_button_press.button_pose_io import load_button_pose_file

    path = write_pose(tmp_path, {
        'orientation': [0.0, 0.0, 0.0, 1.0],
    })

    with pytest.raises(ValueError, match='position'):
        load_button_pose_file(path)


def test_rejects_non_finite_values(tmp_path):
    import pytest
    from rm65_button_press.button_pose_io import load_button_pose_file

    path = write_pose(tmp_path, {
        'position': ['nan', 0.0, 0.0],
        'orientation': [0.0, 0.0, 0.0, 1.0],
    })

    with pytest.raises(ValueError, match='finite'):
        load_button_pose_file(path)


def test_rejects_zero_quaternion(tmp_path):
    import pytest
    from rm65_button_press.button_pose_io import load_button_pose_file

    path = write_pose(tmp_path, {
        'position': [0.0, 0.0, 0.0],
        'orientation': [0.0, 0.0, 0.0, 0.0],
    })

    with pytest.raises(ValueError, match='quaternion'):
        load_button_pose_file(path)
