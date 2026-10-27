import json
import math


def load_button_pose_file(path):
    with open(path, encoding='utf-8') as stream:
        data = json.load(stream)
    try:
        position = [float(value) for value in data['position']]
        orientation = [float(value) for value in data['orientation']]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('button pose position/orientation is missing or invalid: %s' % exc) from exc

    if len(position) != 3:
        raise ValueError('button pose position must have 3 values')
    if len(orientation) != 4:
        raise ValueError('button pose orientation must have 4 values')
    if not all(math.isfinite(value) for value in position + orientation):
        raise ValueError('button pose values must be finite')

    norm = math.sqrt(sum(value * value for value in orientation))
    if norm < 1e-9:
        raise ValueError('button pose quaternion norm is zero')
    orientation = [value / norm for value in orientation]
    return position, orientation, str(data.get('frame_id', ''))
