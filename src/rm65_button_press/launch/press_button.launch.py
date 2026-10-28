import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('rm65_button_press'), 'config',
        'press_button.yaml')

    arguments = [
        DeclareLaunchArgument('dry_run', default_value='false',
                              description='Compute and log targets only'),
        DeclareLaunchArgument('approach_only', default_value='false',
                              description='Move to standoff then return without pressing'),
        DeclareLaunchArgument('press_depth', default_value='0.005',
                              description='Finger travel past the button surface (m)'),
        DeclareLaunchArgument('standoff_distance', default_value='0.10',
                              description='Fingertip gap to the button surface (m)'),
        DeclareLaunchArgument('tool_offset', default_value='0.143',
                              description='Flange to closed fingertip front (m)'),
        DeclareLaunchArgument('align_speed', default_value='3',
                              description='In-place orientation alignment speed (1-100)'),
        DeclareLaunchArgument('approach_speed', default_value='10',
                              description='Standoff move speed (1-100)'),
        DeclareLaunchArgument('press_speed', default_value='3',
                              description='Press move speed (1-100)'),
        DeclareLaunchArgument('retreat_speed', default_value='10',
                              description='Retreat/return speed (1-100)'),
        DeclareLaunchArgument('trigger_on_start', default_value='true',
                              description='Publish /ibvs/capture_trigger after entering WAIT_POSE'),
    ]

    press_button = Node(
        package='rm65_button_press',
        executable='press_button',
        name='press_button',
        output='screen',
        parameters=[
            config,
            {
                'dry_run': ParameterValue(
                    LaunchConfiguration('dry_run'), value_type=bool),
                'approach_only': ParameterValue(
                    LaunchConfiguration('approach_only'), value_type=bool),
                'press_depth': ParameterValue(
                    LaunchConfiguration('press_depth'), value_type=float),
                'standoff_distance': ParameterValue(
                    LaunchConfiguration('standoff_distance'), value_type=float),
                'tool_offset': ParameterValue(
                    LaunchConfiguration('tool_offset'), value_type=float),
                'align_speed': ParameterValue(
                    LaunchConfiguration('align_speed'), value_type=int),
                'approach_speed': ParameterValue(
                    LaunchConfiguration('approach_speed'), value_type=int),
                'press_speed': ParameterValue(
                    LaunchConfiguration('press_speed'), value_type=int),
                'retreat_speed': ParameterValue(
                    LaunchConfiguration('retreat_speed'), value_type=int),
                'trigger_on_start': ParameterValue(
                    LaunchConfiguration('trigger_on_start'), value_type=bool),
            },
        ],
    )

    return LaunchDescription(arguments + [press_button])
