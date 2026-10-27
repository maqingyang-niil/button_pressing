import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('rm65_detect_pose'), 'config', 'detect_pose.yaml')

    start_driver = LaunchConfiguration('start_driver')
    speed = LaunchConfiguration('speed')
    delay = LaunchConfiguration('delay')
    gripper_close = LaunchConfiguration('gripper_close')
    trigger_enabled = LaunchConfiguration('trigger_enabled')

    driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('rm_driver'), 'launch',
            'rm_65_driver.launch.py')),
        condition=IfCondition(start_driver),
    )

    move_to_pose = Node(
        package='rm65_detect_pose',
        executable='move_to_pose',
        name='move_to_pose',
        output='screen',
        parameters=[
            config,
            {
                'speed': ParameterValue(speed, value_type=int),
                'delay': ParameterValue(delay, value_type=float),
                'gripper_close_enabled': ParameterValue(
                    gripper_close, value_type=bool),
                'trigger_enabled': ParameterValue(
                    trigger_enabled, value_type=bool),
            },
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'start_driver', default_value='true',
            description='Start rm_driver (set false when a bringup is already running)'),
        DeclareLaunchArgument('speed', default_value='10',
                              description='movej speed percentage (1-100)'),
        DeclareLaunchArgument('delay', default_value='2.0',
                              description='Seconds to wait after the driver is ready'),
        DeclareLaunchArgument('gripper_close', default_value='true',
                              description='Close the gripper after reaching the detect pose'),
        DeclareLaunchArgument('trigger_enabled', default_value='false',
                              description='Publish a vision capture trigger after arrival'),
        driver,
        move_to_pose,
    ])
