import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('rm65_button_press')
    sim_config = os.path.join(share, 'config', 'sim.yaml')
    press_config = os.path.join(share, 'config', 'press_button.yaml')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('rm_bringup'), 'launch',
            'rm_65_gazebo.launch.py')),
    )

    scene = Node(
        package='rm65_button_press',
        executable='sim_scene',
        name='sim_scene',
        output='screen',
        parameters=[sim_config],
    )
    vision = Node(
        package='rm65_button_press',
        executable='sim_vision',
        name='sim_vision',
        output='screen',
        parameters=[sim_config],
    )
    driver = Node(
        package='rm65_button_press',
        executable='sim_driver',
        name='sim_driver',
        output='screen',
        parameters=[sim_config],
    )
    press = Node(
        package='rm65_button_press',
        executable='press_button',
        name='press_button',
        output='screen',
        parameters=[
            press_config,
            {
                'use_sim_time': True,
                'trigger_on_start': True,
                'gripper_close_enabled': False,
                'dry_run': False,
                'approach_only': False,
            },
        ],
    )

    return LaunchDescription([
        gazebo,
        scene,
        vision,
        driver,
        TimerAction(period=45.0, actions=[press]),
    ])
