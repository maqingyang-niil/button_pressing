import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
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
        parameters=[os.path.join(
            get_package_share_directory('rm65_button_press'), 'config', 'sim.yaml')],
    )

    return LaunchDescription([gazebo, scene])
