import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory('rm65_button_press')
    sim_config = os.path.join(share, 'config', 'sim.yaml')
    default_rviz = os.path.join(share, 'config', 'button_sim.rviz')

    button_pose_file = LaunchConfiguration('button_pose_file')
    rviz_config = LaunchConfiguration('rviz_config')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('rm_bringup'), 'launch',
            'rm_65_gazebo.launch.py')),
        launch_arguments={'rviz_config': rviz_config}.items(),
    )

    scene = Node(
        package='rm65_button_press',
        executable='sim_scene',
        name='sim_scene',
        output='screen',
        parameters=[
            sim_config,
            {'button_pose_file': ParameterValue(
                button_pose_file, value_type=str)},
        ],
    )
    driver = Node(
        package='rm65_button_press',
        executable='sim_driver',
        name='sim_driver',
        output='screen',
        parameters=[sim_config],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'button_pose_file', default_value='',
            description='JSON file with the real detected button pose'),
        DeclareLaunchArgument(
            'rviz_config', default_value=default_rviz,
            description='RViz config including the button MarkerArray display'),
        gazebo,
        scene,
        driver,
    ])
