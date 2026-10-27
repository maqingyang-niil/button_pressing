from setuptools import setup

package_name = 'rm65_button_press'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/press_button.yaml', 'config/sim.yaml', 'config/button_sim.rviz']),
        ('share/' + package_name + '/launch', ['launch/press_button.launch.py', 'launch/sim_scene.launch.py', 'launch/sim_press.launch.py', 'launch/sim_show_detected.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='khadas',
    maintainer_email='khadas@todo.todo',
    description='Press a cabinet button with the RM65-BI arm',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'press_button = rm65_button_press.press_button:main',
            'sim_scene = rm65_button_press.sim_scene:main',
            'sim_vision = rm65_button_press.sim_vision:main',
            'sim_driver = rm65_button_press.sim_driver:main',
            'sim_gripper = rm65_button_press.sim_gripper:main',
        ],
    },
)
