from setuptools import setup

package_name = 'rm65_detect_pose'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/detect_pose.yaml']),
        ('share/' + package_name + '/launch', ['launch/move_to_detect_pose.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='khadas',
    maintainer_email='khadas@todo.todo',
    description='Move the RM65-BI arm to the taught detection pose',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'move_to_pose = rm65_detect_pose.move_to_pose:main',
        ],
    },
)
