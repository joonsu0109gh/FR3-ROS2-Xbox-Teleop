from setuptools import find_packages, setup

package_name = 'my_franka_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rvi',
    maintainer_email='rvi@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'move_to_pose = my_franka_controller.move_to_pose:main',
            'move_to_joint = my_franka_controller.move_to_joint:main',
            'get_pose = my_franka_controller.get_pose:main',
            'xbox_teleop = my_franka_controller.xbox_teleop:main',
            'sgd_planning = my_franka_controller.sgd_planning:main',
            'sgd_planning_smooth = my_franka_controller.sgd_planning_smooth:main',
            'xbox_teleop_gripper = my_franka_controller.xbox_teleop_gripper:main',
            'sgd_planning_smooth_gripper = my_franka_controller.sgd_planning_smooth_gripper:main',
            'sgd_planning_smooth_gripper_initial_pose = my_franka_controller.sgd_planning_smooth_gripper_initial_pose:main',
        ],
    },
)
