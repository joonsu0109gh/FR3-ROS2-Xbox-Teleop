# Franka Research 3 ROS2 Xbox Teleoperation

This repository serves as a personal backup for code related to teleoperating the Franka Research 3 (FR3) robot using an Xbox controller with ROS2. It builds upon the [official FR3 repository](https://github.com/frankarobotics/franka_ros2).

## Adding New Code

To include new code in the project:

1. Add the new file to the `my_franka_controller/my_franka_controller` directory.
2. Update the `my_franka_controller/setup.py` file to include the new file in the `entry_points` section.

Example `setup.py` configuration:

```python
entry_points={
    'console_scripts': [
        'move_to_pose = my_franka_controller.move_to_pose:main',
        'move_to_joint = my_franka_controller.move_to_joint:main',
        'get_pose = my_franka_controller.get_pose:main',
    ],
},
```

## Execution

Follow these steps to run the teleoperation system:

1. Launch the MoveIt configuration for FR3, replacing `{fr3/ip/address}` with the robot's IP address:

```shell
ros2 launch franka_fr3_moveit_config moveit.launch.py robot_ip:={fr3/ip/address}
```

2. Start the joystick node:

```shell
ros2 run joy joy_node
```

3. Run the Xbox teleoperation node:

```shell
ros2 run my_franka_controller xbox_teleop
```