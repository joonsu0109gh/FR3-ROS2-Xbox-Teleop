#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import TransformStamped
from moveit_msgs.srv import GetPositionIK
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from builtin_interfaces.msg import Duration
from scipy.spatial.transform import Rotation as R
import numpy as np
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import threading
# import franky

NSEC = 1e9
EPSILON = 1e-6
DEADZONE = 0.1


class XboxTeleopNode(Node):
    def __init__(self):
        super().__init__('xbox_teleop_node')

        self.translation_increment = 0.03
        self.rotation_increment = self.translation_increment * 5
        self.go_to_home_time = 3.0

        self.frame_id = 'fr3_link0'
        self.ee_link = 'fr3_hand_tcp'
        self.group_name = 'fr3_arm'

        self.home_pose = np.array([0.3, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0])
        self.current_translation = self.home_pose[:3].copy()
        self.current_rotation = self.home_pose[3:].copy()
        self.current_rotation_matrix = R.from_quat(self.current_rotation).as_matrix()

        self.last_pose_sent = self.current_translation.copy()
        self.max_speed = 0.2
        self.min_duration = 0.3
        self.max_duration = 3.0

        self.ik_busy = False
        self.joy_connected = False
        self.home_pose_done = False
        self.prev_button_a = 0
        self.initial_axes = None

        self.latest_joy_msg = None
        self.joy_process_interval_sec = 0.01

        self.last_send_time = self.get_clock().now()
        self.last_joy_time = self.get_clock().now()
        self.joint_order = [f"fr3_joint{i}" for i in range(1, 8)]

        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.ik_client.wait_for_service()

        self.action_client = ActionClient(self, FollowJointTrajectory, '/fr3_arm_controller/follow_joint_trajectory')
        self.action_client.wait_for_server()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info('🕹️ Xbox teleoperation with Cartesian control initialized.')

        # self.gripper = franky.Gripper("172.16.0.2")
        # self.gripper_speed = 0.1
        # self.gripper_force = 20.0
        # self.gripper_opened = True
        # self.gripper.open(self.gripper_speed)

        self.check_joy_connection()
        self.send_home_pose_once()

    def toggle_gripper(self):
        if self.gripper_opened:
            self.get_logger().info("🔒 Grasping object.")
            threading.Thread(target=self.gripper.grasp,
                             args=(0.0, self.gripper_speed, self.gripper_force),
                             kwargs={'epsilon_outer': 1.0}, daemon=True).start()
        else:
            self.get_logger().info("🔓 Opening gripper.")
            threading.Thread(target=self.gripper.open,
                             args=(self.gripper_speed,), daemon=True).start()
        self.gripper_opened = not self.gripper_opened

    def compute_duration(self, dist):
        return max(self.min_duration, min(dist / self.max_speed, self.max_duration))

    def send_home_pose_once(self):
        if not self.home_pose_done:
            self.get_logger().info("🏠 Sending home pose...")
            self.send_pose(is_home=True)
            self.create_timer(self.go_to_home_time, self.set_home_pose_done_once)

    def set_home_pose_done_once(self):
        if not self.home_pose_done:
            self.home_pose_done = True
            self.get_logger().info("✅ Home pose duration elapsed. Ready for teleoperation.")
            self.create_subscription(Joy, '/joy', self.joy_callback, 10)
            self.create_timer(self.joy_process_interval_sec, self.process_joy)

    def joy_callback(self, msg: Joy):
        self.latest_joy_msg = msg

    def apply_deadzone(self, x):
        return x if abs(x) > DEADZONE else 0.0

    def process_joy(self):
        if not self.home_pose_done or self.latest_joy_msg is None:
            return

        msg = self.latest_joy_msg

        if not self.joy_connected:
            self.get_logger().info("✅ Joystick connected!")
            self.joy_connected = True
            self.initial_axes = np.array(msg.axes)
            return

        button_a = msg.buttons[0]
        # if button_a == 1 and self.prev_button_a == 0:
        #     self.toggle_gripper()
        # self.prev_button_a = button_a

        self.last_joy_time = self.get_clock().now()
        delta_axes = np.array(msg.axes) - self.initial_axes

        dx = self.apply_deadzone(delta_axes[0]) * self.translation_increment
        dy = self.apply_deadzone(delta_axes[1]) * self.translation_increment
        dz = self.apply_deadzone(delta_axes[5] - delta_axes[2]) * (self.translation_increment / 2.0)
        dpos = np.array([dy, dx, dz])

        transform = self.get_current_transform()
        if not transform:
            return

        self.current_translation = np.array([
            transform.translation.x,
            transform.translation.y,
            transform.translation.z
        ]) + dpos

        self.current_rotation_matrix = R.from_quat([
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w
        ]).as_matrix()

        drot_xyz = np.array([
            self.apply_deadzone(delta_axes[3]),
            self.apply_deadzone(delta_axes[4]),
            0.0
        ]) * self.rotation_increment

        if np.linalg.norm(drot_xyz) > EPSILON:
            delta_rot = R.from_euler('xyz', drot_xyz).as_matrix()

            # 현재 rotation을 transform에서 가져오지 않고 누적값 사용
            self.current_rotation_matrix = delta_rot @ self.current_rotation_matrix
            self.current_rotation = R.from_matrix(self.current_rotation_matrix).as_quat()

        now = self.get_clock().now()
        if (now - self.last_send_time).nanoseconds > self.joy_process_interval_sec * NSEC and not self.ik_busy:
            self.send_pose()
            self.last_send_time = now

    def get_current_transform(self):
        try:
            return self.tf_buffer.lookup_transform(self.frame_id, self.ee_link, rclpy.time.Time()).transform
        except Exception as e:
            self.get_logger().warning(f"⚠️ TF lookup failed: {e}")
            return None

    def check_joy_connection(self):
        if (self.get_clock().now() - self.last_joy_time).nanoseconds > 5 * NSEC and self.joy_connected:
            self.get_logger().warn('❌ Joystick disconnected!')
            self.joy_connected = False

    def send_pose(self, is_home=False):
        if self.ik_busy:
            return
        self.ik_busy = True

        self.current_duration_sec = self.go_to_home_time if is_home else self.compute_duration(
            np.linalg.norm(self.current_translation - self.last_pose_sent))

        request = GetPositionIK.Request()
        request.ik_request.group_name = self.group_name
        request.ik_request.ik_link_name = self.ee_link
        request.ik_request.pose_stamped.header.frame_id = self.frame_id

        pose = request.ik_request.pose_stamped.pose
        pose.position.x, pose.position.y, pose.position.z = self.current_translation
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = self.current_rotation

        future = self.ik_client.call_async(request)
        future.add_done_callback(self.handle_ik_response)

    def handle_ik_response(self, future):
        try:
            result = future.result()
            joint_state = result.solution.joint_state
            if not joint_state.name:
                raise ValueError("No joint state returned.")
        except Exception as e:
            self.get_logger().error(f'❌ IK service call failed: {e}')
            self.ik_busy = False
            return

        name_to_position = dict(zip(joint_state.name, joint_state.position))
        sorted_positions = [name_to_position[name] for name in self.joint_order]

        traj = JointTrajectory()
        traj.joint_names = self.joint_order
        point = JointTrajectoryPoint()
        point.positions = sorted_positions

        sec = int(self.current_duration_sec)
        nsec = int((self.current_duration_sec - sec) * NSEC)

        if nsec >= NSEC:
            sec += 1
            nsec -= int(NSEC)

        point.time_from_start = Duration(sec=sec, nanosec=nsec)
        traj.points.append(point)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        self.action_client.send_goal_async(goal)
        self.last_pose_sent = self.current_translation.copy()
        self.ik_busy = False


def main():
    rclpy.init()
    node = XboxTeleopNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
