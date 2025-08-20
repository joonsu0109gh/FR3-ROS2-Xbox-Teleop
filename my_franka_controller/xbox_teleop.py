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


class XboxTeleopNode(Node):
    def __init__(self):
        super().__init__('xbox_teleop_node')

        self.translation_increment = 0.03
        self.rotation_increment = 0.2

        self.frame_id = 'fr3_link0'
        self.ee_link = 'fr3_hand_tcp'
        self.group_name = 'fr3_arm'

        self.send_interval_sec = 0.05
        self.home_pose = np.array([0.3, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0])
        self.go_to_home_time = 3.0

        self.current_translation = None
        self.current_rotation = None
        self.last_pose_sent = None

        self.max_speed = 0.01
        self.min_duration = 0.3
        self.max_duration = 3.0

        self.ik_busy = False
        self.joy_connected = False
        self.initial_axes = None
        self.home_pose_done = False
        self.last_send_time = self.get_clock().now()
        self.last_joy_time = self.get_clock().now()

        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.ik_client.wait_for_service()

        self.action_client = ActionClient(self, FollowJointTrajectory,
                                          '/fr3_arm_controller/follow_joint_trajectory')
        self.action_client.wait_for_server()

        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10)

        self.create_timer(1.0, self.send_home_pose_once)
        self.create_timer(2.0, self.check_joy_connection)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info('🕹️ Xbox teleoperation with Cartesian control initialized.')

    def send_home_pose_once(self):
        if not self.home_pose_done:
            self.get_logger().info("🏠 Sending home pose...")
            self.current_translation = self.home_pose[0:3]
            self.current_rotation = self.home_pose[3:7]
            self.send_pose(is_home=True)

            # ⏱️ 홈 포즈 duration(4초) 이후에 조이스틱 입력 허용
            self.create_timer(self.go_to_home_time, self.set_home_pose_done_once)

    def set_home_pose_done_once(self):
        if not self.home_pose_done:
            self.home_pose_done = True
            self.get_logger().info("✅ Home pose duration elapsed. Ready for teleoperation.")

    def joy_callback(self, msg: Joy):
        if not self.home_pose_done:
            return

        if not self.joy_connected:
            self.get_logger().info('✅ Joystick connected!')
            self.joy_connected = True
            self.initial_axes = np.array(msg.axes)
            return

        self.last_joy_time = self.get_clock().now()

        axes = np.array(msg.axes)
        delta_axes = axes - self.initial_axes
        dpos = np.array([delta_axes[1], delta_axes[0], -delta_axes[2] + delta_axes[5]]) * self.translation_increment

        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                self.frame_id, self.ee_link, rclpy.time.Time())
            pose = transform.transform
            self.current_translation = np.array([
                pose.translation.x, pose.translation.y, pose.translation.z]) + dpos
            base_rot = R.from_quat([
                pose.rotation.x, pose.rotation.y,
                pose.rotation.z, pose.rotation.w])
        except Exception as e:
            self.get_logger().warn(f"⚠️ TF lookup failed: {e}")
            return

        drot_xyz = np.array([delta_axes[3], delta_axes[4], 0.0]) * self.rotation_increment
        if np.linalg.norm(drot_xyz) > 1e-6:
            delta_rot = R.from_euler('xyz', drot_xyz)
            self.current_rotation = (delta_rot * base_rot).as_quat()

        now = self.get_clock().now()
        if (now - self.last_send_time).nanoseconds > self.send_interval_sec * 1e9:
            if not self.ik_busy:
                self.send_pose()
                self.last_send_time = now

    def check_joy_connection(self):
        time_since_last = self.get_clock().now() - self.last_joy_time
        if time_since_last.nanoseconds > 5e9 and self.joy_connected:
            self.get_logger().warn('❌ Joystick disconnected!')
            self.joy_connected = False

    def send_pose(self, is_home=False):
        self.ik_busy = True

        if is_home:
            self.current_duration_sec = self.go_to_home_time
        else:
            dist = np.linalg.norm(self.current_translation - self.last_pose_sent)
            duration_sec = dist / self.max_speed
            self.current_duration_sec = max(self.min_duration, min(duration_sec, self.max_duration))

        request = GetPositionIK.Request()
        request.ik_request.group_name = self.group_name
        request.ik_request.pose_stamped.header.frame_id = self.frame_id
        request.ik_request.ik_link_name = self.ee_link

        pose = request.ik_request.pose_stamped.pose
        pose.position.x, pose.position.y, pose.position.z = self.current_translation
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = self.current_rotation

        future = self.ik_client.call_async(request)
        future.add_done_callback(self.handle_ik_response)

    def handle_ik_response(self, future):
        try:
            result = future.result()
        except Exception as e:
            self.get_logger().error(f'❌ IK service call failed: {e}')
            self.ik_busy = False
            return

        joint_state = result.solution.joint_state
        if not joint_state.name:
            self.get_logger().error('❌ IK failed: No joint state returned.')
            self.ik_busy = False
            return

        joint_order = [
            "fr3_joint1", "fr3_joint2", "fr3_joint3",
            "fr3_joint4", "fr3_joint5", "fr3_joint6", "fr3_joint7"
        ]
        name_to_position = dict(zip(joint_state.name, joint_state.position))
        sorted_positions = [name_to_position[name] for name in joint_order]

        traj = JointTrajectory()
        traj.joint_names = joint_order
        point = JointTrajectoryPoint()
        point.positions = sorted_positions

        sec_part = int(self.current_duration_sec)
        nsec_part = int((self.current_duration_sec - sec_part) * 1e9)
        if nsec_part >= 1e9:
            sec_part += 1
            nsec_part -= int(1e9)

        point.time_from_start = Duration(sec=sec_part, nanosec=nsec_part)
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