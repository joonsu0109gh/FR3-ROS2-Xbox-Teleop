#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from builtin_interfaces.msg import Duration

import pickle
import numpy as np

import franky
import time
import torch 

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import TransformStamped

class MoveToJointNode(Node):
    def __init__(self):
        super().__init__('move_to_joint_node')

        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/fr3_arm_controller/follow_joint_trajectory'
        )
        self.get_logger().info('⏳ Waiting for action server...')
        self.action_client.wait_for_server()
        self.get_logger().info('✅ Action server available.')
        
        self.gripper = franky.Gripper("172.16.0.2")

        ## For success data
        # # load trajectory data
        # file_path = "/home/rvi/SDG_planning/real_setup/success_data_dict_SGD.pickle"

        # with open(file_path, 'rb') as f:
        #     success_data_dict = pickle.load(f)

        # self.target_joints_array = success_data_dict["isaac_best_trajs"][:, :7].cpu().numpy()
        # # Position: x=0.405, y=-0.186, z=0.097
        # # Orientation: x=0.514, y=-0.498, z=0.507, w=0.481

        ## Failure data
        file_path = "/home/rvi/SDG_planning/real_setup/results_data_dict.pickle"


        with open(file_path, 'rb') as f:
            failure_data_dict = pickle.load(f)

        self.target_joints_array = failure_data_dict["trajs_final_free"][:, :7].cpu().numpy()

        self.initial_joint_position = self.target_joints_array[0]  # Use the first pose as the initial position

        self.controller_joint_names = [
            "fr3_joint1", "fr3_joint2", "fr3_joint3",
            "fr3_joint4", "fr3_joint5", "fr3_joint6", "fr3_joint7"
        ]

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.max_speed = 0.4          # rad/sec
        self.min_duration = 0.3
        self.max_duration = 3.0
        self.go_to_home_time = 7.0


        # go to initial position
        self.go_to_initial_position()


    def go_to_initial_position(self):
        traj = JointTrajectory()
        traj.joint_names = self.controller_joint_names

        point = JointTrajectoryPoint()
        point.positions = self.initial_joint_position.tolist()
        point.time_from_start = Duration(sec=int(self.go_to_home_time), nanosec=0)
        traj.points.append(point)

        self.get_logger().info('🚀 Sending initial position trajectory')

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        send_goal_future = self.action_client.send_goal_async(goal)
        send_goal_future.add_done_callback(self.goal_response_callback)

        
        self.gripper.move_async(width=0.08, speed=0.05)

        # get input to start grasping
        start_flg = input("Press Enter to grasp the object...")

        if start_flg == '':
            self.grasp_object()
        else:
            self.get_logger().info("No input received, exiting.")
            rclpy.shutdown()

    def grasp_object(self):
        print("Grasping the object with the gripper...")

        # base_link → fr3_hand_tcp까지의 변환을 요청
        transform: TransformStamped = self.tf_buffer.lookup_transform(
            'fr3_link0',       # base frame
            'fr3_hand_tcp',    # end-effector frame
            rclpy.time.Time())

        pose = transform.transform
        self.get_logger().info(f'📍 EE Pose:\n'
                                f'Position: x={pose.translation.x:.3f}, y={pose.translation.y:.3f}, z={pose.translation.z:.3f}\n'
                                f'Orientation: x={pose.rotation.x:.3f}, y={pose.rotation.y:.3f}, z={pose.rotation.z:.3f}, w={pose.rotation.w:.3f}')

        # # wait for the gripper to finish
        # time.sleep(2)  # Adjust the sleep time as necessary
        self.gripper.grasp_async(
                        width=0.02, speed=0.05, force=20, epsilon_inner=0.1, epsilon_outer=0.1
                    )
        
        print("Gripper has grasped the object.")
          # Wait for the gripper to finish grasping

        time.sleep(5)  # Adjust the sleep time as necessary
        self.send_full_trajectory()

    def send_full_trajectory(self):
        traj = JointTrajectory()
        traj.joint_names = self.controller_joint_names

        time_from_start = 0.0
        previous_joints = self.initial_joint_position

        for i, target_joints in enumerate(self.target_joints_array):
            joint_deltas = np.abs(target_joints - previous_joints)
            max_delta = np.max(joint_deltas)
            duration = max(self.min_duration, min(self.max_duration, max_delta / self.max_speed))

            time_from_start += duration

            point = JointTrajectoryPoint()
            point.positions = target_joints.tolist()
            point.time_from_start = Duration(
                sec=int(time_from_start),
                nanosec=int((time_from_start - int(time_from_start)) * 1e9)
            )
            traj.points.append(point)

            previous_joints = target_joints

        self.get_logger().info(f'🚀 Sending full trajectory with {len(traj.points)} waypoints')

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        send_goal_future = self.action_client.send_goal_async(goal)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('❌ Goal rejected.')
            rclpy.shutdown()
            return

        self.get_logger().info('✅ Trajectory goal accepted.')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.trajectory_done_callback)

    def trajectory_done_callback(self, future):
        try:
            result_handle = future.result()
            result = result_handle.result
            status = result_handle.status

            self.get_logger().info(f'🎯 Final goal finished with status: {status}')
            if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
                self.get_logger().error(f'❌ Execution failed: {result.error_code}')
                self.get_logger().error(f'Error string: {result.error_string}')
            else:
                self.get_logger().info('✅ Trajectory execution completed successfully.')

            self.destroy_node()
            rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f'💥 Exception in trajectory_done_callback: {e}')
            rclpy.shutdown()


def main():
    rclpy.init()
    node = MoveToJointNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
