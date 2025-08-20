#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from builtin_interfaces.msg import Duration

import pickle
import numpy as np

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

        # load trajectory data
        file_path = "/home/rvi/SDG_planning/success_data_dict_v2.pickle"
        with open(file_path, 'rb') as f:
            success_data_dict = pickle.load(f)
        self.target_joints_array = success_data_dict["isaac_best_trajs"][:, :7].cpu().numpy()

        self.controller_joint_names = [
            "fr3_joint1", "fr3_joint2", "fr3_joint3",
            "fr3_joint4", "fr3_joint5", "fr3_joint6", "fr3_joint7"
        ]

        self.max_speed = 0.4          # rad/sec
        self.min_duration = 0.3
        self.max_duration = 3.0
        self.go_to_home_time = 5.0

        self.initial_joint_position = np.zeros(7)

        # send entire trajectory at once
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
