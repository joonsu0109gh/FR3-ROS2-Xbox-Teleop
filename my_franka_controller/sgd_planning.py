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

        # Read target joint
        # file_path = "/home/rvi/SDG_planning/success_data_dict_v2.pickle"
        file_path = "/home/rvi/SDG_planning/success_data_dict.pickle"

        with open(file_path, 'rb') as f:
            success_data_dict = pickle.load(f)
        self.target_joints_array = success_data_dict["isaac_best_trajs"][:, :7].cpu().numpy()

        self.current_index = 0
        self.is_initial_move = True

        self.controller_joint_names = [
            "fr3_joint1", "fr3_joint2", "fr3_joint3",
            "fr3_joint4", "fr3_joint5", "fr3_joint6", "fr3_joint7"
        ]

        self.max_speed = 0.4          # rad/sec 기준
        self.min_duration = 0.3
        self.max_duration = 3.0
        self.go_to_home_time = 10.0
        # 초기 위치 (명시적 지정)
        self.initial_joint_position = np.zeros(7)

        self.send_joint_goal()

    def send_joint_goal(self):
        if self.current_index >= len(self.target_joints_array):
            self.get_logger().info('🎉 All trajectories executed successfully.')
            self.destroy_node()
            rclpy.shutdown()
            return

        target_joints = self.target_joints_array[self.current_index].tolist()
        self.get_logger().info(f"Target joints [{self.current_index}]: {target_joints}")

        if self.is_initial_move:
            self.get_logger().info('🚀 Moving to the first joint pose (initial position).')
            self.current_duration_sec = self.go_to_home_time
        else:
            self.get_logger().info(f'👉 Sending waypoint {self.current_index + 1}/{len(self.target_joints_array)}')

            if self.current_index == 0:
                prev_joints = self.initial_joint_position
            else:
                prev_joints = self.target_joints_array[self.current_index - 1]

            # 최대 조인트별 변화량 기준 속도 기반 duration 계산
            joint_deltas = np.abs(np.array(target_joints) - np.array(prev_joints))
            max_delta = np.max(joint_deltas)

            self.current_duration_sec = max(
                self.min_duration,
                min(self.max_duration, max_delta / self.max_speed)
            )
            self.get_logger().info(f"Calculated duration: {self.current_duration_sec:.2f} seconds")

        traj = JointTrajectory()
        traj.joint_names = self.controller_joint_names

        point = JointTrajectoryPoint()
        point.positions = target_joints

        sec_part = int(self.current_duration_sec)
        nsec_part = int((self.current_duration_sec - sec_part) * 1e9)
        if nsec_part >= 1e9:
            sec_part += 1
            nsec_part -= int(1e9)

        point.time_from_start = Duration(sec=sec_part, nanosec=nsec_part)
        traj.points.append(point)

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

            self.get_logger().info(f'Goal finished with status: {status}')
            if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
                self.get_logger().error(f'❌ Trajectory execution failed with error code: {result.error_code}')
                self.get_logger().error(f'Error string: {result.error_string}')
                rclpy.shutdown()
                return

            if self.is_initial_move:
                self.get_logger().info('✅ Initial pose reached. Starting sequential execution.')
                self.is_initial_move = False
            else:
                self.get_logger().info(f'✅ Waypoint {self.current_index + 1} execution finished.')

            self.current_index += 1
            self.send_joint_goal()

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
