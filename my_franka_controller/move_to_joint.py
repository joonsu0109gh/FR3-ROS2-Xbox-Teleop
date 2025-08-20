#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from builtin_interfaces.msg import Duration


class MoveToJointNode(Node):
    def __init__(self):
        super().__init__('move_to_joint_node')

        # JointTrajectory 액션 클라이언트 생성
        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/fr3_arm_controller/follow_joint_trajectory'
        )
        self.action_client.wait_for_server()

        self.send_joint_goal()

    def send_joint_goal(self):
        # 이동할 목표 joint 값 설정 (예: Home 또는 Custom Pose)
        target_joints = [2.5879, -1.2539, -1.9536, -0.4121, -1.4585,  0.7232, -1.2019]  # radian

        controller_joint_names = [
            "fr3_joint1", "fr3_joint2", "fr3_joint3",
            "fr3_joint4", "fr3_joint5", "fr3_joint6", "fr3_joint7"
        ]

        traj = JointTrajectory()
        traj.joint_names = controller_joint_names

        point = JointTrajectoryPoint()
        point.positions = target_joints
        point.time_from_start = Duration(sec=5)
        traj.points.append(point)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        # 목표 trajectory 전송
        send_goal_future = self.action_client.send_goal_async(goal)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('❌ Goal rejected.')
            rclpy.shutdown()
            return

        self.get_logger().info('🚀 Trajectory goal accepted.')

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.trajectory_done_callback)

    def trajectory_done_callback(self, future):
        result = future.result().result
        self.get_logger().info('✅ Trajectory execution finished.')
        self.destroy_node()
        rclpy.shutdown()


def main():
    rclpy.init()
    node = MoveToJointNode()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
