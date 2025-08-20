#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from builtin_interfaces.msg import Duration

class MoveToPoseNode(Node):
    def __init__(self):
        super().__init__('move_to_pose_node')

        # IK 서비스 클라이언트
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.ik_client.wait_for_service()

        # Trajectory 액션 클라이언트
        self.action_client = ActionClient(self, FollowJointTrajectory,
                                          '/fr3_arm_controller/follow_joint_trajectory')
        self.action_client.wait_for_server()

        self.call_ik_service()

    def call_ik_service(self):
        request = GetPositionIK.Request()
        request.ik_request.group_name = 'fr3_arm'
        request.ik_request.pose_stamped.header.frame_id = 'fr3_link0'
        request.ik_request.ik_link_name = 'fr3_hand_tcp'

        self.get_logger().info('header.frame_id: ' + request.ik_request.pose_stamped.header.frame_id)

        target_pose = [0.3, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]

        pose = request.ik_request.pose_stamped.pose
        pose.position.x = target_pose[0]
        pose.position.y = target_pose[1]
        pose.position.z = target_pose[2]
        pose.orientation.x = target_pose[3]
        pose.orientation.y = target_pose[4]
        pose.orientation.z = target_pose[5]
        pose.orientation.w = target_pose[6]  # 🔧 수정됨 (원래 오타)

        future = self.ik_client.call_async(request)
        future.add_done_callback(self.handle_ik_response)

    def handle_ik_response(self, future):
        result = future.result()
        joint_state = result.solution.joint_state

        if not joint_state.name:
            self.get_logger().error('❌ IK failed')
            rclpy.shutdown()
            return

        self.get_logger().info(f'✅ IK Success: {joint_state.position}')

        controller_joint_names = [
            "fr3_joint1", "fr3_joint2", "fr3_joint3",
            "fr3_joint4", "fr3_joint5", "fr3_joint6", "fr3_joint7"
        ]

        name_to_position = dict(zip(joint_state.name, joint_state.position))
        sorted_positions = [name_to_position[name] for name in controller_joint_names]

        traj = JointTrajectory()
        traj.joint_names = controller_joint_names

        point = JointTrajectoryPoint()
        point.positions = sorted_positions
        point.time_from_start = Duration(sec=5)
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
    node = MoveToPoseNode()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
