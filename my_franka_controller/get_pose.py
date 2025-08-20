#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import JointState  # ✅ 추가

class CurrentPoseNode(Node):
    def __init__(self):
        super().__init__('current_pose_node')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.joint_states = None  # 가장 최근 joint state 저장
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',  # 보통 MoveIt이나 ros2_control에서 퍼블리시함
            self.joint_callback,
            10
        )

        self.timer = self.create_timer(1.0, self.timer_callback)

    def joint_callback(self, msg: JointState):
        self.joint_states = msg

    def timer_callback(self):
        try:
            # base_link → fr3_hand_tcp까지의 변환을 요청
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                'fr3_link0',       # base frame
                'fr3_hand_tcp',    # end-effector frame
                rclpy.time.Time())

            pose = transform.transform
            self.get_logger().info(f'📍 EE Pose:\n'
                                   f'Position: x={pose.translation.x:.3f}, y={pose.translation.y:.3f}, z={pose.translation.z:.3f}\n'
                                   f'Orientation: x={pose.rotation.x:.3f}, y={pose.rotation.y:.3f}, z={pose.rotation.z:.3f}, w={pose.rotation.w:.3f}')
        except Exception as e:
            self.get_logger().warn(f'TF transform failed: {e}')

        # ✅ 현재 joint 값 출력
        if self.joint_states is not None:
            joint_str = '\n'.join(
                [f'{name}: {pos:.4f}' for name, pos in zip(self.joint_states.name, self.joint_states.position)]
            )
            self.get_logger().info(f'🦾 Joint Values:\n{joint_str}')
        else:
            self.get_logger().warn('Joint states not received yet.')

def main():
    rclpy.init()
    node = CurrentPoseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
