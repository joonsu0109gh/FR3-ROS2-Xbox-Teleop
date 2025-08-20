#!/usr/bin/env python3
from __future__ import annotations

import os
import pickle
import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from builtin_interfaces.msg import Duration as RosDuration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory

import franky


class MoveToJointNode(Node):
    def __init__(self) -> None:
        super().__init__('move_to_joint_node')

        # ---------- Parameters (CLI/launch에서 override 가능) ----------
        self.declare_parameter('arm_action_ns', '/fr3_arm_controller/follow_joint_trajectory')
        self.declare_parameter('gripper_ip', '172.16.0.2')
        self.declare_parameter('traj_pickle', '/home/rvi/SDG_planning/real_setup/success_data_dict_SGD.pickle')
        self.declare_parameter('traj_key', 'isaac_best_trajs')   # pickled dict key
        self.declare_parameter('go_home_time', 7.0)              # seconds
        self.declare_parameter('max_speed', 0.4)                 # rad/s
        self.declare_parameter('min_duration', 0.30)             # s
        self.declare_parameter('max_duration', 3.00)             # s
        self.declare_parameter('auto_grasp', False)              # True면 입력 대기 없이 바로 grasp

        self.action_ns: str = self.get_parameter('arm_action_ns').get_parameter_value().string_value
        self.gripper_ip: str = self.get_parameter('gripper_ip').get_parameter_value().string_value
        self.traj_pickle: str = self.get_parameter('traj_pickle').get_parameter_value().string_value
        self.traj_key: str = self.get_parameter('traj_key').get_parameter_value().string_value
        self.go_to_home_time: float = self.get_parameter('go_home_time').get_parameter_value().double_value
        self.max_speed: float = self.get_parameter('max_speed').get_parameter_value().double_value
        self.min_duration: float = self.get_parameter('min_duration').get_parameter_value().double_value
        self.max_duration: float = self.get_parameter('max_duration').get_parameter_value().double_value
        self.auto_grasp: bool = self.get_parameter('auto_grasp').get_parameter_value().bool_value

        # ---------- Action client ----------
        self.action_client = ActionClient(self, FollowJointTrajectory, self.action_ns)
        self.get_logger().info(f'⏳ Waiting for action server: {self.action_ns}')
        self.action_client.wait_for_server()
        self.get_logger().info('✅ Action server available.')

        # ---------- Gripper ----------
        self.gripper = franky.Gripper(self.gripper_ip)

        # ---------- Load trajectory ----------
        self.target_joints_array = self._load_trajectory(self.traj_pickle, self.traj_key)
        if self.target_joints_array is None or len(self.target_joints_array) == 0:
            raise RuntimeError('No trajectory points loaded.')

        self.initial_joint_position = self.target_joints_array[0]
        self.controller_joint_names = [
            "fr3_joint1", "fr3_joint2", "fr3_joint3",
            "fr3_joint4", "fr3_joint5", "fr3_joint6", "fr3_joint7"
        ]

        # 시작: 초기 자세 이동
        self.go_to_initial_position()

    # ---------- Helpers ----------
    def _load_trajectory(self, file_path: str, key: str) -> Optional[np.ndarray]:
        """pickle에서 trajectory 텐서를 읽어 numpy (N,7)로 반환."""
        if not os.path.exists(file_path):
            self.get_logger().error(f'❌ Trajectory file not found: {file_path}')
            return None
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            traj = data[key]  # torch.Tensor or np.ndarray expected
            # torch.Tensor인 경우에 대비
            if hasattr(traj, 'cpu'):
                traj = traj.cpu().numpy()
            traj = np.asarray(traj)
            if traj.ndim != 2 or traj.shape[1] < 7:
                self.get_logger().error(f'❌ Invalid trajectory shape: {traj.shape}')
                return None
            traj = traj[:, :7]
            self.get_logger().info(f'📦 Loaded trajectory: {traj.shape[0]} waypoints')
            return traj
        except KeyError:
            self.get_logger().error(f'❌ Key "{key}" not found in pickle.')
            return None
        except Exception as e:
            self.get_logger().error(f'💥 Failed to load trajectory: {e}')
            return None

    def _to_duration(self, seconds: float) -> RosDuration:
        sec = int(seconds)
        nsec = int((seconds - sec) * 1e9)
        return RosDuration(sec=sec, nanosec=nsec)

    def _build_single_point_traj(self, positions: np.ndarray, duration_s: float) -> JointTrajectory:
        traj = JointTrajectory()
        traj.joint_names = self.controller_joint_names

        pt = JointTrajectoryPoint()
        pt.positions = positions.astype(float).tolist()
        pt.time_from_start = self._to_duration(duration_s)
        traj.points.append(pt)
        return traj

    def _build_full_traj(self, waypoints: np.ndarray) -> JointTrajectory:
        traj = JointTrajectory()
        traj.joint_names = self.controller_joint_names

        time_from_start = 0.0
        prev = self.initial_joint_position
        for idx, target in enumerate(waypoints):
            max_delta = float(np.max(np.abs(target - prev)))
            duration = max(self.min_duration, min(self.max_duration, max_delta / max(self.max_speed, 1e-6)))
            time_from_start += duration

            pt = JointTrajectoryPoint()
            pt.positions = target.astype(float).tolist()
            pt.time_from_start = self._to_duration(time_from_start)
            traj.points.append(pt)
            prev = target

        return traj

    # ---------- High-level steps ----------
    def go_to_initial_position(self) -> None:
        traj = self._build_single_point_traj(self.initial_joint_position, self.go_to_home_time)
        self.get_logger().info('🚀 Sending initial position trajectory')
        self._send_goal(traj)

        # 미리 그리퍼 오픈
        try:
            self.gripper.move_async(width=0.08, speed=0.05)
        except Exception as e:
            self.get_logger().warn(f'⚠️ Gripper open failed (continuing): {e}')

        if self.auto_grasp:
            # 입력 대기 없이 바로 진행
            self.grasp_object()
        else:
            # 입력은 blocking이므로, 빠른 종료를 원하면 auto_grasp=True로 사용 권장
            try:
                start_flg = input("Press Enter to grasp the object (or type 'q' to quit)... ")
            except EOFError:
                start_flg = 'q'

            if start_flg.strip().lower() in ('q', 'quit', 'exit'):
                self.get_logger().info('👋 Exit requested, destroying node.')
                self.destroy_node()
                # shutdown은 main()에서 처리
            else:
                self.grasp_object()

    def grasp_object(self) -> None:
        self.get_logger().info("✋ Grasping the object...")
        try:
            self.gripper.grasp_async(width=0.02, speed=0.05, force=20,
                                     epsilon_inner=0.1, epsilon_outer=0.1)
        except Exception as e:
            self.get_logger().error(f'❌ Gripper grasp failed: {e}')
            # 그래도 트래젝토리는 진행 가능하니 계속감
        time.sleep(1.0)

        self.send_full_trajectory()

    def send_full_trajectory(self) -> None:
        traj = self._build_full_traj(self.target_joints_array)
        self.get_logger().info(f'🚀 Sending full trajectory with {len(traj.points)} waypoints')
        self._send_goal(traj)

    # ---------- Action handling ----------
    def _send_goal(self, traj: JointTrajectory) -> None:
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        future = self.action_client.send_goal_async(goal)
        future.add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as e:
            self.get_logger().error(f'💥 Goal response error: {e}')
            self.destroy_node()
            return

        if not goal_handle.accepted:
            self.get_logger().error('❌ Goal rejected.')
            self.destroy_node()
            return

        self.get_logger().info('✅ Trajectory goal accepted.')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._traj_done_cb)

    def _traj_done_cb(self, future) -> None:
        try:
            result_handle = future.result()
            result = result_handle.result
            status = result_handle.status
            self.get_logger().info(f'🎯 Final goal finished with status: {status}')

            if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
                self.get_logger().error(f'❌ Execution failed: {result.error_code}')
                if hasattr(result, 'error_string') and result.error_string:
                    self.get_logger().error(f'Error string: {result.error_string}')
            else:
                self.get_logger().info('✅ Trajectory execution completed successfully.')

        except Exception as e:
            self.get_logger().error(f'💥 Exception in trajectory done callback: {e}')
        finally:
            # 콜백에서는 shutdown 호출하지 않음 (중복 방지)
            self.destroy_node()


def main() -> None:
    rclpy.init()
    node = MoveToJointNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 노드가 콜백에서 destroy될 수 있으므로, 여기서도 안전하게 파괴 시도
        try:
            node.destroy_node()
        except Exception:
            pass
        # shutdown은 단 한 번, 가드로 안전하게
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
