#!/usr/bin/env bash
# 추론 시작 전 로봇 초기 대기 자세 및 그리퍼 개방 스크립트
# ─────────────────────────────────────────────────────────────
# 학습 데이터셋(Task_1_pick_quvi_MCAP_lerobot_v30)의 시작 상태:
#   joint1: 0.06 rad
#   joint2: -1.72 rad (상단 홈)
#   joint3: 1.40 rad
#   joint4: 1.62 rad
#   joint5: -0.17 rad
#   gripper_joint_1: 0.68 rad (완전 개방)

set -e

echo "[prep_inference] 로봇 초기 대기 자세로 이동 및 그리퍼 개방 (3초 소요)..."

docker exec open_manipulator bash -c '
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=30

ros2 topic pub --once /leader/joint_trajectory trajectory_msgs/msg/JointTrajectory "{
  joint_names: [\"joint1\", \"joint2\", \"joint3\", \"joint4\", \"joint5\", \"gripper_joint_1\"],
  points: [{
    positions: [0.06, -1.72, 1.40, 1.62, -0.17, 0.68],
    time_from_start: {sec: 3}
  }]
}"
'

echo "[prep_inference] 준비 완료! 이제 Cyclo UI에서 추론을 시작할 수 있습니다."
