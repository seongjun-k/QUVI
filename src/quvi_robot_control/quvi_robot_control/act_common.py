"""로봇 노드·수집 도구·데이터셋 변환기가 함께 쓰는 상수와 단위 변환.

ROS·lerobot import 금지 — 수집 도구 venv(rclpy 없음)에서도 import 된다.
값을 바꾸면 학습 데이터와 추론 입력이 함께 바뀐다.
"""
import math

# ─── 관절 ───
# lerobot OmxFollower 모터 이름 순서. 데이터셋 state/action 벡터 순서도 이것을 따른다.
JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
               'wrist_flex', 'wrist_roll', 'gripper']

# ─── 그리퍼 raw 위치값 (XL330-M288T) ───
GRIPPER_OPEN  = 2500
GRIPPER_CLOSE = 1800

# ─── 자세 (raw Dynamixel 위치값, 0~4095 = 0~360°) ───
POSE_P1 = {'shoulder_pan': 2037, 'shoulder_lift': 1114, 'elbow_flex': 2771, 'wrist_flex': 3257, 'wrist_roll': 2016, 'gripper': 2077}  # 베드 위 대기 (2026-09-05 HMI 실측으로 상향 조정)

# 현재 학습 데이터(20261108_061014_box2, 112개) 에피소드 첫 프레임 평균(2026-11-08 계산).
# 수집 시작·끝 기준 자세이자 VLA 시작 자세의 단일 출처. 그리퍼도 데이터 평균값(2460)이다.
RECORD_START_POSE = {'shoulder_pan': 2043, 'shoulder_lift': 1009, 'elbow_flex': 2909,
                     'wrist_flex': 3043, 'wrist_roll': 2039, 'gripper': 2460}

# 홈 자세 = 학습 원점. 사이클이 끝나면 정책이 학습한 시작 자세로 돌아온다.
POSE_HOME = dict(RECORD_START_POSE)

# ─── rad↔raw 변환 (학습 데이터셋 단위 — lerobot 정규화와 별개) ───
DXL_TICKS_PER_REV = 4096.0   # XL430/XL330 1회전 틱 수 (변환 분모)
DXL_TICK_CENTER   = 2048.0   # 중립(0 rad) 틱


def raw_to_rad(raw: float) -> float:
    """raw Dynamixel 위치값 → 라디안 (ACT 학습 데이터셋 단위)."""
    return (raw - DXL_TICK_CENTER) * (2.0 * math.pi) / DXL_TICKS_PER_REV


def rad_to_raw(rad: float) -> float:
    """라디안(ACT 정책 출력) → raw Dynamixel 위치값."""
    return DXL_TICK_CENTER + rad * DXL_TICKS_PER_REV / (2.0 * math.pi)


# ─── 수집 에피소드 ───
# Cyclo(제조사) 방식: 녹화는 센서 고유 주기(카메라 30fps, 관절 RECORD_STATE_HZ)로 수신 시각과 함께
# 저장하고, 변환 때 DATASET_FPS 격자에 "그 시각 이전 마지막 샘플"로 맞춘다
# (cyclo_data converter/pipeline_worker.py DEFAULT_CONVERSION_FPS=15, reader/frame_timestamps.py).
# DATASET_FPS 는 로봇 추론 재생 주기(robot_control_node ACT_CONTROL_HZ=30)와 별개 값이다 —
# 그쪽은 2026-09-17 실측(15로 낮추면 떨림 악화)으로 30을 유지 중.
DATASET_FPS = 15
RECORD_STATE_HZ = 30         # 녹화 상태 발행 주기. 격자(15fps)보다 촘촘해야 시각 맞춤 오차가 작다
EPISODE_DURATION_SEC = 5.5
# 학습 지시문. 마침표 하나 차이로 tasks=2 가 된 사고 이력 — 녹화·변환 모두 이 값과 정확히 일치해야 한다
TASK_INSTRUCTION = 'Pick up the printed part from the bed.'
EPISODE_FRAMES = 83          # ceil(5.5 × 15). 변환 결과 길이 점검 기준
