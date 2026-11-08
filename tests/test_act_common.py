"""act_common 상수 스냅샷 — 로봇 노드에서 분리할 때 값이 바뀌지 않았는지, 이후에도 조용히 바뀌지 않는지 고정한다."""
from quvi_robot_control import act_common as ac
from quvi_robot_control import robot_control_node as rcn


def test_constants_snapshot():
    assert ac.JOINT_NAMES == ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
                              'wrist_flex', 'wrist_roll', 'gripper']
    assert (ac.GRIPPER_OPEN, ac.GRIPPER_CLOSE) == (2500, 1800)
    assert [ac.POSE_P1[n] for n in ac.JOINT_NAMES] == [2037, 1114, 2771, 3257, 2016, 2077]
    assert ac.POSE_HOME == ac.RECORD_START_POSE
    assert (ac.DXL_TICKS_PER_REV, ac.DXL_TICK_CENTER) == (4096.0, 2048.0)
    assert ac.TASK_INSTRUCTION == 'Pick up the printed part from the bed.'
    assert (ac.DATASET_FPS, ac.RECORD_STATE_HZ, ac.EPISODE_DURATION_SEC, ac.EPISODE_FRAMES) == (15, 30, 5.5, 83)


def test_robot_node_values_unchanged():
    # 로봇 재생 주기는 데이터셋 fps와 별개로 30 유지(2026-09-17 실측 근거)
    assert rcn.ACT_CONTROL_HZ == 30
    assert rcn.DXL_RAW_MAX == 4095
    assert rcn.POSE_HOME is ac.POSE_HOME


def test_record_start_pose():
    # 수집 시작·끝 기준 = VLA 시작 자세 (현재 학습 데이터 112개 첫 프레임 평균)
    assert [ac.RECORD_START_POSE[n] for n in ac.JOINT_NAMES] == [2043, 1009, 2909, 3043, 2039, 2460]
    assert rcn.ACT_START_POSE == {n: ac.RECORD_START_POSE[n] for n in ac.JOINT_NAMES[:5]}
