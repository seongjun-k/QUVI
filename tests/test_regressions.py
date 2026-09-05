"""회귀 테스트 통합 (구 test_code_review_improvements / test_safety_improvements /
test_reliability_improvements 병합).

- RESET 커맨드 경로: RobotControlNode 리셋 핸들러 + Orchestrator RESET → /robot/reset_command
- 안전: release/home 완료 토픽 분리, E-STOP 상태 전이
- 신뢰성: InspectNode 기준 이미지 부재 NaN 폴백·순차 캡처, Orchestrator 턴테이블 대기 전이
실행: cd <repo> && pytest tests/test_regressions.py (ROS 2 rclpy 필요; 미설치 호스트는 conftest 가 수집 제외)
"""
import json
import math
import time
import threading

import numpy as np
import pytest
import rclpy
from std_msgs.msg import Bool, Int32, String

from quvi_inspect.inspect_node import InspectNode
from quvi_robot_control.robot_control_node import RobotControlNode, RobotState
from quvi_robot_control.main_orchestrator_node import MainOrchestratorNode, FsmState

pytestmark = pytest.mark.usefixtures('_rclpy_session')

def test_code_review_improvements():
    if not rclpy.ok():
        rclpy.init()

    # 1. Test RobotControlNode Reset Topic Handlers
    robot_node = RobotControlNode(
        parameter_overrides=[
            rclpy.parameter.Parameter('use_real_hardware', rclpy.Parameter.Type.BOOL, False),
            rclpy.parameter.Parameter('use_act', rclpy.Parameter.Type.BOOL, False)
        ]
    )

    # Intentionally move to ERROR state to simulate a prior failure
    robot_node._set_state(RobotState.ERROR)
    assert robot_node._get_state() == RobotState.ERROR

    # Execute reset simulation directly
    res = robot_node._execute_reset()
    assert res == True
    assert robot_node._get_state() == RobotState.IDLE

    # 2. Test MainOrchestratorNode RESET Command Triggering Robot Reset Topic
    orch_node = MainOrchestratorNode(
        parameter_overrides=[
            rclpy.parameter.Parameter('use_act', rclpy.Parameter.Type.BOOL, False)
        ]
    )

    # Capture the reset command published by Orchestrator
    reset_command_received = False
    def reset_cb(msg):
        nonlocal reset_command_received
        if msg.data:
            reset_command_received = True

    sub_reset = robot_node.create_subscription(Bool, '/robot/reset_command', reset_cb, 10)

    # Simulate spinning both nodes in background executor
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(orch_node)
    executor.add_node(robot_node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        # Trigger HMI Command RESET via direct callback call
        msg_cmd = String()
        msg_cmd.data = "RESET"
        orch_node._hmi_command_cb(msg_cmd)

        # Give a small window for the message to traverse
        time.sleep(0.5)

        assert reset_command_received == True
        assert orch_node._state in (FsmState.INIT, FsmState.IDLE)
        assert orch_node._act_ready == True

    finally:
        executor.shutdown()
        robot_node.destroy_node()
        orch_node.destroy_node()
        rclpy.shutdown()


def test_safety_and_topic_split():
    # ROS 2 context initialization
    if not rclpy.ok():
        rclpy.init()

    # Create test control node
    # Set parameters to bypass real hardware and ACT model loading
    node = RobotControlNode()
    node.set_parameters([
        rclpy.parameter.Parameter('use_real_hardware', rclpy.Parameter.Type.BOOL, False),
        rclpy.parameter.Parameter('use_act', rclpy.Parameter.Type.BOOL, False)
    ])
    
    # We must start the executor to spin the node
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    
    try:
        # Check initial state
        assert node._get_state() == RobotState.IDLE
        
        # 1. Test release done topic split
        release_received = False
        def release_cb(msg):
            nonlocal release_received
            if msg.data:
                release_received = True
                
        sub_release = node.create_subscription(Bool, '/robot/release_done', release_cb, 10)
        
        # Trigger release in a separate thread
        t_rel = threading.Thread(target=node._execute_release, daemon=True)
        t_rel.start()
        
        # Wait for release to complete (takes 0.8s)
        time.sleep(1.5)
        assert release_received == True
        assert node._get_state() == RobotState.IDLE
        node.destroy_subscription(sub_release)
        
        # 2. Test home done topic split
        home_received = False
        def home_cb(msg):
            nonlocal home_received
            if msg.data:
                home_received = True
                
        sub_home = node.create_subscription(Bool, '/robot/home_done', home_cb, 10)
        
        # Trigger home
        t_home = threading.Thread(target=node._execute_home, daemon=True)
        t_home.start()
        
        time.sleep(2.5) # takes 2.0s
        assert home_received == True
        assert node._get_state() == RobotState.IDLE
        node.destroy_subscription(sub_home)
        
        # 3. Test ESTOP active transition
        msg_estop = Bool()
        msg_estop.data = True
        node._estop_cmd_callback(msg_estop)
        
        # Check if state is ERROR and dxl_ready becomes False
        assert node._get_state() == RobotState.ERROR
        assert node._dxl_ready == False

    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


def test_reliability_improvements():
    if not rclpy.ok():
        rclpy.init()

    # 1. Test inspect_node NaN area_ratio fallback and load validation
    inspect_node = InspectNode()
    # 기준 이미지 미존재 상황 재현: _load_reference_images 는 self._ref_dir 속성을
    # 읽고 기존 dict 를 clear 하지 않으므로, 속성 직접 변경 + 초기화가 필요하다.
    inspect_node._ref_dir = '/tmp/non_existent_ref_dir'
    inspect_node._reference_images.clear()
    inspect_node._load_reference_images()
    
    # Assert reference images are empty
    assert len(inspect_node._reference_images) == 0
    
    # Run mock surface analysis with a dummy captured image
    dummy_captured = np.zeros((240, 320, 3), dtype=np.uint8)
    inspect_node._captured_images = {0: dummy_captured}
    
    # Execute surface analysis and verify that area_ratio is NaN
    res = inspect_node._surface_analysis()
    assert math.isnan(res['area_ratio'])
    assert res['passed'] == False # Because dummy image is empty, other checks (like solidity) will fail
    
    # 2. Test inspect_node sequential capture without pending_angle dependency
    inspect_node._captured_images.clear()
    inspect_node._inspection_active = True
    inspect_node._latest_frame = dummy_captured
    
    # 검사 모드 캡처는 capture_now 콜백 → settle 타이머 발화 경로 (T4 이관분).
    # 타이머 발화는 _on_settle_elapsed 직접 호출로 시뮬레이션한다.
    msg_done = Bool()
    msg_done.data = True
    inspect_node._capture_now_callback(msg_done)
    inspect_node._on_settle_elapsed()

    assert 0 in inspect_node._captured_images
    assert len(inspect_node._captured_images) == 1

    # 두 번째 capture_now 는 다음 미캡처 각도(90)를 채운다
    inspect_node._capture_now_callback(msg_done)
    inspect_node._on_settle_elapsed()
    assert 90 in inspect_node._captured_images
    assert len(inspect_node._captured_images) == 2

    # 3. Test main_orchestrator FSM INSPECTING_WAIT_TURNTABLE state transition
    orch_node = MainOrchestratorNode()
    
    # Trigger active FSM loop inside test
    orch_node._state = FsmState.INSPECTING_ROTATE
    orch_node._inspect_angle_idx = 0
    
    # Run spin step
    orch_node._fsm_loop()
    
    # Verify State transition to INSPECTING_WAIT_TURNTABLE and reset flag
    assert orch_node._state == FsmState.INSPECTING_WAIT_TURNTABLE
    assert orch_node._turntable_done == False
    
    # Trigger done callback and run FSM loop step again
    msg_done_orch = Bool()
    msg_done_orch.data = True
    orch_node._turntable_done_cb(msg_done_orch)
    assert orch_node._turntable_done == True
    
    orch_node._fsm_loop()
    assert orch_node._state == FsmState.INSPECTING_CAPTURE

    inspect_node.destroy_node()
    orch_node.destroy_node()
    rclpy.shutdown()


def test_printer_monitor_done_edge(monkeypatch):
    """프린터 완료 신호는 '출력 중'을 본 뒤 complete 로 바뀔 때 정확히 1회만 나가야 한다.

    재연결·재기동으로 complete 상태를 다시 관측해도 중복 발행되면 무인 루프가
    출력물 없이 재시작된다 — 실물 사고로 이어지는 경로라 회귀로 고정한다.
    """
    from quvi_robot_control.printer_monitor_node import PrinterMonitorNode

    if not rclpy.ok():
        rclpy.init()

    class _Resp:
        def __init__(self, state):
            self._state = state

        def raise_for_status(self):
            pass

        def json(self):
            return {'result': {'status': {
                'print_stats': {'state': self._state, 'filename': 'a.gcode'},
                'virtual_sdcard': {'progress': 0.5},
            }}}

    seq = []

    def _fake_get(*_a, **_kw):
        if not seq:
            raise AssertionError('폴링 횟수가 시나리오보다 많다')
        return _Resp(seq.pop(0))

    monkeypatch.setattr(
        'quvi_robot_control.printer_monitor_node.requests.get', _fake_get)

    node = PrinterMonitorNode()
    try:
        fired = []
        monkeypatch.setattr(node._done_pub, 'publish', lambda m: fired.append(m.data))

        # 기동 직후 이미 complete → 출력 중을 본 적 없으므로 발행 금지
        seq.extend(['complete', 'complete'])
        node._poll()
        node._poll()
        assert fired == [], '기동 직후 잔여 complete 로 오발행'

        # 정상 시나리오: printing → complete 에서 1회
        seq.extend(['printing', 'printing', 'complete'])
        node._poll()
        node._poll()
        node._poll()
        assert fired == [True], f'완료 신호 1회여야 하는데 {fired}'

        # complete 유지 동안 재발행 금지
        seq.extend(['complete', 'complete'])
        node._poll()
        node._poll()
        assert fired == [True], '완료 상태 유지 중 중복 발행'

        # 취소된 출력은 완료로 치지 않는다
        seq.extend(['printing', 'cancelled', 'standby'])
        node._poll()
        node._poll()
        node._poll()
        assert fired == [True], '취소된 출력을 완료로 오판'
    finally:
        node.destroy_node()


def test_auto_start_on_print_done():
    """출력 완료 신호 자동 시작 — 진입 조건과 취소 경로를 고정한다.

    사람 없이 레일이 프린터 베드로 출발하는 동작이라, 조건이 하나라도
    느슨해지면 예기치 않은 시점에 로봇이 움직인다.
    """
    if not rclpy.ok():
        rclpy.init()

    def _node(auto_start):
        return MainOrchestratorNode(parameter_overrides=[
            rclpy.parameter.Parameter('use_act', rclpy.Parameter.Type.BOOL, False),
            rclpy.parameter.Parameter(
                'auto_start_on_print_done', rclpy.Parameter.Type.BOOL, auto_start),
            rclpy.parameter.Parameter(
                'print_done_delay_sec', rclpy.Parameter.Type.DOUBLE, 0.0),
        ])

    def _bed(node, temp):
        node._printer_status_cb(String(data=json.dumps({'bed_temp': temp})))

    node = _node(True)
    try:
        node._motor_homed = True          # 호밍 가드 통과
        node._state = FsmState.IDLE

        # 파라미터가 켜져 있고 IDLE 이면 예약된다
        node._printer_done_cb(Bool(data=True))
        assert node._auto_start_at is not None, '자동 시작이 예약되지 않았다'

        # 베드 온도를 모르면 시작하지 않는다 (fail-closed)
        node._fsm_loop()
        assert node._state == FsmState.IDLE, '베드 온도를 모르는데 시작했다'

        # 아직 뜨거우면 대기한다
        _bed(node, 60.0)
        node._fsm_loop()
        assert node._state == FsmState.IDLE, '베드가 60℃인데 시작했다'

        # 온도 정보가 오래됐으면 모르는 것으로 취급한다
        _bed(node, 30.0)
        node._bed_temp_at = time.time() - 999
        node._fsm_loop()
        assert node._state == FsmState.IDLE, '낡은 온도 정보로 시작했다'

        # 기준 이하로 식으면 시작한다
        _bed(node, 30.0)
        node._fsm_loop()
        assert node._state == FsmState.STARTUP_RAIL_HOME_TRIGGER, \
            f'자동 시작이 START 경로를 타지 않았다: {node._state}'

        # IDLE 이 아닐 때 들어온 신호는 무시된다
        node._state = FsmState.GRASPING_WAIT
        node._auto_start_at = None
        node._printer_done_cb(Bool(data=True))
        assert node._auto_start_at is None, 'IDLE 이 아닌데 자동 시작을 예약했다'

        # 호밍 미완이면 자동 시작도 ERROR 로 차단된다 (HMI START 와 동일 가드)
        node._state = FsmState.IDLE
        node._motor_homed = False
        node._printer_done_cb(Bool(data=True))
        node._fsm_loop()
        assert node._state == FsmState.ERROR, '호밍 미완인데 시퀀스가 시작됐다'
    finally:
        node.destroy_node()

    # 파라미터가 꺼져 있으면 신호를 받아도 아무 일도 없어야 한다
    node = _node(False)
    try:
        node._motor_homed = True
        node._state = FsmState.IDLE
        node._printer_done_cb(Bool(data=True))
        node._fsm_loop()
        assert node._auto_start_at is None and node._state == FsmState.IDLE, \
            '자동 시작이 꺼져 있는데 움직였다'

        # STOP 은 대기 중인 예약을 취소해야 한다 (정지 직후 자가 재시작 방지)
        node._auto_start_at = 1.0
        node._hmi_command_cb(String(data='STOP'))
        assert node._auto_start_at is None, 'STOP 후에도 자동 시작 예약이 남아 있다'
    finally:
        node.destroy_node()
