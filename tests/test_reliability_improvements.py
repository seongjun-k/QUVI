"""InspectNode(기준 이미지 부재 시 NaN 폴백, capture_now 순차 캡처)와
MainOrchestratorNode(INSPECTING_WAIT_TURNTABLE 상태 전이) 회귀 테스트.
실행: cd <repo> && pytest tests/test_reliability_improvements.py (ROS 2 rclpy 필요)
"""
import time
import math
import threading
import pytest
import rclpy
import numpy as np
from std_msgs.msg import Bool, Int32
from quvi_inspect.inspect_node import InspectNode
from quvi_robot_control.main_orchestrator_node import MainOrchestratorNode, FsmState

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
