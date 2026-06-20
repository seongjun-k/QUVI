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
    # Explicitly bypass reference image load by setting dummy empty reference images dir
    inspect_node.set_parameters([
        rclpy.parameter.Parameter('reference_image_dir', rclpy.Parameter.Type.STRING, '/tmp/non_existent_ref_dir')
    ])
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
    
    # First done message triggers capture of the first angle (0)
    msg_done = Bool()
    msg_done.data = True
    inspect_node._turntable_done_callback(msg_done)
    
    assert 0 in inspect_node._captured_images
    assert len(inspect_node._captured_images) == 1
    
    # Second done message triggers capture of the second angle (90)
    inspect_node._turntable_done_callback(msg_done)
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
