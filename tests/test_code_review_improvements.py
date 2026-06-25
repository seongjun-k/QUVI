import time
import threading
import pytest
import rclpy
from std_msgs.msg import Bool, String
from quvi_robot_control.robot_control_node import RobotControlNode, RobotState
from quvi_robot_control.main_orchestrator_node import MainOrchestratorNode, FsmState

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
