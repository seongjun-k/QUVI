import time
import threading
import pytest
import rclpy
from std_msgs.msg import Bool
from quvi_robot_control.robot_control_node import RobotControlNode, RobotState

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
