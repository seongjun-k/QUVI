import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2
import sys

class CamChecker(Node):
    def __init__(self):
        super().__init__('cam_checker')
        self.sub1 = self.create_subscription(
            CompressedImage, '/camera1/image_raw/compressed', self.cb1, 10)
        self.sub2 = self.create_subscription(
            CompressedImage, '/camera2/image_raw/compressed', self.cb2, 10)
        self.received1 = False
        self.received2 = False

    def cb1(self, msg):
        if self.received1:
            return
        print(f"[{self.get_clock().now().to_msg().sec}] Camera 1 message received! Format: {msg.format}, Data size: {len(msg.data)}")
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                print("Error: cv2.imdecode returned None for Camera 1!")
            else:
                print(f"Camera 1 decoded successfully: Shape: {frame.shape}")
                self.received1 = True
        except Exception as e:
            print(f"Exception decoding Camera 1: {e}")

    def cb2(self, msg):
        if self.received2:
            return
        print(f"[{self.get_clock().now().to_msg().sec}] Camera 2 message received! Format: {msg.format}, Data size: {len(msg.data)}")
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                print("Error: cv2.imdecode returned None for Camera 2!")
            else:
                print(f"Camera 2 decoded successfully: Shape: {frame.shape}")
                self.received2 = True
        except Exception as e:
            print(f"Exception decoding Camera 2: {e}")

def main():
    rclpy.init()
    node = CamChecker()
    print("Checking camera topics... Waiting for messages (timeout 5s)...")
    
    # Spin with a timeout
    start_time = node.get_clock().now()
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        elapsed = (node.get_clock().now() - start_time).nanoseconds / 1e9
        if node.received1 and node.received2:
            print("Both cameras successfully verified!")
            break
        if elapsed > 5.0:
            print(f"Timeout reached. Cam1 received: {node.received1}, Cam2 received: {node.received2}")
            break
            
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
