import sys
from dynamixel_sdk import *

def scan_port(port_name, baudrate):
    print(f"\nScanning port {port_name} at {baudrate} bps...")
    portHandler = PortHandler(port_name)
    packetHandler = PacketHandler(2.0)
    
    if not portHandler.openPort():
        print(f"Failed to open port {port_name}")
        return
        
    if not portHandler.setBaudRate(baudrate):
        print(f"Failed to set baudrate to {baudrate}")
        portHandler.closePort()
        return

    found = []
    for dxl_id in range(1, 253):
        model_number, dxl_comm_result, dxl_error = packetHandler.ping(portHandler, dxl_id)
        if dxl_comm_result == COMM_SUCCESS:
            print(f"Found ID: {dxl_id:3d} (Model: {model_number})")
            found.append(dxl_id)
            
    if not found:
        print("No motors found.")
    else:
        print(f"Scan complete. Found IDs: {found}")
        
    portHandler.closePort()

if __name__ == "__main__":
    # Scan Follower port
    scan_port("/dev/ttyFollower", 1000000)
    # Scan Leader port
    scan_port("/dev/ttyLeader", 1000000)
