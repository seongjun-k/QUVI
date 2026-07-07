"""손목(wrist_roll) 모터 하드웨어 진단 스크립트.

follower(ID 15)/leader(ID 5) wrist_roll 모터의 동작모드·토크·하드웨어 에러·
전압·온도를 읽어 출력한다. dynamixel_sdk 로 시리얼 포트를 직접 연다.
실행 위치: quvi-dev 컨테이너 (/dev/ttyFollower, /dev/ttyLeader 접근).
사용법: python3 scripts/diagnose_wrist.py
"""
import sys
from dynamixel_sdk import *

def read_motor_diagnostics(port_name, baudrate, motor_id):
    print(f"\n--- Diagnostics for Motor ID {motor_id} on {port_name} ---")
    portHandler = PortHandler(port_name)
    packetHandler = PacketHandler(2.0)
    
    if not portHandler.openPort():
        print(f"Failed to open port {port_name}")
        return
        
    if not portHandler.setBaudRate(baudrate):
        print(f"Failed to set baudrate to {baudrate}")
        portHandler.closePort()
        return

    # Read operating mode
    op_mode, dxl_comm_result, dxl_error = packetHandler.read1ByteTxRx(portHandler, motor_id, 11)
    if dxl_comm_result == COMM_SUCCESS:
        print(f"Operating Mode: {op_mode}")
    else:
        print(f"Failed to read Operating Mode: {packetHandler.getTxRxResult(dxl_comm_result)}")

    # Read drive mode
    drive_mode, dxl_comm_result, dxl_error = packetHandler.read1ByteTxRx(portHandler, motor_id, 10)
    if dxl_comm_result == COMM_SUCCESS:
        print(f"Drive Mode: {drive_mode}")
    else:
        print(f"Failed to read Drive Mode: {packetHandler.getTxRxResult(dxl_comm_result)}")

    # Read torque enable
    torque_enable, dxl_comm_result, dxl_error = packetHandler.read1ByteTxRx(portHandler, motor_id, 64)
    if dxl_comm_result == COMM_SUCCESS:
        print(f"Torque Enable: {torque_enable}")
    else:
        print(f"Failed to read Torque Enable: {packetHandler.getTxRxResult(dxl_comm_result)}")

    # Read hardware error status
    hw_error, dxl_comm_result, dxl_error = packetHandler.read1ByteTxRx(portHandler, motor_id, 70)
    if dxl_comm_result == COMM_SUCCESS:
        print(f"Hardware Error Status: {hw_error} (Binary: {bin(hw_error)})")
    else:
        print(f"Failed to read Hardware Error: {packetHandler.getTxRxResult(dxl_comm_result)}")

    # Read present position
    pres_pos, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, motor_id, 132)
    if dxl_comm_result == COMM_SUCCESS:
        print(f"Present Position: {pres_pos}")
    else:
        print(f"Failed to read Present Position: {packetHandler.getTxRxResult(dxl_comm_result)}")

    # Read goal position
    goal_pos, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, motor_id, 116)
    if dxl_comm_result == COMM_SUCCESS:
        print(f"Goal Position: {goal_pos}")
    else:
        print(f"Failed to read Goal Position: {packetHandler.getTxRxResult(dxl_comm_result)}")

    # Read voltage
    voltage, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(portHandler, motor_id, 144)
    if dxl_comm_result == COMM_SUCCESS:
        print(f"Present Input Voltage: {voltage / 10.0} V")
    else:
        print(f"Failed to read Voltage: {packetHandler.getTxRxResult(dxl_comm_result)}")

    # Read temp
    temp, dxl_comm_result, dxl_error = packetHandler.read1ByteTxRx(portHandler, motor_id, 146)
    if dxl_comm_result == COMM_SUCCESS:
        print(f"Present Temperature: {temp} C")
    else:
        print(f"Failed to read Temperature: {packetHandler.getTxRxResult(dxl_comm_result)}")

    portHandler.closePort()

if __name__ == "__main__":
    # Check follower wrist_roll (ID 15)
    read_motor_diagnostics("/dev/ttyFollower", 1000000, 15)
    # Check leader wrist_roll (ID 5)
    read_motor_diagnostics("/dev/ttyLeader", 1000000, 5)
