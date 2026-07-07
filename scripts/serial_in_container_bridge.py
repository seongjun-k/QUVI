"""ESP32 물리 시리얼 포트를 컨테이너 내부 PTY(/workspace/ttyV0)로 중계하는 브리지.
호스트에서 컨테이너로 시리얼 장치를 직접 마운트하기 어려운 환경에서, micro-ROS
agent 가 컨테이너 내부 PTY 를 통해 ESP32 와 통신하도록 우회 경로를 만든다.
실행 위치: quvi-dev 컨테이너 (/dev/ttyESP32 접근 필요).
사용법: python3 scripts/serial_in_container_bridge.py
"""
import os
import sys
import time
import serial
import select
import tty
import termios

def main():
    print("Starting Pure Python Serial-to-PTY Bridge INSIDE Container...")
    
    # 1. Create PTY Master/Slave pair
    try:
        master_fd, slave_fd = os.openpty()
        slave_name = os.ttyname(slave_fd)
        print(f"PTY created inside container. Master FD: {master_fd}, Slave Device: {slave_name}")
        
        # Enable Raw Mode on PTY descriptors
        tty.setraw(master_fd)
        tty.setraw(slave_fd)
        print("PTY Raw mode enabled.")
    except Exception as e:
        print("Failed to create PTY or set raw mode:", e)
        sys.exit(1)
        
    # Symlink target path inside the container namespace
    symlink_path = "/workspace/ttyV0"
    
    try:
        if os.path.lexists(symlink_path) or os.path.exists(symlink_path):
            os.remove(symlink_path)
        os.symlink(slave_name, symlink_path)
        print(f"Created symlink inside container: {symlink_path} -> {slave_name}")
    except Exception as e:
        print("Failed to create symlink inside container:", e)
        os.close(master_fd)
        os.close(slave_fd)
        sys.exit(1)

    # 2. Open Physical Serial Port with correct DTR/RTS logic
    try:
        ser_phy = serial.Serial('/dev/ttyESP32', 115200, timeout=0.01)
        ser_phy.dtr = False
        ser_phy.rts = False
        time.sleep(0.5)  # Discharge caps
        
        # Reset pulse
        print("Pulsing hardware Reset (EN=LOW)...")
        ser_phy.rts = True
        ser_phy.dtr = False
        time.sleep(0.2)
        
        # Release Reset
        print("Releasing Reset (EN=HIGH, IO0=HIGH)...")
        ser_phy.rts = False
        ser_phy.dtr = False
        time.sleep(0.5)  # Let boot strap state settle
        print("Physical serial port /dev/ttyESP32 opened successfully inside container.")
    except Exception as e:
        print("Failed to open physical serial port inside container:", e)
        os.close(master_fd)
        os.close(slave_fd)
        if os.path.exists(symlink_path):
            os.remove(symlink_path)
        sys.exit(1)

    # 3. Proxy loop
    print("Proxy loop active. Monitoring bidirectional raw traffic...")
    last_raw_enforce = 0.0
    try:
        while True:
            # Enforce raw binary mode periodically
            current_time = time.time()
            if current_time - last_raw_enforce > 1.0:
                try:
                    tty.setraw(master_fd)
                    tty.setraw(slave_fd)
                except:
                    pass
                last_raw_enforce = current_time

            r, w, x = select.select([master_fd, ser_phy], [], [], 0.005)
            
            # Virtual Master PTY (Agent) -> Physical Serial (ESP32)
            if master_fd in r:
                data = os.read(master_fd, 4096)
                if data:
                    print(f"[VIR->PHY] ({len(data)} bytes) {data.hex()}")
                    ser_phy.write(data)
                    ser_phy.flush()
            
            # Physical Serial (ESP32) -> Virtual Master PTY (Agent)
            if ser_phy in r:
                data = ser_phy.read(ser_phy.in_waiting or 1)
                if data:
                    print(f"[PHY->VIR] ({len(data)} bytes) {data.hex()}")
                    os.write(master_fd, data)
                    
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("Exiting...")
    except Exception as e:
        print("Runtime exception in container proxy loop:", e)
    finally:
        ser_phy.close()
        os.close(master_fd)
        os.close(slave_fd)
        try:
            if os.path.lexists(symlink_path):
                os.remove(symlink_path)
        except:
            pass
        print("Container Bridge closed.")

if __name__ == '__main__':
    main()
