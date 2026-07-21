"""HMI 사이드캠 MJPEG 스트림(/stream/sidecam)에서 프레임 1장을 캡처해 저장하는 확인용 스크립트.
실행 위치: quvi-dev 컨테이너 (HMI Flask 서버가 127.0.0.1:5000 으로 떠 있어야 함).
사용법: python3 scripts/grab_frame.py
"""
import urllib.request
import re

def main():
    print("Connecting to http://127.0.0.1:5000/stream/sidecam...")
    try:
        stream = urllib.request.urlopen('http://127.0.0.1:5000/stream/sidecam', timeout=5)
        bytes_data = b''
        for _ in range(100):
            chunk = stream.read(4096)
            if not chunk:
                break
            bytes_data += chunk
            # Find JPEG start and end
            a = bytes_data.find(b'\xff\xd8')
            b = bytes_data.find(b'\xff\xd9')
            if a != -1 and b != -1 and b > a:
                jpg = bytes_data[a:b+2]
                with open('/workspace/data/test_frame.jpg', 'wb') as f:
                    f.write(jpg)
                print(f"Frame grabbed and saved! Size: {len(jpg)} bytes")
                return
        print("Could not find a valid JPEG frame in the first 100 chunks of the stream.")
    except Exception as e:
        print("Error grabbing frame:", e)

if __name__ == '__main__':
    main()
