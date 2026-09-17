#!/usr/bin/env python3
"""사이드카 서버가 깨진 프레임 한 방에 죽지 않는지 확인하는 자체 점검.

실행: <venv>/bin/python scripts/vla_sidecar/test_server_resilience.py
모델은 로드하지 않는다(_serve 는 sidecar 객체를 그대로 넘기기만 한다).
"""

import socket
import struct
import sys
import tempfile
import threading
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import server                                    # noqa: E402
from protocol import recv_msg, send_msg          # noqa: E402


class _StubSidecar:
    """ping 응답에만 쓰이는 최소 스텁."""
    _policy = object()
    model_path = '/stub'


def main() -> int:
    sock_path = str(pathlib.Path(tempfile.mkdtemp()) / 'test.sock')
    t = threading.Thread(target=server._serve, args=(_StubSidecar(), sock_path),
                         daemon=True)
    t.start()

    def connect():
        for _ in range(100):
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(sock_path)
                return s
            except OSError:
                threading.Event().wait(0.05)
        raise AssertionError('서버 소켓 연결 실패')

    # 1) 길이 헤더는 멀쩡하지만 payload 가 pickle 이 아닌 쓰레기 → unpack 예외
    s = connect()
    junk = b'\xff\xfe not a pickle \x00'
    s.sendall(struct.pack('>I', len(junk)) + junk)
    s.close()

    # 2) 그 뒤에도 서버가 살아서 정상 요청에 답해야 한다
    s = connect()
    send_msg(s, {'cmd': 'ping'})
    resp = recv_msg(s)
    s.close()
    assert resp['success'] and resp['ready'], resp
    assert t.is_alive(), '깨진 프레임 이후 서버 스레드가 죽었다'

    print('OK: 깨진 프레임 이후에도 서버가 계속 서빙한다')
    return 0


if __name__ == '__main__':
    sys.exit(main())
