#!/usr/bin/env python3
"""사이드카가 죽었을 때 SidecarClient.restart() 로 되살아나는지 확인하는 자체 점검.

실제 모델(900MB)은 안 쓴다 — ping 에만 답하는 가짜 서버를 세워 프로세스
라이프사이클(is_alive/restart/stop)만 검증한다.
실행: python3 scripts/vla_sidecar/test_restart.py
"""

import pathlib
import sys
import tempfile
import textwrap
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from client import SidecarClient, SidecarError   # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent

# ping 에만 답하는 최소 서버. 진짜 server.py 와 같은 CLI 인자를 받아야
# SidecarClient 가 그대로 띄울 수 있다.
FAKE_SERVER = textwrap.dedent(f'''
    import argparse, socket, os, sys
    sys.path.insert(0, {str(HERE)!r})
    from protocol import recv_msg, send_msg
    p = argparse.ArgumentParser()
    for a in ("--model-path", "--socket", "--task-default", "--device"):
        p.add_argument(a)
    args = p.parse_args()
    if os.path.exists(args.socket):
        os.remove(args.socket)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(args.socket); s.listen(1)
    while True:
        conn, _ = s.accept()
        try:
            while True:
                recv_msg(conn)
                send_msg(conn, {{"success": True, "ready": True,
                                 "model_path": args.model_path, "message": ""}})
        except Exception:
            pass
        finally:
            conn.close()
''')


def main() -> int:
    tmp = pathlib.Path(tempfile.mkdtemp())
    fake = tmp / 'fake_server.py'
    fake.write_text(FAKE_SERVER)

    c = SidecarClient(
        venv_python=sys.executable, server_script=str(fake),
        model_path='/stub', socket_path=str(tmp / 'sc.sock'),
        task_default='t', device='cpu', log_path=str(tmp / 'sc.log'))

    assert not c.is_alive(), '기동 전인데 alive 로 나온다'
    assert c.start(ready_timeout=30.0), '가짜 서버 기동 실패'
    assert c.is_alive()
    first_pid = c._proc.pid

    # 서버를 밖에서 죽인다 = OOM/크래시 재현
    c._proc.kill()
    c._proc.wait()
    for _ in range(50):
        if not c.is_alive():
            break
        time.sleep(0.05)
    assert not c.is_alive(), '죽였는데 is_alive 가 True — 워치독이 못 알아챈다'

    # 죽은 상태에서는 요청이 실패해야 한다(조용히 성공하면 안 된다)
    try:
        c._request({'cmd': 'ping'}, connect_timeout=1.0)
        raise AssertionError('죽은 사이드카에 요청이 성공해버렸다')
    except SidecarError:
        pass

    assert c.restart(ready_timeout=30.0), '재기동 실패'
    assert c.is_alive()
    assert c._proc.pid != first_pid, '같은 프로세스 — 실제로 새로 안 떴다'
    assert c._request({'cmd': 'ping'}, connect_timeout=5.0)['ready']

    c.stop()
    assert not c.is_alive(), 'stop 후에도 살아 있다'
    print('OK: 죽음 감지 → restart → 통신 재개 → stop 까지 정상')
    return 0


if __name__ == '__main__':
    sys.exit(main())
