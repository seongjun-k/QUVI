"""텔레옵 시연을 15Hz 로 메모리에 모았다가 에피소드 폴더로 저장한다.

저장 형식은 scripts/teleop_episodes_to_lerobot.py 가 LeRobot v3.0 으로 변환한다.
"""
import os
import threading
from datetime import datetime

import numpy as np

RECORD_FPS = 15  # 학습 데이터셋 fps 와 같아야 한다


class TeleopRecorder:
    """start → (due 이면 add 반복) → stop. add 는 텔레옵 루프에서 호출되므로 디스크를 건드리지 않는다."""

    def __init__(self, out_dir: str, home_raw: list, on_error=None):
        self._out_dir = out_dir
        self._home_raw = home_raw
        self._on_error = on_error
        self._lock = threading.Lock()
        self._rows = []
        self._next_t = 0.0
        self.active = False

    def start(self, now: float):
        with self._lock:
            self._rows = []
            self._next_t = now
            self.active = True

    def due(self, now: float) -> bool:
        return self.active and now >= self._next_t

    def add(self, now: float, state_rad: list, action_rad: list, side_jpeg: bytes, top_jpeg: bytes):
        # 밀린 틱을 몰아 찍지 않고 다음 정시로 넘어간다. 프레임 간격 이상은 품질 점검이 잡는다.
        with self._lock:
            if not self.active:
                return
            self._next_t = max(self._next_t + 1.0 / RECORD_FPS, now)
            self._rows.append((now, state_rad, action_rad, side_jpeg, top_jpeg))

    def stop(self):
        """저장 스레드를 시작해 돌려준다. 프레임이 없으면 None."""
        with self._lock:
            self.active = False
            rows, self._rows = self._rows, []
        if not rows:
            return None
        path = os.path.join(self._out_dir, datetime.now().strftime('ep_%Y%m%d_%H%M%S_%f'))
        thread = threading.Thread(target=self._save, args=(path, rows), daemon=True)
        thread.start()
        return thread

    def _save(self, path: str, rows: list):
        try:
            self._write(path, rows)
        except Exception as e:
            if self._on_error:
                self._on_error(e)

    def _write(self, path: str, rows: list):
        # 쓰는 중 죽어도 변환기가 반쪽 에피소드를 읽지 않게 끝난 뒤 이름을 바꾼다.
        tmp = path + '.tmp'
        os.makedirs(tmp, exist_ok=True)
        for i, (_, _, _, side, top) in enumerate(rows):
            for name, jpeg in (('side', side), ('top', top)):
                with open(os.path.join(tmp, f'{i:05d}_{name}.jpg'), 'wb') as f:
                    f.write(jpeg)
        np.savez(os.path.join(tmp, 'data.npz'),
                 t=np.array([r[0] for r in rows]) - rows[0][0],
                 state=np.array([r[1] for r in rows], dtype=np.float32),
                 action=np.array([r[2] for r in rows], dtype=np.float32),
                 home_raw=np.array(self._home_raw))
        os.rename(tmp, path)
