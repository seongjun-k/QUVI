#!/usr/bin/env python3
"""VLA 사이드카 클라이언트 (quvi-dev robot_control_node 에서 사용).

quvi-dev(lerobot 0.3.4/numpy<2)에서는 0.6.1 SmolVLA 모델을 직접 못 올리므로,
격리 venv 에서 도는 사이드카 서버(server.py)에 Unix 소켓으로 관측을 보내고
액션 청크를 받는다. 이 클라이언트는 (1) 사이드카 서버 프로세스의 기동·종료,
(2) 소켓 연결·재연결, (3) load/ping/infer 요청을 담당한다.

정규화·추론 자체는 서버(사이드카)가 cyclo 검증 코드로 수행한다. 이 파일은
프로세스·통신만 다루며 로봇 동작 로직(raw 변환·안전 클립)은 포함하지 않는다.
"""

from __future__ import annotations

import os
import pathlib
import socket
import subprocess
import sys
import threading
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from protocol import recv_msg, send_msg  # noqa: E402


class SidecarError(RuntimeError):
    """사이드카 통신/추론 실패."""


class SidecarClient:
    """사이드카 서버 프로세스 관리 + 소켓 IPC.

    스레드 안전: infer/load 는 단일 요청-응답이 원자적이어야 하므로 내부 락으로
    직렬화한다(제어 루프와 HMI 모델 전환이 다른 스레드에서 올 수 있다).
    """

    def __init__(self, venv_python: str, server_script: str, model_path: str,
                 socket_path: str, task_default: str, device: str = "cuda",
                 log_path: str = "/tmp/quvi_vla_sidecar.log", logger=None):
        self._venv_python = venv_python
        self._server_script = server_script
        self._model_path = model_path
        self._socket_path = socket_path
        self._task_default = task_default
        self._device = device
        self._log_path = log_path
        self._log = logger
        self._proc = None
        self._sock = None
        self._lock = threading.Lock()

    # ── 로깅 헬퍼 ──
    def _info(self, msg):
        (self._log.info if self._log else print)(msg)

    def _warn(self, msg):
        (self._log.warn if self._log else print)(msg)

    # ── 프로세스 라이프사이클 ──
    def start(self, ready_timeout: float = 120.0) -> bool:
        """사이드카 서버를 기동(미기동 시)하고 모델 로드까지 대기한다.

        서버 stdout/stderr 는 PIPE 가 아니라 로그파일로 보낸다 — 모델 로드 시
        tqdm/HF 로그가 PIPE 버퍼(64KB)를 채우면 서버가 write 에서 블록돼
        데드락에 빠지기 때문(실측된 함정).
        """
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return True
            for path in (self._venv_python, self._server_script):
                if not os.path.exists(path):
                    self._warn(f"사이드카 실행 불가 — 경로 없음: {path}")
                    return False
            self._info(f"사이드카 서버 기동: model={self._model_path} log={self._log_path}")
            log_f = open(self._log_path, "ab", buffering=0)
            self._proc = subprocess.Popen(
                [self._venv_python, "-u", self._server_script,
                 "--model-path", self._model_path, "--socket", self._socket_path,
                 "--task-default", self._task_default, "--device", self._device],
                stdout=log_f, stderr=subprocess.STDOUT,
            )
            # ready 대기: ping 이 성공하고 모델이 로드될 때까지.
            deadline = time.monotonic() + ready_timeout
            while time.monotonic() < deadline:
                if self._proc.poll() is not None:
                    self._warn(f"사이드카 서버 조기 종료(exit={self._proc.returncode}). "
                               f"로그 확인: {self._log_path}")
                    self._proc = None
                    return False
                try:
                    resp = self._request_unlocked({"cmd": "ping"}, connect_timeout=2.0)
                    if resp.get("success") and resp.get("ready"):
                        self._info(f"사이드카 ready: {resp.get('model_path')}")
                        return True
                except (SidecarError, OSError):
                    pass
                time.sleep(1.0)
            self._warn(f"사이드카 ready 타임아웃({ready_timeout}s)")
            return False

    def stop(self) -> None:
        with self._lock:
            self._close_sock()
            if self._proc is not None and self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait()
            self._proc = None

    # ── 소켓 ──
    def _close_sock(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _ensure_sock(self, connect_timeout: float):
        if self._sock is not None:
            return
        deadline = time.monotonic() + connect_timeout
        last = None
        while time.monotonic() < deadline:
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(self._socket_path)
                self._sock = s
                return
            except OSError as e:
                last = e
                time.sleep(0.2)
        raise SidecarError(f"사이드카 소켓 연결 실패: {self._socket_path} ({last})")

    def _request_unlocked(self, req: dict, connect_timeout: float = 5.0) -> dict:
        """락을 이미 쥔 상태에서 1회 요청-응답. 소켓 오류 시 1회 재연결 시도."""
        for attempt in (1, 2):
            try:
                self._ensure_sock(connect_timeout if attempt == 1 else 2.0)
                send_msg(self._sock, req)
                return recv_msg(self._sock)
            except (OSError, ConnectionError, EOFError) as e:
                self._close_sock()
                if attempt == 2:
                    raise SidecarError(f"사이드카 통신 실패: {e}") from e
        raise SidecarError("사이드카 통신 실패(도달 불가)")

    def _request(self, req: dict, connect_timeout: float = 5.0) -> dict:
        with self._lock:
            return self._request_unlocked(req, connect_timeout)

    # ── 상위 API ──
    def load(self, model_path: str) -> bool:
        """서버에 모델 (재)로드를 요청. 성공 시 True."""
        self._model_path = model_path
        resp = self._request({"cmd": "load", "model_path": model_path}, connect_timeout=10.0)
        if not resp.get("success"):
            self._warn(f"사이드카 모델 로드 실패: {resp.get('message')}")
            return False
        return True

    def infer(self, cam_images: dict, state, task: str) -> np.ndarray:
        """관측 → 액션 청크 (T, 6) float64. 실패 시 SidecarError."""
        req = {"cmd": "infer",
               "camera1": cam_images.get("camera1"),
               "camera3": cam_images.get("camera3"),
               "state": list(state),
               "task": task}
        resp = self._request(req)
        if not resp.get("success"):
            raise SidecarError(f"사이드카 추론 실패: {resp.get('message')}")
        return np.asarray(resp["action_chunk"], dtype=np.float64)
