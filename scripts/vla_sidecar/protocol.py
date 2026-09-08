#!/usr/bin/env python3
"""VLA 사이드카 IPC 프레이밍/직렬화 (서버·클라이언트 공유용).

4바이트 빅엔디언 길이 프리픽스 + payload(pickle). 핵심 제약: 클라이언트는
quvi-dev(numpy 1.26), 서버는 venv(numpy 2.2)라 **numpy 배열을 pickle에 직접
태우면 안 된다** — numpy 2.0 에서 내부 모듈이 numpy.core→numpy._core 로 바뀌어,
2.x가 pickle 한 ndarray 를 1.26 이 풀 때 `No module named numpy._core.numeric` 로
깨진다(2026-09-08 실측). 그래서 ndarray 는 pickle 전에 dtype 문자열+shape+raw
bytes 의 순수 dict 로 변환해 numpy 모듈 참조가 실리지 않게 한다(양방향 버전 무관).
양단은 /dev/shm 로컬 신뢰 프로세스라 pickle 역직렬화 위험은 없다.
"""

from __future__ import annotations

import pickle
import socket
import struct
from typing import Any

import numpy as np

_LEN_STRUCT = struct.Struct(">I")


def _encode(obj: Any) -> Any:
    """ndarray → {dtype,shape,bytes} 순수 dict 로 재귀 변환(numpy 버전 무관 직렬화)."""
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        return {"__nd__": True, "dtype": str(arr.dtype),
                "shape": tuple(int(s) for s in arr.shape), "data": arr.tobytes()}
    if isinstance(obj, dict):
        return {k: _encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(v) for v in obj]
    return obj


def _decode(obj: Any) -> Any:
    if isinstance(obj, dict):
        if obj.get("__nd__"):
            return np.frombuffer(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    return obj


def pack(obj: Any) -> bytes:
    return pickle.dumps(_encode(obj), protocol=pickle.HIGHEST_PROTOCOL)


def unpack(data: bytes) -> Any:
    return _decode(pickle.loads(data))


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("소켓 연결이 끊어졌습니다 (recv_exact)")
        buf.extend(chunk)
    return bytes(buf)


def send_msg(sock: socket.socket, obj: Any) -> None:
    payload = pack(obj)
    sock.sendall(_LEN_STRUCT.pack(len(payload)) + payload)


def recv_msg(sock: socket.socket) -> Any:
    header = _recv_exact(sock, _LEN_STRUCT.size)
    (length,) = _LEN_STRUCT.unpack(header)
    payload = _recv_exact(sock, length)
    return unpack(payload)
