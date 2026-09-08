#!/usr/bin/env python3
"""VLA 사이드카 IPC 프레이밍/직렬화 (서버·클라이언트 공유용).

4바이트 빅엔디언 길이 프리픽스 + payload. msgpack이 설치돼 있으면 msgpack
(넘파이 배열은 dict로 감싸 raw bytes로 인코딩), 없으면 pickle로 폴백한다.
클라이언트(quvi-dev, numpy<2)와 서버(사이드카 venv, numpy>=2)가 서로 다른
numpy 버전을 쓰므로 pickle 폴백 시에도 numpy 배열은 표준 pickle 프로토콜로
문제없이 오간다 (버전 간 ABI가 아니라 값만 오가므로).
"""

from __future__ import annotations

import pickle
import socket
import struct
from typing import Any

import numpy as np

try:
    import msgpack
    _HAVE_MSGPACK = True
except ImportError:
    _HAVE_MSGPACK = False

_LEN_STRUCT = struct.Struct(">I")


def _default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        return {
            "__ndarray__": True,
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "data": arr.tobytes(),
        }
    raise TypeError(f"직렬화할 수 없는 타입: {type(obj)}")


def _object_hook(obj: dict) -> Any:
    if obj.get("__ndarray__"):
        return np.frombuffer(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])
    return obj


def pack(obj: Any) -> bytes:
    if _HAVE_MSGPACK:
        return msgpack.packb(obj, default=_default, use_bin_type=True)
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def unpack(data: bytes) -> Any:
    if _HAVE_MSGPACK:
        return msgpack.unpackb(data, object_hook=_object_hook, raw=False, strict_map_key=False)
    return pickle.loads(data)


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
