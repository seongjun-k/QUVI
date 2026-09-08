#!/usr/bin/env python3
"""사이드카 오프라인 수치 검증 (하드웨어/RobotClient 없음, venv python으로 실행).

1) 모델 로드 성공
2) 결정성: 동일 입력을 같은 torch 시드로 2회 호출 시 동일 출력
3) 수치 일치: 소켓(IPC) 경로 출력 vs VlaSidecar 직접 호출(인프로세스) 경로 출력
4) 왕복 지연: 소켓 경유 N회 호출의 평균/최대 latency를 30Hz 예산(33ms)과 비교 출력

GPU 메모리를 두 번 동시에 잡지 않도록 소켓 경로(서브프로세스)를 먼저 끝내고
종료한 뒤에 인프로세스 경로를 연다(순차 실행).
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from protocol import recv_msg, send_msg  # noqa: E402
import server as sidecar_server  # noqa: E402

SEED = 20260908
ATOL = 1e-4
TASK = "Pick up the printed part from the bed."
# 학습 데이터셋 시작 자세(prep_inference.sh와 동일) — 임의 랜덤보다 실제
# 궤적 분포에 가까운 state로 검증한다.
FIXED_STATE = np.array([0.06, -1.72, 1.40, 1.62, -0.17, 0.68], dtype=np.float32)


def make_fixed_images() -> dict:
    rng = np.random.default_rng(SEED)
    return {
        cam: rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
        for cam in sidecar_server.CAMERA_POLICY_KEYS
    }


def _connect_with_retry(sock_path: str, proc: subprocess.Popen, timeout: float) -> socket.socket:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(f"사이드카 서버가 조기 종료됨 (exit={proc.returncode}):\n{out}")
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(sock_path)
            return s
        except (FileNotFoundError, ConnectionRefusedError):
            time.sleep(0.5)
    raise TimeoutError(f"{timeout}s 안에 사이드카 서버에 연결하지 못함")


def run_via_socket(model_path: str, device: str, sock_path: str, images: dict, n_calls: int):
    proc = subprocess.Popen(
        [sys.executable, str(_HERE / "server.py"),
         "--model-path", model_path, "--socket", sock_path,
         "--task-default", TASK, "--device", device],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        client = _connect_with_retry(sock_path, proc, timeout=180)
        # seed를 명시해 SmolVLA의 확률적 디노이징 노이즈를 고정한다 - 그래야
        # 인프로세스 경로(아래에서 torch.manual_seed(SEED) 후 호출)와
        # 수치가 정확히 비교 가능해진다. 정상 운용 시에는 seed를 보내지 않는다.
        request = {"camera1": images["camera1"], "camera3": images["camera3"],
                   "state": FIXED_STATE, "task": TASK, "seed": SEED}
        chunk = None
        latencies = []
        for _ in range(n_calls):
            t0 = time.perf_counter()
            send_msg(client, request)
            response = recv_msg(client)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            if not response["success"]:
                raise RuntimeError(f"사이드카 추론 실패: {response['message']}")
            chunk = np.asarray(response["action_chunk"], dtype=np.float64)
        client.close()
        return chunk, latencies
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def run_in_process(model_path: str, device: str, images: dict):
    sidecar = sidecar_server.VlaSidecar(model_path=model_path, device=device, task_default=TASK)
    # flow-matching 계열 정책은 디노이징 노이즈를 내부에서 샘플링할 수 있으므로,
    # "배치 조립 + 파이프라인 재현"이 결정적인지를 보려면 호출 전 시드를 고정해야
    # 한다(모델 자체의 확률적 특성을 우회하기 위한 테스트 전용 조치).
    torch.manual_seed(SEED)
    chunk_a = sidecar.infer(images, FIXED_STATE, TASK)
    torch.manual_seed(SEED)
    chunk_b = sidecar.infer(images, FIXED_STATE, TASK)
    return chunk_a, chunk_b


def main() -> None:
    parser = argparse.ArgumentParser(description="VLA 사이드카 오프라인 수치 검증")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--socket", default="/dev/shm/quvi_vla_test.sock")
    parser.add_argument("--calls", type=int, default=5, help="소켓 왕복 지연 측정 횟수")
    args = parser.parse_args()

    images = make_fixed_images()

    print("[1/3] 소켓(IPC) 경로 실행 중 (서버 서브프로세스 기동)...")
    socket_chunk, latencies = run_via_socket(args.model_path, args.device, args.socket, images, args.calls)
    print(f"  action_chunk shape={socket_chunk.shape}, 왕복 latency(ms): "
          f"avg={np.mean(latencies):.1f} max={np.max(latencies):.1f} "
          f"(30Hz 예산 33ms 대비 {'OK' if np.mean(latencies) < 33 else '초과'})")

    print("[2/3] 인프로세스 경로 + 결정성 확인 중 (서버 종료 후 순차 실행)...")
    direct_chunk_a, direct_chunk_b = run_in_process(args.model_path, args.device, images)
    deterministic = np.allclose(direct_chunk_a, direct_chunk_b, atol=ATOL)
    print(f"  결정성(동일입력 2회 호출 일치, atol={ATOL}): {'PASS' if deterministic else 'FAIL'}")
    assert deterministic, "동일 시드/입력인데 두 호출 결과가 다름 - 파이프라인에 숨은 상태 의존이 있을 수 있음"

    print("[3/3] 소켓 경로 vs 인프로세스 경로 수치 일치 확인 중...")
    match = np.allclose(socket_chunk, direct_chunk_a, atol=ATOL)
    print(f"  IPC 경로 vs 직접호출 경로 일치(atol={ATOL}): {'PASS' if match else 'FAIL'}")
    if not match:
        diff = np.abs(socket_chunk - direct_chunk_a)
        print(f"  max abs diff={diff.max():.6f}, shape socket={socket_chunk.shape} direct={direct_chunk_a.shape}")
    assert match, "소켓 경로와 인프로세스 경로 출력이 불일치함"

    print("모든 검증 통과.")


if __name__ == "__main__":
    main()
