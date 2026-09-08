#!/usr/bin/env python3
"""QUVI VLA(SmolVLA) 추론 사이드카 서버.

quvi-dev 컨테이너의 lerobot(0.3.4)은 lerobot 0.6.1로 학습·export된 SmolVLA
2캠 체크포인트를 읽지 못하고, numpy<2 / transformers<4.52 핀 때문에 메인
파이썬에 0.6.1을 올릴 수도 없다. 그래서 격리 venv에서 도는 별도 프로세스로
모델을 1회 로드해 상주시키고, Unix domain socket으로 관측→액션 청크를
주고받는다.

전처리·정규화 로직은 cyclo_intelligence에서 실기 검증(2026-09-07, jitter 없음
·박스 파지 성공)을 통과한 lerobot_engine 모듈(loading/prediction/
image_preprocessing/constants)을 QUVI에 vendoring 한 사본(같은 디렉토리
lerobot_engine/)에서 그대로 import해 재사용한다 — 손으로 재구현하면 학습 시
정규화 통계와 de-sync될 위험이 있다. RobotClient에 묶인 io_mapping.py/engine.py
는 vendoring 대상에서 제외했다.
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import socket
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from protocol import recv_msg, send_msg  # noqa: E402

logger = logging.getLogger("vla_sidecar")

DEFAULT_SOCKET = "/dev/shm/quvi_vla.sock"
DEFAULT_TASK = "Pick up the printed part from the bed."

# IPC 요청의 짧은 카메라 이름 -> 정책 입력 키. 체크포인트 config.json의
# input_features 키(observation.images.rgb.camera1/camera3)와 일치해야 한다.
CAMERA_POLICY_KEYS = {
    "camera1": "observation.images.rgb.camera1",
    "camera3": "observation.images.rgb.camera3",
}


# vendoring 한 lerobot_engine/ 는 이 파일과 같은 디렉토리에 있고 __init__.py 가
# import-clean 이라, 위 sys.path(=이 파일의 부모) 삽입만으로 바로 임포트된다.
from lerobot_engine import loading, prediction, constants  # noqa: E402
from lerobot_engine.image_preprocessing import (  # noqa: E402
    prepare_policy_image,
    infer_image_resize_targets,
)


class VlaSidecar(prediction.PredictionMixin):
    """cyclo lerobot_engine의 로딩/추론 믹스인을 재사용하는 상주 추론기."""

    def __init__(self, model_path: str, device: str, task_default: str = DEFAULT_TASK):
        self._device = torch.device(device)
        self._task_default = task_default
        self.model_path = None
        self._policy = self._preprocessor = self._postprocessor = None
        self._image_resize: dict = {}
        self.load(model_path)

    def load(self, model_path: str) -> None:
        """정책 + 전처리기를 (재)로드. HMI 모델 전환 시 클라이언트가 load 명령으로 호출.

        실패 시 예외를 올려 호출자가 처리한다(기존 정책은 교체 직전까지 유지).
        """
        resolved = loading.LoadingMixin._resolve_model_dir(model_path)
        logger.info("정책 로드 중: %s (device=%s)", resolved, self._device)
        policy, preprocessor, postprocessor = (
            loading.LoadingMixin._load_policy_assets(resolved, self._device)
        )
        self._policy, self._preprocessor, self._postprocessor = (
            policy, preprocessor, postprocessor)
        features = getattr(self._policy.config, "input_features", {}) or {}
        self._image_resize = infer_image_resize_targets(features)
        self.model_path = resolved
        logger.info("로드 완료: %s | 리사이즈 타겟 %s", resolved, self._image_resize)

    def _build_batch(self, cam_images: dict, state, task: str) -> dict:
        """preprocessing.py:44-118의 RobotClient-free 부분과 동일하게 재현."""
        batch: dict = {}
        for cam_name, policy_key in CAMERA_POLICY_KEYS.items():
            img = cam_images.get(cam_name)
            if img is None:
                raise ValueError(f"카메라 프레임 누락: {cam_name}")
            img = prepare_policy_image(
                img,
                rotation_deg=0,
                target_size=self._image_resize.get(policy_key, (640, 480)),
            )
            tensor = torch.from_numpy(img.copy()).to(torch.float32) / 255.0
            tensor = tensor.permute(2, 0, 1).contiguous().unsqueeze(0)
            batch[policy_key] = tensor.to(self._device)

        state_arr = np.asarray(state, dtype=np.float32)
        batch[constants.STATE_KEY] = torch.from_numpy(state_arr).unsqueeze(0).to(self._device)
        batch["task"] = [(task or self._task_default).strip() or self._task_default]
        return batch

    def infer(self, cam_images: dict, state, task: str) -> np.ndarray:
        """engine.py get_action_chunk와 동일한 3단 파이프라인. (T, 6) float64 반환."""
        batch = self._build_batch(cam_images, state, task)
        with torch.inference_mode():
            preprocessed = self._preprocessor(batch)
            action = self._predict_chunk(preprocessed)
            action = self._postprocessor(action)
        return self._to_numpy_chunk(action)


def _serve(sidecar: VlaSidecar, sock_path: str) -> None:
    if os.path.exists(sock_path):
        os.remove(sock_path)
    server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_sock.bind(sock_path)
    server_sock.listen(1)
    logger.info("리스닝: %s", sock_path)

    try:
        while True:
            conn, _ = server_sock.accept()
            logger.info("클라이언트 연결됨")
            try:
                _handle_connection(sidecar, conn)
            except ConnectionError:
                logger.info("클라이언트 연결 종료")
            finally:
                conn.close()
    finally:
        server_sock.close()
        if os.path.exists(sock_path):
            os.remove(sock_path)


def _handle_connection(sidecar: VlaSidecar, conn: socket.socket) -> None:
    while True:
        request = recv_msg(conn)
        cmd = request.get("cmd", "infer")
        try:
            if cmd == "ping":
                # 클라이언트 연결·ready 확인용. 모델 로드 상태와 경로를 돌려준다.
                response = {"success": True, "ready": sidecar._policy is not None,
                            "model_path": sidecar.model_path, "message": ""}
            elif cmd == "load":
                # HMI 모델 전환 — 사이드카가 정책을 재로드한다.
                sidecar.load(request["model_path"])
                response = {"success": True, "ready": True,
                            "model_path": sidecar.model_path, "message": "loaded"}
            else:  # "infer"
                seed = request.get("seed")
                if seed is not None:
                    # SmolVLA는 디노이징 초기 노이즈를 전역 RNG에서 샘플링해 호출마다
                    # 확률적이다. 정상 운용에서는 시드를 주지 않지만(자연스러운
                    # 샘플링), 오프라인 수치검증에서 재현성이 필요할 때만 클라이언트가
                    # seed를 넘겨 결정적으로 만든다.
                    torch.manual_seed(int(seed))
                chunk = sidecar.infer(
                    {"camera1": request.get("camera1"), "camera3": request.get("camera3")},
                    request.get("state"),
                    request.get("task", ""),
                )
                t, d = chunk.shape
                response = {
                    "success": True,
                    "action_chunk": chunk,
                    "chunk_size": int(t),
                    "action_dim": int(d),
                    "message": "",
                }
        except Exception as exc:  # noqa: BLE001 - 클라이언트에 실패 사유 전달
            logger.error("요청 처리 실패(cmd=%s): %s", cmd, exc, exc_info=True)
            response = {"success": False, "action_chunk": None, "chunk_size": 0,
                        "action_dim": 0, "message": str(exc)}
        send_msg(conn, response)


def main() -> None:
    parser = argparse.ArgumentParser(description="QUVI VLA 추론 사이드카 서버")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--socket", default=DEFAULT_SOCKET)
    parser.add_argument("--task-default", default=DEFAULT_TASK)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    sidecar = VlaSidecar(args.model_path, args.device, args.task_default)
    logger.info("모델 로드 완료. 서빙 시작.")
    _serve(sidecar, args.socket)


if __name__ == "__main__":
    main()
