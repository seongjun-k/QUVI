#!/usr/bin/env bash
# SmolVLA(lerobot 0.6.1) 추론 사이드카용 격리 venv 구성.
# quvi-dev 메인 파이썬은 numpy<2(cv_bridge ABI)/transformers<4.52(lerobot 0.3.4)
# 핀이라 0.6.1을 올릴 수 없다 - 그래서 별도 venv + 별도 프로세스로 분리한다.
#
# 핀은 lerobot_server 컨테이너(2026-09-08 docker exec pip freeze / 실측 import
# 버전 확인)와 동일하게 맞춘다: torch 2.11.0+cu128, numpy 2.2.6,
# transformers 5.5.4, lerobot 0.6.1.
set -euo pipefail

VENV_DIR="${1:-/home/ksj/QUVI/data/vla_sidecar/venv}"

echo "[setup_venv] venv 생성: $VENV_DIR"
python3.12 -m venv "$VENV_DIR"
# venv 는 워크스페이스(data/) 안이라 colcon 이 site-packages 의 numpy/cmake 테스트용
# setup.py·CMakeLists 를 ROS 패키지로 오인한다 — 상위 디렉토리째 colcon 스캔에서 제외.
touch "$(dirname "$VENV_DIR")/COLCON_IGNORE"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install --upgrade pip

pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install numpy==2.2.6 transformers==5.5.4 lerobot==0.6.1
# num2words: transformers 5.5.4의 SmolVLM 프로세서가 하드 의존성으로 요구하지만
# lerobot/transformers 어느 쪽도 pip 의존성에 선언하지 않아 별도 설치가
# 필요하다(2026-09-08 실측 - "Package num2words is required" ImportError).
pip install num2words
# lerobot이 의존성으로 끌어온 torchvision은 PyPI 기본 빌드라 torch(cu128)와
# ABI가 어긋나 "operator torchvision::nms does not exist"로 죽는다(2026-09-08
# 실측). cu128 인덱스 빌드로 강제 재설치해 torch와 짝을 맞춘다.
pip install --index-url https://download.pytorch.org/whl/cu128 --no-deps --force-reinstall torchvision==0.26.0+cu128

echo "[setup_venv] 완료. 버전 확인:"
python -c "
import lerobot, torch, numpy, transformers
print('lerobot', lerobot.__version__)
print('torch', torch.__version__)
print('numpy', numpy.__version__)
print('transformers', transformers.__version__)
print('cuda available', torch.cuda.is_available())
"
