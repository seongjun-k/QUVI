#!/usr/bin/env bash
# 수집 도구용 venv 생성 — 컨테이너 quvi-dev 안에서 실행:
#   docker exec quvi-dev bash /workspace/scripts/collect/setup_tools_venv.sh
# 녹화 앱(컨테이너 안)이 이 venv 의 python 을 subprocess 로 불러 변환기를 돌린다.
# 사이드카 venv(data/vla_sidecar/venv)와 별도 — datasets 가 없어 못 쓴다.
# 재실행 안전: 이미 import 가능하면 아무것도 하지 않는다. FORCE=1 이면 pip 설치만 다시 한다.
set -euo pipefail

VENV=/workspace/data/collect_tools/venv
PY=$VENV/bin/python

if [ ! -d /workspace/src ]; then
  echo "컨테이너 quvi-dev 안에서 실행해야 합니다." >&2
  exit 1
fi

check() { "$PY" -c "import lerobot, datasets, pytest; assert lerobot.__version__ == '0.6.1'" 2>/dev/null; }

if [ -x "$PY" ] && check && [ "${FORCE:-0}" != "1" ]; then
  echo "이미 설치됨: $VENV"
  "$PY" -c "import lerobot, datasets, torch; print('lerobot', lerobot.__version__, 'datasets', datasets.__version__, 'torch', torch.__version__)"
  exit 0
fi

# 컨테이너에 python3-venv(ensurepip)가 없다 — pip 없이 만들고 시스템 pip 로 venv 에 pip 를 심는다
[ -x "$PY" ] || python3 -m venv --without-pip "$VENV"
"$PY" -m pip --version >/dev/null 2>&1 || python3 -m pip --python "$PY" install -q pip

# GPU 불필요 — CPU 휠을 먼저 깔아 용량을 줄인다(lerobot 요구 torch<2.12 범위)
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q 'torch>=2.7,<2.12' 'torchvision>=0.22,<0.27' --index-url https://download.pytorch.org/whl/cpu
# [dataset] extra = datasets·pandas·pyarrow·av·torchcodec·jsonlines
"$PY" -m pip install -q 'lerobot[dataset]==0.6.1' pytest

check
du -sh "$VENV"
"$PY" -c "import lerobot, datasets, torch; print('lerobot', lerobot.__version__, 'datasets', datasets.__version__, 'torch', torch.__version__)"
