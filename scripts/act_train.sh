#!/bin/bash
# QUVI ACT Grasp Model Training Helper
# act_record.sh 로 녹화한 데이터셋으로 ACT 정책을 학습한다.
# 실행 위치: 호스트 또는 quvi-dev 컨테이너 (스크립트가 /.dockerenv 로 자동 판별).
# 사용법: ./scripts/act_train.sh [HF_USER]

HF_USER=${1:-"ksj"}
REPO_ID="${HF_USER}/quvi_act_grasp"
OUTPUT_DIR="outputs/train/quvi_act_grasp"

echo "=================================================="
echo "Starting LeRobot ACT Policy Training"
echo "Dataset ID: $REPO_ID"
echo "Output Dir: $OUTPUT_DIR"
echo "=================================================="

# Check if CUDA is available, otherwise default to cpu
DEVICE="cuda"
if [ -f /.dockerenv ]; then
  # Inside Docker
  python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)"
  if [ $? -ne 0 ]; then
    echo "Warning: CUDA is not available. Training on CPU will be extremely slow!"
    DEVICE="cpu"
  fi

  python3 /workspace/lerobot/src/lerobot/scripts/train.py \
    --dataset.repo_id="$REPO_ID" \
    --policy.type=act \
    --output_dir="$OUTPUT_DIR" \
    --device="$DEVICE" \
    --env.type=real \
    --wandb.enable=false
else
  # Host
  docker exec -it quvi-dev python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)"
  if [ $? -ne 0 ]; then
     echo "Warning: CUDA is not available inside Docker. Training on CPU will be extremely slow!"
     DEVICE="cpu"
  fi

  docker exec -it quvi-dev python3 /workspace/lerobot/src/lerobot/scripts/train.py \
    --dataset.repo_id="$REPO_ID" \
    --policy.type=act \
    --output_dir="$OUTPUT_DIR" \
    --device="$DEVICE" \
    --env.type=real \
    --wandb.enable=false
fi
