#!/bin/bash
# QUVI SmolVLA Grasp Model Training Helper
# cyclo 의 lerobot 이미지(lerobot 0.6.1)로 SmolVLA 2캠 정책을 fine-tuning 한다.
# smolvla_218ep 학습 설정(train_config.json)을 그대로 따르고, bf16 mixed precision 만 추가했다.
# 실행 위치: 호스트. QUVI 가 GPU 메모리를 쓰므로 먼저 내릴 것(8GB VRAM).
#
# 사용법: ./scripts/smolvla_train.sh [데이터셋 디렉토리명] [출력 이름] [steps] [batch]
#   예: ./scripts/smolvla_train.sh Task_1_pick_quvi_MCAP_lerobot_v30 smolvla_218ep_bf16 20000 16

DATASET=${1:-"Task_1_pick_quvi_MCAP_lerobot_v30"}   # 218 에피소드
JOB_NAME=${2:-"smolvla_$(date +%m%d_%H%M)"}
STEPS=${3:-10000}
BATCH=${4:-16}   # RTX 5070 Laptop(8GB)은 bf16 이어도 batch 32 에서 OOM(2026-10-27 실측 7.49GB) — smolvla_218ep 은 32 로 학습됨

# 8GB VRAM 을 거의 다 쓴다. QUVI 사이드카(~1GB) 등이 남아 있으면 첫 스텝에서 OOM 난다.
GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "${GPU_USED:-0}" -gt 500 ]; then
  echo "경고: GPU 메모리를 이미 ${GPU_USED} MiB 쓰는 중 — 아래 프로세스를 먼저 종료할 것"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  exit 1
fi

CYCLO_DOCKER_DIR="${HOME}/cyclo_intelligence/docker"
IMAGE="robotis/lerobot-zenoh:1.4.1-amd64"

if [ ! -d "${CYCLO_DOCKER_DIR}/workspace/lerobot/${DATASET}" ]; then
  echo "데이터셋 없음: ${CYCLO_DOCKER_DIR}/workspace/lerobot/${DATASET}"
  exit 1
fi
if [ -d "${CYCLO_DOCKER_DIR}/workspace/model/lerobot/${JOB_NAME}" ]; then
  echo "출력 디렉토리가 이미 있음: workspace/model/lerobot/${JOB_NAME} — 다른 이름을 쓸 것"
  exit 1
fi

echo "=================================================="
echo "Starting LeRobot SmolVLA Policy Training"
echo "Dataset : /workspace/lerobot/${DATASET}"
echo "Output  : /workspace/model/lerobot/${JOB_NAME}"
echo "Steps   : ${STEPS}   Batch: ${BATCH}"
echo "=================================================="

# SmolVLA 설정에는 dtype 필드가 없어 lerobot_train 이 mixed_precision=None 으로 Accelerator 를 만든다.
# 그러면 accelerate 가 ACCELERATE_MIXED_PRECISION 을 읽는다 — --policy.use_amp 는 이 버전에서 무시된다.
TTY_FLAG=""
[ -t 0 ] && TTY_FLAG="-it"   # 백그라운드(nohup 등) 실행 시 -it 는 실패한다
docker run --rm ${TTY_FLAG} --gpus all --ipc=host \
  -v "${CYCLO_DOCKER_DIR}/workspace:/workspace" \
  -v "${CYCLO_DOCKER_DIR}/huggingface:/root/.cache/huggingface" \
  -e ACCELERATE_MIXED_PRECISION=bf16 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash "${IMAGE}" -lc "
    lerobot-train \
      --dataset.repo_id=quvi/pick_quvi \
      --dataset.root=/workspace/lerobot/${DATASET} \
      --dataset.video_backend=torchcodec \
      --policy.path=/workspace/smolvla_base_2cam \
      --policy.freeze_vision_encoder=true \
      --policy.train_expert_only=true \
      --policy.device=cuda \
      --policy.push_to_hub=false \
      --batch_size=${BATCH} \
      --steps=${STEPS} \
      --save_freq=2000 \
      --log_freq=200 \
      --num_workers=8 \
      --seed=1000 \
      --output_dir=/workspace/model/lerobot/${JOB_NAME} \
      --job_name=${JOB_NAME} \
      --wandb.enable=false
  "
