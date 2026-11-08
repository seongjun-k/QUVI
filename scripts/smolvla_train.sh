#!/bin/bash
# QUVI SmolVLA Grasp Model Training Helper
# lerobot 0.6.1 이미지(robotis/lerobot-zenoh)로 SmolVLA 2캠 정책을 fine-tuning 한다.
# 입출력은 QUVI data/ 아래만 쓴다(~/cyclo_intelligence 폴더 불필요):
#   데이터셋 data/datasets/<이름>, 베이스 data/models/smolvla_base_2cam, 출력 data/models/<출력 이름>
# smolvla_218ep과 같이 이미지 증강을 켠다. 차이는 기본 batch 16(218ep은 32)과 bf16이다.
# 실행 위치: 호스트. QUVI 가 GPU 메모리를 쓰므로 먼저 내릴 것(8GB VRAM).
#
# 사용법: ./scripts/smolvla_train.sh [데이터셋 디렉토리명] [출력 이름] [steps] [batch]
#   예: ./scripts/smolvla_train.sh Task_1_pick_quvi_MCAP_lerobot_v30 smolvla_218ep_bf16 20000 16

DATASET=${1:-"Task_1_pick_quvi_MCAP_lerobot_v30"}   # 218 에피소드
JOB_NAME=${2:-"smolvla_$(date +%m%d_%H%M)"}
STEPS=${3:-10000}
BATCH=${4:-16}   # RTX 5070 Laptop(8GB)은 bf16 이어도 batch 32 에서 OOM(2026-10-27 실측 7.49GB) — smolvla_218ep 은 32 로 학습됨
# 시작 모델(data/models 기준 경로). 데이터가 적으면 학습된 모델에서 이어서 학습:
#   BASE_MODEL=smolvla_218ep/checkpoints/010000/pretrained_model ./scripts/smolvla_train.sh ...
BASE_MODEL=${BASE_MODEL:-smolvla_base_2cam}

# 8GB VRAM 을 거의 다 쓴다. QUVI 사이드카(~1GB) 등이 남아 있으면 첫 스텝에서 OOM 난다.
# 화면(Xorg·gnome-shell 등 그래픽 프로세스)이 쓰는 1GB 남짓은 빼고, 연산 프로세스가 있을 때만 막는다.
GPU_APPS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
if [ -n "${GPU_APPS}" ]; then
  echo "경고: GPU 를 쓰는 연산 프로세스가 있음 — 먼저 종료할 것"
  echo "${GPU_APPS}"
  exit 1
fi

DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"
IMAGE="robotis/lerobot-zenoh:1.4.1-amd64"

if [ ! -d "${DATA_DIR}/datasets/${DATASET}" ]; then
  echo "데이터셋 없음: ${DATA_DIR}/datasets/${DATASET}"
  exit 1
fi
if [ ! -d "${DATA_DIR}/models/${BASE_MODEL}" ]; then
  echo "베이스 모델 없음: ${DATA_DIR}/models/${BASE_MODEL}"
  exit 1
fi
if [ -d "${DATA_DIR}/models/${JOB_NAME}" ]; then
  echo "출력 디렉토리가 이미 있음: data/models/${JOB_NAME} — 다른 이름을 쓸 것"
  exit 1
fi

echo "=================================================="
echo "Starting LeRobot SmolVLA Policy Training"
echo "Dataset : data/datasets/${DATASET}"
echo "Output  : data/models/${JOB_NAME}"
echo "Steps   : ${STEPS}   Batch: ${BATCH}"
echo "=================================================="

# SmolVLA 설정에는 dtype 필드가 없어 lerobot_train 이 mixed_precision=None 으로 Accelerator 를 만든다.
# 그러면 accelerate 가 ACCELERATE_MIXED_PRECISION 을 읽는다 — --policy.use_amp 는 이 버전에서 무시된다.
TTY_FLAG=""
[ -t 0 ] && TTY_FLAG="-it"   # 백그라운드(nohup 등) 실행 시 -it 는 실패한다
# 노트북 snap 도커는 CDI 모드라 --gpus 만 주면 "invoking the NVIDIA Container Runtime Hook directly" 로 실패한다
docker run --rm ${TTY_FLAG} --runtime=nvidia --gpus all --ipc=host \
  -v "${DATA_DIR}/datasets:/data/datasets:ro" \
  -v "${DATA_DIR}/models:/data/models" \
  -v "${HOME}/.cache/huggingface/hub:/root/.cache/huggingface/hub:ro" \
  -e HF_HUB_OFFLINE=1 \
  -e ACCELERATE_MIXED_PRECISION=bf16 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash "${IMAGE}" -lc "
    lerobot-train \
      --dataset.repo_id=quvi/pick_quvi \
      --dataset.root=/data/datasets/${DATASET} \
      --dataset.video_backend=torchcodec \
      --dataset.image_transforms.enable=true \
      --policy.path=/data/models/${BASE_MODEL} \
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
      --output_dir=/data/models/${JOB_NAME} \
      --job_name=${JOB_NAME} \
      --wandb.enable=false
  "
