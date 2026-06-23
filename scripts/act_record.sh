#!/bin/bash
# QUVI ACT Grasp Dataset Recording Helper

HF_USER=${1:-"ksj"}
EPISODES=${2:-50}
EPISODE_TIME=${3:-15}
REPO_ID="${HF_USER}/quvi_act_grasp"

echo "=================================================="
echo "Starting LeRobot Dataset Recording for ACT"
echo "HF Repo ID: $REPO_ID"
echo "Num Episodes: $EPISODES"
echo "Episode Time: $EPISODE_TIME sec"
echo "=================================================="

# Create log directories if on host
mkdir -p log outputs

if [ -f /.dockerenv ]; then
  # Inside Docker Container
  python3 /workspace/lerobot/src/lerobot/record.py \
    --robot.path /workspace/lerobot/configs/robot/omx.yaml \
    --teleop.path /workspace/lerobot/configs/robot/omx_leader.yaml \
    --dataset.repo_id="$REPO_ID" \
    --dataset.single_task="Grasp the 3D printed object from the bed" \
    --dataset.fps=30 \
    --dataset.episode_time_s="$EPISODE_TIME" \
    --dataset.reset_time_s=5 \
    --dataset.num_episodes="$EPISODES" \
    --display_data=true
else
  # On Host System
  docker exec -it quvi-dev python3 /workspace/lerobot/src/lerobot/record.py \
    --robot.path /workspace/lerobot/configs/robot/omx.yaml \
    --teleop.path /workspace/lerobot/configs/robot/omx_leader.yaml \
    --dataset.repo_id="$REPO_ID" \
    --dataset.single_task="Grasp the 3D printed object from the bed" \
    --dataset.fps=30 \
    --dataset.episode_time_s="$EPISODE_TIME" \
    --dataset.reset_time_s=5 \
    --dataset.num_episodes="$EPISODES" \
    --display_data=true
fi
