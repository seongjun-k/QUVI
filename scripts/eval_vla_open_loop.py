#!/usr/bin/env python3
"""SmolVLA 개루프 평가 — 데이터셋 영상을 넣어 예측 액션을 녹화 액션과 비교하고 떨림을 잰다.

로봇·ROS 불필요, GPU 만 쓴다. 호스트에서 학습 이미지(lerobot 0.6.1)로 실행한다.
    docker run --rm --runtime=nvidia --gpus all --ipc=host --entrypoint bash \
        -v ~/QUVI:/quvi -v ~/.cache/huggingface/hub:/root/.cache/huggingface/hub:ro -e HF_HUB_OFFLINE=1 \
        robotis/lerobot-zenoh:1.4.1-amd64 -lc "python3 /quvi/scripts/eval_vla_open_loop.py \
        --dataset /quvi/data/datasets/Task_1_pick_quvi_MCAP_lerobot_v30 \
        --model /quvi/data/models/smolvla_218ep/checkpoints/010000/pretrained_model"

개루프라 예측 오차가 쌓이지 않는다. 실기 성공률을 대신하지 않고 모델 간 상대 비교에만 쓴다.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']


def reversal_rate(values) -> float:
    """연속한 두 변화량의 부호가 뒤집힌 비율. 값이 그대로인 스텝은 건너뛴다."""
    deltas = np.diff(np.asarray(values, dtype=float))
    deltas = deltas[deltas != 0]
    if len(deltas) < 2:
        return 0.0
    return float(np.mean(np.sign(deltas[1:]) != np.sign(deltas[:-1])))


def pick_episodes(total: int, count: int) -> list:
    """학습 앞뒤 구간이 고루 들어가도록 일정 간격으로 고른다."""
    return sorted({int(i) for i in np.linspace(0, total - 1, min(count, total))})


def evaluate(model: str, dataset_root: str, episodes: int, seed: int) -> dict:
    # pylint: disable=import-outside-toplevel
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'vla_sidecar'))
    from server import VlaSidecar, CAMERA_POLICY_KEYS

    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sidecar = VlaSidecar(str(Path(model).expanduser().resolve()), device)
    dataset = LeRobotDataset('quvi/pick_quvi', root=str(Path(dataset_root).expanduser()))
    starts = dataset.meta.episodes['dataset_from_index']
    stops = dataset.meta.episodes['dataset_to_index']

    errors, predicted_rev, recorded_rev = [], [], []
    for ep in pick_episodes(dataset.meta.total_episodes, episodes):
        start, stop = int(starts[ep]), int(stops[ep])
        recorded = np.stack([dataset[i]['action'].numpy() for i in range(start, stop)])
        predicted = []
        frame = start
        while frame < stop:
            item = dataset[frame]
            images = {
                name: (item[key].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                for name, key in CAMERA_POLICY_KEYS.items()
            }
            chunk = sidecar.infer(images, item['observation.state'].numpy(), '')
            predicted.extend(chunk[:stop - frame])
            frame += len(chunk)
        predicted = np.asarray(predicted)
        errors.append(np.abs(predicted - recorded).mean(axis=0))
        predicted_rev.append([reversal_rate(predicted[:, j]) for j in range(len(JOINTS))])
        recorded_rev.append([reversal_rate(recorded[:, j]) for j in range(len(JOINTS))])
    return {
        'model': model,
        'mae': np.mean(errors, axis=0),
        'predicted_reversal': np.mean(predicted_rev, axis=0),
        'recorded_reversal': np.mean(recorded_rev, axis=0),
        'episodes': len(errors),
    }


def table(result: dict) -> str:
    lines = [
        f"### {result['model']} (에피소드 {result['episodes']}개)", '',
        '| 관절 | 평균 절대 오차 | 예측 반전율 | 녹화 반전율 |', '|---|---:|---:|---:|',
    ]
    for j, name in enumerate(JOINTS):
        lines.append(f"| {name} | {result['mae'][j]:.4f} | {result['predicted_reversal'][j] * 100:.1f}% "
                     f"| {result['recorded_reversal'][j] * 100:.1f}% |")
    lines.append(f"| 평균 | {result['mae'].mean():.4f} | {result['predicted_reversal'].mean() * 100:.1f}% "
                 f"| {result['recorded_reversal'].mean() * 100:.1f}% |")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--model', action='append', required=True, help='여러 번 지정해 비교')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--out', help='결과를 저장할 Markdown 경로')
    args = parser.parse_args()
    text = '\n\n'.join(table(evaluate(m, args.dataset, args.episodes, args.seed)) for m in args.model)
    print(text)
    if args.out:
        Path(args.out).write_text(text + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
