#!/usr/bin/env python3
"""텔레옵 에피소드 폴더를 점검하고 LeRobot v3.0 데이터셋으로 변환한다.

점검 전용 (호스트):
  python3 scripts/teleop_episodes_to_lerobot.py --episodes data/teleop_episodes

변환 (호스트에서):
  docker run --rm --entrypoint bash -v ~/QUVI:/quvi -v ~/cyclo_intelligence/docker/workspace:/workspace robotis/lerobot-zenoh:1.4.1-amd64 -lc "python3 /quvi/scripts/teleop_episodes_to_lerobot.py --episodes /quvi/data/teleop_episodes --out /workspace/lerobot/quvi_teleop_v30"

합격 기준은 docs/vla_episode_collection_guide.md 2단계. 불합격 에피소드는 변환에서 제외한다.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

FPS = 15
MIN_SEC, MAX_SEC = 3.0, 7.0
MAX_GAP_SEC = 2.5 / FPS
HOME_TOLERANCE_RAW = 20
RAW_PER_RAD = 4096 / (2 * np.pi)
TASK = 'Pick up the printed part from the bed.'  # 기존 데이터셋과 같은 지시문(마침표 포함)
JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'gripper_joint_1']


def check_episode(t, state, action, home_raw) -> list:
    if len(t) < 2:
        return ['프레임 부족']
    if state.shape != (len(t), 6) or action.shape != (len(t), 6):
        return [f'shape 불일치 t{len(t)} state{state.shape} action{action.shape}']
    if not (np.isfinite(t).all() and np.isfinite(state).all() and np.isfinite(action).all()):
        return ['NaN/inf 포함']
    problems = []
    duration = t[-1] - t[0]
    if not MIN_SEC <= duration <= MAX_SEC:
        problems.append(f'길이 {duration:.1f}초 ({MIN_SEC:.0f}~{MAX_SEC:.0f}초 밖)')
    gap = np.diff(t).max()
    if gap > MAX_GAP_SEC:
        problems.append(f'프레임 간격 최대 {gap:.2f}초')
    start_raw = state[0][:5] * RAW_PER_RAD + 2048
    off = np.abs(start_raw - np.array(home_raw[:5])).max()
    if off > HOME_TOLERANCE_RAW:
        problems.append(f'시작 자세가 홈에서 {off:.0f} raw 벗어남')
    # 그리퍼 닫힘 = raw 감소(GRIPPER_OPEN 2500 → GRIPPER_CLOSE 1800) = rad 감소
    if action[:, 5].min() >= action[0, 5]:
        problems.append('그리퍼를 닫지 않음')
    return problems


def load_episode(path) -> dict:
    with np.load(Path(path) / 'data.npz') as d:
        return {k: d[k] for k in ('t', 'state', 'action', 'home_raw')}


def convert(accepted, out):
    import cv2
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    image = {'dtype': 'video', 'shape': (3, 480, 640), 'names': ['channels', 'height', 'width']}
    vec = {'dtype': 'float32', 'shape': (6,), 'names': JOINT_NAMES}
    ds = LeRobotDataset.create(
        'quvi/pick_quvi', FPS, root=out, robot_type='omx_f', use_videos=True,
        features={'observation.state': vec, 'action': vec,
                  'observation.images.rgb.camera1': image,
                  'observation.images.rgb.camera3': image})
    for path, ep in accepted:
        for i in range(len(ep['t'])):
            imgs = {}
            for cam, name in (('camera1', 'side'), ('camera3', 'top')):
                f = path / f'{i:05d}_{name}.jpg'
                bgr = cv2.imread(str(f))
                if bgr is None:
                    raise ValueError(f'이미지를 읽을 수 없음: {f}')
                imgs[f'observation.images.rgb.{cam}'] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            ds.add_frame({'observation.state': ep['state'][i], 'action': ep['action'][i],
                          'task': TASK, **imgs})
        ds.save_episode()
    ds.finalize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', required=True)
    ap.add_argument('--out')
    args = ap.parse_args()

    accepted = []
    for path in sorted(Path(args.episodes).glob('ep_*')):
        if path.suffix == '.tmp':
            continue
        try:
            ep = load_episode(path)
            problems = check_episode(ep['t'], ep['state'], ep['action'], ep['home_raw'])
            for cam in ('side', 'top'):
                n = len(list(path.glob(f'*_{cam}.jpg')))
                if n != len(ep['t']):
                    problems.append(f'{cam} 이미지 {n}장 != 프레임 {len(ep["t"])}')
        except Exception as e:
            ep, problems = None, [f'읽기 실패: {e}']
        print(f"{path.name}: {'불합격 - ' + ', '.join(problems) if problems else '합격'}")
        if not problems:
            accepted.append((path, ep))
    print(f'합격 {len(accepted)}개')
    if not accepted:
        sys.exit(1)
    if args.out:
        convert(accepted, args.out)


if __name__ == '__main__':
    main()
