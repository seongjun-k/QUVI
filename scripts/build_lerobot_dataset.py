#!/usr/bin/env python3
"""수집 원본(docs/_local/collect_raw_format.md) → LeRobot v3.0 데이터셋 변환 + 품질 점검.

사용: <도구 venv>/bin/python scripts/build_lerobot_dataset.py <세션 폴더…> --out <경로>
Cyclo 변환 데이터셋과 같은 키·형식(omx_f, h264 yuv420p)을 만들어 학습 명령을 바꾸지 않게 한다.
lerobot·cv2 는 함수 안에서 import — 리샘플 순수 함수는 numpy 만으로 테스트된다.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# 수집 도구 venv 에는 rclpy 가 없다 — act_common 은 ROS 무관이라 소스 경로를 직접 붙인다
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'quvi_robot_control'))
from quvi_robot_control.act_common import (  # noqa: E402  pylint: disable=wrong-import-position
    DATASET_FPS, EPISODE_FRAMES, JOINT_NAMES, POSE_HOME, TASK_INSTRUCTION, raw_to_rad)

# ─── 상수 ───
TASK = TASK_INSTRUCTION   # 정확히 일치 검사
ROBOT_TYPE = 'omx_f'
# 기존 Cyclo 데이터셋 info.json 의 names/키 그대로 (학습 명령 호환)
DATASET_JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'gripper_joint_1']
CAMERAS = {'camera1': 'observation.images.rgb.camera1', 'camera3': 'observation.images.rgb.camera3'}
IMG_H, IMG_W = 480, 640
SESSION_SAME_KEYS = ('dataset_fps', 'record_state_hz', 'episode_sec', 'pre_roll_sec')
STATES_HEADER = (['t'] + [f'present_{j}' for j in JOINT_NAMES] + [f'goal_{j}' for j in JOINT_NAMES])

START_TOL_RAD = 0.03     # 시작 자세 편차 경고 기준(지시값). 시작 흔들림에도 같은 값을 쓴다(임의 가정)
MOVE_THRESH_RAD = 0.02   # 첫 움직임 판정: 첫 프레임 대비 관절 최대 변화량(약 13 raw, 가정값)
GAP_WARN_SEC = 0.5       # 격자 칸에 쓰인 샘플이 이보다 오래됐으면 공백으로 경고(원본 형식 문서의 카메라 공백 기준)


# ─── 리샘플 (순수 함수) ───
def resample_indices(t, n_frames=EPISODE_FRAMES, fps=DATASET_FPS):
    """격자 g_k=k/fps 마다 "g_k 이하 마지막 샘플" 인덱스. t 는 오름차순.

    g_k 이전 샘플이 없으면 첫 샘플(0)로 대체하고 그 개수를 clamped 로 돌려준다.
    """
    grid = np.arange(n_frames) / fps
    idx = np.searchsorted(np.asarray(t), grid, side='right') - 1
    clamped = int((idx < 0).sum())
    return np.maximum(idx, 0), clamped


def sample_ages(t, idx, fps=DATASET_FPS):
    """격자 시각 − 선택된 샘플 시각(초). clamped 칸은 음수."""
    return np.arange(len(idx)) / fps - np.asarray(t)[idx]


# ─── 입력 읽기 ───
def load_session(path):
    meta = json.loads((path / 'session.json').read_text(encoding='utf-8'))
    eps = sorted(p for p in path.glob('ep_*') if p.is_dir())
    if not eps:
        raise SystemExit(f'에피소드 없음: {path}')
    return meta, eps


def read_states(ep):
    f = ep / 'states.csv'
    header = f.read_text(encoding='utf-8').splitlines()[0].split(',')
    if header != STATES_HEADER:
        raise SystemExit(f'states.csv 열이 형식 문서와 다름: {f}\n  {header}')
    a = np.loadtxt(f, delimiter=',', skiprows=1, ndmin=2)
    return a[:, 0], a[:, 1:7].astype(np.float32), a[:, 7:13].astype(np.float32)


def read_stamps(ep, cam):
    d = ep / cam
    a = np.loadtxt(d / 'stamps.csv', delimiter=',', skiprows=1, ndmin=2)
    if (np.diff(a[:, 1]) < 0).any():
        raise SystemExit(f't 가 오름차순이 아님: {d / "stamps.csv"}')
    n = len(list(d.glob('*.jpg')))
    if n != len(a):
        raise SystemExit(f'stamps 행 수({len(a)}) != jpg 수({n}): {d}')
    return a[:, 0].astype(int), a[:, 1]


def decode_rgb(path):
    import cv2
    bgr = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if bgr is None or bgr.shape != (IMG_H, IMG_W, 3):
        raise SystemExit(f'JPEG 디코드 실패 또는 해상도 불일치({IMG_H}x{IMG_W} 아님): {path}')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ─── 검증 ───
def check_sessions(session_dirs):
    """task 정확 일치 + 세션 간 설정 동일 검사. (session meta 목록, 에피소드 경로 목록) 반환."""
    metas, eps, ref = [], [], None
    for sd in session_dirs:
        meta, e = load_session(sd)
        if meta.get('task') != TASK:
            raise SystemExit(f'지시문 불일치: {sd}\n  기대 {TASK!r}\n  실제 {meta.get("task")!r}')
        cur = tuple(meta.get(k) for k in SESSION_SAME_KEYS)
        if ref is None:
            ref = cur
        elif cur != ref:
            raise SystemExit(f'세션 설정 불일치: {sd}\n  {dict(zip(SESSION_SAME_KEYS, cur))} != {dict(zip(SESSION_SAME_KEYS, ref))}')
        metas.append(meta)
        eps += [(sd, p) for p in e]
    return metas, eps


# ─── 에피소드 변환 ───
def build_episode(ep, n_frames, fps, home_rad):
    """에피소드 하나를 격자 리샘플해 (프레임 dict 생성기용 데이터, 품질 지표)로 돌려준다."""
    ts, present, goal = read_states(ep)
    if (np.diff(ts) < 0).any():
        raise SystemExit(f'states.csv t 가 오름차순이 아님: {ep}')
    s_idx, s_clamp = resample_indices(ts, n_frames, fps)
    state, action = present[s_idx], goal[s_idx]

    cams, clamped, gap = {}, {'state': s_clamp}, {'state': float(sample_ages(ts, s_idx, fps).max())}
    for cam in CAMERAS:
        jpg_idx, tc = read_stamps(ep, cam)
        c_idx, c_clamp = resample_indices(tc, n_frames, fps)
        cams[cam] = [ep / cam / f'{jpg_idx[i]:06d}.jpg' for i in c_idx]
        clamped[cam] = c_clamp
        gap[cam] = float(sample_ages(tc, c_idx, fps).max())

    start_dev = np.abs(state[0] - home_rad)
    shake = float((state[:3].max(axis=0) - state[:3].min(axis=0)).max())
    moved = np.abs(state - state[0]).max(axis=1) > MOVE_THRESH_RAD
    q = {
        'length': n_frames,
        'start_dev_max_rad': round(float(start_dev.max()), 4),
        'start_dev_joint': JOINT_NAMES[int(start_dev.argmax())],
        'start_shake_rad': round(shake, 4),
        'first_move_frame': int(moved.argmax()) if moved.any() else None,
        'clamped': clamped,
        'max_sample_age_sec': {k: round(v, 3) for k, v in gap.items()},
    }
    warns = []
    if q['start_dev_max_rad'] > START_TOL_RAD:
        warns.append(f'시작 편차 {q["start_dev_max_rad"]} rad({q["start_dev_joint"]})')
    if shake > START_TOL_RAD:
        warns.append(f'시작 흔들림 {q["start_shake_rad"]} rad')
    if q['first_move_frame'] is None:
        warns.append('움직임 없음')
    if sum(clamped.values()):
        warns.append(f'clamped {sum(clamped.values())}')
    for k, v in gap.items():
        if v >= GAP_WARN_SEC:
            warns.append(f'{k} 공백 {v:.2f}s')
    q['warnings'] = warns
    return state, action, cams, q


def write_dataset(session_dirs, out, fps, repo_id):
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if out.exists():
        raise SystemExit(f'--out 이 이미 있음(덮어쓰기 금지): {out}')
    metas, eps = check_sessions(session_dirs)
    if any(m.get('dataset_fps') != fps for m in metas):
        raise SystemExit(f'--fps {fps} != session.json dataset_fps {metas[0].get("dataset_fps")}')
    n_frames = math.ceil(metas[0]['episode_sec'] * fps - 1e-9)
    home_rad = np.array([raw_to_rad(POSE_HOME[j]) for j in JOINT_NAMES], np.float32)

    vec = {'dtype': 'float32', 'shape': (6,), 'names': DATASET_JOINT_NAMES}
    features = {'observation.state': dict(vec), 'action': dict(vec)}
    for key in CAMERAS.values():
        features[key] = {'dtype': 'video', 'shape': (3, IMG_H, IMG_W), 'names': ['channels', 'height', 'width']}

    # lerobot 0.6.1 기본 vcodec 은 libsvtav1 이다 — 기존 데이터셋과 같은 h264/yuv420p 를 명시
    ds = LeRobotDataset.create(repo_id, fps, features, root=out, robot_type=ROBOT_TYPE, use_videos=True,
                               rgb_encoder=RGBEncoderConfig(vcodec='h264', pix_fmt='yuv420p'))
    report = []
    for sd, ep in eps:
        state, action, cams, q = build_episode(ep, n_frames, fps, home_rad)
        for k in range(n_frames):
            frame = {'observation.state': state[k], 'action': action[k], 'task': TASK}
            for cam, key in CAMERAS.items():
                frame[key] = decode_rgb(cams[cam][k])
            ds.add_frame(frame)
        ds.save_episode()
        q['episode'] = f'{sd.name}/{ep.name}'
        report.append(q)
        print(f'  변환: {q["episode"]}', flush=True)
    ds.finalize()
    return report, n_frames


# ─── 출력 후 로드 확인 ───
def verify_load(out, repo_id, n_episodes, n_frames):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id, root=out)
    item = ds[0]
    shapes = {k: tuple(item[k].shape) for k in ('observation.state', 'action', *CAMERAS.values())}
    expect = {'observation.state': (6,), 'action': (6,), **{k: (3, IMG_H, IMG_W) for k in CAMERAS.values()}}
    if shapes != expect:
        raise SystemExit(f'프레임 shape 불일치: {shapes}')
    tasks = list(ds.meta.tasks.index)
    if len(ds) != n_episodes * n_frames or ds.meta.total_episodes != n_episodes or tasks != [TASK]:
        raise SystemExit(f'로드 결과 불일치: frames={len(ds)} episodes={ds.meta.total_episodes} tasks={tasks}')
    return len(tasks)


# ─── 품질 점검 출력 ───
def print_report(report, n_tasks):
    cols = ['episode', 'len', 'start_dev(rad)', 'shake', 'move@', 'clamped', 'age_max(s)', 'warn']
    rows = []
    for q in report:
        rows.append([q['episode'], str(q['length']), f'{q["start_dev_max_rad"]:.3f} {q["start_dev_joint"]}',
                     f'{q["start_shake_rad"]:.3f}', str(q['first_move_frame']), str(sum(q['clamped'].values())),
                     f'{max(q["max_sample_age_sec"].values()):.2f}', '; '.join(q['warnings']) or '-'])
    w = [max(len(c), *(len(r[i]) for r in rows)) for i, c in enumerate(cols)]
    print('\n품질 점검')
    print('  '.join(c.ljust(w[i]) for i, c in enumerate(cols)))
    for r in rows:
        print('  '.join(c.ljust(w[i]) for i, c in enumerate(r)))
    len_ok = all(q['length'] == EPISODE_FRAMES for q in report)
    n_warn = sum(bool(q['warnings']) for q in report)
    print(f'\n길이=={EPISODE_FRAMES}: {"ok" if len_ok else "bad"} / tasks==1: {"ok" if n_tasks == 1 else "bad"} / 경고 에피소드 {n_warn}/{len(report)}')
    print(json.dumps({'episodes': len(report), 'length_ok': len_ok, 'tasks': n_tasks,
                      'warn_episodes': n_warn, 'per_episode': report}, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('sessions', nargs='+', type=Path, help='세션 폴더(data/episodes_raw/<session_id>)')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--fps', type=int, default=DATASET_FPS)
    ap.add_argument('--repo-id', default=None, help='기본: local/<out 폴더명>')
    a = ap.parse_args()
    out = a.out.resolve()
    repo_id = a.repo_id or f'local/{out.name}'

    report, n_frames = write_dataset(a.sessions, out, a.fps, repo_id)
    n_tasks = verify_load(out, repo_id, len(report), n_frames)
    print_report(report, n_tasks)


if __name__ == '__main__':
    main()
