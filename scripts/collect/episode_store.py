"""수집 원본 폴더(docs/_local/collect_raw_format.md)의 디스크 입출력·편집·판정 순수 함수.

Flask·rclpy 에 의존하지 않는다 — 단위 테스트가 이 모듈만 import 한다.
시간은 전부 에피소드 기준 상대 초(t0=0)로 받는다.
"""
import csv
import json
import math
import os
import re
import shutil
from datetime import datetime

from quvi_robot_control.act_common import (
    DATASET_FPS, EPISODE_DURATION_SEC, GRIPPER_OPEN, JOINT_NAMES, POSE_HOME,
    RECORD_STATE_HZ, TASK_INSTRUCTION, rad_to_raw)

# ─── 형식 상수 ───
PRE_ROLL_SEC = 0.2
START_TOL_RAW = 20
START_HOLD_SEC = 0.5
END_TOL_RAW = 40
TARGET_RETURN_SEC = 4.5
MAX_GAP_SEC = 0.5            # 카메라·상태 공백 허용 한도(초과 시 verdict bad)
SPEED_DEV_WARN_SEC = 1.0     # 형식 문서에 임계값 없음 — 복귀가 목표보다 이만큼 어긋나면 warn
GRIPPER_CLOSED_MARGIN = 200
CAMERAS = ('camera1', 'camera3')
DEFAULT_TASK = TASK_INSTRUCTION

# 합치기 호환 검사 대상 — 하나라도 다르면 학습 데이터가 섞이면 안 되는 설정
COMPAT_KEYS = ('task', 'dataset_fps', 'record_state_hz', 'episode_sec', 'pre_roll_sec',
               'start_tol_raw', 'start_hold_sec', 'end_tol_raw', 'target_return_sec')

_EP_RE = re.compile(r'^ep_(\d{4})$')
_HOME_RAW = [POSE_HOME[j] for j in JOINT_NAMES]


def sanitize_name(name: str) -> str:
    """폴더명에 쓸 수 있게 정리(한글 허용). 비면 빈 문자열."""
    return re.sub(r'[^\w\-]+', '_', name.strip()).strip('_')


def new_session_json(product: str, task: str = DEFAULT_TASK, cameras: dict = None) -> dict:
    return {
        'product': product,
        'created': datetime.now().astimezone().isoformat(timespec='seconds'),
        'task': task,
        'dataset_fps': DATASET_FPS, 'record_state_hz': RECORD_STATE_HZ,
        'episode_sec': EPISODE_DURATION_SEC, 'pre_roll_sec': PRE_ROLL_SEC,
        'start_tol_raw': START_TOL_RAW, 'start_hold_sec': START_HOLD_SEC,
        'end_tol_raw': END_TOL_RAW, 'target_return_sec': TARGET_RETURN_SEC,
        'cameras': cameras or {},
    }


# ─── 판정 ───

def _max_gap(ts: list) -> float:
    """저장 범위 [0, EPISODE_DURATION_SEC] 안의 최대 공백(초). 샘플이 없으면 inf."""
    pts = [t for t in ts if 0.0 <= t <= EPISODE_DURATION_SEC]
    if not pts:
        return math.inf
    gaps = [b - a for a, b in zip(pts, pts[1:])]
    gaps += [pts[0], EPISODE_DURATION_SEC - pts[-1]]
    return max(gaps)


def compute_meta(t0_wall: str, state: list, cams: dict) -> dict:
    """state=[(t, present6_rad, goal6_rad)], cams={camera: [(t, jpeg_bytes)]} → meta.json 내용."""
    warnings = []
    verdict = 'ok'

    def _raw_dev(rad6):
        return [rad_to_raw(r) - h for r, h in zip(rad6, _HOME_RAW)]

    if not state:
        return {'t0_wall': t0_wall, 'duration_sec': EPISODE_DURATION_SEC,
                'counts': {'state': 0, **{c: len(cams.get(c, [])) for c in CAMERAS}},
                'start_dev_raw': None, 'end_dev_raw': None, 'gripper_closed': None,
                'return_time_sec': None, 'speed_dev_sec': None, 'verdict': 'bad',
                'warnings': ['상태 샘플 없음']}

    # 시작 편차: t=0 직전 마지막 present (없으면 첫 샘플)
    before = [s for s in state if s[0] <= 0.0]
    start_dev = [round(d) for d in _raw_dev((before[-1] if before else state[0])[1])]
    last_raw = _raw_dev(state[-1][1])
    end_dev = [round(d) for d in last_raw[:5]]
    gripper_raw = rad_to_raw(state[-1][1][5])
    gripper_closed = bool(gripper_raw < GRIPPER_OPEN - GRIPPER_CLOSED_MARGIN)

    # 복귀 시각: 끝에서부터 팔 5관절이 허용 안인 구간이 시작되는 첫 시각
    return_time = None
    for t, present, _ in reversed(state):
        if all(abs(d) <= END_TOL_RAW for d in _raw_dev(present)[:5]):
            return_time = round(max(t, 0.0), 3)
        else:
            break
    speed_dev = None if return_time is None else round(return_time - TARGET_RETURN_SEC, 3)

    if return_time is None:
        verdict = 'warn'
        warnings.append(f'끝 자세가 ±{END_TOL_RAW} raw 안에 들어오지 못함')
    elif abs(speed_dev) > SPEED_DEV_WARN_SEC:
        verdict = 'warn'
        warnings.append(f'복귀 시각이 목표보다 {speed_dev:+.1f}초 어긋남')
    if not gripper_closed:
        verdict = 'warn'
        warnings.append('끝에서 그리퍼가 닫혀 있지 않음')

    # 데이터 결손은 warn 보다 우선
    for name, ts in (('상태', [s[0] for s in state]),
                     *((c, [f[0] for f in cams.get(c, [])]) for c in CAMERAS)):
        gap = _max_gap(ts)
        if gap > MAX_GAP_SEC:
            verdict = 'bad'
            warnings.append(f'{name} 공백 {gap:.2f}초' if math.isfinite(gap) else f'{name} 샘플 없음')

    return {
        't0_wall': t0_wall, 'duration_sec': EPISODE_DURATION_SEC,
        'counts': {'state': len(state), **{c: len(cams.get(c, [])) for c in CAMERAS}},
        'start_dev_raw': start_dev, 'end_dev_raw': end_dev,
        'gripper_closed': gripper_closed, 'return_time_sec': return_time,
        'speed_dev_sec': speed_dev, 'verdict': verdict, 'warnings': warnings,
    }


# ─── 디스크 기록 ───

def list_episode_dirs(session_dir: str) -> list:
    """번호 오름차순 [(index, path)]. 임시 폴더(.tmp)는 무시한다."""
    out = []
    for name in os.listdir(session_dir):
        m = _EP_RE.match(name)
        if m and os.path.isdir(os.path.join(session_dir, name)):
            out.append((int(m.group(1)), os.path.join(session_dir, name)))
    return sorted(out)


def write_episode(session_dir: str, state: list, cams: dict, meta: dict) -> int:
    """에피소드를 임시 폴더에 다 쓴 뒤 rename — 중간에 죽어도 반쪽 에피소드가 목록에 안 뜬다."""
    eps = list_episode_dirs(session_dir)
    idx = eps[-1][0] + 1 if eps else 0
    final = os.path.join(session_dir, f'ep_{idx:04d}')
    tmp = final + '.tmp'
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    try:
        with open(os.path.join(tmp, 'states.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t'] + [f'present_{j}' for j in JOINT_NAMES] + [f'goal_{j}' for j in JOINT_NAMES])
            for t, present, goal in state:
                w.writerow([f'{t:.6f}'] + [f'{v:.8f}' for v in list(present) + list(goal)])
        for cam in CAMERAS:
            cdir = os.path.join(tmp, cam)
            os.makedirs(cdir)
            with open(os.path.join(cdir, 'stamps.csv'), 'w', newline='') as sf:
                sw = csv.writer(sf)
                sw.writerow(['idx', 't'])
                for i, (t, data) in enumerate(cams.get(cam, [])):
                    with open(os.path.join(cdir, f'{i:06d}.jpg'), 'wb') as jf:
                        jf.write(data)
                    sw.writerow([i, f'{t:.6f}'])
        with open(os.path.join(tmp, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        os.rename(tmp, final)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return idx


# ─── 조회 ───

def read_json(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def list_sessions(root: str) -> list:
    """세션 요약 목록(최신순). 에피소드 meta 가 깨져 있어도 목록은 나온다."""
    out = []
    if not os.path.isdir(root):
        return out
    for sid in sorted(os.listdir(root), reverse=True):
        sdir = os.path.join(root, sid)
        sj = os.path.join(sdir, 'session.json')
        if not os.path.isfile(sj):
            continue
        try:
            info = read_json(sj)
        except (OSError, ValueError):
            info = {'error': 'session.json 읽기 실패'}
        eps = []
        for idx, edir in list_episode_dirs(sdir):
            try:
                meta = read_json(os.path.join(edir, 'meta.json'))
            except (OSError, ValueError):
                meta = {'verdict': 'bad', 'warnings': ['meta.json 읽기 실패']}
            eps.append({'index': idx, **meta})
        out.append({'id': sid, 'product': info.get('product'), 'created': info.get('created'),
                    'merged_from': [m.get('session') for m in info.get('merged_from', [])],
                    'error': info.get('error'), 'episodes': eps})
    return out


def frame_path(session_dir: str, ep_index: int, cam: str, pos: str) -> str:
    """처음/중간/끝 또는 프레임 번호(숫자 문자열) jpg 경로. 없으면 FileNotFoundError."""
    cdir = os.path.join(session_dir, f'ep_{ep_index:04d}', cam)
    if pos.isdigit():
        path = os.path.join(cdir, f'{int(pos):06d}.jpg')
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        return path
    frames = sorted(n for n in os.listdir(cdir) if n.endswith('.jpg'))
    if not frames:
        raise FileNotFoundError(cdir)
    return os.path.join(cdir, {'first': frames[0], 'mid': frames[len(frames) // 2],
                               'last': frames[-1]}[pos])


def episode_stamps(session_dir: str, ep_index: int) -> dict:
    """재생용 카메라별 수신 시각 [t…]. 리스트 위치 = 프레임 번호."""
    edir = os.path.join(session_dir, f'ep_{ep_index:04d}')
    out = {}
    for cam in CAMERAS:
        with open(os.path.join(edir, cam, 'stamps.csv'), newline='', encoding='utf-8') as f:
            out[cam] = [float(r['t']) for r in csv.DictReader(f)]
    return out


# ─── 편집 ───

REVIEWS = ('keep', 'discard')


def set_review(session_dir: str, ep_index: int, review) -> None:
    """사람의 사용/안 씀 표시를 meta.json 'review' 에 기록(None = 해제). 변환기는 discard 를 건너뛴다.

    지우지 않고 표시만 하므로 언제든 되돌릴 수 있다. 실제 삭제는 delete_episodes.
    """
    if review is not None and review not in REVIEWS:
        raise ValueError(f'review 는 {REVIEWS} 또는 null: {review!r}')
    path = os.path.join(session_dir, f'ep_{ep_index:04d}', 'meta.json')
    meta = read_json(path)
    if review is None:
        meta.pop('review', None)
    else:
        meta['review'] = review
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _renumber(session_dir: str):
    """ep_* 를 0부터 빈칸 없이. 목표 번호 ≤ 현재 번호라 오름차순 rename 이 서로 안 덮는다."""
    for new, (old, path) in enumerate(list_episode_dirs(session_dir)):
        if new != old:
            os.rename(path, os.path.join(session_dir, f'ep_{new:04d}'))


def delete_episodes(session_dir: str, indices: list, renumber: bool = False) -> int:
    """지정 번호 에피소드 삭제. 없는 번호가 하나라도 있으면 아무것도 지우지 않고 ValueError."""
    existing = dict(list_episode_dirs(session_dir))
    missing = [i for i in indices if i not in existing]
    if missing:
        raise ValueError(f'없는 에피소드: {missing}')
    for i in indices:
        shutil.rmtree(existing[i])
    if renumber:
        _renumber(session_dir)
    return len(indices)


def merge_sessions(sources: list, out_dir: str, move: bool = False) -> int:
    """여러 세션을 새 세션 폴더로 합친다. 에피소드는 소스 순서·번호 순으로 0..N-1 재번호.

    이미 내용이 있는 out_dir 은 거부, 설정(COMPAT_KEYS)이 다른 세션도 거부한다.
    """
    if not sources:
        raise ValueError('합칠 세션이 없음')
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise FileExistsError(f'대상 폴더에 이미 내용이 있음: {out_dir}')
    infos = []
    for s in sources:
        if not os.path.isfile(os.path.join(s, 'session.json')):
            raise FileNotFoundError(f'session.json 없음: {s}')
        infos.append(read_json(os.path.join(s, 'session.json')))
    base = infos[0]
    for s, info in zip(sources, infos):
        diff = [k for k in COMPAT_KEYS if info.get(k) != base.get(k)]
        if diff:
            raise ValueError(f'설정이 다른 세션은 합칠 수 없음: {os.path.basename(s)} ({", ".join(diff)})')
    per_source = [list_episode_dirs(s) for s in sources]
    if not any(per_source):
        raise ValueError('에피소드가 하나도 없음')

    created_dir = not os.path.exists(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    try:
        n = 0
        for eps in per_source:
            for _, path in eps:
                dest = os.path.join(out_dir, f'ep_{n:04d}')
                (shutil.move if move else shutil.copytree)(path, dest)
                n += 1
        merged = dict(base)
        merged['product'] = '+'.join(dict.fromkeys(i.get('product', '') for i in infos))
        merged['created'] = datetime.now().astimezone().isoformat(timespec='seconds')
        merged['merged_from'] = [
            {'session': os.path.basename(s), 'product': i.get('product'),
             'created': i.get('created'), 'cameras': i.get('cameras'), 'episodes': len(e)}
            for s, i, e in zip(sources, infos, per_source)]
        with open(os.path.join(out_dir, 'session.json'), 'w', encoding='utf-8') as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
    except Exception:
        # 복사 모드의 실패만 되돌릴 수 있다(이동은 이미 소스에서 빠져나감)
        if not move:
            shutil.rmtree(out_dir, ignore_errors=True)
            if not created_dir:
                os.makedirs(out_dir, exist_ok=True)
        raise
    return n
