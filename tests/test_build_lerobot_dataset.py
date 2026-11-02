"""합성 세션 2개 → 변환 → info.json features 를 기존 Cyclo 데이터셋과 비교.

도구 venv(lerobot 0.6.1) python 으로 실행한다:
  docker exec quvi-dev bash -c 'cd /workspace && /workspace/data/collect_tools/venv/bin/python -m pytest -q tests/test_build_lerobot_dataset.py'
lerobot 이 없으면 skip.
"""
import ast
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest


def _has_lerobot_06():
    # lerobot.configs.video 는 0.6.x 에만 있다 — 시스템 python 의 구버전 lerobot 도 skip 대상
    try:
        return importlib.util.find_spec('lerobot.configs.video') is not None
    except ModuleNotFoundError:
        return False


# 모듈 수준 importorskip 은 tests/ 전체 수집을 중단시켜 쓰지 않는다
pytestmark = pytest.mark.skipif(not _has_lerobot_06(), reason='lerobot 0.6.x(도구 venv) 필요')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import build_lerobot_dataset as bld  # noqa: E402
from quvi_robot_control.act_common import JOINT_NAMES, POSE_HOME, raw_to_rad  # noqa: E402

CYCLO_INFO = ROOT / 'data/datasets/Task_1_pick_quvi_MCAP_lerobot_v30/meta/info.json'
# 기존 Cyclo info.json 에서 비교 대상만 옮긴 스냅샷(데이터셋 사본이 없는 곳에서도 비교하도록)
VEC = {'dtype': 'float32', 'shape': [6], 'names': ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'gripper_joint_1']}
VID = {'dtype': 'video', 'shape': [3, 480, 640], 'names': ['channels', 'height', 'width'],
       'info': {'video.fps': 15.0, 'video.height': 480, 'video.width': 640, 'video.channels': 3,
                'video.codec': 'h264', 'video.pix_fmt': 'yuv420p', 'video.is_depth_map': False, 'has_audio': False}}
REF_FEATURES = {'observation.state': VEC, 'action': VEC,
                'observation.images.rgb.camera1': VID, 'observation.images.rgb.camera3': VID}


def make_session(root, name, n_eps, seed):
    """형식 문서대로: 카메라 30fps·상태 30Hz, t = -0.2~5.5."""
    import cv2
    rng = np.random.default_rng(seed)
    sd = root / name
    sd.mkdir(parents=True)
    (sd / 'session.json').write_text(json.dumps({
        'product': 'box', 'task': bld.TASK, 'dataset_fps': 15, 'record_state_hz': 30,
        'episode_sec': 5.5, 'pre_roll_sec': 0.2}), encoding='utf-8')
    home = np.array([raw_to_rad(POSE_HOME[j]) for j in JOINT_NAMES])
    for e in range(n_eps):
        ep = sd / f'ep_{e:04d}'
        ts = np.arange(-0.2, 5.5 + 1e-9, 1 / 30)
        move = np.clip((ts - 0.5) / 4.0, 0, 1)[:, None] * rng.uniform(-0.3, 0.3, 6)
        present = home + move
        goal = home + np.clip((ts + 0.1 - 0.5) / 4.0, 0, 1)[:, None] * (move[-1] if len(move) else 0)
        rows = np.column_stack([ts, present, goal])
        ep.mkdir()
        np.savetxt(ep / 'states.csv', rows, delimiter=',', header=','.join(bld.STATES_HEADER), comments='')
        for cam in bld.CAMERAS:
            (ep / cam).mkdir()
            ct = ts + rng.uniform(0, 0.005, len(ts))
            for i in range(len(ct)):
                img = np.full((480, 640, 3), (i * 3) % 256, np.uint8)
                (ep / cam / f'{i:06d}.jpg').write_bytes(cv2.imencode('.jpg', img)[1].tobytes())
            np.savetxt(ep / cam / 'stamps.csv', np.column_stack([np.arange(len(ct)), ct]), delimiter=',',
                       header='idx,t', comments='', fmt=['%d', '%.6f'])
    return sd


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    raw = tmp_path_factory.mktemp('raw')
    s1, s2 = make_session(raw, '20261107_100000_box', 2, 1), make_session(raw, '20261107_110000_box', 1, 2)
    out = tmp_path_factory.mktemp('out') / 'ds'
    report, n_frames = bld.write_dataset([s1, s2], out, 15, 'local/test_ds')
    n_tasks = bld.verify_load(out, 'local/test_ds', len(report), n_frames)
    return out, report, n_tasks, (s1, s2)


def test_features_match_cyclo(built):
    out = built[0]
    info = json.loads((out / 'meta/info.json').read_text())
    assert (info['fps'], info['robot_type'], info['codebase_version']) == (15, 'omx_f', 'v3.0')
    refs = [(REF_FEATURES, 'snapshot')]
    if CYCLO_INFO.exists():
        live = json.loads(CYCLO_INFO.read_text())['features']
        refs.append(({k: live[k] for k in REF_FEATURES}, 'live'))
    for ref, label in refs:
        for key, rf in ref.items():
            f = info['features'][key]
            assert (f['dtype'], f['shape'], f['names']) == (rf['dtype'], rf['shape'], rf['names']), (label, key)
            if rf['dtype'] == 'video':
                for k in ('video.fps', 'video.height', 'video.width', 'video.channels', 'video.codec', 'video.pix_fmt'):
                    assert f['info'][k] == rf['info'][k], (label, key, k)
    assert 'subtask_index' not in info['features']   # Cyclo 변환기 전용 — 만들지 않는다


def test_quality_report_and_counts(built):
    _, report, n_tasks, _ = built
    assert n_tasks == 1 and len(report) == 3
    assert all(q['length'] == 83 and not q['warnings'] for q in report)
    assert all(q['first_move_frame'] is not None for q in report)


def test_sidecar_input_keys_match(built):
    """사이드카 _build_batch 가 요구하는 키(CAMERA_POLICY_KEYS, STATE_KEY)와 데이터셋 프레임 키 일치."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    src = (ROOT / 'scripts/vla_sidecar/server.py').read_text(encoding='utf-8')
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.Assign) and n.targets[0].id == 'CAMERA_POLICY_KEYS')
    policy_keys = ast.literal_eval(node.value)
    assert policy_keys == bld.CAMERAS
    from lerobot.utils.constants import OBS_STATE
    item = LeRobotDataset('local/test_ds', root=built[0])[0]
    for key in policy_keys.values():
        assert tuple(item[key].shape) == (3, 480, 640) and 0.0 <= float(item[key].min()) and float(item[key].max()) <= 1.0
    assert tuple(item[OBS_STATE].shape) == (6,)


def test_rejects_wrong_task_and_existing_out(built, tmp_path):
    out, _, _, (s1, _) = built
    with pytest.raises(SystemExit, match='이미 있음'):
        bld.write_dataset([s1], out, 15, 'local/x')
    meta = json.loads((s1 / 'session.json').read_text(encoding='utf-8'))
    bad = tmp_path / 'bad'
    (bad / 'ep_0000').mkdir(parents=True)
    (bad / 'session.json').write_text(json.dumps(dict(meta, task=bld.TASK.rstrip('.'))), encoding='utf-8')
    with pytest.raises(SystemExit, match='지시문 불일치'):
        bld.check_sessions([bad])
    other = tmp_path / 'other'
    (other / 'ep_0000').mkdir(parents=True)
    (other / 'session.json').write_text(json.dumps(dict(meta, dataset_fps=30)), encoding='utf-8')
    with pytest.raises(SystemExit, match='세션 설정 불일치'):
        bld.check_sessions([s1, other])
