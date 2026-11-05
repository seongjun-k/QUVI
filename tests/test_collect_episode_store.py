"""수집 도구 편집·판정 로직 테스트 (rclpy·Flask 불필요)."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts', 'collect'))
import episode_store as st  # noqa: E402
from quvi_robot_control.act_common import JOINT_NAMES, POSE_HOME, raw_to_rad  # noqa: E402

HOME = [POSE_HOME[j] for j in JOINT_NAMES]


def _make_ep(sdir, idx, tag=b'x'):
    d = os.path.join(sdir, f'ep_{idx:04d}')
    os.makedirs(os.path.join(d, 'camera1'))
    with open(os.path.join(d, 'camera1', '000000.jpg'), 'wb') as f:
        f.write(tag)
    with open(os.path.join(d, 'meta.json'), 'w') as f:
        json.dump({'verdict': 'ok', 'warnings': []}, f)


def _make_session(root, sid, n, product='box', **override):
    sdir = os.path.join(root, sid)
    os.makedirs(sdir)
    info = st.new_session_json(product)
    info.update(override)
    with open(os.path.join(sdir, 'session.json'), 'w') as f:
        json.dump(info, f)
    for i in range(n):
        _make_ep(sdir, i, f'{sid}-{i}'.encode())
    return sdir


def _names(sdir):
    return [i for i, _ in st.list_episode_dirs(sdir)]


def test_delete_without_renumber_keeps_gap(tmp_path):
    s = _make_session(str(tmp_path), 'a', 4)
    assert st.delete_episodes(s, [1, 2]) == 2
    assert _names(s) == [0, 3]


def test_delete_with_renumber_compacts(tmp_path):
    s = _make_session(str(tmp_path), 'a', 5)
    st.delete_episodes(s, [0, 3], renumber=True)
    assert _names(s) == [0, 1, 2]
    # 남은 원본은 1,2,4 → 새 번호 0,1,2 (내용 순서 보존)
    tags = [open(os.path.join(s, f'ep_{i:04d}', 'camera1', '000000.jpg'), 'rb').read() for i in range(3)]
    assert tags == [b'a-1', b'a-2', b'a-4']


def test_delete_missing_index_deletes_nothing(tmp_path):
    s = _make_session(str(tmp_path), 'a', 3)
    with pytest.raises(ValueError):
        st.delete_episodes(s, [1, 9])
    assert _names(s) == [0, 1, 2]


def test_merge_copy_keeps_sources_and_records_origin(tmp_path):
    r = str(tmp_path)
    a, b = _make_session(r, 'a', 2, 'box'), _make_session(r, 'b', 3, 'gear')
    out = os.path.join(r, 'm')
    assert st.merge_sessions([a, b], out) == 5
    assert _names(out) == [0, 1, 2, 3, 4]
    assert _names(a) == [0, 1] and _names(b) == [0, 1, 2]
    info = json.load(open(os.path.join(out, 'session.json')))
    assert [m['session'] for m in info['merged_from']] == ['a', 'b']
    assert info['product'] == 'box+gear'
    assert open(os.path.join(out, 'ep_0002', 'camera1', '000000.jpg'), 'rb').read() == b'b-0'


def test_merge_move_empties_sources(tmp_path):
    r = str(tmp_path)
    a, b = _make_session(r, 'a', 2), _make_session(r, 'b', 1)
    out = os.path.join(r, 'm')
    assert st.merge_sessions([a, b], out, move=True) == 3
    assert _names(a) == [] and _names(b) == []
    assert _names(out) == [0, 1, 2]


def test_merge_rejects_nonempty_target(tmp_path):
    r = str(tmp_path)
    a = _make_session(r, 'a', 1)
    out = os.path.join(r, 'm')
    os.makedirs(out)
    open(os.path.join(out, 'keep.txt'), 'w').close()
    with pytest.raises(FileExistsError):
        st.merge_sessions([a], out)
    assert os.listdir(out) == ['keep.txt']
    assert _names(a) == [0]


def test_merge_accepts_empty_existing_target(tmp_path):
    r = str(tmp_path)
    a = _make_session(r, 'a', 1)
    out = os.path.join(r, 'm')
    os.makedirs(out)
    assert st.merge_sessions([a], out) == 1


@pytest.mark.parametrize('key,val', [('dataset_fps', 30), ('episode_sec', 6.0),
                                     ('task', 'Pick up the printed part from the bed')])
def test_merge_rejects_config_mismatch(tmp_path, key, val):
    r = str(tmp_path)
    a, b = _make_session(r, 'a', 1), _make_session(r, 'b', 1, **{key: val})
    out = os.path.join(r, 'm')
    with pytest.raises(ValueError):
        st.merge_sessions([a, b], out)
    assert not os.path.exists(out)


def test_write_episode_roundtrip_and_numbering(tmp_path):
    s = _make_session(str(tmp_path), 'a', 0)
    state = [(i / 30 - 0.2, [raw_to_rad(h) for h in HOME], [raw_to_rad(h) for h in HOME]) for i in range(10)]
    cams = {c: [(i / 30, b'\xff\xd8jpg%d' % i) for i in range(3)] for c in st.CAMERAS}
    meta = st.compute_meta('2026-01-01T00:00:00.000+09:00', state, cams)
    assert st.write_episode(s, state, cams, meta) == 0
    assert st.write_episode(s, state, cams, meta) == 1
    ep = os.path.join(s, 'ep_0000')
    header = open(os.path.join(ep, 'states.csv')).readline().strip().split(',')
    assert header[0] == 't' and len(header) == 13 and header[1] == 'present_shoulder_pan'
    assert open(os.path.join(ep, 'camera3', '000002.jpg'), 'rb').read() == b'\xff\xd8jpg2'
    assert open(os.path.join(ep, 'camera1', 'stamps.csv')).readline().strip() == 'idx,t'
    assert not [n for n in os.listdir(s) if n.endswith('.tmp')]


def _traj(shift_end=0, step=1 / 30):
    """t∈[-0.2,5.5] 30Hz. 3.0초에 홈에서 먼 자세로 갔다가 4.4초에 복귀, 그리퍼는 닫힘."""
    out, t = [], -0.2
    while t <= st.EPISODE_DURATION_SEC:
        away = 1.0 <= t < 4.4
        raw = [h + (300 if away else shift_end) for h in HOME[:5]] + [1800]
        out.append((t, [raw_to_rad(r) for r in raw], [raw_to_rad(r) for r in raw]))
        t += step
    return out


def _cams(gap=None):
    f = [(i / 30 - 0.2, b'j') for i in range(int(5.7 * 30))]
    if gap:
        f = [x for x in f if not gap[0] <= x[0] <= gap[1]]
    return {c: f for c in st.CAMERAS}


def test_meta_ok():
    m = st.compute_meta('w', _traj(), _cams())
    assert m['verdict'] == 'ok' and m['gripper_closed'] is True
    assert abs(m['return_time_sec'] - 4.4) < 0.05 and abs(m['speed_dev_sec'] + 0.1) < 0.05
    assert len(m['start_dev_raw']) == 6 and len(m['end_dev_raw']) == 5


def test_meta_warn_when_not_returned():
    m = st.compute_meta('w', _traj(shift_end=100), _cams())
    assert m['verdict'] == 'warn' and m['return_time_sec'] is None and m['speed_dev_sec'] is None


def test_meta_bad_on_camera_gap():
    m = st.compute_meta('w', _traj(), _cams(gap=(2.0, 2.8)))
    assert m['verdict'] == 'bad' and any('camera1' in w for w in m['warnings'])


def test_meta_bad_on_state_gap_and_empty():
    state = [s for s in _traj() if not 2.0 <= s[0] <= 2.8]
    assert st.compute_meta('w', state, _cams())['verdict'] == 'bad'
    assert st.compute_meta('w', [], _cams())['verdict'] == 'bad'


def test_review_mark_and_clear(tmp_path):
    s = _make_session(str(tmp_path), 'a', 2)
    st.set_review(s, 1, 'discard')
    eps = st.list_sessions(str(tmp_path))[0]['episodes']
    assert [e.get('review') for e in eps] == [None, 'discard']
    assert eps[1]['verdict'] == 'ok'  # 기존 판정은 보존
    st.set_review(s, 1, None)
    assert 'review' not in st.read_json(os.path.join(s, 'ep_0001', 'meta.json'))
    with pytest.raises(ValueError):
        st.set_review(s, 0, 'maybe')


def test_playback_stamps_and_numbered_frame(tmp_path):
    s = _make_session(str(tmp_path), 'a', 1)
    for cam in st.CAMERAS:
        os.makedirs(os.path.join(s, 'ep_0000', cam), exist_ok=True)
        with open(os.path.join(s, 'ep_0000', cam, 'stamps.csv'), 'w') as f:
            f.write('idx,t\n0,-0.100000\n1,0.033000\n')
    assert st.episode_stamps(s, 0)['camera1'] == [-0.1, 0.033]
    assert st.frame_path(s, 0, 'camera1', '0').endswith('000000.jpg')
    with pytest.raises(FileNotFoundError):
        st.frame_path(s, 0, 'camera1', '7')


def test_start_tolerance_gripper_only():
    assert [st.start_tol(i) for i in range(6)] == [20] * 5 + [50]
    assert st.start_aligned([20] * 5 + [-43])
    assert st.start_aligned([-20] * 5 + [50])
    assert not st.start_aligned([21] + [0] * 5)
    assert not st.start_aligned([0] * 5 + [-51])
    assert not st.start_aligned([0] * 5)
    assert st.new_session_json('box')['start_tol_gripper_raw'] == 50
