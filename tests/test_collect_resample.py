"""격자 리샘플 순수 함수 — numpy 만 필요(venv 불필요)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_lerobot_dataset import resample_indices, sample_ages  # noqa: E402


def test_at_or_before_selection():
    t = np.array([-0.2, 0.0, 0.05, 0.07, 0.2])
    idx, clamped = resample_indices(t, n_frames=4, fps=15)   # g = 0, 1/15, 2/15, 3/15
    # g0=0 → t=0.0(정확히 일치 포함), g1=0.0667 → 0.05, g2=0.133 → 0.07, g3=0.2 → 0.2(경계 포함)
    assert idx.tolist() == [1, 2, 3, 4]
    assert clamped == 0


def test_clamp_when_no_sample_before():
    t = np.array([0.1, 0.2, 0.3])
    idx, clamped = resample_indices(t, n_frames=4, fps=15)   # g0=0, g1=0.067 은 첫 샘플(0.1) 이전
    assert idx.tolist() == [0, 0, 0, 1]      # g3=0.2 는 t=0.2 와 일치 → 그 샘플
    assert clamped == 2                       # g0, g1 만 0.1 이전 (g2=0.133 은 0.1 이후라 정상 선택)
    assert (sample_ages(t, idx, 15)[:2] < 0).all()


def test_default_grid_is_83_frames_and_holds_last_sample():
    t = np.arange(-0.2, 5.5, 1 / 30)                          # 카메라 30fps, t -0.2~5.5
    idx, clamped = resample_indices(t)
    assert len(idx) == 83 and clamped == 0
    assert (np.diff(idx) > 0).all()
    assert sample_ages(t, idx).max() < 1 / 30 + 1e-9          # 30fps 소스면 샘플 나이는 한 프레임 미만


def test_discarded_episodes_are_skipped(tmp_path):
    import json
    from build_lerobot_dataset import load_session
    (tmp_path / 'session.json').write_text('{}')
    for i, review in enumerate([None, 'discard', 'keep']):
        ep = tmp_path / f'ep_{i:04d}'
        ep.mkdir()
        (ep / 'meta.json').write_text(json.dumps({'review': review} if review else {}))
    _, eps = load_session(tmp_path)
    assert [p.name for p in eps] == ['ep_0000', 'ep_0002']


def test_start_warning_gripper_tolerance(tmp_path):
    import build_lerobot_dataset as b
    from quvi_robot_control.act_common import raw_to_rad
    home = np.array([raw_to_rad(b.RECORD_START_POSE[j]) for j in b.JOINT_NAMES])
    for cam in b.CAMERAS:
        (tmp_path / cam).mkdir()
        (tmp_path / cam / 'stamps.csv').write_text('idx,t\n0,0\n')
        (tmp_path / cam / '000000.jpg').touch()
    for index, delta, warned in [(5, -43, False), (5, -51, True), (0, 75, False), (0, 81, True)]:
        pose = home.copy()
        pose[index] += delta * 2 * np.pi / 4096
        row = '0,' + ','.join(map(str, list(pose) * 2))
        (tmp_path / 'states.csv').write_text(','.join(b.STATES_HEADER) + '\n' + row + '\n')
        q = b.build_episode(tmp_path, 1, 15, home)[3]
        assert any(w.startswith('시작 편차') for w in q['warnings']) == warned
