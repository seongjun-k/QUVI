"""ROS 없이 메타데이터 분포와 빈 입력을 확인한다."""
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/collect/threshold_report.py'


def run(root):
    return subprocess.run([sys.executable, str(SCRIPT), str(root)], capture_output=True, text=True)


def test_empty(tmp_path):
    result = run(tmp_path)
    assert result.returncode == 0
    assert '에피소드 0건' in result.stdout


def test_distribution_and_reviews(tmp_path):
    for i, review in enumerate(['keep', 'discard', None]):
        ep = tmp_path / 'session' / f'ep_{i:04d}'
        ep.mkdir(parents=True)
        (ep / 'meta.json').write_text(json.dumps(dict(
            start_dev_raw=[-i, 2, 3, 4, 5, -43], end_dev_raw=[1]*5,
            return_time_sec=i, speed_dev_sec=None if i == 0 else -i,
            gripper_closed=True, verdict='ok', warnings=['경고'], review=review)))
    result = run(tmp_path)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data['전체']['return_time_sec'] == {'개수': 3, '중앙값': 1, 'p95': 2, '최대': 2}
    assert data['전체']['start_dev_raw']['gripper']['최대'] == 43
    assert data['전체']['speed_dev_sec']['개수'] == 2
    assert data['전체']['warnings'] == {'경고': 3}
    assert data['사용']['에피소드'] == data['안 씀']['에피소드'] == data['미정']['에피소드'] == 1
    assert data['현재 기준']['START_TOL_RAW'] == 80
