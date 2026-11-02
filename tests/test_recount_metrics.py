"""README 주요 수치가 저장소 원자료에서 그대로 재현되는지 확인한다."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    'recount_metrics', Path(__file__).resolve().parent.parent / 'scripts' / 'recount_metrics.py')
recount = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recount)


@pytest.mark.skipif(not recount.METRICS_DIR.exists(), reason='quvi-dev 컨테이너는 docs/ 를 마운트하지 않는다')
def test_readme_numbers_and_full_totals():
    inspections, grasps = recount.load_inspections(), recount.load_grasps()
    snapshot = recount.summarize(
        [r for r in inspections if r[0] < recount.README_CUTOFF],
        [g for g in grasps if g[0] < recount.README_CUTOFF])
    assert snapshot == {'cycles': 133, 'agree': 91, 'compared': 103, 'recent_agree': 13,
                        'recent_compared': 13, 'grasp_ok': 11, 'grasp_fail': 2, 'grasp_manual_stop': 4}
    full = recount.summarize(inspections, grasps)
    assert (full['cycles'], full['agree'], full['compared'], full['grasp_ok']) == (137, 95, 107, 15)
