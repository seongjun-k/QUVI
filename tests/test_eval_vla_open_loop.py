"""개루프 평가의 떨림 지표와 에피소드 선택을 확인한다."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    'eval_vla_open_loop', Path(__file__).resolve().parent.parent / 'scripts' / 'eval_vla_open_loop.py')
evaluator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluator)


def test_reversal_rate_counts_direction_flips_only():
    assert evaluator.reversal_rate([0, 1, 2, 3]) == 0.0
    assert evaluator.reversal_rate([0, 1, 0, 1]) == 1.0
    assert evaluator.reversal_rate([0, 1, 1, 2]) == 0.0  # 정지 구간은 방향 변화가 아니다
    assert evaluator.reversal_rate([5, 5]) == 0.0


def test_pick_episodes_spans_dataset_without_duplicates():
    assert evaluator.pick_episodes(218, 3) == [0, 108, 217]
    assert evaluator.pick_episodes(2, 10) == [0, 1]
