#!/usr/bin/env python3
"""원본 meta.json 분포를 읽기 전용으로 출력한다. p95는 최근접 순위 방식이다.

사용: python3 scripts/collect/threshold_report.py [data/episodes_raw 경로]
현재 기준은 소스에서 읽으며, 과거 verdict를 재판정하지 않는다.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import median
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src/quvi_robot_control'))
import episode_store as store  # noqa: E402


def stats(values):
    values = sorted(v for v in values if isinstance(v, (int, float))
                    and not isinstance(v, bool) and math.isfinite(v))
    return {'개수': len(values), '중앙값': median(values) if values else None,
            'p95': values[math.ceil(len(values) * .95) - 1] if values else None,
            '최대': max(values) if values else None}


def summarize(rows):
    out = {'에피소드': len(rows)}
    for field in ('start_dev_raw', 'end_dev_raw'):
        out[field] = {}
        for i, joint in enumerate(store.JOINT_NAMES[:6 if field == 'start_dev_raw' else 5]):
            out[field][joint] = stats(abs(r[field][i]) for r in rows
                                     if isinstance(r.get(field), list) and len(r[field]) > i
                                     and isinstance(r[field][i], (int, float)))
    for field in ('return_time_sec', 'speed_dev_sec'):
        out[field] = stats(r.get(field) for r in rows)
    for field in ('gripper_closed', 'verdict'):
        out[field] = dict(Counter(str(r.get(field)) for r in rows))
    out['warnings'] = dict(Counter(w for r in rows for w in (r.get('warnings') or [])))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path, default=REPO / 'data/episodes_raw')
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.glob('*/ep_[0-9][0-9][0-9][0-9]/meta.json')):
        try:
            row = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(row, dict):
                raise ValueError('객체가 아님')
            rows.append(row)
        except (OSError, ValueError) as exc:
            parser.exit(1, f'meta.json 읽기 실패: {path}: {exc}\n')
    if not rows:
        print('에피소드 0건')
        return
    names = ('START_TOL_RAW', 'START_TOL_GRIPPER_RAW', 'END_TOL_RAW', 'START_HOLD_SEC',
             'TARGET_RETURN_SEC', 'SPEED_DEV_WARN_SEC', 'GRIPPER_CLOSED_MARGIN',
             'GRIPPER_OPEN', 'MAX_GAP_SEC', 'EPISODE_DURATION_SEC')
    out = {'현재 기준': {n: getattr(store, n) for n in names if hasattr(store, n)},
           '비교 안내': '편차는 절대값, 시간은 부호 유지. 시작/끝 raw는 허용치 이하, '
                      'speed_dev_sec는 절대값이 경고 기준 초과인지 비교. '
                      '그리퍼 닫힘은 raw < GRIPPER_OPEN - GRIPPER_CLOSED_MARGIN. '
                      '현재 기준이며 과거 verdict와 다를 수 있음. p95=최근접 순위.',
           '전체': summarize(rows)}
    if any(r.get('review') for r in rows):
        for value, label in [('keep', '사용'), ('discard', '안 씀'), (None, '미정')]:
            out[label] = summarize([r for r in rows if r.get('review') == value])
    print(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
