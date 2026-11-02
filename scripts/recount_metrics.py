#!/usr/bin/env python3
"""README 주요 수치를 docs/metrics 원자료에서 다시 센다. 사용법: python3 scripts/recount_metrics.py"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

METRICS_DIR = Path(__file__).resolve().parent.parent / 'docs' / 'metrics'
# README 수치(커밋 74145ef)는 07-30 마지막 두 세션 전 기록과 일치한다. 로그 시각은 UTC.
README_CUTOFF = datetime(2026, 7, 30, 11, 43, tzinfo=timezone.utc)
AGREE_SINCE = datetime(2026, 7, 15, tzinfo=timezone.utc)
GRASP_RE = re.compile(r'\[(\d+\.\d+)\].*GRASPING_WAIT -> FsmState\.([A-Z_]+)')


def load_inspections(metrics_dir=METRICS_DIR):
    """검사 1회당 (시각, result.json 내용)."""
    rows = []
    for path in sorted((metrics_dir / 'inspection_results').glob('*/result.json')):
        when = datetime.strptime(path.parent.name[:15], '%Y%m%d_%H%M%S').replace(tzinfo=timezone.utc)
        rows.append((when, json.loads(path.read_text(encoding='utf-8'))))
    return rows


def load_grasps(metrics_dir=METRICS_DIR):
    """'파지 대기' 다음 상태 1건당 (시각, 상태)."""
    rows = []
    for path in sorted((metrics_dir / 'grasp_ros_logs').glob('*.log')):
        for match in GRASP_RE.finditer(path.read_text(encoding='utf-8', errors='replace')):
            when = datetime.fromtimestamp(float(match.group(1)), tz=timezone.utc)
            rows.append((when, match.group(2)))
    return rows


def summarize(inspections, grasps):
    # ML 판정이 없는 기록(07-08 이전)은 일치율 분모에서 뺀다.
    compared = [(t, r) for t, r in inspections if r.get('ml_passed') is not None]
    recent = [r for t, r in compared if t >= AGREE_SINCE]
    states = [s for _, s in grasps]
    return {
        'cycles': len(inspections),
        'agree': sum(r['passed'] == r['ml_passed'] for _, r in compared),
        'compared': len(compared),
        'recent_agree': sum(r['passed'] == r['ml_passed'] for r in recent),
        'recent_compared': len(recent),
        'grasp_ok': states.count('PLACING_CHAMBER_TRIGGER'),
        'grasp_fail': states.count('ERROR'),
        # IDLE 복귀 = 사람이 시험을 멈춤. 로봇 실패가 아니라 성공률 분모에서 뺀다.
        'grasp_manual_stop': states.count('IDLE'),
    }


def report(name, s):
    attempts = s['grasp_ok'] + s['grasp_fail']
    with_stops = attempts + s['grasp_manual_stop']
    print(f"[{name}]")
    print(f"  검사 사이클: {s['cycles']}회")
    print(f"  룰-ML 일치: {s['agree']}/{s['compared']} = {s['agree'] / s['compared'] * 100:.1f}% "
          f"(07-15 이후 {s['recent_agree']}/{s['recent_compared']})")
    print(f"  파지 성공: {s['grasp_ok']}/{attempts} = {s['grasp_ok'] / attempts * 100:.1f}% "
          f"(수동 중단 {s['grasp_manual_stop']}건 포함 시 "
          f"{s['grasp_ok']}/{with_stops} = {s['grasp_ok'] / with_stops * 100:.1f}%)")


def main():
    inspections, grasps = load_inspections(), load_grasps()
    report('README 집계 범위 (07-30 11:43 UTC 이전)', summarize(
        [r for r in inspections if r[0] < README_CUTOFF], [g for g in grasps if g[0] < README_CUTOFF]))
    report('원자료 전체', summarize(inspections, grasps))


if __name__ == '__main__':
    main()
