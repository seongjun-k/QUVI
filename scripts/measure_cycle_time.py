#!/usr/bin/env python3
"""북극성 지표 집계 — '시작(출력 완료 신호) → 분류 완료' 사이클 시간.

main_orchestrator_node 의 FSM 전이 로그([FSM] 상태 변경: A -> B)에서
IDLE→STARTUP_RAIL_HOME_TRIGGER 를 시작, FINISHED 진입을 종료로 본다.
중간에 ERROR/IDLE 로 빠진 사이클은 집계에서 뺀다.
시연에서는 HMI 시작 버튼이 출력 완료 신호를 대신하므로 이 시간이
'출력 완료 → 판정·분류 완료' 시간이다.

한계: 로그만으로는 실기 사이클과 모의(use_real_hardware=false) 사이클을 구분하지 못한다.

실행 위치: quvi-dev 컨테이너
  python3 /workspace/scripts/measure_cycle_time.py [--log-dir /root/.ros/log] [--since 2026-09-01]
"""
import argparse
import datetime as dt
import glob
import os
import re
import statistics as st

FSM_RE = re.compile(r'\[(\d+\.\d+)\] \[main_orchestrator_node\]: \[FSM\] 상태 변경: '
                    r'FsmState\.(\w+) -> FsmState\.(\w+)')
# (표시 이름, 구간 시작 키, 구간 끝 키)
PHASES = [('시작 → 파지 시작', 'start', 'grasp'), ('파지 시작 → 검사 시작', 'grasp', 'insp'),
          ('검사 시작 → 분류 시작', 'insp', 'sort'), ('분류 시작 → 종료', 'sort', 'end'),
          ('전체 (시작 → 종료)', 'start', 'end')]
MARKS = {'GRASPING_TRIGGER': 'grasp', 'INSPECTING_TRIGGER': 'insp', 'SORTING_TRIGGER': 'sort'}


def collect(log_dir):
    cycles = []
    for path in glob.glob(os.path.join(log_dir, '*.log')):
        events = []
        with open(path, encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = FSM_RE.search(line)
                if m:
                    events.append((float(m[1]), m[2], m[3]))
        cur = None
        for t, src, dst in sorted(events):
            if src == 'IDLE' and dst == 'STARTUP_RAIL_HOME_TRIGGER':
                cur = {'start': t}
            elif cur is not None:
                if dst in MARKS:
                    cur.setdefault(MARKS[dst], t)
                if dst == 'FINISHED':
                    cur['end'] = t
                    cycles.append(cur)
                    cur = None
                elif dst in ('ERROR', 'IDLE'):
                    cur = None
    return [c for c in cycles if all(k in c for k in ('grasp', 'insp', 'sort'))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log-dir', default='/root/.ros/log')
    ap.add_argument('--since', help='YYYY-MM-DD 이후 시작한 사이클만')
    a = ap.parse_args()
    cycles = collect(a.log_dir)
    if a.since:
        since = dt.datetime.strptime(a.since, '%Y-%m-%d').timestamp()
        cycles = [c for c in cycles if c['start'] >= since]
    if not cycles:
        print('완주 사이클 없음')
        return
    days = sorted({dt.datetime.fromtimestamp(c['start']).strftime('%Y-%m-%d') for c in cycles})
    print(f'완주 사이클 {len(cycles)}회 · {days[0]} ~ {days[-1]} ({len(days)}일)')
    for name, a_key, b_key in PHASES:
        v = [c[b_key] - c[a_key] for c in cycles]
        print(f'{name}: 중앙값 {st.median(v):.0f}s  최소 {min(v):.0f}s  최대 {max(v):.0f}s')


if __name__ == '__main__':
    main()
