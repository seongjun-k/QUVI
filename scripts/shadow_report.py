#!/usr/bin/env python3
"""
QUVI 섀도우 모드 리포트
──────────────────────
data/inspection_logs/*/result.txt 를 순회해 룰 판정과 ML 판정(섀도우, 참고용)의
일치율/불일치 케이스를 집계한다. passed 판정 자체에는 영향 없음 — 순수 리포팅.

참고: docs/ml_anomaly_inspection_plan.md §4 Phase 3
사용법: python3 scripts/shadow_report.py [--log-dir DIR]
"""
import argparse
import glob
import os
import re


def _parse_result_txt(path):
    """result.txt 한 개를 파싱해 dict 반환 (누락 필드는 None)."""
    with open(path, encoding='utf-8') as f:
        text = f.read()
    fields = dict(re.findall(r'^(판정|ML판정|ML점수\(worst\)):\s*(\S+)', text, re.MULTILINE))
    rule = fields.get('판정')
    ml = fields.get('ML판정')
    ml_score_raw = fields.get('ML점수(worst)')
    ml_score = None
    if ml_score_raw is not None and ml_score_raw != 'N/A':
        try:
            ml_score = float(ml_score_raw)
        except ValueError:
            pass
    return {'rule': rule, 'ml': ml, 'ml_score': ml_score}


def _fmt(s):
    return f'{s:.2f}' if s is not None else 'N/A'


def main():
    parser = argparse.ArgumentParser(description='QUVI 섀도우 모드 룰 vs ML 리포트')
    parser.add_argument('--log-dir', default='/workspace/data/inspection_logs')
    args = parser.parse_args()

    result_files = sorted(glob.glob(os.path.join(args.log_dir, '*', 'result.txt')))
    total = len(result_files)

    ml_recorded = 0
    agree = 0
    disagree_cases = []

    for path in result_files:
        folder = os.path.basename(os.path.dirname(path))
        info = _parse_result_txt(path)
        if info['ml'] is None or info['ml'] == 'N/A':
            continue
        ml_recorded += 1
        if info['rule'] == info['ml']:
            agree += 1
        else:
            disagree_cases.append((folder, info['rule'], info['ml'], info['ml_score']))

    print(f'총 검사 건수: {total}')
    print(f'ML 기록 건수: {ml_recorded}')
    if ml_recorded == 0:
        print('ML 기록이 있는 검사 로그가 없습니다 (anomaly_enabled=False 였거나 아직 섀도우 실행 전).')
        return

    print(f'룰·ML 일치율: {agree}/{ml_recorded} ({100.0 * agree / ml_recorded:.1f}%)')

    print(f'\n불일치 목록 ({len(disagree_cases)}건):')
    for folder, rule, ml, score in disagree_cases:
        print(f'  {folder} | 룰={rule} ML={ml} score={_fmt(score)}')

    false_accept = [(f, s) for f, r, m, s in disagree_cases if r == 'FAIL' and m == 'PASS']
    false_reject = [(f, s) for f, r, m, s in disagree_cases if r == 'PASS' and m == 'FAIL']

    print(f'\n룰FAIL·MLPASS (false-accept 후보, 컷오버 게이트 §4 대상): {len(false_accept)}건')
    for folder, score in false_accept:
        print(f'  {folder} | score={_fmt(score)}')

    print(f'\n룰PASS·MLFAIL (false-reject 후보): {len(false_reject)}건')
    for folder, score in false_reject:
        print(f'  {folder} | score={_fmt(score)}')


if __name__ == '__main__':
    main()
