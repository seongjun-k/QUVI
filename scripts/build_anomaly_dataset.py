#!/usr/bin/env python3
"""
QUVI 이상탐지 데이터셋 빌더 (Phase 0-a)
──────────────────────────────────
`inspection_logs`의 `*_PASS` 폴더(룰 판정기가 PASS 로 붙인 검사 로그)를
각도별로 모아 `anomaly_dataset/raw/{angle}/`로 정리하고, 사람이 눈으로
불량 혼입을 검수할 수 있도록 각도별 썸네일 그리드 시트를 생성한다.

배경(docs/ml_anomaly_inspection_plan.md §0-3, Phase 0):
  수집되는 "정상" 라벨은 이번에 교체하려는 바로 그 룰 판정기가 붙인 것이라
  라벨 오염 위험이 있다. 이 스크립트는 정리만 하고, 실제 오염 제거(불량
  혼입 이미지 삭제)는 review_sheet_{angle}.png 를 보고 사람이 raw/ 에서
  직접 파일을 지우는 방식으로 수행한다.

동작:
  - raw/ 복사는 idempotent — 대상 파일이 이미 있으면 건너뛴다. 즉, 사람이
    raw/ 에서 삭제한 파일이라도 로그 폴더명이 같으면 재실행 시 그대로
    남아있는 원본에서 다시 복사되어 되살아날 수 있다(같은 이름의 로그
    폴더가 그대로 있는 한). 검수 후에는 원본 inspection_logs 를 건드리지
    않는 한 안전하다.
  - review_sheet_{angle}.png 는 매 실행마다 raw/ 의 현재 상태를 반영해
    다시 생성한다(검수 진행 상황을 계속 확인할 수 있도록).

torch 를 import하지 않는다 (호스트에서도 실행 가능하도록 cv2/numpy만 사용).

사용법:
  python3 build_anomaly_dataset.py \
    --logs-dir /workspace/data/inspection_logs \
    --out-dir  /workspace/data/anomaly_dataset \
    --thumb-size 160
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import shutil
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

ANGLES = (0, 90, 180, 270)
MIN_IMAGES_PER_ANGLE = 20   # Phase 0 게이트 (계획서 §4)
GRID_COLS_MAX = 8


# ─────────────────────────────────────────────
# raw/ 데이터셋 구성
# ─────────────────────────────────────────────
def find_pass_folders(logs_dir: str) -> List[str]:
    """`*_PASS` 로 끝나는 검사 로그 폴더 목록(정렬됨)을 반환."""
    pattern = os.path.join(logs_dir, '*_PASS')
    return sorted(d for d in glob.glob(pattern) if os.path.isdir(d))


def build_raw_dataset(
    logs_dir: str, out_dir: str, angles: Tuple[int, ...],
) -> Dict[int, int]:
    """PASS 폴더의 각도별 `captured_{angle}.png` 를 raw/{angle}/ 로 복사한다.

    이미 대상 파일이 있으면 건너뛴다(idempotent).

    Returns:
        각도별 raw/{angle}/ 폴더 내 최종 이미지 수.
    """
    pass_folders = find_pass_folders(logs_dir)
    if not pass_folders:
        print(f'[경고] PASS 로그 폴더가 없습니다: {logs_dir}')

    for folder in pass_folders:
        folder_name = os.path.basename(folder)
        for angle in angles:
            src = os.path.join(folder, f'captured_{angle}.png')
            if not os.path.isfile(src):
                continue
            dst_dir = os.path.join(out_dir, 'raw', str(angle))
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, f'{folder_name}.png')
            if not os.path.isfile(dst):
                shutil.copy2(src, dst)

    counts: Dict[int, int] = {}
    for angle in angles:
        dst_dir = os.path.join(out_dir, 'raw', str(angle))
        if os.path.isdir(dst_dir):
            counts[angle] = len([
                f for f in os.listdir(dst_dir) if f.lower().endswith('.png')])
        else:
            counts[angle] = 0
    return counts


# ─────────────────────────────────────────────
# 검수용 썸네일 시트
# ─────────────────────────────────────────────
def build_review_sheet(
    out_dir: str, angle: int, thumb_size: int,
) -> Optional[Tuple[str, List[str]]]:
    """raw/{angle}/ 의 이미지들을 그리드로 배치한 검수 시트를 생성한다.

    각 셀에 인덱스 번호를 라벨로 표기한다 — 시트를 보고 사람이 불량 혼입
    이미지를 찾으면 인덱스로 원본 파일명을 대조해 raw/ 에서 삭제한다.

    Returns:
        (시트 저장 경로, 인덱스 순서의 파일명 리스트). 이미지가 없으면 None.
    """
    img_dir = os.path.join(out_dir, 'raw', str(angle))
    if not os.path.isdir(img_dir):
        return None

    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith('.png'))
    if not files:
        return None

    cols = min(GRID_COLS_MAX, len(files))
    rows = math.ceil(len(files) / cols)
    label_h = 20
    cell_h = thumb_size + label_h

    sheet = np.full((rows * cell_h, cols * thumb_size, 3), 40, dtype=np.uint8)

    for idx, fname in enumerate(files):
        r, c = divmod(idx, cols)
        img = cv2.imread(os.path.join(img_dir, fname))
        if img is None:
            continue
        thumb = cv2.resize(img, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
        y0, x0 = r * cell_h, c * thumb_size
        sheet[y0:y0 + thumb_size, x0:x0 + thumb_size] = thumb
        cv2.putText(
            sheet, f'#{idx}', (x0 + 4, y0 + thumb_size + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    sheet_path = os.path.join(out_dir, f'review_sheet_{angle}.png')
    cv2.imwrite(sheet_path, sheet)
    return sheet_path, files


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description='QUVI 이상탐지 데이터셋 빌더 (Phase 0-a)')
    parser.add_argument('--logs-dir', default='/workspace/data/inspection_logs')
    parser.add_argument('--out-dir', default='/workspace/data/anomaly_dataset')
    parser.add_argument('--thumb-size', type=int, default=160)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f'[1/2] raw/ 데이터셋 구성 중 ... (logs: {args.logs_dir})')
    counts = build_raw_dataset(args.logs_dir, args.out_dir, ANGLES)

    print(f'[2/2] 각도별 검수 시트 생성 중 ... (out: {args.out_dir})')
    print('─' * 50)
    for angle in ANGLES:
        result = build_review_sheet(args.out_dir, angle, args.thumb_size)
        n = counts.get(angle, 0)
        if result is None:
            print(f'  {angle:>3}° : 이미지 0장 — 시트 생성 생략')
        else:
            sheet_path, _files = result
            print(f'  {angle:>3}° : {n}장 → {sheet_path}')
        if n < MIN_IMAGES_PER_ANGLE:
            print(
                f'    [경고] {angle}° 이미지 {n}장 < 최소 {MIN_IMAGES_PER_ANGLE}장 '
                f'— Phase 0 게이트 미달, 정식 학습 전 추가 촬영 필요')
    print('─' * 50)
    print('완료. review_sheet_{angle}.png 를 확인해 불량 혼입 이미지를 raw/{angle}/ 에서 직접 삭제하세요.')


if __name__ == '__main__':
    main()
