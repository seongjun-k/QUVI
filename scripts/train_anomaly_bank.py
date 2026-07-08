#!/usr/bin/env python3
"""
QUVI 이상탐지 메모리뱅크 학습 (Phase 1)
──────────────────────────────────
정상품 데이터셋(build_anomaly_dataset.py 로 정리·검수된 raw/{angle}/)으로부터
각도별 PatchCore 메모리뱅크를 학습하고, held-out 정상 점수 분포로 임계값을
산정한다. 기존 `*_FAIL` 로그는 참고용 sanity check 로만 사용한다(통계적
검증이 아님 — FAIL 표본이 극소수이기 때문).

임계값 산정(docs/ml_anomaly_inspection_plan.md §4 Phase 1):
  1. 정상셋을 시드 고정 셔플 후 8:2 분할
  2. 80% 로 fit()
  3. held-out 20% 점수의 최댓값 × threshold-margin 을 임계값으로 채택
  4. held-out 이 0장(데이터 극소)이면 학습셋 점수로 대체 + "신뢰 불가" 경고

산출물:
  {models-dir}/bank_{angle}.pt   — 각도별 PatchCore 메모리뱅크
  {models-dir}/thresholds.json   — 각도별 임계값 + 메타
  콘솔 리포트                    — 각도별 정상/불량 점수 분포

sys.path 에 quvi_inspect / quvi_robot_control 소스 경로를 직접 추가해
colcon 빌드 없이 동작한다.

실행 위치: quvi-dev 컨테이너 (기본 --device cuda, GPU 필요).

사용법:
  python3 train_anomaly_bank.py \
    --dataset-dir /workspace/data/anomaly_dataset/raw \
    --models-dir  /workspace/data/models \
    --logs-dir    /workspace/data/inspection_logs \
    --coreset-ratio 0.1 --threshold-margin 1.15 --seed 42 --device cuda
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np

# ─── colcon 빌드 없이 동작: 소스 경로 직접 추가 ───
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.normpath(os.path.join(_THIS_DIR, '..', 'src'))
for _pkg in ('quvi_inspect', 'quvi_robot_control'):
    _pkg_path = os.path.join(_SRC_DIR, _pkg)
    if _pkg_path not in sys.path:
        sys.path.insert(0, _pkg_path)

from quvi_inspect.anomaly_detector import PatchCoreDetector  # noqa: E402
from quvi_inspect.ml_preprocess import preprocess_for_ml  # noqa: E402

ANGLES = (0, 90, 180, 270)
BACKBONE_WEIGHTS_FILENAME = 'wide_resnet50.pth'
BIN_THRESH = 127   # inspect_node 기본 binary_threshold 와 동일
MIN_RELIABLE_VAL_SAMPLES = 5   # held-out 표본이 이보다 적으면 max() 임계값 과적합 위험 — 경고만, 학습은 계속


# ─────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────
def load_and_preprocess(img_dir: str) -> List[Tuple[str, np.ndarray]]:
    """디렉토리의 png 이미지를 로드 후 ml_preprocess 로 전처리한다.

    Returns:
        (파일명, 전처리된 256×256 RGB) 리스트 — 파일명 정렬 순서(재현성).
    """
    if not os.path.isdir(img_dir):
        return []
    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith('.png'))
    result = []
    for fname in files:
        bgr = cv2.imread(os.path.join(img_dir, fname))
        if bgr is None:
            print(f'    [경고] 이미지 로드 실패: {fname}')
            continue
        result.append((fname, preprocess_for_ml(bgr, bin_thresh=BIN_THRESH)))
    return result


def split_train_val(
    items: List[Tuple[str, np.ndarray]], val_ratio: float, seed: int,
) -> Tuple[List[Tuple[str, np.ndarray]], List[Tuple[str, np.ndarray]]]:
    """시드 고정 셔플 후 8:2(기본) 분할. 데이터가 적으면 held-out 이 0장일 수 있다."""
    shuffled = items.copy()
    random.Random(seed).shuffle(shuffled)
    n_val = int(len(shuffled) * val_ratio)
    return shuffled[n_val:], shuffled[:n_val]


def _fmt_scores(scores: List[float]) -> str:
    if not scores:
        return '(없음)'
    return (
        f'min={min(scores):.4f} max={max(scores):.4f} '
        f'mean={sum(scores) / len(scores):.4f} (n={len(scores)})')


# ─────────────────────────────────────────────
# 각도별 학습
# ─────────────────────────────────────────────
def train_one_angle(
    angle: int, args: argparse.Namespace, backbone_weights_path: str,
) -> Dict:
    """단일 각도에 대해 뱅크 학습 + 임계값 산정 + FAIL sanity check 를 수행한다."""
    print(f'\n{"=" * 50}\n[{angle}°] 학습 시작\n{"=" * 50}')

    img_dir = os.path.join(args.dataset_dir, str(angle))
    items = load_and_preprocess(img_dir)
    n_total = len(items)
    print(f'  정상 이미지: {n_total}장 ({img_dir})')

    if n_total == 0:
        print(f'  [경고] {angle}° 이미지가 0장 — 학습 생략')
        return {'angle': angle, 'status': 'skipped', 'n_images': 0, 'threshold': None,
                'threshold_reliable': False}

    train_items, val_items = split_train_val(items, val_ratio=0.2, seed=args.seed)
    print(f'  분할: train {len(train_items)}장 / held-out {len(val_items)}장')

    detector = PatchCoreDetector(
        device=args.device, backbone_weights_path=backbone_weights_path)
    train_images = [img for _f, img in train_items]
    detector.fit(train_images, coreset_ratio=args.coreset_ratio, seed=args.seed)
    print(
        f'  뱅크 구성 완료: 패치 {detector.bank.shape[0]}개 '
        f'(coreset_ratio={args.coreset_ratio}, device={detector.device})')

    threshold_reliable = True
    if val_items:
        val_scores = [detector.score(img) for _f, img in val_items]
        if len(val_items) < MIN_RELIABLE_VAL_SAMPLES:   # ponytail: 소표본 max() 임계값은 과적합 위험 — 절대 개수 기준
            print(f'  [경고] held-out {len(val_items)}장 — 소표본, 임계값 신뢰 불가')
            threshold_reliable = False
    else:
        print('  [경고] held-out 0장 — 학습셋 점수로 대체, 임계값 신뢰 불가')
        threshold_reliable = False
        val_scores = [detector.score(img) for img in train_images]

    threshold = max(val_scores) * args.threshold_margin

    detector.meta = {
        'angle': angle,
        'n_train': len(train_items),
        'n_val': len(val_items),
        'coreset_ratio': args.coreset_ratio,
        'threshold_margin': args.threshold_margin,
        'threshold': threshold,
        'threshold_reliable': threshold_reliable,
        'seed': args.seed,
    }
    bank_path = os.path.join(args.models_dir, f'bank_{angle}.pt')
    detector.save(bank_path)
    print(f'  뱅크 저장: {bank_path}')
    print(f'  held-out(대체 포함) 점수: {_fmt_scores(val_scores)}')
    print(
        f'  임계값 = max(점수) × {args.threshold_margin} = {threshold:.4f}'
        + ('' if threshold_reliable else '  [신뢰 불가 — 데이터 부족]'))

    # ── FAIL 로그 sanity check (통계적 검증 아님) ──
    fail_scores = _score_fail_logs(angle, args, detector)
    if fail_scores:
        exceed = sum(1 for s in fail_scores if s > threshold)
        print(
            f'  FAIL 로그 sanity check: {len(fail_scores)}장 중 {exceed}장이 '
            f'임계값 초과 — {_fmt_scores(fail_scores)} (참고용, 통계적 검증 아님)')
    else:
        print('  FAIL 로그 sanity check: 대상 이미지 없음')

    return {
        'angle': angle,
        'status': 'ok',
        'n_images': n_total,
        'n_train': len(train_items),
        'n_val': len(val_items),
        'threshold': threshold,
        'threshold_reliable': threshold_reliable,
        'bank_path': bank_path,
    }


def _score_fail_logs(
    angle: int, args: argparse.Namespace, detector: PatchCoreDetector,
) -> List[float]:
    """`*_FAIL` 로그 폴더의 해당 각도 이미지들에 대한 이상점수 리스트."""
    pattern = os.path.join(args.logs_dir, '*_FAIL')
    scores = []
    for folder in sorted(glob.glob(pattern)):
        path = os.path.join(folder, f'captured_{angle}.png')
        if not os.path.isfile(path):
            continue
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        img = preprocess_for_ml(bgr, bin_thresh=BIN_THRESH)
        scores.append(detector.score(img))
    return scores


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description='QUVI 이상탐지 메모리뱅크 학습 (Phase 1)')
    parser.add_argument('--dataset-dir', default='/workspace/data/anomaly_dataset/raw')
    parser.add_argument('--models-dir', default='/workspace/data/models')
    parser.add_argument('--logs-dir', default='/workspace/data/inspection_logs')
    parser.add_argument('--coreset-ratio', type=float, default=0.1)
    parser.add_argument('--threshold-margin', type=float, default=1.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    backbone_weights_path = os.path.join(args.models_dir, BACKBONE_WEIGHTS_FILENAME)

    results = {}
    for angle in ANGLES:
        results[angle] = train_one_angle(angle, args, backbone_weights_path)

    thresholds = {
        str(angle): {
            'threshold': r['threshold'],
            'threshold_reliable': r.get('threshold_reliable'),
            'status': r['status'],
            'n_images': r['n_images'],
        }
        for angle, r in results.items()
    }
    thresholds_path = os.path.join(args.models_dir, 'thresholds.json')
    with open(thresholds_path, 'w', encoding='utf-8') as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)

    print(f'\n{"=" * 50}')
    print(f'thresholds.json 저장: {thresholds_path}')
    print('=' * 50)


if __name__ == '__main__':
    main()
