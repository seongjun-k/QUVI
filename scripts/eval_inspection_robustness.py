#!/usr/bin/env python3
"""검사 판정 강건성 오프라인 시험 — 정상품 사진에 잘림·밝기 변화를 줘서 현재 판정이
얼마나 '불량'으로 뒤집히는지(오탐률) 잰다. 로봇·카메라 없이 돈다.

판정은 inspect_node 와 같게 재현한다: 룰(Solidity·구멍·텍스처) AND ML(PatchCore 임계값).
오탐률은 학습에 안 쓴 held-out 정상품(train_anomaly_bank.py 와 같은 시드 분할)으로만 낸다.
학습 사진 결과는 뱅크가 이미 본 사진이라 낙관적인 참고값이다.

실행 위치: quvi-dev 컨테이너
  python3 /workspace/scripts/eval_inspection_robustness.py --products box cube
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path[:0] = ['/workspace/src/quvi_robot_control', '/workspace/src/quvi_inspect',
                os.path.dirname(os.path.abspath(__file__))]
from quvi_robot_control.utils import BinaryCache  # noqa: E402
from quvi_inspect.ml_preprocess import preprocess_for_ml  # noqa: E402
from quvi_inspect.anomaly_detector import PatchCoreDetector  # noqa: E402
from train_anomaly_bank import split_train_val  # noqa: E402

# inspect_node 기본값 / inspect_params.yaml 과 동일
BIN, BLUR = 127, 5
SOL_MIN, SOL_MAX = 0.85, 1.00
HOLE_MAX, HOLE_AREA_MAX, MIN_HOLE_PX = 0, 0.05, 50
TEX_MAX = 500.0
ALIGN = dict(max_dim=200, padding_pct=0.15, min_area=500)
ANGLES = [0, 90, 180, 270]
SEED, VAL_RATIO = 42, 0.2


# ─── 변형 ───
def shift_down(bgr, frac):
    """물체를 '물체 높이 × frac' 만큼 아래로 밀어 아랫부분을 화면 밖으로 내보낸다(편심 잘림 재현)."""
    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (BLUR, BLUR), 0)
    cs = BinaryCache(g, BIN).contours_external
    if not cs:
        return bgr
    _, _, _, h = cv2.boundingRect(max(cs, key=cv2.contourArea))
    dy = int(h * frac)
    out = np.zeros_like(bgr)
    out[dy:] = bgr[:bgr.shape[0] - dy]
    return out


def brightness(bgr, f):
    return np.clip(bgr.astype(np.float32) * f, 0, 255).astype(np.uint8)


VARIANTS = [('원본', lambda b: b)]
VARIANTS += [(f'아래 잘림 {int(f * 100)}%', lambda b, f=f: shift_down(b, f)) for f in (0.1, 0.2, 0.3, 0.4)]
VARIANTS += [(f'밝기 ×{f}', lambda b, f=f: brightness(b, f)) for f in (0.7, 0.85, 1.15, 1.3)]


# ─── 판정 (inspect_node._surface_analysis 재현) ───
def rule_reasons(bgr):
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (BLUR, BLUR), 0)
    cache = BinaryCache(gray, BIN)
    aligned = cache.get_aligned_roi(**ALIGN)
    if aligned is not None:
        cache = BinaryCache(aligned, BIN)
    out = []
    if not SOL_MIN <= cache.solidity() <= SOL_MAX:
        out.append('sol')
    n_holes, hole_ar = cache.holes(MIN_HOLE_PX)
    if n_holes > HOLE_MAX or hole_ar > HOLE_AREA_MAX:
        out.append('hole')
    if cv2.Laplacian(gray, cv2.CV_64F).var() > TEX_MAX:  # inspect_node 와 같이 전체 프레임
        out.append('tex')
    return out


def evaluate(product, products_dir, backbone_path, device):
    model_dir = os.path.join(products_dir, product, 'models')
    with open(os.path.join(model_dir, 'thresholds.json'), encoding='utf-8') as f:
        thr = {int(k): v['threshold'] for k, v in json.load(f).items() if isinstance(v, dict)}
    rows = []   # (split, variant, rule_fail, ml_fail, score/thr)
    backbone = None
    for angle in ANGLES:
        det = PatchCoreDetector.load(os.path.join(model_dir, f'bank_{angle}.pt'), device=device,
                                     backbone_weights_path=backbone_path, backbone=backbone)
        backbone = det.backbone if hasattr(det, 'backbone') else backbone
        img_dir = os.path.join(products_dir, product, 'anomaly_dataset', 'raw', str(angle))
        files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith('.png'))
        train, val = split_train_val([(f, None) for f in files], VAL_RATIO, SEED)
        split_of = {f: 'train' for f, _ in train} | {f: 'val' for f, _ in val}
        for fname in files:
            src = cv2.imread(os.path.join(img_dir, fname))
            if src is None:
                continue
            for name, fn in VARIANTS:
                img = fn(src)
                r = rule_reasons(img)
                ratio = det.score(preprocess_for_ml(img, BIN)) / thr[angle]
                rows.append((split_of[fname], name, bool(r), ratio > 1.0, ratio))
    return rows


def table(rows, split):
    lines = ['| 변형 | n | 불량 판정(오탐) | 룰 | ML | ML 점수/임계값 중앙값 |', '|---|---|---|---|---|---|']
    for name, _ in VARIANTS:
        sel = [r for r in rows if r[0] == split and r[1] == name]
        if not sel:
            continue
        n = len(sel)
        fp = sum(1 for r in sel if r[2] or r[3])
        lines.append(f'| {name} | {n} | {fp} ({fp / n:.0%}) | {sum(r[2] for r in sel)} | '
                     f'{sum(r[3] for r in sel)} | {np.median([r[4] for r in sel]):.2f} |')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--products', nargs='+', default=['box', 'cube'])
    ap.add_argument('--products-dir', default='/workspace/data/inspection_products')
    ap.add_argument('--backbone', default='/workspace/data/models/wide_resnet50.pth')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', help='결과 마크다운 저장 경로')
    a = ap.parse_args()
    report = []
    for p in a.products:
        rows = evaluate(p, a.products_dir, a.backbone, a.device)
        report += [f'## {p}', '', '### held-out 정상품 (오탐률 기준)', '', table(rows, 'val'), '',
                   '### 학습 사진 (참고, 낙관적)', '', table(rows, 'train'), '']
    text = '\n'.join(report)
    print(text)
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            f.write(text)


if __name__ == '__main__':
    main()
