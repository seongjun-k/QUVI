"""
QUVI ML 전처리 (공유 모듈)
──────────────────────
학습 스크립트(scripts/train_anomaly_bank.py)와 inspect_node ML 추론(섀도우 모드)이
**동일한 함수**를 공유하여 train/infer skew 를 방지하기 위한 모듈.

주의:
  이 모듈은 torch 를 import하지 않는다 (cv2/numpy만 사용).
  노드가 ML 비활성 상태(anomaly_enabled=False)일 때도 항상 import 되므로
  여기서 torch 를 로드하면 불필요한 GPU/CUDA 초기화 비용이 발생한다.
"""
from __future__ import annotations

import cv2
import numpy as np

from quvi_robot_control.utils import BinaryCache, compute_aligned_crop

# 윤곽 면적이 이 값 미만이면 "객체 미검출"로 간주하고 전체 이미지 폴백을 사용한다.
_MIN_CONTOUR_AREA_PX = 500


# ─────────────────────────────────────────────
# 학습·추론 공용 전처리
# ─────────────────────────────────────────────
def preprocess_for_ml(
    bgr: np.ndarray,
    bin_thresh: int = 127,
    padding_pct: float = 0.15,
    out_size: int = 256,
) -> np.ndarray:
    """검사 원본(BGR 컬러)을 이상탐지 백본 입력용 256×256 RGB로 변환한다.

    학습 스크립트(train_anomaly_bank.py)와 추론(inspect_node ML 경로)이
    이 함수 하나를 공유해야 한다 — 서로 다른 전처리를 쓰면 학습/추론 분포가
    어긋나 이상탐지 오탐의 최대 원인이 된다.

    처리 순서:
      1. BGR → grayscale → GaussianBlur(5) 로 룰 파이프라인과 동일하게 정렬
      2. BinaryCache 로 이진화 + 최대 외부 윤곽의 minAreaRect(각도 포함) 계산
      3. 그 각도로 **컬러 원본**(bgr, 블러 전)에 warpAffine 역회전 적용
      4. 역회전된 rect 크기에 padding_pct 마진을 두고 crop
      5. 장축을 out_size 에 맞춘 종횡비 보존 리사이즈 후 out_size×out_size
         검은 캔버스 중앙에 배치(letterbox) → BGR → RGB 변환

    윤곽을 못 찾거나 최대 윤곽 면적이 너무 작으면(객체 미검출) 전체 이미지를
    그대로 리사이즈하는 폴백을 사용한다. 이 함수는 파이프라인이 끊기지 않도록
    항상 유효한 이미지를 반환하며 실패를 None 으로 알리지 않는다 — 폴백 여부를
    로그로 남기는 것은 호출자의 책임이다.

    Args:
        bgr: 원본 BGR 이미지 (예: 1920×1080 검사 캡처).
        bin_thresh: BinaryCache 이진화 임계값.
        padding_pct: 회전 정렬된 rect 크롭 시 여유 마진 비율 (0.15 = 15%).
        out_size: 출력 정사각 이미지 한 변 픽셀 수.

    Returns:
        out_size × out_size × 3 RGB uint8 배열.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cache = BinaryCache(blurred, bin_thresh)

    # 각도 정규화 + 역회전 크롭 (컬러 원본, get_aligned_roi 와 동일 로직 공유)
    crop = compute_aligned_crop(
        cache.contours_external, bgr, padding_pct, _MIN_CONTOUR_AREA_PX)

    if crop is None:
        # 객체 미검출 또는 윤곽 과소 — 전체 이미지 폴백 (경고는 호출자 몫).
        crop = bgr

    # 장축 기준 종횡비 보존 리사이즈 후 검은 캔버스 중앙에 배치 (letterbox)
    crop_h, crop_w = crop.shape[:2]
    long_side = max(crop_w, crop_h)
    scale = out_size / long_side if long_side > 0 else 1.0
    target_w = max(1, int(crop_w * scale))
    target_h = max(1, int(crop_h * scale))
    resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.zeros((out_size, out_size, 3), dtype=np.uint8)
    off_x = (out_size - target_w) // 2
    off_y = (out_size - target_h) // 2
    canvas[off_y:off_y + target_h, off_x:off_x + target_w] = resized

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
