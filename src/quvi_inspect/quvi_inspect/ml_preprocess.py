"""
QUVI ML 전처리 (공유 모듈)
──────────────────────
학습 스크립트(scripts/train_anomaly_bank.py)와 (미래의) inspect_node ML 추론이
**동일한 함수**를 공유하여 train/infer skew 를 방지하기 위한 모듈.

주의:
  이 모듈은 torch 를 import하지 않는다 (cv2/numpy만 사용).
  노드가 ML 비활성 상태(anomaly_enabled=False)일 때도 항상 import 되므로
  여기서 torch 를 로드하면 불필요한 GPU/CUDA 초기화 비용이 발생한다.
"""
from __future__ import annotations

import cv2
import numpy as np

from quvi_robot_control.utils import BinaryCache

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

    학습 스크립트(train_anomaly_bank.py)와 추론(inspect_node ML 경로, Phase 2)이
    이 함수 하나를 공유해야 한다 — 서로 다른 전처리를 쓰면 학습/추론 분포가
    어긋나 이상탐지 오탐의 최대 원인이 된다.

    처리 순서:
      1. BGR → grayscale → GaussianBlur(5) 로 룰 파이프라인과 동일하게 정렬
      2. BinaryCache 로 이진화 + 최대 외부 윤곽의 boundingRect 계산
      3. boundingRect 에 padding_pct 만큼 여유를 두고 이미지 경계로 클램프
      4. **컬러 원본**(bgr, 블러 전)에서 해당 영역을 crop
      5. out_size × out_size 로 리사이즈 후 BGR → RGB 변환

    윤곽을 못 찾거나 최대 윤곽 면적이 너무 작으면(객체 미검출) 전체 이미지를
    그대로 리사이즈하는 폴백을 사용한다. 이 함수는 파이프라인이 끊기지 않도록
    항상 유효한 이미지를 반환하며 실패를 None 으로 알리지 않는다 — 폴백 여부를
    로그로 남기는 것은 호출자의 책임이다.

    Args:
        bgr: 원본 BGR 이미지 (예: 1920×1080 검사 캡처).
        bin_thresh: BinaryCache 이진화 임계값.
        padding_pct: bbox 크롭 시 여유 마진 비율 (0.15 = 15%).
        out_size: 출력 정사각 이미지 한 변 픽셀 수.

    Returns:
        out_size × out_size × 3 RGB uint8 배열.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cache = BinaryCache(blurred, bin_thresh)

    img_h, img_w = bgr.shape[:2]
    crop = None

    if cache.contours_external:
        largest = max(cache.contours_external, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area >= _MIN_CONTOUR_AREA_PX:
            x, y, w, h = cv2.boundingRect(largest)
            pad_w = int(w * padding_pct)
            pad_h = int(h * padding_pct)
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(img_w, x + w + pad_w)
            y2 = min(img_h, y + h + pad_h)
            if x2 > x1 and y2 > y1:
                crop = bgr[y1:y2, x1:x2]

    if crop is None or crop.size == 0:
        # 객체 미검출 또는 윤곽 과소 — 전체 이미지 폴백 (경고는 호출자 몫).
        crop = bgr

    resized = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
