"""
QUVI 공통 유틸리티
────────────────
모든 노드에서 공유하는 이미지 처리 + 파라미터 헬퍼.

포함 내용:
  - decode_compressed / decode_raw : CompressedImage / Image → BGR ndarray
  - BinaryCache                    : 단일 이미지 이진화+윤곽 결과 캐싱
                                     (동일 이미지에 대한 중복 threshold/findContours 제거)
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image

_bridge = CvBridge()


# ─────────────────────────────────────────────
# 이미지 디코딩
# ─────────────────────────────────────────────

def decode_compressed(msg: CompressedImage) -> Optional[np.ndarray]:
    """CompressedImage → BGR numpy array. 실패 시 None 반환."""
    np_arr = np.frombuffer(msg.data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def decode_raw(msg: Image) -> Optional[np.ndarray]:
    """sensor_msgs/Image → BGR numpy array. 실패 시 None 반환."""
    try:
        return _bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    except Exception as e:
        logging.getLogger(__name__).warning(f'decode_raw 실패: {e}')
        return None


def encode_bgr(frame: np.ndarray) -> Image:
    """BGR numpy array → sensor_msgs/Image (bgr8 encoding)."""
    return _bridge.cv2_to_imgmsg(frame, encoding='bgr8')


# ─────────────────────────────────────────────
# 이진화 캐시
# ─────────────────────────────────────────────

class BinaryCache:
    """단일 회색조 이미지에 대한 이진화 + 윤곽 검출 결과를 캐싱한다.

    inspect_node 에서 largest_external_area() / solidity() / holes() 가
    동일 이미지에 대해 각각 threshold + findContours 를 반복 호출하던 문제를 제거.
    BinaryCache 를 1 회 생성하면 세 메서드가 결과를 공유한다.

    Attributes:
        binary            : 이진화된 이미지 (uint8)
        contours_external : RETR_EXTERNAL 윤곽 리스트
        contours_tree     : RETR_TREE 윤곽 리스트
        hierarchy         : RETR_TREE 계층 구조 (shape: [1, N, 4])
    """

    __slots__ = ('gray', 'binary', 'contours_external', 'contours_tree',
                 'hierarchy', '_aligned_cache')

    def __init__(self, gray: np.ndarray, thresh: int) -> None:
        self.gray = gray
        _, self.binary = cv2.threshold(
            gray, thresh, 255, cv2.THRESH_BINARY)

        self.contours_external, _ = cv2.findContours(
            self.binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # RETR_TREE (구멍 검출용)는 holes() 호출 시에만 lazy 초기화
        self.contours_tree = None
        self.hierarchy = None

        # 정렬 결과 캐시 (get_aligned_roi 호출 시 lazy 초기화)
        self._aligned_cache = None

    # ── 편의 메서드 ──────────────────────────────

    def largest_external_area(self) -> float:
        """외부 윤곽 중 가장 큰 면적을 반환. 윤곽 없으면 0.0."""
        if not self.contours_external:
            return 0.0
        return float(cv2.contourArea(
            max(self.contours_external, key=cv2.contourArea)))

    def largest_external_width(self) -> float:
        """가장 큰 외부 윤곽의 boundingRect 폭(px). 윤곽 없으면 0.0.

        턴테이블 편심으로 물체-카메라 거리가 회전 위상마다 변해 픽셀 면적이
        거리 제곱으로 흔들린다. 면적을 폭²로 나누면 거리 배율이 상쇄되므로
        면적비 검사의 거리 불변 정규화 분모로 쓴다.
        """
        if not self.contours_external:
            return 0.0
        largest = max(self.contours_external, key=cv2.contourArea)
        return float(cv2.boundingRect(largest)[2])

    def solidity(self) -> float:
        """컨벡스 헐 대비 윤곽 면적 비율 (워핑 감지 지표)."""
        if not self.contours_external:
            return 0.0
        largest = max(self.contours_external, key=cv2.contourArea)
        c_area = cv2.contourArea(largest)
        hull_area = cv2.contourArea(cv2.convexHull(largest))
        return c_area / hull_area if hull_area > 0 else 0.0

    def holes(self, min_hole_px: int) -> Tuple[int, float]:
        """내부 공동 개수와 면적 비율을 반환.

        Args:
            min_hole_px: 구멍으로 인정할 최소 픽셀 면적.

        Returns:
            (hole_count, hole_area_ratio)
        """
        # lazy 초기화: holes() 호출 시에만 RETR_TREE findContours 실행
        if self.hierarchy is None:
            self.contours_tree, self.hierarchy = cv2.findContours(
                self.binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if self.hierarchy is None:
            return 0, 0.0

        h_count = 0
        h_area = 0.0
        total = self.binary.shape[0] * self.binary.shape[1]

        for i, h in enumerate(self.hierarchy[0]):
            if h[3] != -1:  # parent 있음 → 내부 윤곽
                area = cv2.contourArea(self.contours_tree[i])
                if area >= min_hole_px:
                    h_count += 1
                    h_area += area

        return h_count, (h_area / total if total > 0 else 0.0)

    def get_aligned_roi(
        self,
        max_dim: int = 200,
        padding_pct: float = 0.15,
        min_area: int = 500,
    ) -> Optional[np.ndarray]:
        """역회전 정렬 + 종횡비 보존 크롭된 ROI를 반환한다.

        원본 그레이스케일에 warpAffine을 적용하여 인플레인 회전을 보정하고,
        minAreaRect 기반으로 종횡비를 보존한 채 크롭 + 리사이즈한다.

        Note:
            이진 이미지가 아닌 **원본 그레이스케일**에 역회전을 적용합니다.
            이진 이미지에 INTER_CUBIC 보간을 적용하면 경계에 회색 아티팩트가
            생기기 때문입니다.

        Args:
            max_dim: 출력 이미지의 장축 최대 픽셀 수.
            padding_pct: 크롭 시 여유 마진 비율 (0.15 = 15%).
            min_area: 정렬을 수행할 최소 윤곽 면적(픽셀). 미만이면 None 반환.

        Returns:
            정렬된 ROI 이미지 (그레이스케일). 정렬 불가 시 None.
        """
        if self._aligned_cache is not None:
            return self._aligned_cache

        if not self.contours_external:
            return None

        largest = max(self.contours_external, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < min_area:
            return None

        # ── minAreaRect → 각도 / 중심 / (w, h) ──
        (cx, cy), (w, h), angle = cv2.minAreaRect(largest)

        # 각도 정규화 [-45, 45]: 가로축을 장축으로 통일
        if w < h:
            angle += 90
            w, h = h, w
        if angle > 45:
            angle -= 90

        # ── 원본 그레이스케일에 역회전 ──
        img_h, img_w = self.gray.shape[:2]
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(
            self.gray, M, (img_w, img_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # ── 종횡비 보존 크롭 + 경계 클램핑 ──
        pad_w = int(w * (1 + padding_pct))
        pad_h = int(h * (1 + padding_pct))
        x1 = max(0, int(cx - pad_w / 2))
        y1 = max(0, int(cy - pad_h / 2))
        x2 = min(img_w, x1 + pad_w)
        y2 = min(img_h, y1 + pad_h)

        cropped = rotated[y1:y2, x1:x2]
        if cropped.size == 0:
            return None

        # 장축 기준 종횡비 보존 리사이즈
        crop_h, crop_w = cropped.shape[:2]
        long_side = max(crop_w, crop_h)
        scale = max_dim / long_side if long_side > 0 else 1.0
        target_w = max(1, int(crop_w * scale))
        target_h = max(1, int(crop_h * scale))
        aligned = cv2.resize(
            cropped, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        self._aligned_cache = aligned
        return self._aligned_cache
