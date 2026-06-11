"""
QUVI 공통 유틸리티
────────────────
모든 노드에서 공유하는 이미지 처리 + 파라미터 헬퍼.

포함 내용:
  - decode_compressed / decode_raw : CompressedImage / Image → BGR ndarray
  - declare_and_get                : 파라미터 선언+로드 1단계 처리
  - BinaryCache                    : 단일 이미지 이진화+윤곽 결과 캐싱
                                     (동일 이미지에 대한 중복 threshold/findContours 제거)
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image

_bridge = CvBridge()


# ─────────────────────────────────────────────
# 이미지 디코딩
# ─────────────────────────────────────────────

def decode_compressed(msg: CompressedImage) -> Optional[np.ndarray]:
    """CompressedImage → BGR numpy array. 실패 시 None 반환."""
    np_arr = np.frombuffer(msg.data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame if frame is not None else None


def decode_raw(msg: Image) -> Optional[np.ndarray]:
    """sensor_msgs/Image → BGR numpy array. 실패 시 None 반환."""
    try:
        return _bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    except Exception:
        return None


# ─────────────────────────────────────────────
# 파라미터 헬퍼
# ─────────────────────────────────────────────

def declare_and_get(node: Node, name: str, default):
    """파라미터 선언과 로드를 한 번에 처리한다."""
    node.declare_parameter(name, default)
    return node.get_parameter(name).value


# ─────────────────────────────────────────────
# 이진화 캐시
# ─────────────────────────────────────────────

class BinaryCache:
    """단일 회색조 이미지에 대한 이진화 + 윤곽 검출 결과를 캐싱한다.

    inspect_node 에서 _get_object_area / _compute_solidity / _compute_holes 가
    동일 이미지에 대해 각각 threshold + findContours 를 반복 호출하던 문제를 제거.
    BinaryCache 를 1 회 생성하면 세 함수가 결과를 공유한다.

    Attributes:
        binary            : 이진화된 이미지 (uint8)
        contours_external : RETR_EXTERNAL 윤곽 리스트
        contours_tree     : RETR_TREE 윤곽 리스트
        hierarchy         : RETR_TREE 계층 구조 (shape: [1, N, 4])
    """

    __slots__ = ('binary', 'contours_external', 'contours_tree', 'hierarchy')

    def __init__(self, gray: np.ndarray, thresh: int) -> None:
        _, self.binary = cv2.threshold(
            gray, thresh, 255, cv2.THRESH_BINARY)

        self.contours_external, _ = cv2.findContours(
            self.binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.contours_tree, self.hierarchy = cv2.findContours(
            self.binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # ── 편의 메서드 ──────────────────────────────

    def largest_external_area(self) -> float:
        """외부 윤곽 중 가장 큰 면적을 반환. 윤곽 없으면 0.0."""
        if not self.contours_external:
            return 0.0
        return float(cv2.contourArea(
            max(self.contours_external, key=cv2.contourArea)))

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
