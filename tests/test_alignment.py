#!/usr/bin/env python3
"""
BinaryCache 정렬 파이프라인 테스트
──────────────────────────────────
ROS 없이 순수 OpenCV + numpy 만으로 BinaryCache.get_aligned_roi() 를
검증하는 독립 실행형 테스트 스크립트.

사용법:
    cd /home/ksj/QUVI && python3 tests/test_alignment.py
"""
from __future__ import annotations

import sys
import math
from typing import Optional, Tuple, List

import cv2
import numpy as np
import pytest

# ── ROS 없이 BinaryCache 만 임포트 ──────────────────
# utils.py 가 모듈 수준에서 cv_bridge / rclpy 를 import 하므로,
# 해당 모듈이 없을 때도 BinaryCache 만 사용할 수 있도록 stub 처리.
_STUBS_INSTALLED = False


def _install_ros_stubs() -> None:
    """cv_bridge / rclpy / sensor_msgs 를 가짜 모듈로 등록."""
    global _STUBS_INSTALLED
    if _STUBS_INSTALLED:
        return

    # 실제 라이브러리가 이미 존재한다면 스텁을 등록하지 않음
    try:
        import rclpy
        import cv_bridge
        import sensor_msgs
        return
    except ImportError:
        pass

    import types

    for mod_name in (
        'cv_bridge',
        'rclpy', 'rclpy.node',
        'sensor_msgs', 'sensor_msgs.msg',
    ):
        if mod_name not in sys.modules:
            stub = types.ModuleType(mod_name)
            # cv_bridge.CvBridge 가 클래스처럼 호출됨
            if mod_name == 'cv_bridge':
                stub.CvBridge = type('CvBridge', (), {})
            # rclpy.node.Node
            if mod_name == 'rclpy.node':
                stub.Node = type('Node', (), {})
            # sensor_msgs.msg.CompressedImage / Image
            if mod_name == 'sensor_msgs.msg':
                stub.CompressedImage = type('CompressedImage', (), {})
                stub.Image = type('Image', (), {})
            sys.modules[mod_name] = stub
    _STUBS_INSTALLED = True


_install_ros_stubs()

sys.path.insert(0, '/home/ksj/QUVI/src/quvi_robot_control')
from quvi_robot_control.utils import BinaryCache  # noqa: E402


# ─────────────────────────────────────────────
# 합성 이미지 생성 헬퍼
# ─────────────────────────────────────────────

def make_rotated_rect(
    img_h: int,
    img_w: int,
    rect_w: int,
    rect_h: int,
    angle_deg: float,
    cx: Optional[int] = None,
    cy: Optional[int] = None,
) -> np.ndarray:
    """검정 배경 위에 흰색 회전 사각형이 그려진 그레이스케일 이미지를 생성한다.

    Args:
        img_h, img_w: 전체 이미지 크기.
        rect_w, rect_h: 사각형의 (폭, 높이) — 회전 전 기준.
        angle_deg: 반시계 방향 회전 각도(도).
        cx, cy: 사각형 중심 좌표. None 이면 이미지 중앙.

    Returns:
        uint8 그레이스케일 ndarray (shape: img_h × img_w).
    """
    if cx is None:
        cx = img_w // 2
    if cy is None:
        cy = img_h // 2

    img = np.zeros((img_h, img_w), dtype=np.uint8)

    # 사각형 네 꼭짓점 (중심 기준)
    half_w, half_h = rect_w / 2.0, rect_h / 2.0
    corners = np.array([
        [-half_w, -half_h],
        [ half_w, -half_h],
        [ half_w,  half_h],
        [-half_w,  half_h],
    ], dtype=np.float64)

    # 회전 적용
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    R = np.array([[cos_a, -sin_a],
                  [sin_a,  cos_a]], dtype=np.float64)
    rotated = (R @ corners.T).T  # (4, 2)

    # 이미지 좌표로 이동
    pts = rotated + np.array([cx, cy], dtype=np.float64)
    pts_int = pts.astype(np.int32)

    cv2.fillConvexPoly(img, pts_int, 255)
    return img


def _measure_residual_angle(aligned: np.ndarray) -> float:
    """정렬된 ROI 에서 여전히 남아 있는 회전 각도를 측정한다.

    정렬이 완벽하면 minAreaRect 각도 ≈ 0° (또는 ±90° 정수배).
    """
    _, binary = cv2.threshold(aligned, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 999.0  # 측정 불가

    largest = max(contours, key=cv2.contourArea)
    (_, _), (w, h), angle = cv2.minAreaRect(largest)

    # 90도 주기로 최적 각도 정규화 (축 정렬 기준)
    angle_mod = (angle + 45) % 90 - 45
    return abs(angle_mod)


# ─────────────────────────────────────────────
# 테스트 함수들
# ─────────────────────────────────────────────

@pytest.mark.parametrize("angle_deg", [5.0, 10.0, 15.0])
def test_normal_rotation(angle_deg: float):
    """일반 회전 보정 테스트: 잔여 각도 < 0.5°"""
    img = make_rotated_rect(480, 640, 200, 200, angle_deg)
    bc = BinaryCache(img, thresh=127)
    aligned = bc.get_aligned_roi(min_area=100)
    assert aligned is not None, "get_aligned_roi() 가 None 반환"

    residual = _measure_residual_angle(aligned)
    assert residual < 0.5, f"residual: {residual:.2f}°"


def test_empty_frame():
    """빈 프레임(전부 검정)일 때 None 반환"""
    img = np.zeros((480, 640), dtype=np.uint8)
    bc = BinaryCache(img, thresh=127)
    result = bc.get_aligned_roi()
    assert result is None, "None 이 아닌 값 반환됨"


def test_boundary_object():
    """이미지 경계에 걸쳐 잘린 사각형 — 크래시 없이 동작"""
    # 왼쪽 상단 모서리에 걸치게 배치 (cx=30, cy=30)
    img = make_rotated_rect(480, 640, 200, 200, 5.0, cx=30, cy=30)
    bc = BinaryCache(img, thresh=127)
    # 크래시만 아니면 통과
    bc.get_aligned_roi(min_area=100)


def test_multi_contour():
    """노이즈 블롭 + 메인 사각형 → 가장 큰 윤곽 사용 확인"""
    img = make_rotated_rect(480, 640, 200, 200, 7.0)

    # 작은 노이즈 블롭 추가 (좌측 하단)
    cv2.circle(img, (50, 430), 15, 255, -1)
    cv2.rectangle(img, (580, 20), (620, 50), 255, -1)

    bc = BinaryCache(img, thresh=127)
    aligned = bc.get_aligned_roi(min_area=100)
    assert aligned is not None, "get_aligned_roi() 가 None 반환"

    residual = _measure_residual_angle(aligned)
    assert residual < 0.5, f"residual: {residual:.2f}° (노이즈 블롭 무시됨)"


def test_aspect_ratio():
    """100×200 직사각형(종횡비 2:1) 보존 확인 (허용 오차 10%)"""
    rect_w, rect_h = 200, 100  # 가로 200, 세로 100 → 종횡비 2:1
    img = make_rotated_rect(480, 640, rect_w, rect_h, 8.0)
    bc = BinaryCache(img, thresh=127)
    aligned = bc.get_aligned_roi(min_area=100)
    assert aligned is not None, "get_aligned_roi() 가 None 반환"

    # 출력 ROI 에서 실제 객체의 종횡비를 측정
    _, binary = cv2.threshold(aligned, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert contours, "정렬된 이미지에서 윤곽 미검출"

    largest = max(contours, key=cv2.contourArea)
    (_, _), (mw, mh), _ = cv2.minAreaRect(largest)
    long_side = max(mw, mh)
    short_side = min(mw, mh)

    assert short_side >= 1, "short_side ≈ 0"

    ratio = long_side / short_side
    target = 2.0
    err = abs(ratio - target) / target
    assert err < 0.10, f"종횡비: {ratio:.2f} (목표: 2.00, 오차: {err*100:.1f}%)"


def test_kill_switch():
    """min_area 를 매우 높게 설정 → None 반환"""
    img = make_rotated_rect(480, 640, 200, 200, 5.0)
    bc = BinaryCache(img, thresh=127)
    result = bc.get_aligned_roi(min_area=999_999)
    assert result is None, "None 이 아닌 값 반환됨"


def test_cache_consistency():
    """get_aligned_roi() 를 두 번 호출해도 동일 결과"""
    img = make_rotated_rect(480, 640, 200, 200, 10.0)
    bc = BinaryCache(img, thresh=127)
    r1 = bc.get_aligned_roi(min_area=100)
    r2 = bc.get_aligned_roi(min_area=100)
    assert r1 is not None and r2 is not None, "get_aligned_roi() 가 None 반환"
    assert np.array_equal(r1, r2), "결과가 다름"


def test_rotation_info():
    """get_aligned_roi() 후 get_rotation_info() 가 유효한 값 반환"""
    angle_in = 12.0
    img = make_rotated_rect(480, 640, 200, 200, angle_in)
    bc = BinaryCache(img, thresh=127)
    aligned = bc.get_aligned_roi(min_area=100)
    assert aligned is not None, "get_aligned_roi() 가 None 반환"

    info = bc.get_rotation_info()
    assert info is not None, "get_rotation_info() 가 None 반환"

    angle, (cx, cy), (w, h) = info
    ok = (
        isinstance(angle, float)
        and isinstance(cx, float) and isinstance(cy, float)
        and isinstance(w, float) and isinstance(h, float)
        and w > 0 and h > 0
    )
    assert ok, f"angle={angle:.2f}°, center=({cx:.0f},{cy:.0f}), size=({w:.0f}×{h:.0f})"

