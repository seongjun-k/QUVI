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

# ── ROS 없이 BinaryCache 만 임포트 ──────────────────
# utils.py 가 모듈 수준에서 cv_bridge / rclpy 를 import 하므로,
# 해당 모듈이 없을 때도 BinaryCache 만 사용할 수 있도록 stub 처리.
_STUBS_INSTALLED = False


def _install_ros_stubs() -> None:
    """cv_bridge / rclpy / sensor_msgs 를 가짜 모듈로 등록."""
    global _STUBS_INSTALLED
    if _STUBS_INSTALLED:
        return
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

    # 정규화: 정렬 후 이상적 각도는 0°
    if w < h:
        angle += 90
    if angle > 45:
        angle -= 90
    return abs(angle)


# ─────────────────────────────────────────────
# 테스트 함수들
# ─────────────────────────────────────────────

TestResult = Tuple[bool, str]  # (passed, detail_message)


def test_normal_rotation(angle_deg: float) -> TestResult:
    """일반 회전 보정 테스트: 잔여 각도 < 0.5°"""
    img = make_rotated_rect(480, 640, 200, 200, angle_deg)
    bc = BinaryCache(img, thresh=127)
    aligned = bc.get_aligned_roi(min_area=100)
    if aligned is None:
        return False, "get_aligned_roi() 가 None 반환"

    residual = _measure_residual_angle(aligned)
    passed = residual < 0.5
    return passed, f"residual: {residual:.2f}°"


def test_empty_frame() -> TestResult:
    """빈 프레임(전부 검정)일 때 None 반환"""
    img = np.zeros((480, 640), dtype=np.uint8)
    bc = BinaryCache(img, thresh=127)
    result = bc.get_aligned_roi()
    passed = result is None
    return passed, "None 반환" if passed else "None 이 아닌 값 반환됨"


def test_boundary_object() -> TestResult:
    """이미지 경계에 걸쳐 잘린 사각형 — 크래시 없이 동작"""
    # 왼쪽 상단 모서리에 걸치게 배치 (cx=30, cy=30)
    img = make_rotated_rect(480, 640, 200, 200, 5.0, cx=30, cy=30)
    bc = BinaryCache(img, thresh=127)
    try:
        result = bc.get_aligned_roi(min_area=100)
        # None 이어도 되고, 유효한 이미지여도 됨 — 크래시만 아니면 통과
        if result is not None:
            detail = f"유효한 이미지 반환 ({result.shape})"
        else:
            detail = "None 반환 (경계 객체 무시)"
        return True, detail
    except Exception as e:
        return False, f"예외 발생: {e}"


def test_multi_contour() -> TestResult:
    """노이즈 블롭 + 메인 사각형 → 가장 큰 윤곽 사용 확인"""
    img = make_rotated_rect(480, 640, 200, 200, 7.0)

    # 작은 노이즈 블롭 추가 (좌측 하단)
    cv2.circle(img, (50, 430), 15, 255, -1)
    cv2.rectangle(img, (580, 20), (620, 50), 255, -1)

    bc = BinaryCache(img, thresh=127)
    aligned = bc.get_aligned_roi(min_area=100)
    if aligned is None:
        return False, "get_aligned_roi() 가 None 반환"

    residual = _measure_residual_angle(aligned)
    passed = residual < 0.5
    return passed, f"residual: {residual:.2f}° (노이즈 블롭 무시됨)"


def test_aspect_ratio() -> TestResult:
    """100×200 직사각형(종횡비 2:1) 보존 확인 (허용 오차 10%)"""
    rect_w, rect_h = 200, 100  # 가로 200, 세로 100 → 종횡비 2:1
    img = make_rotated_rect(480, 640, rect_w, rect_h, 8.0)
    bc = BinaryCache(img, thresh=127)
    aligned = bc.get_aligned_roi(min_area=100)
    if aligned is None:
        return False, "get_aligned_roi() 가 None 반환"

    h_out, w_out = aligned.shape[:2]
    # 출력 ROI 에서 실제 객체의 종횡비를 측정
    _, binary = cv2.threshold(aligned, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, "정렬된 이미지에서 윤곽 미검출"

    largest = max(contours, key=cv2.contourArea)
    (_, _), (mw, mh), _ = cv2.minAreaRect(largest)
    long_side = max(mw, mh)
    short_side = min(mw, mh)

    if short_side < 1:
        return False, "short_side ≈ 0"

    ratio = long_side / short_side
    target = 2.0
    err = abs(ratio - target) / target
    passed = err < 0.10
    return passed, f"종횡비: {ratio:.2f} (목표: 2.00, 오차: {err*100:.1f}%)"


def test_kill_switch() -> TestResult:
    """min_area 를 매우 높게 설정 → None 반환"""
    img = make_rotated_rect(480, 640, 200, 200, 5.0)
    bc = BinaryCache(img, thresh=127)
    result = bc.get_aligned_roi(min_area=999_999)
    passed = result is None
    return passed, "None 반환" if passed else "None 이 아닌 값 반환됨"


def test_cache_consistency() -> TestResult:
    """get_aligned_roi() 를 두 번 호출해도 동일 결과"""
    img = make_rotated_rect(480, 640, 200, 200, 10.0)
    bc = BinaryCache(img, thresh=127)
    r1 = bc.get_aligned_roi(min_area=100)
    r2 = bc.get_aligned_roi(min_area=100)
    if r1 is None or r2 is None:
        return False, "get_aligned_roi() 가 None 반환"

    identical = np.array_equal(r1, r2)
    return identical, "동일 결과" if identical else "결과가 다름"


def test_rotation_info() -> TestResult:
    """get_aligned_roi() 후 get_rotation_info() 가 유효한 값 반환"""
    angle_in = 12.0
    img = make_rotated_rect(480, 640, 200, 200, angle_in)
    bc = BinaryCache(img, thresh=127)
    aligned = bc.get_aligned_roi(min_area=100)
    if aligned is None:
        return False, "get_aligned_roi() 가 None 반환"

    info = bc.get_rotation_info()
    if info is None:
        return False, "get_rotation_info() 가 None 반환"

    angle, (cx, cy), (w, h) = info
    # 기본 유효성: 타입 확인 + 값 범위 확인
    ok = (
        isinstance(angle, float)
        and isinstance(cx, float) and isinstance(cy, float)
        and isinstance(w, float) and isinstance(h, float)
        and w > 0 and h > 0
    )
    detail = f"angle={angle:.2f}°, center=({cx:.0f},{cy:.0f}), size=({w:.0f}×{h:.0f})"
    return ok, detail


# ─────────────────────────────────────────────
# 실행 + 결과 출력
# ─────────────────────────────────────────────

def main() -> None:
    tests: List[Tuple[str, callable]] = [
        ("Normal rotation 5°",          lambda: test_normal_rotation(5.0)),
        ("Normal rotation 10°",         lambda: test_normal_rotation(10.0)),
        ("Normal rotation 15°",         lambda: test_normal_rotation(15.0)),
        ("Empty frame",                 test_empty_frame),
        ("Boundary object",             test_boundary_object),
        ("Multi contour",               test_multi_contour),
        ("Aspect ratio preservation",   test_aspect_ratio),
        ("Kill switch",                 test_kill_switch),
        ("Cache consistency",           test_cache_consistency),
        ("Rotation info",               test_rotation_info),
    ]

    results: List[Tuple[str, bool, str]] = []

    for name, fn in tests:
        try:
            passed, detail = fn()
        except Exception as e:
            passed, detail = False, f"예외: {e}"
        results.append((name, passed, detail))

    # ── 요약 출력 ──
    print()
    print("=== Alignment Test Results ===")
    pass_count = 0
    for name, passed, detail in results:
        tag = "[PASS]" if passed else "[FAIL]"
        if passed:
            pass_count += 1
        print(f"  {tag} {name} — {detail}")

    total = len(results)
    print()
    if pass_count == total:
        print(f"Result: {pass_count}/{total} PASSED ✅")
    else:
        print(f"Result: {pass_count}/{total} PASSED ❌")
    print()

    # 실패가 하나라도 있으면 종료 코드 1
    sys.exit(0 if pass_count == total else 1)


if __name__ == '__main__':
    main()
