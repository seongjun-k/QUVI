"""다품종 검사 판정 안전 게이트 오프라인 테스트.

배경: 검사 PASS/FAIL 은 로봇 분류 동작(레일 PASS/FAIL 스테이션)을 gate 하는
안전 관련 판정이다. 다품종 전환(2026-09) 으로 도입된 3가지 안전 조건이
실제로 지켜지는지 하드웨어/노드 실행 없이 로직 단위로 검증한다.

(a) UNSELECTED 상태에서 검사 실행 시 PASS 가 나오지 않고
    fail_reason 에 "품종 미선택" 이 포함된다.
(b) 불완전 품종(기준이미지 일부)은 캡처 대상으로 선택은 되지만(자산을 쌓아
    완성하려면 선택돼야 하므로), 그 상태로 검사하면 자산 미비 FAIL 이 나간다.
(c) `_inspection_active=True` 일 때 품종 스왑 요청이 거부된다.

실행: cd <repo> && pytest tests/test_multiproduct_gate.py
(ROS 2 rclpy 필요; 미설치 호스트는 conftest 가 test_regressions.py 만 제외하므로
 이 파일도 rclpy 를 직접 요구한다 — 없으면 전체가 스킵되도록 모듈 상단에서 처리)
"""
import os
import shutil
import tempfile

import pytest

rclpy = pytest.importorskip('rclpy')

from std_msgs.msg import String  # noqa: E402

from quvi_inspect.inspect_node import InspectNode  # noqa: E402

pytestmark = pytest.mark.usefixtures('_rclpy_session')


def _make_node(products_dir: str) -> InspectNode:
    return InspectNode(
        parameter_overrides=[
            rclpy.parameter.Parameter(
                'inspection_products_dir', rclpy.Parameter.Type.STRING, products_dir),
            rclpy.parameter.Parameter(
                'anomaly_enabled', rclpy.Parameter.Type.BOOL, False),
        ]
    )


def _make_complete_product(products_dir: str, pid: str, angles=(0, 90, 180, 270)):
    """4각도 기준이미지가 전부 존재하는 완전한 품종을 만든다 (더미 PNG)."""
    import cv2
    import numpy as np
    ref_dir = os.path.join(products_dir, pid, 'reference_images')
    os.makedirs(ref_dir, exist_ok=True)
    dummy = np.zeros((32, 32), dtype=np.uint8)
    for angle in angles:
        cv2.imwrite(os.path.join(ref_dir, f'ref_{angle}.png'), dummy)


def _make_incomplete_product(products_dir: str, pid: str):
    """기준이미지 3장만 있는(4장 미만) 불완전한 품종을 만든다."""
    import cv2
    import numpy as np
    ref_dir = os.path.join(products_dir, pid, 'reference_images')
    os.makedirs(ref_dir, exist_ok=True)
    dummy = np.zeros((32, 32), dtype=np.uint8)
    for angle in (0, 90, 180):   # 270 누락
        cv2.imwrite(os.path.join(ref_dir, f'ref_{angle}.png'), dummy)


@pytest.fixture
def products_dir():
    d = tempfile.mkdtemp(prefix='quvi_test_products_')
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_unselected_inspection_blocks_pass(products_dir):
    """(a) 품종 미선택 상태에서 검사가 트리거되면 정상 판정을 실행하지 않고
    passed=False, fail_reason='품종 미선택 — 검사 불가' 를 발행해야 한다."""
    node = _make_node(products_dir)
    try:
        assert node._current_product is None
        assert node._reference_images == {}

        published = {}
        node._result_pub.publish = lambda msg: published.update(
            passed=msg.passed, fail_reason=msg.fail_reason)

        node._run_inspection_inner()

        assert published['passed'] is False
        assert '품종 미선택' in published['fail_reason']
    finally:
        node.destroy_node()


def test_incomplete_product_selectable_but_inspection_blocked(products_dir):
    """(b) 기준이미지가 3장만 있는 불완전 품종은 캡처 대상으로 '선택은' 되지만,
    그 상태로 검사가 트리거되면 정상 판정을 실행하지 않고 자산 미비 FAIL 이 나가야 한다.
    (선택을 막으면 기준이미지 캡처 자체가 불가능해지는 교착을 피하기 위함 —
     안전은 '선택 거부'가 아니라 '판정 실행 게이트'로 지킨다.)"""
    node = _make_node(products_dir)
    try:
        _make_incomplete_product(products_dir, 'partial_product')

        status = node._product_status('partial_product')
        assert status['ready'] is False
        assert 'ref_270' in status['missing']

        # 선택 요청 → 백그라운드 스레드 대신 로직을 동기 호출로 검증
        node._reload_product('partial_product')

        # 선택은 성공(캡처 대상 지정), 있는 기준이미지 3장만 로드됨
        assert node._current_product == 'partial_product'
        assert len(node._reference_images) == 3

        # 그러나 불완전(4각도 미만)이라 검사는 자산 미비 FAIL
        published = {}
        node._result_pub.publish = lambda msg: published.update(
            passed=msg.passed, fail_reason=msg.fail_reason)
        node._run_inspection_inner()
        assert published['passed'] is False
        assert '자산 미비' in published['fail_reason']
    finally:
        node.destroy_node()


def test_product_swap_rejected_during_active_inspection(products_dir):
    """(c) _inspection_active=True 인 동안 들어온 품종 전환 요청은 거부되고
    현재 품종이 바뀌지 않아야 한다(구/신 자산 혼합 방지)."""
    node = _make_node(products_dir)
    try:
        _make_complete_product(products_dir, 'product_a')
        node._reload_product('product_a')
        assert node._current_product == 'product_a'

        _make_complete_product(products_dir, 'product_b')
        node._inspection_active = True
        try:
            node._on_product_select(String(data='product_b'))
            # _on_product_select 는 검사 활성 시 스레드조차 띄우지 않고 즉시 거부한다.
            assert node._current_product == 'product_a'
        finally:
            node._inspection_active = False
    finally:
        node.destroy_node()


if __name__ == '__main__':
    # ponytail: 프레임워크 없이도 핵심 3가지를 바로 확인할 수 있는 최소 self-check.
    import sys
    sys.exit(pytest.main([__file__, '-v']))
