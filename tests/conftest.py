"""tests 공통 설정.

- rclpy 미설치 호스트에서는 test_regressions.py 를 수집 대상에서 제외한다.
- _rclpy_session: ROS 2 컨텍스트 init/shutdown 를 테스트 단위로 감싼다
  (테스트 본문이 자체 init/shutdown 을 호출해도 안전하도록 ok() 가드).
"""
import importlib.util

import pytest

if importlib.util.find_spec('rclpy') is None:
    collect_ignore = ['test_regressions.py']


@pytest.fixture
def _rclpy_session():
    import rclpy
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()
