import math
import pytest
from quvi_robot_control.robot_control_node import raw_to_rad, rad_to_raw

# 기존 3곳(관측/실HW 액션/SIM 액션)에 인라인으로 있던 변환식을 리터럴로 재현 —
# raw_to_rad/rad_to_raw 헬퍼가 이 식들과 완전히 동일한 결과를 내는지 검증한다.
legacy_raw_to_rad = lambda raw: (raw - 2048.0) * (2.0 * math.pi) / 4096.0
legacy_rad_to_raw = lambda rad: 2048.0 + rad * 4096.0 / (2.0 * math.pi)

RAW_SAMPLES = [0, 830, 1258, 2048, 3129, 4095]
RAD_SAMPLES = [0.0, math.pi / 2, -math.pi / 2, math.pi, -math.pi, -1.56, 1.6]


@pytest.mark.parametrize('raw', RAW_SAMPLES)
def test_raw_to_rad_matches_legacy(raw):
    assert raw_to_rad(raw) == legacy_raw_to_rad(raw)


@pytest.mark.parametrize('rad', RAD_SAMPLES)
def test_rad_to_raw_matches_legacy(rad):
    assert rad_to_raw(rad) == legacy_rad_to_raw(rad)


@pytest.mark.parametrize('rad', RAD_SAMPLES)
def test_round_trip_rad(rad):
    assert abs(raw_to_rad(rad_to_raw(rad)) - rad) < 1e-9


@pytest.mark.parametrize('raw', RAW_SAMPLES)
def test_round_trip_raw_tick(raw):
    assert int(round(rad_to_raw(raw_to_rad(raw)))) == raw
