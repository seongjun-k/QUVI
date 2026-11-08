"""물리 비상정지 버튼 해제(/system/estop=False) 시 자동 복귀 규칙.

실제 노드 메서드를 가짜 self 로 호출한다(하드웨어·ROS 그래프 불필요).
"""
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip('rclpy')
from std_msgs.msg import Bool  # noqa: E402

from quvi_robot_control import main_orchestrator_node as orch  # noqa: E402
from quvi_robot_control import robot_control_node as rc  # noqa: E402


class _Log:
    def error(self, _m): pass
    def info(self, _m): pass


# ─── 오케스트레이터 ───
def _fsm(state, msg):
    led = []
    return SimpleNamespace(_state=state, _error_msg=msg, get_logger=lambda: _Log(),
                           _auto_start_at=123.0, led=led,
                           _led_pub=SimpleNamespace(publish=lambda m: led.append(m.data)))


def test_fsm_estop_then_release_returns_idle():
    f = _fsm(orch.FsmState.GRASPING_WAIT, '')
    orch.MainOrchestratorNode._estop_system_cb(f, Bool(data=True))
    assert f._state == orch.FsmState.ERROR and f._error_msg == 'ESTOP ACTIVE'
    assert f._auto_start_at is None and f.led == [False]
    orch.MainOrchestratorNode._estop_system_cb(f, Bool(data=False))
    assert f._state == orch.FsmState.IDLE and f._error_msg == ''


def test_fsm_release_ignored_for_other_errors_and_idle():
    f = _fsm(orch.FsmState.ERROR, 'GRASP_TIMEOUT')
    orch.MainOrchestratorNode._estop_system_cb(f, Bool(data=False))
    assert f._state == orch.FsmState.ERROR
    f = _fsm(orch.FsmState.IDLE, '')
    orch.MainOrchestratorNode._estop_system_cb(f, Bool(data=False))
    assert f._state == orch.FsmState.IDLE


# ─── 로봇 노드 ───
def _robot(state, latched):
    r = SimpleNamespace(_estop_latched=latched, _abort_event=threading.Event(), resets=0)
    r.get_logger = lambda: _Log()
    r._get_state = lambda: state
    r._safe_estop_cleanup = lambda: None
    done = threading.Event()

    def _reset():
        r.resets += 1
        done.set()
    r._execute_reset = _reset
    r.done = done
    return r


def test_robot_release_resets_only_after_estop():
    r = _robot(rc.RobotState.ERROR, latched=False)
    rc.RobotControlNode._estop_cmd_callback(r, Bool(data=True))
    assert r._estop_latched and r._abort_event.is_set()
    rc.RobotControlNode._estop_cmd_callback(r, Bool(data=False))
    assert r.done.wait(2) and r.resets == 1 and not r._estop_latched


def test_robot_release_ignored_without_estop_latch():
    r = _robot(rc.RobotState.ERROR, latched=False)   # 예: RESET FAILED
    rc.RobotControlNode._estop_cmd_callback(r, Bool(data=False))
    assert not r.done.wait(0.2) and r.resets == 0


def test_robot_release_ignored_when_not_error():
    r = _robot(rc.RobotState.IDLE, latched=True)
    rc.RobotControlNode._estop_cmd_callback(r, Bool(data=False))
    assert not r.done.wait(0.2) and r.resets == 0


def test_idle_usb_loss_starts_recovery_once():
    """대기 중 포트 핸들이 죽어 EIO 가 이어지면 20회째에 한 번만 재연결 스레드를 띄운다."""
    class _Bus:
        port_handler = SimpleNamespace(is_using=True)

        def sync_read(self, *_a, **_k):
            raise OSError(5, 'Input/output error')

    log = SimpleNamespace(error=lambda *_a, **_k: None)
    started = []
    r = SimpleNamespace(_use_real_hardware=True, _dxl_ready=True, _dxl_io_lock=threading.Lock(),
                        _follower=SimpleNamespace(bus=_Bus()), _read_io_errors=0,
                        _usb_recovering=False, _estop_latched=False,
                        get_logger=lambda: log, _get_state=lambda: rc.RobotState.IDLE,
                        _recover_after_usb_loss=lambda: started.append(1))
    for _ in range(25):
        assert rc.RobotControlNode._read_raw_positions(r) is None
    for _ in range(50):
        if started:
            break
        threading.Event().wait(0.01)
    assert started == [1] and r._usb_recovering
