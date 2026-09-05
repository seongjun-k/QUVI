#!/usr/bin/env python3
"""
QUVI 프린터 모니터 노드 — Klipper/Moonraker 출력 상태 감시.

Moonraker HTTP API 를 주기적으로 폴링해 출력 진행 상태를 발행하고,
"출력 중 → 완료" 전이가 관측된 순간에 완료 신호를 1회 발행한다.
심사 피드백의 "출력 직후 자동 검증" 무인 루프에서 시작 트리거로 쓰인다.

이 노드는 **아무 것도 움직이지 않는다** — 관측과 발행만 한다.
완료 신호를 받아 실제 시퀀스를 시작할지는 orchestrator 쪽 정책이다.

Klipper 쪽 준비물은 Moonraker 가 떠 있는 것뿐이다(Mainsail/Fluidd 설치면 기본 포함).
프린터 매크로(PRINT_END 등) 수정은 필요 없다 — 매크로에 의존하면 매크로를
안 거치고 끝난 출력에서 신호를 놓친다.
"""

import json

import requests
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

import quvi_robot_control.topics as topics

# Klipper print_stats.state 값. 완료 판정은 'complete' 하나만 인정한다 —
# 'cancelled'/'error' 는 출력물이 정상 완성되지 않았으므로 검사를 걸지 않는다.
STATE_PRINTING = 'printing'
STATE_COMPLETE = 'complete'


class PrinterMonitorNode(Node):
    def __init__(self):
        super().__init__('printer_monitor_node')

        self.declare_parameter('moonraker_url', 'http://localhost:7125')
        self.declare_parameter('poll_sec', 2.0)
        self.declare_parameter('request_timeout_sec', 3.0)
        # 출력 시작은 사람 없이 히터를 켜는 동작이다. 기본은 차단이고,
        # 켜는 책임은 명시적으로 파라미터를 true 로 준 쪽에 있다.
        self.declare_parameter('allow_print_start', False)

        self._url = str(self.get_parameter('moonraker_url').value).rstrip('/')
        poll_sec = float(self.get_parameter('poll_sec').value)
        self._timeout = float(self.get_parameter('request_timeout_sec').value)
        self._allow_start = bool(self.get_parameter('allow_print_start').value)

        self._status_pub = self.create_publisher(String, topics.TOPIC_PRINTER_STATUS, 10)
        # 완료 신호는 놓치면 무인 루프가 통째로 멈춘다 — 구독자가 늦게 떠도
        # 받을 수 있도록 depth 를 여유있게 둔다.
        self._done_pub = self.create_publisher(Bool, topics.TOPIC_PRINTER_PRINT_DONE, 10)

        # 이전 폴링에서 본 상태. None = 아직 한 번도 못 봄(기동 직후).
        self._prev_state = None
        # 마지막으로 관측된 gcode 파일명. 재출력 대상은 슬라이서가 이미
        # Moonraker 에 올려둔 이 파일이다 — 슬라이서를 다시 거칠 필요가 없다.
        self._last_filename = ''
        # 마지막 완료 발행 이후 'printing' 을 실제로 관측했는지. 완료 상태로
        # 머무는 동안 재연결·재기동으로 완료가 중복 발행되는 것을 막는다.
        self._saw_printing = False
        self._connected = False

        # 출력 시작/취소는 토픽이 아니라 서비스다 — 성공·실패 사유를 호출자가
        # 반드시 받아야 하는 외부 하드웨어 동작이라 fire-and-forget 이면 안 된다.
        self.create_service(Trigger, topics.SRV_PRINTER_START_PRINT, self._start_print_cb)
        self.create_service(Trigger, topics.SRV_PRINTER_CANCEL_PRINT, self._cancel_print_cb)

        self.create_timer(poll_sec, self._poll)
        self.get_logger().info(
            f'프린터 모니터 시작 | Moonraker={self._url} | 폴링={poll_sec}s | '
            f'출력시작 허용={self._allow_start}')

    # ─── Moonraker 폴링 ───
    def _poll(self):
        try:
            resp = requests.get(
                f'{self._url}/printer/objects/query',
                params={'print_stats': '', 'virtual_sdcard': '', 'heater_bed': '', 'extruder': ''},
                timeout=self._timeout)
            resp.raise_for_status()
            status = resp.json()['result']['status']
        except Exception as e:
            if self._connected:
                self.get_logger().warning(f'Moonraker 연결 끊김: {e}')
            else:
                self.get_logger().warning(
                    f'Moonraker 응답 없음({self._url}): {e}',
                    throttle_duration_sec=30.0)
            self._connected = False
            # 연결이 끊긴 동안의 상태 변화는 알 수 없다. 재연결 직후를
            # 새 전이로 오인하지 않도록 이전 상태를 지운다.
            self._prev_state = None
            self._publish_status(None, '', 0.0, None, None)
            return

        if not self._connected:
            self.get_logger().info('Moonraker 연결됨')
            self._connected = True

        stats = status.get('print_stats', {})
        state = str(stats.get('state', '')).lower()
        filename = str(stats.get('filename', '') or '')
        if filename:
            self._last_filename = filename
        progress = float(status.get('virtual_sdcard', {}).get('progress', 0.0) or 0.0)
        # 베드 온도. 히터가 없거나 응답에 없으면 None — 소비자가 '모름'과 '차갑다'를
        # 구분할 수 있어야 한다(모르면 파지를 시작하면 안 된다).
        bed = status.get('heater_bed') or {}
        bed_temp = bed.get('temperature')
        bed_temp = float(bed_temp) if bed_temp is not None else None
        extruder = status.get('extruder') or {}
        nozzle_temp = extruder.get('temperature')
        nozzle_temp = float(nozzle_temp) if nozzle_temp is not None else None

        self._publish_status(state, filename, progress, bed_temp, nozzle_temp)
        self._check_done_edge(state, filename)
        self._prev_state = state

    def _check_done_edge(self, state: str, filename: str):
        """'출력 중'을 관측한 뒤 'complete' 로 바뀐 순간에만 1회 발행."""
        if state == STATE_PRINTING:
            self._saw_printing = True
            return
        if state != STATE_COMPLETE:
            return
        if self._prev_state == STATE_COMPLETE or not self._saw_printing:
            return
        self._saw_printing = False
        self._done_pub.publish(Bool(data=True))
        self.get_logger().info(f'출력 완료 감지 → 완료 신호 발행 | 파일={filename}')

    # ─── 출력 제어 서비스 ───
    def _start_print_cb(self, _req, resp):
        """마지막으로 출력한 gcode 를 다시 출력한다. 자동 호출 경로는 아직 없다."""
        if not self._allow_start:
            resp.success = False
            resp.message = '출력 시작이 차단돼 있습니다 (allow_print_start=false)'
            return resp
        if self._prev_state == STATE_PRINTING:
            resp.success = False
            resp.message = '이미 출력 중입니다'
            return resp
        if not self._last_filename:
            resp.success = False
            resp.message = '출력할 gcode 파일명을 모릅니다 (한 번도 출력 이력이 없음)'
            return resp
        resp.success, resp.message = self._post(
            '/printer/print/start', {'filename': self._last_filename})
        if resp.success:
            self.get_logger().warning(f'출력 시작 요청: {self._last_filename}')
        return resp

    def _cancel_print_cb(self, _req, resp):
        """출력 취소. 정지 방향이라 allow_print_start 와 무관하게 항상 허용한다."""
        resp.success, resp.message = self._post('/printer/print/cancel', None)
        if resp.success:
            self.get_logger().warning('출력 취소 요청 전송')
        return resp

    def _post(self, path: str, params):
        try:
            r = requests.post(f'{self._url}{path}', params=params, timeout=self._timeout)
            r.raise_for_status()
            return True, f'Moonraker {path} OK'
        except Exception as e:
            self.get_logger().error(f'Moonraker {path} 실패: {e}')
            return False, f'Moonraker {path} 실패: {e}'

    def _publish_status(self, state, filename: str, progress: float, bed_temp, nozzle_temp):
        self._status_pub.publish(String(data=json.dumps({
            'connected': self._connected,
            'state': state or 'unknown',
            'filename': filename,
            'progress': round(progress, 4),
            'bed_temp': round(bed_temp, 1) if bed_temp is not None else None,
            'nozzle_temp': round(nozzle_temp, 1) if nozzle_temp is not None else None,
        }, ensure_ascii=False)))


def main(args=None):
    rclpy.init(args=args)
    node = PrinterMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
