#!/usr/bin/env python3
"""
QUVI MAIN_ORCHESTRATOR_NODE
────────────────────────────────────────────────────────────────
전체 3D 프린터 출력물 자동 양불 판정 시스템의 핵심 두뇌.
비전 탐지(YOLO), 로봇 파지(ACT), 다방향 품질 검사(SSIM), 분류 레일 제어를
FSM(유한상태머신)으로 자율 조율 및 통합합니다.

구현 규칙 준수: 이모지 금지, 한국어 주석, 간결한 핵심 논리 중심.
"""

import time
from enum import Enum
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Int32, String
from quvi_msgs.msg import GraspGoal, InspectionResult, ObjectArray, SystemStatus
import quvi_robot_control.topics as topics


class FsmState(Enum):
    INIT = "INIT"
    IDLE = "IDLE"
    STARTUP_RAIL_HOME_TRIGGER  = "STARTUP_RAIL_HOME_TRIGGER"   # 시작: 레일 0mm 홈이동
    STARTUP_RAIL_HOME_WAIT     = "STARTUP_RAIL_HOME_WAIT"
    STARTUP_INSPECT_TRIGGER    = "STARTUP_INSPECT_TRIGGER"     # 시작: 레일 12.5mm 검사장
    STARTUP_INSPECT_WAIT       = "STARTUP_INSPECT_WAIT"
    STARTUP_TURNTABLE_TRIGGER  = "STARTUP_TURNTABLE_TRIGGER"   # 시작: 턴테이블 0°
    STARTUP_TURNTABLE_WAIT     = "STARTUP_TURNTABLE_WAIT"
    START_RAIL_MOVE_BED_TRIGGER = "START_RAIL_MOVE_BED_TRIGGER"
    START_RAIL_MOVE_BED_WAIT = "START_RAIL_MOVE_BED_WAIT"
    GRASPING_TRIGGER = "GRASPING_TRIGGER"
    GRASPING_WAIT = "GRASPING_WAIT"
    PLACING_CHAMBER_TRIGGER = "PLACING_CHAMBER_TRIGGER"
    PLACING_CHAMBER_WAIT = "PLACING_CHAMBER_WAIT"
    INSPECTING_TRIGGER = "INSPECTING_TRIGGER"
    INSPECTING_ROTATE = "INSPECTING_ROTATE"
    INSPECTING_WAIT_TURNTABLE = "INSPECTING_WAIT_TURNTABLE"
    INSPECTING_CAPTURE = "INSPECTING_CAPTURE"
    INSPECTING_WAIT_RESULT = "INSPECTING_WAIT_RESULT"
    PICKING_CHAMBER_TRIGGER = "PICKING_CHAMBER_TRIGGER"
    PICKING_CHAMBER_WAIT = "PICKING_CHAMBER_WAIT"
    SORTING_TRIGGER = "SORTING_TRIGGER"
    SORTING_WAIT_RAIL = "SORTING_WAIT_RAIL"
    RELEASING_TRIGGER = "RELEASING_TRIGGER"
    RELEASING_WAIT = "RELEASING_WAIT"
    HOMING_RAIL_TRIGGER = "HOMING_RAIL_TRIGGER"
    HOMING_RAIL_WAIT = "HOMING_RAIL_WAIT"
    HOMING_ARM_TRIGGER = "HOMING_ARM_TRIGGER"
    HOMING_ARM_WAIT = "HOMING_ARM_WAIT"
    FINISHED = "FINISHED"
    ERROR = "ERROR"


class MainOrchestratorNode(Node):
    """자율 자동화 시퀀스를 총괄 제어하는 메인 오케스트레이터 노드."""

    def __init__(self, **kwargs):
        super().__init__('main_orchestrator_node', **kwargs)

        # ─── 파라미터 선언 ───
        self.declare_parameter('use_act', False)
        self.declare_parameter('px_to_mm_x', 0.5)
        self.declare_parameter('px_to_mm_y', 0.5)
        self.declare_parameter('offset_x', 100.0)
        self.declare_parameter('offset_y', 100.0)
        self.declare_parameter('target_z', 15.0)
        self.declare_parameter('step_delay_sec', 2.0)
        self.declare_parameter('loop_rate_hz', 10.0)
        self.declare_parameter('grasp_timeout_sec', 20.0)
        self.declare_parameter('release_timeout_sec', 10.0)
        self.declare_parameter('home_timeout_sec', 15.0)
        self.declare_parameter('rail_timeout_sec', 25.0)
        self.declare_parameter('inspect_timeout_sec', 15.0)
        self.declare_parameter('detecting_timeout_sec', 10.0)

        # ─── 파라미터 로드 ───
        self._use_act = self.get_parameter('use_act').value
        self._px_to_mm_x = self.get_parameter('px_to_mm_x').value
        self._px_to_mm_y = self.get_parameter('px_to_mm_y').value
        self._offset_x = self.get_parameter('offset_x').value
        self._offset_y = self.get_parameter('offset_y').value
        self._target_z = self.get_parameter('target_z').value
        self._step_delay = self.get_parameter('step_delay_sec').value
        self._loop_rate = self.get_parameter('loop_rate_hz').value
        self._grasp_timeout = self.get_parameter('grasp_timeout_sec').value
        self._release_timeout = self.get_parameter('release_timeout_sec').value
        self._home_timeout = self.get_parameter('home_timeout_sec').value
        self._rail_timeout = self.get_parameter('rail_timeout_sec').value
        self._inspect_timeout = self.get_parameter('inspect_timeout_sec').value
        self._detecting_timeout = self.get_parameter('detecting_timeout_sec').value

        # ─── 내부 상태 변수 ───
        self._state = FsmState.INIT
        self._prev_state = None
        self._error_msg = ""

        # 작업 대상 객체 관련 데이터
        self._detected_objects = []
        self._current_object_idx = 0
        self._total_objects = 0
        self._processed_count = 0
        self._pass_count = 0
        self._fail_count = 0

        # 검사 로직 관련
        self._inspect_angles = [0, 90, 180, 270]
        self._inspect_angle_idx = 0

        # 검사 결과 임시 저장
        self._latest_inspection_passed = False

        # 하위 노드 헬스 체크용 플래그
        self._yolo_online = False
        self._grasp_online = False
        self._inspect_online = False
        self._motor_online = False
        self._act_ready = False

        # 완료 토픽 수신 플래그
        self._yolo_received = False
        self._robot_grasp_done = False
        self._robot_rail_done = False
        self._robot_release_done = False
        self._robot_home_done = False
        self._inspect_done = False
        self._turntable_done = False
        self._robot_rotate_done = False
        self._place_chamber_done = False
        self._pick_chamber_done = False
        self._startup_rail_done = False      # 시작 초기화 레일 done
        self._startup_turntable_done = False # 시작 초기화 턴테이블 done

        # 타이머 카운터 (FSM 딜레이 대기용)
        self._state_timer_counter = 0

        # ─── ROS 2 인터페이스 설정 ───
        self._setup_publishers()
        self._setup_subscribers()

        # FSM 주기 제어 타이머 기동 (10 Hz)
        self._fsm_timer = self.create_timer(1.0 / self._loop_rate, self._fsm_loop)

        # HMI 상태 전송 타이머 (2 Hz)
        self._hmi_pub_timer = self.create_timer(0.5, self._publish_hmi_status)

        self.get_logger().info('MAIN_ORCHESTRATOR_NODE 기동 완료 | 자율 제어 모드 대기')

    def _setup_publishers(self):
        # HMI 대시보드 상태 통보
        self._hmi_status_pub = self.create_publisher(SystemStatus, topics.TOPIC_HMI_STATUS, 10)

        # 하위 노드 트리거 발행
        self._yolo_trigger_pub = self.create_publisher(Bool, topics.TOPIC_DETECTION_TRIGGER, 10)
        self._inspect_trigger_pub = self.create_publisher(Bool, topics.TOPIC_INSPECTION_TRIGGER, 10)

        # 로봇 및 구동부 명령 발행
        self._robot_grasp_pub = self.create_publisher(GraspGoal, topics.TOPIC_ROBOT_GRASP_CMD, 10)
        self._robot_rail_pub = self.create_publisher(Int32, topics.TOPIC_ROBOT_RAIL_CMD, 10)
        self._robot_rotate_pub = self.create_publisher(Bool, topics.TOPIC_ROBOT_ROTATE_CMD, 10)
        self._robot_release_pub = self.create_publisher(Bool, topics.TOPIC_ROBOT_RELEASE_CMD, 10)
        self._robot_home_pub = self.create_publisher(Bool, topics.TOPIC_ROBOT_HOME_CMD, 10)
        self._robot_reset_pub = self.create_publisher(Bool, topics.TOPIC_ROBOT_RESET_CMD, 10)
        self._robot_place_chamber_pub = self.create_publisher(Bool, topics.TOPIC_ROBOT_PLACE_CHAMBER_CMD, 10)
        self._robot_pick_chamber_pub = self.create_publisher(Bool, topics.TOPIC_ROBOT_PICK_CHAMBER_CMD, 10)

        # STARTUP 시퀀스 전용 구동부 퍼블리셔
        # _turntable_pub 과 동일 토픽이므로 _turntable_pub 을 재사용한다.
        self._turntable_pub = self.create_publisher(Int32, topics.TOPIC_MOTOR_TURNTABLE_CMD, 10)
        self._startup_rail_pub = self.create_publisher(Int32, topics.TOPIC_MOTOR_RAIL_CMD, 10)

    def _setup_subscribers(self):
        # 레일/턴테이블 done 구독 (시작 초기화 시)
        self.create_subscription(Bool, '/motor/rail_done', self._startup_rail_home_done_cb, 10)
        self.create_subscription(Bool, '/motor/rail_done', self._startup_inspect_done_cb, 10)
        self.create_subscription(Bool, '/motor/turntable_done', self._startup_turntable_done_cb, 10)
        self.create_subscription(String, topics.TOPIC_HMI_COMMAND, self._hmi_command_cb, 10)

        # YOLO 탐지 결과 구독
        self.create_subscription(ObjectArray, '/detection/objects', self._yolo_objects_cb, 10)

        # 로봇 피드백 완료 토픽 구독
        self.create_subscription(Bool, topics.TOPIC_ROBOT_ACT_DONE, self._robot_act_done_cb, 10)
        self.create_subscription(Bool, topics.TOPIC_ROBOT_GRASP_DONE, self._robot_grasp_done_cb, 10)
        self.create_subscription(Bool, topics.TOPIC_ROBOT_RELEASE_DONE, self._robot_release_done_cb, 10)
        self.create_subscription(Bool, topics.TOPIC_ROBOT_HOME_DONE, self._robot_home_done_cb, 10)
        self.create_subscription(Bool, topics.TOPIC_MOTOR_TURNTABLE_DONE, self._turntable_done_cb, 10)
        self.create_subscription(Bool, topics.TOPIC_ROBOT_RAIL_DONE, self._robot_rail_done_cb, 10)
        self.create_subscription(Bool, topics.TOPIC_ROBOT_ROTATE_DONE, self._robot_rotate_done_cb, 10)
        self.create_subscription(Bool, topics.TOPIC_ROBOT_PLACE_CHAMBER_DONE, self._place_chamber_done_cb, 10)
        self.create_subscription(Bool, topics.TOPIC_ROBOT_PICK_CHAMBER_DONE, self._pick_chamber_done_cb, 10)

        # 검사 결과 토픽 구독
        self.create_subscription(InspectionResult, '/inspection/result', self._inspect_result_cb, 10)

        # 헬스 체크용 정보 노드 상태 구독
        self.create_subscription(String, topics.TOPIC_ROBOT_STATUS, self._robot_node_status_cb, 10)

        # 비상정지 구독
        self.create_subscription(Bool, topics.TOPIC_ESTOP, self._estop_system_cb, 10)

    # ─── ROS 2 구독 콜백 함수 정의 ───
    def _hmi_command_cb(self, msg: String):
        command = msg.data.upper()
        self.get_logger().info(f'HMI 명령 수신: {command}')

        if command == "START":
            if self._state == FsmState.IDLE:
                if self._use_act and not self._act_ready:
                    self.get_logger().error('ACT 노드가 아직 준비되지 않았습니다. 시작을 차단합니다. ERROR 상태로 전환.')
                    self._error_msg = "ACT NOT READY"
                    self._state = FsmState.ERROR
                    return
                self._startup_rail_done = False
                self._startup_turntable_done = False
                self._state = FsmState.STARTUP_RAIL_HOME_TRIGGER
                self.get_logger().info('시작 초기화 시퀀스: 레일 0mm 홈 이동 시작')
        elif command == "STOP":
            self.get_logger().warn('자율 구동 시퀀스가 정지되었습니다. IDLE 상태로 복귀합니다.')
            self._state = FsmState.IDLE
        elif command == "ESTOP":
            self.get_logger().error('비상 정지 명령(ESTOP)이 작동했습니다! 비상 에러 상태로 강제 천이합니다.')
            self._state = FsmState.ERROR
            self._error_msg = "ESTOP ACTIVE"
        elif command == "RESET":
            self.get_logger().info('시스템 리셋을 시도합니다. 초기화 단계로 진입합니다.')
            reset_msg = Bool()
            reset_msg.data = True
            self._robot_reset_pub.publish(reset_msg)

            self._state = FsmState.INIT
            self._error_msg = ""
            self._processed_count = 0
            self._pass_count = 0
            self._fail_count = 0
            self._act_ready = False

    def _estop_system_cb(self, msg: Bool):
        if msg.data:
            self.get_logger().error('시스템 비상 정지(ESTOP) 수신! 비상 에러 상태로 강제 천이합니다.')
            self._state = FsmState.ERROR
            self._error_msg = "ESTOP ACTIVE"

    def _yolo_objects_cb(self, msg: ObjectArray):
        if self._state.value.startswith("DETECTING_"):
            self.get_logger().info(f'YOLO 객체 수집 완료: {msg.total_count}개 발견')
            self._detected_objects = msg.objects
            self._total_objects = msg.total_count
            self._current_object_idx = 0
            self._yolo_received = True
        # 주기적으로 감지가 수행되고 있으면 헬스체크 정상 처리
        self._yolo_online = True

    def _robot_act_done_cb(self, msg: Bool):
        if msg.data and self._state == FsmState.GRASPING_WAIT:
            self._robot_grasp_done = True

    def _robot_grasp_done_cb(self, msg: Bool):
        # 파지 완료 전용
        if msg.data and self._state == FsmState.GRASPING_WAIT:
            self._robot_grasp_done = True

    def _robot_rotate_done_cb(self, msg: Bool):
        if msg.data and (self._state == FsmState.PICKING_CHAMBER_WAIT or self._state == FsmState.PLACING_CHAMBER_WAIT):
            self._robot_rotate_done = True

    def _place_chamber_done_cb(self, msg: Bool):
        if msg.data and self._state == FsmState.PLACING_CHAMBER_WAIT:
            self._place_chamber_done = True

    def _pick_chamber_done_cb(self, msg: Bool):
        if msg.data and self._state == FsmState.PICKING_CHAMBER_WAIT:
            self._pick_chamber_done = True

    def _robot_release_done_cb(self, msg: Bool):
        # 투하 완료 전용
        if msg.data and self._state == FsmState.RELEASING_WAIT:
            self._robot_release_done = True

    def _robot_home_done_cb(self, msg: Bool):
        # 홈 복귀 완료 전용
        if msg.data and (self._state == FsmState.HOMING_ARM_WAIT):
            self._robot_home_done = True

    def _turntable_done_cb(self, msg: Bool):
        # 턴테이블 회전 완료 전용
        if msg.data and self._state == FsmState.INSPECTING_WAIT_TURNTABLE:
            self._turntable_done = True

    def _robot_rail_done_cb(self, msg: Bool):
        if msg.data:
            if self._state in (FsmState.SORTING_WAIT_RAIL, FsmState.HOMING_RAIL_WAIT, FsmState.START_RAIL_MOVE_BED_WAIT):
                self._robot_rail_done = True

    def _inspect_result_cb(self, msg: InspectionResult):
        if self._state.value.startswith("INSPECTING_"):
            self.get_logger().info(f'검사 결과 수신: passed={msg.passed}')
            self._latest_inspection_passed = msg.passed
            self._inspect_done = True
        self._inspect_online = True

    def _robot_node_status_cb(self, msg: String):
        # 로봇 상태 노드가 작동 중임을 기록
        self._grasp_online = True
        self._motor_online = True
        if "ACT_READY" in msg.data.upper():
            self._act_ready = True

    def _startup_rail_home_done_cb(self, msg: Bool):
        """STARTUP_RAIL_HOME_WAIT 상태에서만 레일 done 을 수락한다."""
        if msg.data and self._state == FsmState.STARTUP_RAIL_HOME_WAIT:
            self._startup_rail_done = True

    def _startup_inspect_done_cb(self, msg: Bool):
        """STARTUP_INSPECT_WAIT 상태에서만 레일 done 을 수락한다."""
        if msg.data and self._state == FsmState.STARTUP_INSPECT_WAIT:
            self._startup_rail_done = True

    def _startup_turntable_done_cb(self, msg: Bool):
        """START 초기화 시퀀스에서 턴테이블 done 수신."""
        if msg.data and self._state == FsmState.STARTUP_TURNTABLE_WAIT:
            self._startup_turntable_done = True

    # ─── FSM 루프 및 상태 천이 제어 ───
    def _fsm_loop(self):
        if self._state != self._prev_state:
            self.get_logger().info(f'[FSM] 상태 변경: {self._prev_state} -> {self._state}')
            self._prev_state = self._state

        # FSM 구현
        if self._state == FsmState.INIT:
            self._state = FsmState.IDLE

        elif self._state == FsmState.IDLE:
            # HMI로부터 START 대기 (Callback에서 처리)
            pass

        # ── 시작 초기화 시퀀스: 레일 0mm → 12.5mm → 턴테이블 0° ──
        elif self._state == FsmState.STARTUP_RAIL_HOME_TRIGGER:
            self._startup_rail_done = False
            self._state_timer_counter = 0
            rail_msg = Int32()
            rail_msg.data = 0  # 0mm 홈 = 0 steps
            self._startup_rail_pub.publish(rail_msg)
            self.get_logger().info('[STARTUP] 레일 0mm 홈 이동 명령 발행')
            self._state = FsmState.STARTUP_RAIL_HOME_WAIT

        elif self._state == FsmState.STARTUP_RAIL_HOME_WAIT:
            self._state_timer_counter += 1
            if self._startup_rail_done:
                self.get_logger().info('[STARTUP] 레일 0mm 도신. 12.5mm 검사장 이동 시작')
                self._state = FsmState.STARTUP_INSPECT_TRIGGER
            elif self._state_timer_counter > int(self._rail_timeout * self._loop_rate):
                self.get_logger().error('[STARTUP] 레일 홈 타임아웃! ERROR')
                self._error_msg = 'STARTUP_RAIL_HOME_TIMEOUT'
                self._state = FsmState.ERROR

        elif self._state == FsmState.STARTUP_INSPECT_TRIGGER:
            self._startup_rail_done = False
            self._state_timer_counter = 0
            rail_msg = Int32()
            rail_msg.data = int(12.5 * 80)  # INSPECT(A) = 1000 steps
            self._startup_rail_pub.publish(rail_msg)
            self.get_logger().info('[STARTUP] 레일 12.5mm INSPECT 이동 명령 발행')
            self._state = FsmState.STARTUP_INSPECT_WAIT

        elif self._state == FsmState.STARTUP_INSPECT_WAIT:
            self._state_timer_counter += 1
            if self._startup_rail_done:
                self.get_logger().info('[STARTUP] 레일 12.5mm 도신. 턴테이블 0° 초기화 시작')
                self._state = FsmState.STARTUP_TURNTABLE_TRIGGER
            elif self._state_timer_counter > int(self._rail_timeout * self._loop_rate):
                self.get_logger().error('[STARTUP] 레일 검사장 타임아웃! ERROR')
                self._error_msg = 'STARTUP_RAIL_INSPECT_TIMEOUT'
                self._state = FsmState.ERROR

        elif self._state == FsmState.STARTUP_TURNTABLE_TRIGGER:
            self._startup_turntable_done = False
            self._state_timer_counter = 0
            turn_msg = Int32()
            turn_msg.data = 0  # 0°
            self._turntable_pub.publish(turn_msg)
            self.get_logger().info('[STARTUP] 턴테이블 0° 초기화 명령 발행')
            self._state = FsmState.STARTUP_TURNTABLE_WAIT

        elif self._state == FsmState.STARTUP_TURNTABLE_WAIT:
            self._state_timer_counter += 1
            if self._startup_turntable_done:
                self.get_logger().info('[STARTUP] 터테이블 0° 완료. 자율 구동 시퀀스 진입')
                self._state = FsmState.START_RAIL_MOVE_BED_TRIGGER
            elif self._state_timer_counter > int(10.0 * self._loop_rate):  # 10초 타임아웃
                self.get_logger().error('[STARTUP] 턴테이블 done 타임아웃. ERROR 상태로 전환.')
                self._error_msg = 'STARTUP_TURNTABLE_TIMEOUT'
                self._state = FsmState.ERROR

        elif self._state == FsmState.START_RAIL_MOVE_BED_TRIGGER:
            self._robot_rail_done = False
            self._state_timer_counter = 0
            rail_cmd = Int32()
            rail_cmd.data = 0  # RailPosition.BED
            self._robot_rail_pub.publish(rail_cmd)
            self.get_logger().info('레일 베드 위치(30,500스텝) 이동 명령 전송')
            self._state = FsmState.START_RAIL_MOVE_BED_WAIT

        elif self._state == FsmState.START_RAIL_MOVE_BED_WAIT:
            self._state_timer_counter += 1
            if self._robot_rail_done:
                self.get_logger().info('레일 베드 위치 이동 완료')
                self._state = FsmState.GRASPING_TRIGGER
            elif self._state_timer_counter > int(self._rail_timeout * self._loop_rate):
                self.get_logger().error('레일 베드 이동 타임아웃! ERROR 상태로 천이')
                self._error_msg = 'START_RAIL_BED_TIMEOUT'
                self._state = FsmState.ERROR

        elif self._state == FsmState.GRASPING_TRIGGER:
            self._total_objects = 1
            self._current_object_idx = 0
            
            goal = GraspGoal()
            goal.header.stamp = self.get_clock().now().to_msg()
            goal.header.frame_id = "camera_handcam"
            goal.target_x = 0.0
            goal.target_y = 0.0
            goal.target_z = float(self._target_z)
            goal.object_index = 0

            self.get_logger().info('로봇 ACT 파지 명령 발행')
            self._robot_grasp_done = False
            self._state_timer_counter = 0
            self._robot_grasp_pub.publish(goal)
            self._state = FsmState.GRASPING_WAIT

        elif self._state == FsmState.GRASPING_WAIT:
            self._state_timer_counter += 1
            if self._robot_grasp_done:
                self.get_logger().info('로봇 ACT 파지 완료. 검사장 안착 단계로 진입')
                self._state = FsmState.PLACING_CHAMBER_TRIGGER
            elif self._state_timer_counter > int(self._grasp_timeout * self._loop_rate):
                self.get_logger().error('로봇 파지 대기 타임아웃! ERROR 상태로 천이')
                self._error_msg = 'GRASP_TIMEOUT'
                self._state = FsmState.ERROR

        elif self._state == FsmState.PLACING_CHAMBER_TRIGGER:
            self._place_chamber_done = False
            self._state_timer_counter = 0
            cmd = Bool()
            cmd.data = True
            self._robot_place_chamber_pub.publish(cmd)
            self.get_logger().info('검사장(턴테이블) 안착 명령 전송')
            self._state = FsmState.PLACING_CHAMBER_WAIT

        elif self._state == FsmState.PLACING_CHAMBER_WAIT:
            self._state_timer_counter += 1
            if self._place_chamber_done:
                self.get_logger().info('검사장 안착 완료. 품질 검사 단계로 진입')
                self._state = FsmState.INSPECTING_TRIGGER
            elif self._state_timer_counter > int(self._grasp_timeout * self._loop_rate):
                self.get_logger().error('검사장 안착 대기 타임아웃! ERROR 상태로 천이')
                self._error_msg = 'PLACE_CHAMBER_TIMEOUT'
                self._state = FsmState.ERROR

        elif self._state == FsmState.INSPECTING_TRIGGER:
            self._inspect_done = False
            # 챔버 품질 검사 노드 활성화
            inspect_trigger = Bool()
            inspect_trigger.data = True
            self._inspect_trigger_pub.publish(inspect_trigger)

            # 턴테이블 회전 및 캡처 제어 진입
            self._inspect_angle_idx = 0
            self._state_timer_counter = 0
            self._state = FsmState.INSPECTING_ROTATE

        elif self._state == FsmState.INSPECTING_ROTATE:
            angle = self._inspect_angles[self._inspect_angle_idx]
            angle_msg = Int32()
            angle_msg.data = angle
            self._turntable_pub.publish(angle_msg)
            self.get_logger().info(f'턴테이블 {angle}도 회전 명령 전송')

            self._turntable_done = False
            self._state_timer_counter = 0
            self._state = FsmState.INSPECTING_WAIT_TURNTABLE

        elif self._state == FsmState.INSPECTING_WAIT_TURNTABLE:
            self._state_timer_counter += 1
            if self._turntable_done:
                self.get_logger().info(f'턴테이블 {self._inspect_angles[self._inspect_angle_idx]}도 회전 완료 확인')
                self._state_timer_counter = 0
                self._state = FsmState.INSPECTING_CAPTURE
            elif self._state_timer_counter > int(self._step_delay * self._loop_rate * 2):
                self.get_logger().warn(
                    f'턴테이블 완료 대기 타임아웃 — '
                    f'{self._inspect_angles[self._inspect_angle_idx]}도 회전이 늦어지고 있습니다. '
                    f'그래도 다음 단계로 진행합니다.')
                self._state_timer_counter = 0
                self._state = FsmState.INSPECTING_CAPTURE

        elif self._state == FsmState.INSPECTING_CAPTURE:
            self._state_timer_counter += 1
            # 기구 안정화 및 이미지 캡처 대기 (step_delay_sec)
            if self._state_timer_counter >= int(self._step_delay * self._loop_rate):
                self._inspect_angle_idx += 1
                if self._inspect_angle_idx < len(self._inspect_angles):
                    self._state = FsmState.INSPECTING_ROTATE
                else:
                    self._state_timer_counter = 0
                    self._state = FsmState.INSPECTING_WAIT_RESULT

        elif self._state == FsmState.INSPECTING_WAIT_RESULT:
            self._state_timer_counter += 1
            if self._inspect_done:
                self._state = FsmState.PICKING_CHAMBER_TRIGGER
            elif self._state_timer_counter > int(self._inspect_timeout * self._loop_rate):
                self.get_logger().error('품질 검사 대기 타임아웃! ERROR 상태로 천이')
                self._error_msg = 'INSPECT_TIMEOUT'
                self._state = FsmState.ERROR

        elif self._state == FsmState.PICKING_CHAMBER_TRIGGER:
            self._pick_chamber_done = False
            self._state_timer_counter = 0
            cmd = Bool()
            cmd.data = True
            self._robot_pick_chamber_pub.publish(cmd)
            self.get_logger().info('검사장 재파지 명령 전송')
            self._state = FsmState.PICKING_CHAMBER_WAIT

        elif self._state == FsmState.PICKING_CHAMBER_WAIT:
            self._state_timer_counter += 1
            if self._pick_chamber_done:
                self.get_logger().info('검사장 재파지 완료. 분류 이송 단계로 진입')
                self._state = FsmState.SORTING_TRIGGER
            elif self._state_timer_counter > int(self._grasp_timeout * self._loop_rate):
                self.get_logger().error('검사장 재파지 대기 타임아웃! ERROR 상태로 천이')
                self._error_msg = 'PICK_CHAMBER_TIMEOUT'
                self._state = FsmState.ERROR

        elif self._state == FsmState.SORTING_TRIGGER:
            self._robot_rail_done = False
            self._state_timer_counter = 0
            rail_cmd = Int32()

            # 양불 결과에 따라 레일 목표 위치 분류
            if self._latest_inspection_passed:
                rail_cmd.data = 2  # RailPosition.PASS
                self._pass_count += 1
                self.get_logger().info('검사 결과: PASS -> PASS 적재함(X=B)으로 레일 이송')
            else:
                rail_cmd.data = 3  # RailPosition.FAIL
                self._fail_count += 1
                self.get_logger().info('검사 결과: FAIL -> FAIL 적재함(X=C)으로 레일 이송')

            self._robot_rail_pub.publish(rail_cmd)
            self._state = FsmState.SORTING_WAIT_RAIL

        elif self._state == FsmState.SORTING_WAIT_RAIL:
            self._state_timer_counter += 1
            if self._robot_rail_done:
                self.get_logger().info('레일 분류 목적지 이동 완료')
                self._state = FsmState.RELEASING_TRIGGER
            elif self._state_timer_counter > int(self._rail_timeout * self._loop_rate):
                self.get_logger().error('레일 이송 대기 타임아웃! ERROR 상태로 천이')
                self._error_msg = 'RAIL_TIMEOUT'
                self._state = FsmState.ERROR

        elif self._state == FsmState.RELEASING_TRIGGER:
            self._robot_release_done = False
            self._state_timer_counter = 0
            release_cmd = Bool()
            release_cmd.data = True
            self._robot_release_pub.publish(release_cmd)
            self.get_logger().info('분류 적재함 위 출력물 투하 명령 전송')
            self._state = FsmState.RELEASING_WAIT

        elif self._state == FsmState.RELEASING_WAIT:
            self._state_timer_counter += 1
            if self._robot_release_done:
                self.get_logger().info('적재 및 그리퍼 해제 완료')
                self._processed_count += 1
                self._current_object_idx += 1
                self._state = FsmState.HOMING_RAIL_TRIGGER
            elif self._state_timer_counter > int(self._release_timeout * self._loop_rate):
                self.get_logger().error('그리퍼 해제 대기 타임아웃! ERROR 상태로 천이')
                self._error_msg = 'RELEASE_TIMEOUT'
                self._state = FsmState.ERROR

        elif self._state == FsmState.HOMING_RAIL_TRIGGER:
            self._robot_rail_done = False
            self._state_timer_counter = 0
            # 레일을 다시 3D 프린터 베드로 원점 복귀
            rail_cmd = Int32()
            rail_cmd.data = 0  # RailPosition.BED
            self._robot_rail_pub.publish(rail_cmd)
            self.get_logger().info('레일 홈 복귀 명령 전송')
            self._state = FsmState.HOMING_RAIL_WAIT

        elif self._state == FsmState.HOMING_RAIL_WAIT:
            self._state_timer_counter += 1
            if self._robot_rail_done:
                self.get_logger().info('레일 홈 복귀 완료')
                self._state = FsmState.HOMING_ARM_TRIGGER
            elif self._state_timer_counter > int(self._home_timeout * self._loop_rate):
                self.get_logger().error('레일 홈 복귀 대기 타임아웃! ERROR 상태로 천이')
                self._error_msg = 'HOME_RAIL_TIMEOUT'
                self._state = FsmState.ERROR

        elif self._state == FsmState.HOMING_ARM_TRIGGER:
            self._robot_home_done = False  # arm home done flag
            self._state_timer_counter = 0
            # 로봇팔 홈 복귀 실행
            home_cmd = Bool()
            home_cmd.data = True
            self._robot_home_pub.publish(home_cmd)
            self.get_logger().info('로봇팔 홈 복귀 명령 전송')
            self._state = FsmState.HOMING_ARM_WAIT

        elif self._state == FsmState.HOMING_ARM_WAIT:
            self._state_timer_counter += 1
            if self._robot_home_done:
                self.get_logger().info('로봇팔 홈 복귀 완료')
                # 다음 감지된 오브젝트가 남았으면 순회 구동
                if self._current_object_idx < self._total_objects:
                    self._state = FsmState.GRASPING_TRIGGER
                else:
                    self._state = FsmState.FINISHED
            elif self._state_timer_counter > int(self._home_timeout * self._loop_rate):
                self.get_logger().error('로봇팔 홈 복귀 대기 타임아웃! ERROR 상태로 천이')
                self._error_msg = 'HOME_ARM_TIMEOUT'
                self._state = FsmState.ERROR

        elif self._state == FsmState.FINISHED:
            self.get_logger().info('모든 탐지된 출력물의 검사 및 적재 분류가 완료되었습니다.')
            self._state = FsmState.IDLE

        elif self._state == FsmState.ERROR:
            # 예외 및 ESTOP 상태
            pass

    # ─── HMI 전송 유틸리티 ───
    def _publish_hmi_status(self):
        # Dynamically query the ROS 2 graph to check if other nodes are online
        yolo_online = self.count_publishers('/detection/objects') > 0
        inspect_online = self.count_publishers('/inspection/result') > 0
        grasp_online = self.count_publishers('/robot/status') > 0
        motor_online = self.count_publishers('/robot/status') > 0

        msg = SystemStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "main_controller"

        msg.current_state = str(self._state.value)
        msg.total_objects = int(self._total_objects)
        msg.processed_count = int(self._processed_count)
        msg.pass_count = int(self._pass_count)
        msg.fail_count = int(self._fail_count)

        msg.yolo_ready = bool(yolo_online)
        msg.grasp_ready = bool(grasp_online)
        msg.inspect_ready = bool(inspect_online)
        msg.motor_ready = bool(motor_online)

        msg.error_message = str(self._error_msg)

        self._hmi_status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MainOrchestratorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
