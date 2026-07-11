# 2026-07-11 코드 리뷰 문제 해결 계획서

대상 커밋: `ada720f` (master = claude/code-review-9g5rl1)
리뷰 범위: robot_control_node / main_orchestrator_node / inspect_node / hmi_node / ESP32 펌웨어 / dashboard.js·html
이전 리뷰(2026-07-08, 07-10) 반영분은 제외한 신규 발견 13건.

## 우선순위 요약

| # | 우선순위 | 위치 | 요약 |
|---|---------|------|------|
| 1 | **P1** | dashboard.js | 검사 이력 테이블 중복 폭주 (초당 10행) |
| 2 | **P2** | robot_control_node | Trigger 서비스가 상태머신 선점 우회 |
| 3 | **P2** | robot_control_node | 레일 대기 루프 abort 비인지 + 잔존 스레드 상태 덮어쓰기 |
| 4 | **P2** | robot_control_node | 안착/재파지 시퀀스 저속(SEQ) 프로파일 누락 |
| 5 | **P2** | hmi_node | 기준 캡처·데이터셋·텔레옵 API에 FSM 가드 누락 |
| 6 | **P2** | 펌웨어 | AccelStepper 크로스코어 동시 접근 |
| 7 | P3 | robot_control_node | 룰베이스 파지가 gripper 포함 pose 로 `_wait_motion_done` |
| 8 | P3 | robot_control_node | ESTOP 후 텔레옵 리더 포트 미해제 |
| 9 | P3 | hmi_node | teleop_active 플래그 UI 고착 |
| 10 | P3 | orchestrator | STOP 시 LED 방치 + `_estop_pub` 죽은 코드 |
| 11 | P3 | dashboard.js/html ↔ inspect_node | 면적비 임계값 표기 불일치 (1.10 vs 1.50) |
| 12 | P3 | inspect_node | 캡처/워치독 동시 판정 실행 가능 |
| 13 | P3 | 전반 | 표기·주석·로그 오타 일괄 정리 |

---

## P1

### #1 검사 이력 테이블 중복 폭주

- **증상**: 검사 1건 완료 후 히스토리 테이블에 같은 결과가 초당 10행씩 쌓여
  약 5초 만에 50행 전체가 동일 레코드로 덮인다. `historyCount` 는 항상 "50건" 포화.
- **원인**:
  - `hmi_node.py` `_ws_broadcast` 가 0.1s 주기로 `latest_inspection: history[-1]` 을 매번 전송.
  - `dashboard.js` `status_update` 핸들러가 수신 시마다 `updateLatestInspection()` →
    `addHistoryRow()` 호출. `addHistoryRow` 에 중복 제거 없음.
  - 부가: `loadInitialData()` 가 `historyData` 채운 뒤 `updateLatestInspection(historyData[0])` 을
    호출해 같은 건을 한 번 더 unshift.
- **수정 계획**:
  1. dashboard.js 에 `let _lastInspectionTs = null;` 추가. `updateLatestInspection()` 진입부에서
     `result.timestamp === _lastInspectionTs` 이면 즉시 return (메트릭 갱신은 무해하므로
     `addHistoryRow` 호출만 건너뛰어도 됨 — 단순성을 위해 전체 skip 권장, 값이 같으므로 표시 불변).
  2. `loadInitialData()` 는 `updateLatestInspection()` 대신 테이블 재렌더 함수 분리
     (`renderHistoryTable()`) 후 최신 결과 카드만 별도 갱신 — 또는 1번 dedup 이 자연 흡수하므로
     `_lastInspectionTs` 를 초기 로드 시 세팅.
  3. (선택, 서버측 방어) `_ws_broadcast` 에서 마지막 전송 timestamp 를 기억해 신규 결과일 때만
     `latest_inspection` 포함. 클라이언트 dedup 만으로 충분하면 생략 가능.
- **검증**: 검사 단독 테스트 1회 실행 → 히스토리 행이 정확히 1행 증가하는지,
  새로고침 후에도 중복 없는지 확인.

---

## P2

### #2 Trigger 서비스가 상태머신 선점 우회 (`/robot/act_grasp`, `/robot/go_home`)

- **증상/위험**: 서비스 핸들러가 `_try_start_command` 를 거치지 않고 `_execute_act_grasp()` /
  `_execute_home()` 을 직접 호출. IDLE 확인 없이 `_set_state` 로 현재 상태를 덮어쓰고,
  ReentrantCallbackGroup + MultiThreadedExecutor 조합이라 토픽 기반 동작 스레드와
  실제 병렬 실행되어 동일 Dynamixel 버스에 이중 목표를 쓸 수 있다.
  그리퍼 서비스 2종도 상태 무관하게 즉시 sync_write.
- **수정 계획**:
  1. `_act_grasp_service` / `_go_home_service`: 실행 전 `_state_lock` 안에서 IDLE 선점
     (기존 `_try_start_command` 의 선점 로직을 동기 실행용으로 재사용할 수 있게
     `_try_acquire_state(target_state) -> bool` 헬퍼로 분리). 선점 실패 시
     `response.success=False, message='현재 {state} 동작 중'` 반환.
  2. `_open/_close_gripper_service`: 최소한 `_get_state() in (IDLE, TELEOPING 제외)` 가드 +
     `_abort_event` 확인. ACT/시퀀스 진행 중이면 거부.
- **검증**: `tests/test_orchestrator_logic.py` 스타일로 상태 선점 단위 테스트 추가 —
  상태를 MOVING_RAIL 로 만든 뒤 서비스 호출 시 success=False 인지.

### #3 레일 대기 루프 abort 비인지 + 잔존 스레드의 상태 덮어쓰기

- **증상/위험**: `_execute_rail_move` 의 done 폴링 루프가 `_should_abort()` 를 확인하지 않아
  STOP/ESTOP 후에도 최대 30s(MOVING_RAIL) 유지. 최악 시나리오:
  ESTOP → 즉시 RESET(IDLE 복귀) → 30s 시점에 잔존 스레드가 타임아웃하며
  `_set_state(ERROR)` 로 리셋된 상태를 덮어씀. 성공 경로의 `_set_state(IDLE)` 도 동일 위험.
- **수정 계획**:
  1. 폴링 루프에 `if self._should_abort(): return False` 추가 (done 발행 없이 종료 —
     오케스트레이터는 자체 타임아웃/STOP 경로로 처리).
  2. 세대(generation) 토큰 도입: `_try_start_command` 에서 `self._cmd_gen += 1` 후 스레드에 전달,
     실행 함수 말미의 `_set_state(...)` 를 `_set_state_if_gen(state, gen)` 으로 교체해
     자신이 최신 명령일 때만 상태를 쓴다. RESET(`_execute_reset`)도 `_cmd_gen` 증가.
     (#2 의 서비스 경로도 동일 토큰 사용 — 잔존 스레드의 상태 덮어쓰기를 구조적으로 차단.)
- **검증**: SIM 모드에서 rail 이동 중 abort_event set → 즉시 루프 탈출 확인.
  이동 중 RESET → 30s 후 상태가 IDLE 유지되는지 (기존에는 ERROR 로 뒤집힘).

### #4 안착/재파지 시퀀스 저속(SEQ) 프로파일 누락

- **증상/위험**: `PROFILE_VELOCITY_SEQ`(2000ms) 는 "물체 파지 중 낙하/파손 방지" 목적인데,
  정작 물체를 쥐고 이동하는 `_execute_place_in_chamber`(P1→P4) 와
  `_execute_pick_from_chamber` 가 `_write_raw_position` 기본값(1200ms) 으로 이동.
  `_execute_taught_sequence` 에서 P 시퀀스를 분리할 때 프로파일 지정이 누락된 것.
- **수정 계획**: 두 함수의 모든 팔 이동 `_write_raw_position(...)` 호출에
  `velocity=PROFILE_VELOCITY_SEQ, accel=PROFILE_ACCEL_SEQ` 명시.
  두 함수의 중복 `_arm_only` 로컬 정의도 이참에 모듈 레벨 헬퍼로 통합.
  (참고: pick 의 P4 접근·P3 복귀는 빈손/파지 혼재 — 전 구간 SEQ 로 통일하는 것이
  안전 측면에서 단순하고, 시간 손실은 이동당 0.8s 수준.)
- **검증**: 실기에서 안착 시퀀스 실행하며 이동 시간 육안 확인(기존 대비 감속),
  `_wait_motion_done` 타임아웃 경고 없는지 로그 확인.

### #5 기준 캡처·데이터셋·텔레옵 API 에 FSM 가드 누락

- **증상/위험**: 레일/턴테이블/LED/검사테스트는 `_manual_trigger_allowed()` 로 자율 시퀀스 중
  거부하는데 `/api/capture/reference/start`, `/api/capture/dataset/start`, `/api/teleop/on` 은
  가드가 없다. 자율 검사(INSPECTING) 중 기준 캡처를 누르면 HMI 와 오케스트레이터가 동시에
  턴테이블 명령을 발행하고, 공유 이벤트(`_ref_turntable_done_event`)를 서로 소모하며,
  inspect_node 캡처가 오염될 수 있다.
- **수정 계획**:
  1. 세 라우트 진입부에 기존 409 응답 패턴 그대로 가드 추가
     (`if not hmi_node._manual_trigger_allowed(): return 409`).
     `/api/teleop/off` 는 가드 제외(끄는 것은 항상 허용).
  2. 기준 캡처 ↔ 데이터셋 촬영 ↔ 검사 단독 테스트 상호 배제:
     `_inspection_test_active` 패턴을 일반화한 `_hmi_busy_lock`(name 문자열 보관) 하나로
     세 시퀀스가 서로 진행 중이면 409 반환.
- **검증**: FSM 을 GRASPING_WAIT 상태로 만든 뒤 세 API 호출 → 409 확인.
  검사 테스트 진행 중 기준 캡처 시작 → 409 확인.

### #6 펌웨어 AccelStepper 크로스코어 동시 접근

- **증상/위험**: 호밍은 `homingRequested` 플래그로 Core 0(vMotorTask) 위임해 해결했지만,
  일반 이동은 여전히 vCommTask(Core 1) 의 `setTargetPosition()`(내부 `moveTo()` 가
  speed/step 재계산) 과 Core 0 의 `run()` 이 같은 AccelStepper 객체에 비동기 접근.
  AccelStepper 는 thread-safe 가 아니므로 이론상 스텝 누락·순간 역방향 가능.
- **수정 계획**: 호밍과 동일한 위임 패턴으로 통일.
  ```
  volatile bool railTargetPending = false;
  volatile long railPendingTarget = 0;   // turn 동일
  ```
  - 구독 콜백: 클램프/최단경로 계산까지만 수행 후 pending 변수 세트 (모터 객체 비접촉).
  - vMotorTask 루프 선두: pending 이면 `setTargetPosition()` 적용 후 플래그 해제.
  - `done_pending` 세트는 기존 위치 유지 (isMoving 판정은 적용 이후에만 false→true 전이가
    일어나므로, done 조기 발행 방지를 위해 pending 적용 전에는 done 체크를 건너뛰는 분기 추가).
  - CLI 모드(R/T 명령)도 동일 경로 사용.
- **검증**: 실기에서 이동 중 반복 재명령(레일 왕복 스트레스) 시 위치 어긋남 없는지,
  `S` 명령/motor_status 의 position 이 명령 누적과 일치하는지 확인.

---

## P3

### #7 룰베이스 파지의 gripper 포함 `_wait_motion_done`

- `_execute_rule_based_grasp` 가 `p1_pose['gripper']=GRIPPER_CLOSE` 포함 전체 pose 로
  `_wait_motion_done` 호출 → 물체를 쥐면 매번 10s 풀타임아웃. place/pick 에서 이미 고친 패턴.
- **수정**: `_arm_only(p1_pose)` 로 대기하고 그리퍼는 목표 쓰기만 유지 (이미 닫힌 상태 유지 목적).

### #8 ESTOP 후 텔레옵 리더 포트 미해제

- `_safe_estop_cleanup` 이 `_teleop_running=False` 로 만들면 이후 `_stop_teleop` 이
  조기 반환해 `_leader.disconnect()` 미호출 → 포트 점유 잔존, 다음 텔레옵 연결 실패 소지.
- **수정**: `_stop_teleop` 조기 반환 조건을 `not self._teleop_running and self._leader is None`
  으로 변경 — running 여부와 무관하게 leader 가 남아 있으면 disconnect 수행.
  (또는 `_safe_estop_cleanup` 에서 leader disconnect 를 함께 수행.)

### #9 teleop_active 플래그 UI 고착

- REST 로만 세트/해제되어 robot_control 이 텔레옵을 거부해도 UI 는 TELEOPING 고정.
- **수정**: `_status_cb` 에서 `/hmi/status` 기반 보정은 어렵고(오케스트레이터는 텔레옵 모름),
  `_robot_node_status_cb` 처럼 HMI 가 `/robot/status` 를 구독해
  `'텔레옵 에러'`/`'텔레오퍼레이션 종료'` 수신 시 `teleop_active=False` 로 동기화.
  (robot_control 의 상태 문자열은 이미 발행 중이므로 구독 1개 추가로 충분.)

### #10 STOP 시 LED 방치 + `_estop_pub` 죽은 코드

- **수정**:
  1. `_hmi_command_cb` 의 STOP/ESTOP 분기에서 `self._led_pub.publish(Bool(data=False))` 발행
     (INSPECTING 계열 상태에서 정지해도 링 조명 소등 보장). RESET 분기에도 동일 추가.
  2. `_estop_pub`(피드백 루프 롤백 후 발행처 없음) 제거 — 주석의 (#6) 이력도 함께 정리.

### #11 면적비 임계값 표기 불일치

- dashboard.js `THRESHOLDS.areaRatio=[0.90, 1.10]`, dashboard.html 기준표 "0.90 ~ 1.10"
  vs inspect_node 기본 `feature_area_ratio_max=1.50`. 노드가 PASS 한 1.2 를 UI 상세표는 FAIL 표기.
- **수정**: 판정 SSoT 는 inspect_node 파라미터이므로 UI 를 1.50 으로 맞춘다
  (js THRESHOLDS + html 기준표 2곳). 장기적으로는 `/api/status` 에 임계값을 실어
  UI 가 동적 표시하는 것이 정석이나 이번 범위에선 상수 정합만.

### #12 inspect_node 캡처/워치독 동시 판정

- MultiThreadedExecutor 에서 4번째 캡처(`_capture_angle`)와 워치독(`_watchdog_cb`)이
  동시에 `_run_inspection()` 진입 가능 → 결과 이중 발행 (오케스트레이터는 멱등이라 실피해 낮음).
- **수정**: `self._inspection_lock = threading.Lock()` 추가, `_run_inspection` 을
  `with self._inspection_lock:` + 진입 직후 `if not self._inspection_active: return` 재확인으로 감싼다.
  `_captured_images` 를 만지는 콜백들(`_capture_angle`, `_trigger_callback` 등)도 동일 락 사용.

### #13 표기·주석·로그 정리 (일괄 커밋 1건)

- `robot_control_node.py` "관절 상태 발행 (30 Hz 타이머)" 섹션 주석 → 실제 10 Hz 로 정정.
- dashboard.js `jointSyncTime` "실시간 (30Hz)" → "실시간 (10Hz)".
- dashboard.js 그리퍼 ticks 환산 분모 4095 → 4096 (발행 측 `DXL_TICKS_PER_REV` 와 일치).
- `robot_control_node.py` ACT 관절 읽기 실패 로그 f-string 에 `{e}` 누락 보완.
- orchestrator 로그 오타: "도신" → "도착"(2곳), "터테이블" → "턴테이블".
- `main_orchestrator_node.py` 의 리터럴 `'/motor/rail_done'` → `topics.TOPIC_MOTOR_RAIL_DONE`
  (`'/motor/turntable_done'`, `'/inspection/result'` 리터럴도 topics.py 등재 검토).

---

## 작업 순서 제안

1. **1차 (동작 결함·안전)**: #1 → #4 → #5 → #3 — 시연에서 바로 체감/사고 소지 순.
2. **2차 (구조 보강)**: #2 (+#3 의 세대 토큰 통합), #6 (펌웨어 재플래시 필요하므로 별도 커밋).
3. **3차 (정리)**: #7 ~ #13 을 P3 일괄 커밋 2~3건으로.

각 커밋은 기존 컨벤션(`fix:`/`refactor:` + 리뷰 번호 참조)을 따르고,
#2·#3 은 `tests/` 에 상태 선점·abort 단위 테스트를 함께 추가한다.
펌웨어(#6)는 플래시 후 실기 레일 왕복 스트레스 테스트를 통과해야 머지.
