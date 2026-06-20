# 🛡️ QUVI 1순위 안전성 개선 계획서

**작성일**: 2026-06-20
**대상 파일**:
- `src/quvi_robot_control/quvi_robot_control/robot_control_node.py`
- `src/quvi_robot_control/quvi_robot_control/main_orchestrator_node.py`

---

## 개요

코드 리뷰에서 식별된 1순위(안전/정확성) 이슈 3건에 대한 상세 해결 계획입니다. 이 세 가지 문제는 실제 로봇 하드웨어 운용 시 **인명 피해나 장비 파손**으로 이어질 수 있는 안전-critical 이슈입니다.

| # | 이슈 | 영향도 | 예상 공수 |
|---|------|--------|----------|
| 1 | ACT 실행 루프 내 ESTOP 체크 부재 | **심각** - 비상정지 무시됨 | 0.5일 |
| 2 | `_read_raw_positions()` 에러 시 가짜 값 반환 | **심각** - 엉뚱한 동작 유발 | 0.5일 |
| 3 | `/robot/grasp_done` 토픽 다중 용도로 혼선 | **중간** - 잘못된 상태 전이 | 1일 |

---

## 이슈 1: ACT 실행 루프 내 ESTOP 체크 부재

### 현재 상태

`robot_control_node.py`의 `_execute_act_grasp()` 메서드는 별도 스레드에서 실행됩니다.

```python
# L446-448: grasp_cmd 콜백
t = threading.Thread(
    target=self._execute_act_grasp, daemon=True)
t.start()
```

ESTOP 수신 시 `_estop_cmd_callback`(L497-502)이 `RobotState.ERROR`로 설정하지만, **ACT 청크 실행 루프는 상태를 전혀 확인하지 않고 끝까지 실행**됩니다:

```python
# L609-636: 현재 코드 — ESTOP 체크 없음
dt = 1.0 / ACT_CONTROL_HZ
for i, action in enumerate(action_chunk):
    step_start = time.time()

    if self._use_real_hardware and self._dxl_ready:
        action_dict = {f"{name}.pos": float(action[j]) for j, name in enumerate(JOINT_NAMES)}
        with self._dxl_io_lock:
            self._follower.send_action(action_dict)  # ← ESTOP 중에도 계속 전송됨
    else:
        # 시뮬레이션 모드...
        ...

    # 30 Hz 타이밍 유지
    elapsed = time.time() - step_start
    remaining = dt - elapsed
    if remaining > 0:
        time.sleep(remaining)
```

**결과**: 사용자가 ESTOP을 눌러도 ACT 파지 동작이 멈추지 않고 청크 전체(약 0.67초 분량)를 끝까지 실행합니다. Dynamixel 토크 오프도 일어나지 않습니다.

### 해결 방안

액션 청크 실행 루프의 **3곳**에 ESTOP 체크를 추가합니다:

#### 위치 ①: 루프 진입 직전 (청크 실행 자체를 스킵)

```python
# ACT 추론 완료 후, 청크 실행 전
if self._get_state() == RobotState.ERROR:
    self.get_logger().error('ESTOP 감지 — ACT 청크 실행을 중단합니다.')
    self._safe_estop_cleanup()
    return False

# ── 액션 청크 실행 ──
dt = 1.0 / ACT_CONTROL_HZ
for i, action in enumerate(action_chunk):
    ...
```

#### 위치 ②: 각 스텝의 모터 명령 전송 직전 (가장 중요)

```python
for i, action in enumerate(action_chunk):
    step_start = time.time()

    # ★ 매 스텝마다 ESTOP 체크
    if self._get_state() == RobotState.ERROR:
        self.get_logger().error(
            f'ESTOP 감지 — ACT 실행 중단 (스텝 {i}/{len(action_chunk)})')
        self._safe_estop_cleanup()
        return False

    if self._use_real_hardware and self._dxl_ready:
        ...
```

#### 위치 ③: `_safe_estop_cleanup()` 신규 메서드

```python
def _safe_estop_cleanup(self):
    """ESTOP 발생 시 안전하게 Dynamixel 토크를 해제하고 상태를 정리한다."""
    try:
        if self._dxl_ready and self._follower:
            # lerobot OmxFollower의 disconnect는 내부적으로 torque disable을 수행
            # 개별 모터에 Torque_Enable = 0 을 직접 쓰는 것이 더 빠름
            if hasattr(self._follower, 'bus'):
                torque_off = {name: 0 for name in JOINT_NAMES}
                with self._dxl_io_lock:
                    self._follower.bus.sync_write('Torque_Enable', torque_off, normalize=False)
            self._dxl_ready = False  # 재연결 전까지 하드웨어 접근 차단
    except Exception as e:
        self.get_logger().error(f'ESTOP cleanup 중 오류: {e}')

    self._set_state(RobotState.ERROR)
    self._publish_status('ERROR: ESTOP ACTIVE — 토크 해제됨')
```

> **참고**: `OmxFollower.disconnect()`를 호출할 수도 있지만, 메서드 내부 검증 로직 때문에 지연이 있을 수 있습니다. ESTOP 상황에서는 `sync_write('Torque_Enable', 0)` 으로 즉시 토크를 해제하는 것이 더 안전합니다.

### 변경 위치 요약

| 파일 | 위치 | 변경 내용 |
|------|------|----------|
| `robot_control_node.py` | L609 (액션 청크 실행 전) | ESTOP 체크 + early return 추가 |
| `robot_control_node.py` | L614 (모터 전송 직전) | 매 스텝 ESTOP 체크 추가 |
| `robot_control_node.py` | 신규 (클래스 하단) | `_safe_estop_cleanup()` 메서드 추가 |

### 예상 동작 흐름

```
[ESTOP 버튼 누름]
  → /system/estop 토픽 발행 (Bool: True)
  → RobotControlNode._estop_cmd_callback()
    → self._set_state(RobotState.ERROR)   # 상태 변경
  → (ACT 스레드) 매 루프 반복에서 체크
    → _get_state() == ERROR → _safe_estop_cleanup()
      → Torque_Enable = 0  (전 관절 토크 해제)
      → dxl_ready = False  (재접근 차단)
      → done 신호 발행하지 않음 (오케스트레이터는 timeout으로 ERROR 감지)
```

### 주의: ESTOP 해제(복구) 절차

ESTOP 이후에는 **자동 복구하지 않도록** 합니다:
1. `_dxl_ready = False`로 설정되어 이후 명령은 자동으로 무시됨
2. 복구하려면 `RESET` 명령 → `_init_follower()` 재호출 필요
3. `main_orchestrator_node`의 `_hmi_command_cb`에서 RESET 시 `_state = FsmState.INIT`으로 초기화하는 로직과 연계

---

## 이슈 2: `_read_raw_positions()` 에러 시 가짜 기본값 반환

### 현재 상태

```python
# L790-805
def _read_raw_positions(self) -> dict:
    if not self._use_real_hardware or not self._dxl_ready:
        return {name: 2048 for name in JOINT_NAMES}

    try:
        with self._dxl_io_lock:
            return self._follower.bus.sync_read(
                'Present_Position', normalize=False)
    except Exception as e:
        self.get_logger().error(f'lerobot sync_read 오류: {e}')
        return {name: 2048 for name in JOINT_NAMES}  # ← 위험한 폴백
```

**문제 시나리오**:
1. ACT 추론 중 통신 케이블 단선
2. `_read_raw_positions()` 예외 발생 → 모든 관절값 2048 반환
3. ACTPolicy가 가짜 관절값(2048)과 실제 이미지를 조합해 추론
4. 실제 관절 각도와 전혀 다른 액션 생성 → **충돌 가능성**

이 메서드는 3곳에서 호출됩니다:
- `_execute_act_grasp()` L581: ACT 추론용 obs 구성
- `_publish_joint_states()` L812: HMI 표시용

### 해결 방안

#### 2-A) 반환 타입을 `dict` → `Optional[dict]`로 변경

```python
from typing import Optional

def _read_raw_positions(self) -> Optional[dict]:
    """lerobot bus를 통해 현재 raw 위치값 읽기 (normalize=False).

    Returns:
        {'shoulder_pan': 2048, ...} 형태의 dict. 실패 시 None.
    """
    if not self._use_real_hardware or not self._dxl_ready:
        return {name: 2048 for name in JOINT_NAMES}

    try:
        with self._dxl_io_lock:
            return self._follower.bus.sync_read(
                'Present_Position', normalize=False)
    except Exception as e:
        self.get_logger().error(f'lerobot sync_read 오류: {e}')
        return None  # ← 실패를 명시적으로 전파
```

#### 2-B) 호출 측에서 None 처리

**`_execute_act_grasp()` (L581-588):**

```python
# 관절 상태: lerobot bus에서 raw 위치 읽기
raw_positions = self._read_raw_positions()
if raw_positions is None:
    self.get_logger().error(
        '관절 위치 읽기 실패 — 통신 오류. 파지를 중단합니다.')
    self._set_state(RobotState.ERROR)
    self._publish_status('ERROR: Dynamixel 통신 오류')
    return False

joint_rad = [
    (raw_positions[name] / 4095.0) * 2 * math.pi
    for name in JOINT_NAMES
]
```

**`_publish_joint_states()` (L812-823):**

```python
def _publish_joint_states(self):
    """현재 관절 위치를 JointState 토픽으로 발행."""
    raw_positions = self._read_raw_positions()

    if raw_positions is None:
        # 읽기 실패 시 이전 값 유지 또는 발행 스킵
        return  # HMI에서 마지막 수신값이 유지됨

    msg = JointState()
    msg.header.stamp = self.get_clock().now().to_msg()
    msg.header.frame_id = 'base_link'
    msg.name = JOINT_NAMES
    msg.position = [
        (raw_positions[name] / 4095.0) * 2 * math.pi
        for name in JOINT_NAMES
    ]
    self._joint_state_pub.publish(msg)
```

> **참고**: `_publish_joint_states()`는 30Hz 타이머 콜백이므로, 일시적 통신 오류에서 스킵하는 것이 전체 시스템에 더 안전합니다. HMI는 마지막으로 수신한 정상 값을 계속 표시하게 됩니다.

#### 2-C) `_use_real_hardware=False` 또는 `_dxl_ready=False` 경우는 유지

시뮬레이션 모드이거나 초기화 실패 상태에서는 현재처럼 중립값(2048)을 반환하는 것이 적절합니다. 이 경우는 예외가 아니라 **의도된 동작**이기 때문입니다.

### 변경 위치 요약

| 파일 | 위치 | 변경 내용 |
|------|------|----------|
| `robot_control_node.py` | L790, L805 | `return None` 으로 변경, 타입 힌트 `Optional[dict]` |
| `robot_control_node.py` | L581 | `_read_raw_positions()` 호출 후 None 체크 추가 |
| `robot_control_node.py` | L812 | `_read_raw_positions()` 호출 후 None → early return |

---

## 이슈 3: `/robot/grasp_done` 토픽의 다중 용도 분리

### 현재 상태

하나의 토픽 `/robot/grasp_done`이 **3가지 완료 신호**를 모두 전달합니다:

```python
# robot_control_node.py L740 — _execute_release()
done_msg = Bool()
done_msg.data = True
self._grasp_done_pub.publish(done_msg)  # ← /robot/grasp_done

# robot_control_node.py L761 — _execute_home()
done_msg = Bool()
done_msg.data = True
self._grasp_done_pub.publish(done_msg)  # ← /robot/grasp_done (같은 토픽!)
```

오케스트레이터 측 콜백에서 상태 분기:

```python
# main_orchestrator_node.py L213-221
def _robot_grasp_done_cb(self, msg: Bool):
    # release, grasp, home 동작의 피드백 완료 통합 처리
    if msg.data:
        if self._state.value.startswith("GRASPING_"):
            self._robot_grasp_done = True
        elif self._state.value.startswith("RELEASING_"):
            self._robot_release_done = True
        elif self._state.value.startswith("HOMING_"):
            self._robot_home_done = True
```

**문제 시나리오**:
1. 오케스트레이터가 `HOMING_TRIGGER` → `HOMING_WAIT` 전이
2. 동시에 이전 사이클에서 발행된 `/robot/grasp_done` 메시지가 DDS 레이어에 남아있음 (QoS: keep_last=10)
3. 콜백이 이 오래된 메시지를 수신 → `_robot_home_done = True` 로 잘못 설정
4. `HOMING_WAIT`에서 `_robot_rail_done`과 `_robot_home_done`이 모두 True → 실제 이동 완료 전에 다음 상태로 전이

현재는 FSM 상태 prefix로 구분하고 있어 어느 정도 방어되지만, **토픽 수준에서 분리하는 것이 근본적인 해결책**입니다.

### 해결 방안

#### 3-A) 새로운 완료 토픽 추가

`robot_control_node.py`에 두 개의 새로운 퍼블리셔를 추가합니다:

| 신규 토픽 | 타입 | 용도 | 발행 위치 |
|-----------|------|------|----------|
| `/robot/release_done` | `std_msgs/Bool` | 그리퍼 투하 완료 | `_execute_release()` |
| `/robot/home_done` | `std_msgs/Bool` | 홈 복귀 완료 (팔) | `_execute_home()` |

#### 3-B) robot_control_node.py 변경

**신규 퍼블리셔 선언** (`_setup_ros_interfaces()` 내, 기존 퍼블리셔 하단):

```python
# 기존 퍼블리셔 (유지)
self._grasp_done_pub = self.create_publisher(
    Bool, '/robot/grasp_done', 10)

# 신규 퍼블리셔 추가
self._release_done_pub = self.create_publisher(
    Bool, '/robot/release_done', 10)

self._home_done_pub = self.create_publisher(
    Bool, '/robot/home_done', 10)
```

**`_execute_act_grasp()` — 기존과 동일 (변경 없음)**:
```python
# ACT 파지 완료 → /robot/act_done + /robot/grasp_done 발행
done_msg = Bool()
done_msg.data = True
self._act_done_pub.publish(done_msg)
self._grasp_done_pub.publish(done_msg)
```

**`_execute_release()` — `/robot/release_done` 사용**:
```python
def _execute_release(self) -> bool:
    """분류함 위에서 그리퍼를 열어 출력물 투하."""
    self._set_state(RobotState.RELEASING)
    self._publish_status('출력물 투하')
    self.get_logger().info('출력물 투하: 그리퍼 열기 (OmxFollower gripper)')

    self._write_raw_position({'gripper': GRIPPER_OPEN})
    time.sleep(0.8)

    done_msg = Bool()
    done_msg.data = True
    self._release_done_pub.publish(done_msg)  # ← 변경: release_done

    self._set_state(RobotState.IDLE)
    self._publish_status('투하 완료')
    return True
```

**`_execute_home()` — `/robot/home_done` 사용**:
```python
def _execute_home(self) -> bool:
    """전체 관절을 홈 자세로 복귀."""
    self._set_state(RobotState.HOMING)
    self._publish_status('홈 복귀')
    self.get_logger().info('홈 복귀 시작')

    success = self._write_raw_position(POSE_HOME)
    time.sleep(2.0)

    # 홈 복귀 완료 신호 발행
    done_msg = Bool()
    done_msg.data = True
    self._home_done_pub.publish(done_msg)  # ← 변경: home_done

    self._set_state(RobotState.IDLE)
    self._publish_status('홈 복귀 완료')
    return success
```

#### 3-C) main_orchestrator_node.py 변경

**신규 구독 추가** (`_setup_subscribers()` 내):

```python
# 로봇 피드백 완료 토픽 구독 (분리된 토픽)
self.create_subscription(Bool, '/robot/act_done', self._robot_act_done_cb, 10)
self.create_subscription(Bool, '/robot/grasp_done', self._robot_grasp_done_cb, 10)
self.create_subscription(Bool, '/robot/release_done', self._robot_release_done_cb, 10)  # 신규
self.create_subscription(Bool, '/robot/home_done', self._robot_home_done_cb, 10)        # 신규
self.create_subscription(Bool, '/robot/rail_done', self._robot_rail_done_cb, 10)
```

**콜백 단순화** — 상태 prefix 분기 제거:

```python
# 기존 _robot_grasp_done_cb → 단순화
def _robot_grasp_done_cb(self, msg: Bool):
    # ACT 파지 완료 전용 (act_done과 별도로 하드코딩 grasp 완료)
    if msg.data and self._state.value.startswith("GRASPING_"):
        self._robot_grasp_done = True

# 신규 콜백 추가
def _robot_release_done_cb(self, msg: Bool):
    """그리퍼 투하 완료 수신."""
    if msg.data:
        self._robot_release_done = True

def _robot_home_done_cb(self, msg: Bool):
    """로봇팔 홈 복귀 완료 수신."""
    if msg.data:
        self._robot_home_done = True
```

> **참고**: `_robot_release_done_cb`와 `_robot_home_done_cb`는 상태 prefix 체크 없이 항상 플래그를 True로 설정해도 됩니다. 오케스트레이터 FSM은 `RELEASING_WAIT`나 `HOMING_WAIT` 상태에서만 이 플래그를 확인하기 때문입니다. 이렇게 하면 오래된 메시지가 도착해도 FSM이 WAIT 상태가 아니면 무시됩니다.

#### 3-D) `_robot_act_done_cb` 역할 명확화

현재 `_robot_act_done_cb`(L209-211)와 `_robot_grasp_done_cb`(L213-221)는 중복 역할을 합니다:

```python
# _robot_act_done_cb
if msg.data and self._state.value.startswith("GRASPING_"):
    self._robot_grasp_done = True

# _robot_grasp_done_cb
if self._state.value.startswith("GRASPING_"):
    self._robot_grasp_done = True
```

`_execute_act_grasp()`가 두 토픽을 모두 발행하므로(L644-645), `_robot_act_done_cb`는 **제거하고** `_robot_grasp_done_cb`로 통일하거나, `_robot_act_done_cb`가 별도 플래그(`_act_done`)를 관리하도록 변경합니다. 현재는 두 콜백이 동일한 플래그를 설정하므로 하나는 불필요합니다.

**권장**: `_robot_act_done_cb` → `_act_done` 플래그 관리, `_robot_grasp_done_cb` → `_robot_grasp_done` 플래그 관리. 현재 FSM은 `_robot_grasp_done`만 확인하므로, ACT 완료 콜백을 통합:

```python
# 단순화: ACT 완료는 /robot/grasp_done 하나로 통일
# /robot/act_done 구독 제거 (또는 별도 로깅/모니터링 용도로 유지)
def _robot_grasp_done_cb(self, msg: Bool):
    if msg.data and self._state == FsmState.GRASPING_WAIT:
        self._robot_grasp_done = True
```

### 최종 토픽 체계

변경 후 완료 신호 흐름:

```
로봇 제어 노드                    오케스트레이터 노드
─────────────                    ────────────────
_execute_act_grasp()
  → /robot/grasp_done  ──────→  _robot_grasp_done_cb  → grasp_done=True

_execute_release()
  → /robot/release_done ─────→  _robot_release_done_cb → release_done=True

_execute_home()
  → /robot/home_done    ─────→  _robot_home_done_cb    → home_done=True

_execute_rail_move()
  → /robot/rail_done    ─────→  _robot_rail_done_cb    → rail_done=True
```

### 변경 위치 요약

| 파일 | 위치 | 변경 내용 |
|------|------|----------|
| `robot_control_node.py` | `_setup_ros_interfaces()` | `_release_done_pub`, `_home_done_pub` 추가 |
| `robot_control_node.py` | `_execute_release()` L740 | `/robot/grasp_done` → `/robot/release_done` |
| `robot_control_node.py` | `_execute_home()` L761 | `/robot/grasp_done` → `/robot/home_done` |
| `main_orchestrator_node.py` | `_setup_subscribers()` L150-153 | `release_done`, `home_done` 구독 추가 |
| `main_orchestrator_node.py` | 콜백 L209-221 | 콜백 단순화 + 신규 콜백 2개 추가 |

---

## 테스트 계획

### 이슈 1 (ESTOP) 테스트

```
[시뮬레이션 모드 테스트]
1. use_real_hardware:=false 로 실행
2. 자율 시퀀스 시작 → ACT 파지 단계 진입
3. ACT 청크 실행 중 HMI에서 ESTOP 버튼 클릭
4. 확인사항:
   - RobotControlNode 상태가 ERROR로 변경됨
   - ACT 청크 루프가 즉시 중단됨 (로그 "ESTOP 감지 — ACT 실행 중단")
   - _safe_estop_cleanup() 호출됨
   - _execute_act_grasp()가 False 반환

[실제 하드웨어 테스트]
1. Dynamixel 전원이 켜진 상태에서 테스트
2. ACT 파지 실행 중 물리적 ESTOP 버튼 누름 (또는 HMI ESTOP)
3. 확인사항:
   - 모든 Dynamixel 토크가 즉시 해제되는지 (손으로 관절 움직여 확인)
   - 모터가 완전히 멈추는지
   - _dxl_ready가 False가 되어 이후 명령이 무시되는지
```

### 이슈 2 (가짜 값) 테스트

```
[통신 오류 시뮬레이션]
1. 정상 실행 중 Dynamixel USB 케이블 분리
2. 확인사항:
   - _publish_joint_states: 발행 스킵, HMI는 마지막 값 유지
   - ACT 실행 시도: "관절 위치 읽기 실패" 로그 + ERROR 상태
   - 이전처럼 가짜 값(2048)으로 ACT 추론이 실행되지 않음
```

### 이슈 3 (토픽 분리) 테스트

```
[기능 테스트]
1. 정상 자율 시퀀스 전체 실행 (DETECT → GRASP → INSPECT → SORT → RELEASE → HOME)
2. 각 단계에서 올바른 done 토픽이 발행되고 FSM이 정상 전이하는지 확인

[오래된 메시지 테스트]
1. 이전 사이클의 /robot/grasp_done 메시지가 DDS 캐시에 남은 상태에서
2. HOMING_TRIGGER 진입
3. 확인사항:
   - /robot/grasp_done 수신해도 _robot_home_done이 True로 설정되지 않음
   - /robot/home_done 수신 시에만 _robot_home_done = True
```

---

## 구현 순서

이슈 1, 2, 3은 동일 파일들을 수정하므로 **한 번에 구현**하는 것이 효율적입니다:

1. **1단계** (1시간): 이슈 1 — `_safe_estop_cleanup()` + ACT 루프 체크 추가
2. **2단계** (1시간): 이슈 2 — `_read_raw_positions()` 반환 타입 변경 + 호출부 처리
3. **3단계** (1.5시간): 이슈 3 — 완료 토픽 분리 (robot_control_node + orchestrator 양쪽)
4. **4단계** (1시간): 통합 테스트 (시뮬레이션 + 실제 하드웨어)

---

## 관련 파일 요약

| 파일 | 변경 규모 | 위험도 |
|------|----------|--------|
| `robot_control_node.py` | **약 80줄 수정/추가** (ESTOP 체크 30줄, read 오류처리 10줄, 토픽 분리 40줄) | 낮음 |
| `main_orchestrator_node.py` | **약 30줄 수정/추가** (신규 구독 + 콜백 추가) | 낮음 |

기존 FSM 타이머 로직, YOLO 노드, Inspect 노드, HMI 노드, ESP32 펌웨어는 **변경 없음**.