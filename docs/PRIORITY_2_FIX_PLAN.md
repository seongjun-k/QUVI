# QUVI 2순위 신뢰성 개선 계획서

**작성일**: 2026-06-20
**대상 파일**:
- `src/quvi_robot_control/quvi_robot_control/main_orchestrator_node.py`
- `src/quvi_inspect/quvi_inspect/inspect_node.py`

---

## 개요

코드 리뷰에서 식별된 2순위(신뢰성) 이슈 3건에 대한 상세 해결 계획입니다. 이 이슈들은 시스템이 정상 동작하는 중에도 잘못된 타이밍이나 부정확한 데이터로 인해 검사 누락, 잘못된 판정, 불필요한 ERROR 전이가 발생할 수 있는 문제들입니다.

| # | 이슈 | 영향도 | 예상 공수 |
|---|------|--------|----------|
| 1 | 오케스트레이터가 턴테이블 실제 완료 피드백을 기다리지 않고 고정 타이머로 전이 | **중간** - 타이밍 불일치로 캡처 누락 가능 | 1일 |
| 2 | `inspect_node`의 `_turntable_cmd_callback` 불필요 (이미 `turntable_done` 기반 동작) | **낮음** - dead code, 혼란 유발 | 0.25일 |
| 3 | 기준 이미지 없을 때 `area_ratio=1.0` 폴백 → 검사 신뢰도 저하 | **중간** - 잘못된 PASS 판정 가능 | 0.5일 |

---

## 이슈 1: 오케스트레이터 턴테이블 완료 피드백 부재

### 현재 상태

`main_orchestrator_node.py`는 턴테이블 회전 명령을 보낸 후 **고정 시간(`step_delay_sec=2.0`초)만 대기**하고 다음 각도로 넘어갑니다. ESP32가 실제로 회전을 완료했는지 확인하지 않습니다.

```python
# main_orchestrator_node.py L336-355 (INSPECTING_ROTATE / INSPECTING_CAPTURE)
elif self._state == FsmState.INSPECTING_ROTATE:
    angle = self._inspect_angles[self._inspect_angle_idx]
    angle_msg = Int32()
    angle_msg.data = angle
    self._turntable_pub.publish(angle_msg)       # ← 명령 발행
    self.get_logger().info(f'턴테이블 {angle}도 회전 명령 전송')

    self._state_timer_counter = 0
    self._state = FsmState.INSPECTING_CAPTURE     # ← 바로 CAPTURE 로 전이

elif self._state == FsmState.INSPECTING_CAPTURE:
    self._state_timer_counter += 1
    # 고정 타이머만 사용 — 실제 회전 완료 여부 무시
    if self._state_timer_counter >= int(self._step_delay * self._loop_rate):
        self._inspect_angle_idx += 1
        if self._inspect_angle_idx < len(self._inspect_angles):
            self._state = FsmState.INSPECTING_ROTATE
        else:
            self._state_timer_counter = 0
            self._state = FsmState.INSPECTING_WAIT_RESULT
```

**반면 `inspect_node`는 이미 제대로 구현되어 있습니다** — `_turntable_done_callback`이 ESP32의 실제 완료 신호를 받아 캡처를 수행합니다 (L180-185).

### 문제 시나리오

```
[오케스트레이터]                     [ESP32]                    [inspect_node]
     │                                  │                           │
     ├─ 0° 회전 명령 ──────────────────→│                           │
     ├─ 2초 대기 시작                    │                           │
     │                                  ├─ 회전 시작 (0.8초 소요)    │
     │                                  ├─ 회전 완료                 │
     │                                  ├─ /motor/turntable_done ──→├─ 캡처 0° (정상)
     ├─ 2초 경과 → 90° 회전 명령 ──────→│                           │
     ├─ 2초 대기 시작                    │                           │
     │                                  ├─ 회전 시작 (부하로 2.3초)   │
     │                                  │                           │
     ├─ 2초 경과 → 180° 명령 ──────────→│  ← 아직 90° 회전 중!      │
     │                                  ├─ 새 명령 수신 → 90° 중단   │
     │                                  ├─ 180° 로 바로 이동         │
     │                                  │                           ├─ _pending_angle = 90
     │                                  │                           ├─ turntable_done(180) 수신
     │                                  │                           │   → pending_angle !=
     │                                  │                           │      180 이므로 캡처 안 함
     │                                  │                           │
     │                                  │                           ├─ 90° 캡처 누락!
```

**결과**: 90° 각도 이미지가 누락되어 검사가 3방향만으로 수행되거나, `_captured_images` 길이가 4가 되지 않아 검사가 영원히 트리거되지 않습니다. `inspect_timeout`이 발동하여 `ERROR`로 전이됩니다.

### 해결 방안

#### 1-A) `_setup_subscribers()`에 턴테이블 완료 구독 추가

```python
# main_orchestrator_node.py _setup_subscribers() 내
self.create_subscription(
    Bool, topics.TOPIC_MOTOR_TURNTABLE_DONE,  # '/motor/turntable_done'
    self._turntable_done_cb, 10)
```

#### 1-B) 콜백 및 내부 플래그 추가

```python
# main_orchestrator_node.py __init__() 내 상태 변수 영역
self._turntable_done = False

# 신규 콜백
def _turntable_done_cb(self, msg: Bool):
    """ESP32 턴테이블 회전 완료 수신."""
    if msg.data and self._state == FsmState.INSPECTING_CAPTURE:
        self._turntable_done = True
```

#### 1-C) FSM 상태 재설계

현재 두 개의 상태(`INSPECTING_ROTATE` + `INSPECTING_CAPTURE`)를 세 개로 확장하여 **명령 → 대기 → 완료확인** 패턴으로 변경합니다:

```
기존: INSPECTING_ROTATE → INSPECTING_CAPTURE → (루프) → INSPECTING_WAIT_RESULT
변경: INSPECTING_ROTATE → INSPECTING_WAIT_TURNTABLE → INSPECTING_CAPTURE → (루프) → INSPECTING_WAIT_RESULT
```

**새로운 FSM 상태 추가**:

```python
class FsmState(Enum):
    # ... 기존 상태 유지 ...
    INSPECTING_ROTATE = "INSPECTING_ROTATE"
    INSPECTING_WAIT_TURNTABLE = "INSPECTING_WAIT_TURNTABLE"  # ← 신규
    INSPECTING_CAPTURE = "INSPECTING_CAPTURE"
    # ...
```

**FSM 루프 변경**:

```python
elif self._state == FsmState.INSPECTING_TRIGGER:
    self._inspect_done = False
    inspect_trigger = Bool()
    inspect_trigger.data = True
    self._inspect_trigger_pub.publish(inspect_trigger)

    self._inspect_angle_idx = 0
    self._state_timer_counter = 0
    self._state = FsmState.INSPECTING_ROTATE

elif self._state == FsmState.INSPECTING_ROTATE:
    # 턴테이블 회전 명령 발행
    angle = self._inspect_angles[self._inspect_angle_idx]
    angle_msg = Int32()
    angle_msg.data = angle
    self._turntable_pub.publish(angle_msg)
    self.get_logger().info(f'턴테이블 {angle}도 회전 명령 전송')

    self._turntable_done = False
    self._state_timer_counter = 0
    self._state = FsmState.INSPECTING_WAIT_TURNTABLE  # ← 신규 상태로 전이

elif self._state == FsmState.INSPECTING_WAIT_TURNTABLE:
    # ESP32 실제 완료 피드백 대기 (with timeout)
    self._state_timer_counter += 1
    if self._turntable_done:
        self.get_logger().info(
            f'턴테이블 {self._inspect_angles[self._inspect_angle_idx]}도 회전 완료 확인')
        self._state_timer_counter = 0
        self._state = FsmState.INSPECTING_CAPTURE
    elif self._state_timer_counter > int(self._step_delay * self._loop_rate * 2):
        # timeout: step_delay * 2 (회전에 더 긴 시간 허용)
        self.get_logger().warn(
            f'턴테이블 완료 대기 타임아웃 — '
            f'{self._inspect_angles[self._inspect_angle_idx]}도 회전이 '
            f'늦어지고 있습니다. 그래도 다음 단계로 진행합니다.')
        self._state_timer_counter = 0
        self._state = FsmState.INSPECTING_CAPTURE

elif self._state == FsmState.INSPECTING_CAPTURE:
    # 기구 안정화 대기 (짧은 시간: step_delay_sec)
    self._state_timer_counter += 1
    if self._state_timer_counter >= int(self._step_delay * self._loop_rate):
        self._inspect_angle_idx += 1
        if self._inspect_angle_idx < len(self._inspect_angles):
            self._state = FsmState.INSPECTING_ROTATE
        else:
            self._state_timer_counter = 0
            self._state = FsmState.INSPECTING_WAIT_RESULT
```

#### 1-D) `topics.py`에 턴테이블 완료 토픽 상수 확인

`TOPIC_MOTOR_TURNTABLE_DONE = '/motor/turntable_done'`이 이미 정의되어 있습니다 (L28). 별도 추가 필요 없음.

> **설계 결정**: timeout 발생 시에도 다음 각도로 진행합니다 (ERROR로 가지 않음). 이는 일시적인 통신 지연으로 전체 사이클이 중단되는 것을 방지하기 위함입니다. 대신 경고 로그를 남기고, `inspect_node`는 캡처를 시도합니다. 만약 `_captured_images`가 4장 미만이면 `inspect_timeout`에 의해 `INSPECT_TIMEOUT` → `ERROR`가 발생하므로, 이중 안전장치가 됩니다.

### 변경 위치 요약

| 파일 | 위치 | 변경 내용 |
|------|------|----------|
| `main_orchestrator_node.py` | `FsmState` Enum | `INSPECTING_WAIT_TURNTABLE` 상태 추가 |
| `main_orchestrator_node.py` | `__init__()` 상태 변수 | `self._turntable_done = False` 추가 |
| `main_orchestrator_node.py` | `_setup_subscribers()` | 턴테이블 done 구독 추가 |
| `main_orchestrator_node.py` | 신규 콜백 | `_turntable_done_cb` 추가 |
| `main_orchestrator_node.py` | `_fsm_loop()` INSPECTING 영역 | 3단계로 재설계 |
| `main_orchestrator_node.py` | `full_system.launch.py` | 파라미터에 `step_delay_sec` 적절히 조정 (회전 대기와 캡처 대기를 분리했으므로) |

### 예상 시퀀스 다이어그램

```
[오케스트레이터]                [ESP32]                    [inspect_node]
     │                             │                           │
     ├─ INSPECTING_ROTATE          │                           │
     ├─ 0° 명령 ──────────────────→│                           │
     ├─ WAIT_TURNTABLE             │                           │
     │                             ├─ 회전 시작                │
     │                             ├─ 회전 완료                │
     │                             ├─ /motor/turntable_done ──→├─ 캡처 0°
     │← _turntable_done=True ─────┤                           │
     ├─ INSPECTING_CAPTURE         │                           │
     ├─ 2초 안정화 대기             │                           │
     ├─ INSPECTING_ROTATE          │                           │
     ├─ 90° 명령 ─────────────────→│                           │
     ├─ WAIT_TURNTABLE             │                           │
     │                             ├─ 회전 시작 (부하로 2.3초) │
     │                             ├─ 회전 완료                │
     │                             ├─ /motor/turntable_done ──→├─ 캡처 90°
     │← _turntable_done=True ─────┤                           │
     ├─ INSPECTING_CAPTURE         │                           │
     ├─ ...                        │                           │
     │  (4방향 모두 완료)           │                           │
     ├─ INSPECTING_WAIT_RESULT     ├─ 대기                      ├─ _run_inspection()
```

---

## 이슈 2: `inspect_node`의 `_turntable_cmd_callback` 정리

### 현재 상태

`inspect_node.py`는 두 가지 방식으로 턴테이블 각도를 추적합니다:

```python
# L55-57: 명령 토픽 구독 (오케스트레이터 → ESP32)
self._turntable_cmd_sub = self.create_subscription(
    Int32, topics.TOPIC_MOTOR_TURNTABLE_CMD,  # '/motor/turntable_cmd'
    self._turntable_cmd_callback, 10)

# L59-61: 완료 토픽 구독 (ESP32 → 시스템)
self._turntable_done_sub = self.create_subscription(
    Bool, topics.TOPIC_MOTOR_TURNTABLE_DONE,  # '/motor/turntable_done'
    self._turntable_done_callback, 10)
```

`_turntable_cmd_callback`(L175-178)은 단순히 `_pending_angle`을 설정하고, `_turntable_done_callback`(L180-185)이 실제 캡처를 트리거합니다. 하지만 **오케스트레이터가 `INSPECTING_ROTATE`에서 바로 `INSPECTING_CAPTURE`로 전이**하기 때문에, 오케스트레이터가 이미 다음 각도 명령을 보낸 후에 ESP32의 이전 각도 done이 도착하는 타이밍 문제가 발생할 수 있습니다.

ESTOP 콜백과 마찬가지로, `_turntable_cmd_callback`은 더 이상 필요하지 않습니다. `_turntable_done_callback`만으로 충분합니다.

### 해결 방안

`_turntable_cmd_callback`과 `_turntable_cmd_sub`를 완전히 제거하고, `_pending_angle` 상태도 `_turntable_done_callback`에서 직접 판단하도록 단순화합니다.

#### 2-A) 제거할 코드

```python
# 제거: L55-57
self._turntable_cmd_sub = self.create_subscription(
    Int32, topics.TOPIC_MOTOR_TURNTABLE_CMD,
    self._turntable_cmd_callback, 10)

# 제거: L79
self._pending_angle: Optional[int] = None

# 제거: L175-178
def _turntable_cmd_callback(self, msg: Int32):
    """턴테이블 목표 각도 수신."""
    if self._inspection_active and msg.data in self._angles:
        self._pending_angle = msg.data
```

#### 2-B) `_turntable_done_callback` 간소화

턴테이블 완료 신호가 오면, **현재 대기 중인 각도를 `self._inspect_angle_idx`로 추적**합니다 (이슈 1의 개선 후 오케스트레이터가 순차적으로 각도를 보내므로, done도 순서대로 도착한다고 가정할 수 있습니다):

```python
def _turntable_done_callback(self, msg: Bool):
    """턴테이블 이동 완료 시 즉시 캡처."""
    if not self._inspection_active or not msg.data:
        return

    # 어떤 각도가 완료되었는지는 ESP32 상태 메시지로 확인 가능하나,
    # 오케스트레이터가 순차적으로 명령을 보내므로, done 순서 = 각도 순서로 가정.
    # 이미 캡처된 각도는 건너뛰고, 아직 캡처되지 않은 가장 빠른 각도를 캡처.
    for angle in self._angles:
        if angle not in self._captured_images:
            self._capture_angle(angle)
            break
```

> **참고**: 만약 순서가 보장되지 않는 환경이라면, ESP32의 `/motor/status`(`MotorStatus.msg`)에 포함된 `turntable_angle` 값을 사용하여 어떤 각도가 완료되었는지 확인할 수 있습니다. 현재 `MotorStatus` 메시지에는 `turntable_angle` 필드가 포함되어 있으므로(펌웨어 L402), 향후 이 방식으로 전환할 수 있습니다.

#### 2-C) `_trigger_callback`, `_run_inspection`에서 `_pending_angle` 초기화 제거

```python
def _trigger_callback(self, msg: Bool):
    if msg.data:
        self._inspection_active = True
        self._captured_images.clear()
        # self._pending_angle = None  ← 제거
        self.get_logger().info('검사 모드 활성화 — 턴테이블 회전 대기 중')
    else:
        self._inspection_active = False
        # self._pending_angle = None  ← 제거

def _run_inspection(self):
    # ... 검사 로직 ...
    self._inspection_active = False
    self._captured_images.clear()
    # self._pending_angle = None  ← 제거
```

### 변경 위치 요약

| 파일 | 위치 | 변경 내용 |
|------|------|----------|
| `inspect_node.py` | L55-57 | `_turntable_cmd_sub` 구독 제거 |
| `inspect_node.py` | L79 | `_pending_angle` 변수 제거 |
| `inspect_node.py` | L175-178 | `_turntable_cmd_callback` 메서드 제거 |
| `inspect_node.py` | L180-185 | `_turntable_done_callback` 간소화 |
| `inspect_node.py` | L203-208 | `_pending_angle` 초기화 제거 |
| `inspect_node.py` | L271-273 | `_pending_angle` 초기화 제거 |
| `inspect_node.py` | L54 (import) | `Int32` import 제거 (다른 곳에서 사용하지 않는 경우) |

> **참고**: `Int32`는 다른 import가 없으면 제거합니다. `topics` import는 `TOPIC_MOTOR_TURNTABLE_CMD`가 제거되면 사용하지 않을 수 있으므로 확인이 필요합니다. `TOPIC_MOTOR_TURNTABLE_DONE`만 사용한다면 `topics` import는 유지합니다.

---

## 이슈 3: 기준 이미지 없을 때 `area_ratio=1.0` 폴백 개선

### 현재 상태

`inspect_node.py`의 `_surface_analysis()` 메서드에서 기준 이미지가 없을 때:

```python
# L487-494
ref = self._reference_images.get(angle)
if ref is not None:
    ref_resized = cv2.resize(ref, (cache.gray.shape[1], cache.gray.shape[0]))
    ref_area = BinaryCache(ref_resized, self._bin_thresh).largest_external_area()
    cap_area = cache.largest_external_area()
    a_ratio = cap_area / ref_area if ref_area > 0 else 0.0
else:
    a_ratio = 1.0  # ← 기준 이미지 없으면 정상으로 간주
```

**문제**: 기준 이미지가 하나도 없을 때, 모든 면적 비율이 `1.0`이 되어 "정상 범위"(`0.90~1.10`)에 들어갑니다. 표면 특징 분석은 통과하고, CAD 비교는 실패하지만, 전체 판정은 `cad_pass AND surface_pass`이므로 CAD 비교에서 이미 실패하기 때문에 실제로는 PASS로 오판될 위험은 낮습니다.

그러나 **일부 각도만 기준 이미지가 있는 경우**(예: `ref_0.png`만 존재)에는 심각한 문제가 됩니다. 0°는 정상 비교, 90°~270°는 `a_ratio=1.0`으로 통과 → CAD 비교만 실패 → "CAD비교실패" 원인만 표시되고 표면 특징은 정상으로 나옵니다. 디버깅이 어려워집니다.

### 해결 방안

#### 3-A) 검사 전 기준 이미지 완전성 검증

```python
# inspect_node.py __init__() 내
def _load_reference_images(self):
    """기준 이미지(STL 렌더링 결과)를 로드하고 완전성을 검증한다."""
    if not os.path.isdir(self._ref_dir):
        self.get_logger().warn(
            f'기준 이미지 디렉토리 없음: {self._ref_dir} — '
            f'stl_renderer로 먼저 생성하세요.')
        return

    for angle in self._angles:
        path = os.path.join(self._ref_dir, f'ref_{angle}.png')
        if os.path.isfile(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self._reference_images[angle] = img
                self.get_logger().info(f'기준 이미지 로드: {path}')
            else:
                self.get_logger().warn(f'기준 이미지 읽기 실패: {path}')
        else:
            self.get_logger().warn(f'기준 이미지 없음: {path}')

    loaded = len(self._reference_images)
    expected = len(self._angles)
    self.get_logger().info(f'기준 이미지 {loaded}/{expected}개 로드됨')

    # 완전성 검증: 일부만 로드된 경우 경고
    if 0 < loaded < expected:
        missing = [a for a in self._angles if a not in self._reference_images]
        self.get_logger().error(
            f'기준 이미지가 불완전합니다! '
            f'누락된 각도: {missing}. '
            f'검사 결과의 신뢰도가 떨어집니다.')
```

#### 3-B) `_surface_analysis()`에서 `area_ratio` 폴백 제거

```python
# inspect_node.py _surface_analysis() L487-494 변경
ref = self._reference_images.get(angle)
if ref is not None:
    ref_resized = cv2.resize(ref, (cache.gray.shape[1], cache.gray.shape[0]))
    ref_area = BinaryCache(ref_resized, self._bin_thresh).largest_external_area()
    cap_area = cache.largest_external_area()
    a_ratio = cap_area / ref_area if ref_area > 0 else 0.0
else:
    # 기준 이미지가 없으면 면적 비교를 건너뛰고,
    # 해당 각도는 다른 지표(Solidity, Holes, Texture)로만 판정
    a_ratio = float('nan')  # ← NaN 으로 표시하여 "측정 불가"임을 명시
```

#### 3-C) 판정 로직에서 NaN 처리

```python
# inspect_node.py _surface_analysis() 판정 부분 (L519-533 영역)
for angle, feats in angle_features.items():
    sol   = feats['solidity']
    area  = feats['area_ratio']
    holes = feats['hole_count']
    h_ar  = feats['hole_area_ratio']
    tex   = feats['texture_variance']

    if not (self._sol_min <= sol <= self._sol_max):
        all_pass = False
        fail_details.append(f'{angle}°워핑:Solidity={sol:.3f}')

    # area_ratio 가 NaN 이면 면적 비교를 건너뜀
    if not math.isnan(area):
        if not (self._f_area_min <= area <= self._f_area_max):
            all_pass = False
            fail_details.append(f'{angle}°미출력:면적비={area:.3f}')
    # else: 기준 이미지 없음 — 면적 비교 불가, 다른 지표만으로 판정

    if holes > self._hole_max:
        all_pass = False
        fail_details.append(f'{angle}°레이어분리:구멍={holes}개')
    if h_ar > self._hole_area_max:
        all_pass = False
        fail_details.append(f'{angle}°레이어분리:구멍면적={h_ar:.3f}')
    if tex > self._tex_var_max:
        all_pass = False
        fail_details.append(f'{angle}°스트링잉:텍스처={tex:.1f}')
```

> **참고**: `math.isnan()`을 사용하려면 파일 상단에 `import math`가 필요합니다. 현재 `inspect_node.py`에는 `import math`가 없으므로 추가해야 합니다.

#### 3-D) worst_area 계산 시 NaN 제외

```python
# inspect_node.py _surface_analysis() 결과 집계 부분 (L536-544)
# worst_area: NaN 제외하고 계산
valid_areas = [a for a in all_area if not math.isnan(a)]
worst_area = max(valid_areas, key=lambda a: abs(1.0 - a)) if valid_areas else float('nan')
```

#### 3-E) HMI/로그에서 NaN 표시 처리

`InspectionResult` 메시지의 `area_ratio` 필드는 `float` 타입이므로 NaN을 그대로 전송할 수 있습니다. HMI 대시보드에서는 NaN을 "N/A"로 표시하도록 처리합니다. 로그 저장 시에도 `math.isnan()` 체크 후 "N/A"로 기록합니다.

```python
# inspect_node.py _save_inspection_log() 결과 기록 부분
f.write(f'면적비(표면): {"N/A" if math.isnan(surface["area_ratio"]) else surface["area_ratio"]:.4f}\n')
```

### 변경 위치 요약

| 파일 | 위치 | 변경 내용 |
|------|------|----------|
| `inspect_node.py` | 상단 import | `import math` 추가 |
| `inspect_node.py` | `_load_reference_images()` L155-156 | 불완전 로드 시 ERROR 로그 추가 |
| `inspect_node.py` | `_surface_analysis()` L494 | `a_ratio = 1.0` → `a_ratio = float('nan')` |
| `inspect_node.py` | `_surface_analysis()` L522-524 | `math.isnan()` 체크로 면적 비교 건너뜀 |
| `inspect_node.py` | `_surface_analysis()` L543 | worst_area 계산 시 NaN 제외 |
| `inspect_node.py` | `_save_inspection_log()` L667 | NaN → "N/A" 기록 |
| `dashboard.html` / `dashboard.js` | HMI 영역 | NaN → "N/A" 표시 (선택적) |

---

## 테스트 계획

### 이슈 1 (턴테이블 피드백) 테스트

```
[시뮬레이션 모드]
1. use_real_hardware:=false 로 실행
2. ESP32 없이 /motor/turntable_done 토픽을 수동 발행하며 테스트
3. 확인사항:
   - INSPECTING_WAIT_TURNTABLE 상태에서 turntable_done 수신 전까지 대기
   - turntable_done 수신 후 즉시 INSPECTING_CAPTURE로 전이
   - timeout 발생 시 경고 로그와 함께 다음 단계로 진행
   - 4방향 모두 정상적으로 순회

[실제 하드웨어]
1. ESP32 연결 상태에서 전체 자율 시퀀스 실행
2. 확인사항:
   - 각 각도에서 ESP32 회전 완료 후에만 다음 각도로 전이
   - 이전보다 전체 INSPECTING 단계 시간이 다소 증가 (턴테이블 회전 시간만큼)
   - 캡처 누락 없이 4방향 모두 검사 수행
```

### 이슈 2 (코드 정리) 테스트

```
[회귀 테스트]
1. 기존 검사 시퀀스 정상 동작 확인
2. /motor/turntable_cmd 토픽을 구독하지 않아도 캡처 누락 없음 확인
3. inspect_node 로그에서 turntable_done 기반 캡처 메시지 정상 출력

[코드 품질]
4. pylint/flake8: 미사용 import(Int32, topics) 정리 확인
```

### 이슈 3 (기준 이미지) 테스트

```
[기준 이미지 없는 상태]
1. reference_image_dir를 빈 디렉토리로 설정
2. 검사 실행
3. 확인사항:
   - CAD 비교: 모든 각도 "이미지없음" → fail
   - 표면 특징: area_ratio = NaN, 면적 비교 건너뜀
   - Solidity, Holes, Texture는 정상 측정
   - 전체 판정: FAIL (CAD 비교 실패로)
   - 로그 파일에 area_ratio "N/A" 기록

[기준 이미지 일부만 있는 상태]
4. ref_0.png만 존재하도록 설정
5. 확인사항:
   - 0°: 정상 면적 비교
   - 90°, 180°, 270°: 면적 비교 건너뜀 (NaN)
   - worst_area: 0°의 area_ratio만으로 계산
   - _load_reference_images()에서 "불완전" ERROR 로그 출력

[기준 이미지 정상]
6. 4방향 모두 존재
7. 확인사항:
   - 기존과 동일하게 동작
   - area_ratio가 NaN이 아닌 정상 값
```

---

## 구현 순서

1. **1단계** (15분): 이슈 2 — `inspect_node`에서 `_turntable_cmd_callback` 제거 (가장 간단, 의존성 없음)
2. **2단계** (1시간): 이슈 1 — 오케스트레이터 FSM에 `INSPECTING_WAIT_TURNTABLE` 상태 추가
3. **3단계** (45분): 이슈 3 — 기준 이미지 NaN 처리 + `_load_reference_images` 완전성 검증
4. **4단계** (30분): 통합 테스트 (시뮬레이션 전체 시퀀스)

---

## 관련 파일 요약

| 파일 | 변경 규모 | 위험도 |
|------|----------|--------|
| `main_orchestrator_node.py` | **약 50줄** (상태 추가 + FSM 변경 + 구독 추가) | 중간 — FSM 로직 변경 |
| `inspect_node.py` | **약 30줄** (코드 제거 15줄 + NaN 처리 15줄) | 낮음 |
| `topics.py` | 변경 없음 (이미 `TOPIC_MOTOR_TURNTABLE_DONE` 정의됨) | - |
| `dashboard.html` / `dashboard.js` | **약 5줄** (NaN → "N/A" 표시) | 낮음 — cosmetic |