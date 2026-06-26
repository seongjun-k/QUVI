# QUVI 자율 시퀀스 구현 계획서

## 배경 및 목적

ACT 추론 오류로 인해 LeRobot ACT 기반 파지를 우회한다.
대신 `test_sequence.py`에 티칭된 웨이포인트·속도 값을 그대로 활용하여
대시보드 START 버튼 → 자율 시퀀스 실행 → 양불 분기 배출 흐름을 구현한다.

ACT 추론은 Physical AI Tools에서 별도 실행하며 QUVI와 USB 포트를 분리한다.

---

## 브랜치 운영 규칙

| 브랜치 | 용도 | 수정 가능 여부 |
|---|---|---|
| `master` | 원본 보존 | **절대 수정 금지** |
| `demo/sequence-no-act` | 발표용 구현 작업 브랜치 | 이 브랜치에서만 작업 |

- 에이전트는 반드시 `demo/sequence-no-act` 브랜치에서만 파일을 수정한다.
- `master`로 직접 커밋하지 않는다.
- 발표 후 검증이 완료되면 별도로 PR을 열어 master에 병합한다.

---

## 전제 조건

| 항목 | 내용 |
|---|---|
| 참고 파일 | `/home/ksj/QUVI/scripts/test_sequence.py` (위치·속도·이동방식 그대로 사용) |
| 실제 구현 위치 | `src/quvi_hmi/quvi_hmi/hmi_node.py` 에만 작성 |
| ACT | 분리 운영 — QUVI와 동시에 실행하지 않음 |
| 레일 위치 단일 출처 | `hmi_node.py` 상단 `RAIL_STATION_MAP` 상수 |

### RAIL_STATION_MAP (변경 금지)

```python
RAIL_STATION_MAP = [
    {'name': 'INSPECT (A)', 'mm': 12.5},   # 베드 = 검사 위치 = 복귀 홈
    {'name': 'PASS (B)',    'mm': 25.0},
    {'name': 'FAIL (C)',    'mm': 125.0},
    {'name': 'BED (D)',     'mm': 381.25},
]
```

> 베드 위치와 검사 위치는 동일하다 → `INSPECT (A) 12.5 mm` 사용.

---

## 시퀀스 흐름

```
[대시보드 START 버튼]
        │
        ▼
① 팔: P1 → P2 → P3 → P4       # 베드(=검사 위치)에서 출력물 접근
② 그리퍼 열기 → 출력물 내려놓기
③ 팔: P3 퇴출 대기
        │
        ▼
④ /inspection/result 수신 대기  # 최대 30초, 이벤트 기반
        │
        ├── passed=True  → 레일 → PASS  (25.0 mm)
        └── passed=False → 레일 → FAIL  (125.0 mm)
        │
        ▼
⑤ 팔: P4 재파지 → 그리퍼 닫기
⑥ 팔: P3 → P5 → P1 → P6
⑦ 그리퍼 열기 (배출)
        │
        ▼
⑧ 레일 → HOME (12.5 mm)        # 베드=검사 위치로 복귀
```

레일은 **판정 분기 1회 + 복귀 1회**, 총 2회만 이동한다.

---

## 수정 대상

**파일 1개만 수정한다.**

```
src/quvi_hmi/quvi_hmi/hmi_node.py
```

`test_sequence.py`는 참고 자료이므로 수정하지 않는다.

---

## 구체적 변경 내용

### 1. `HmiNode.__init__` 끝에 추가

```python
# ─── 시퀀스 제어 ───
self._seq_thread = None
self._seq_stop_event = threading.Event()
self._inspect_result_event = threading.Event()
self._inspect_passed = False
```

---

### 2. `_inspection_cb` — with 블록 끝에 2줄 추가

```python
# 기존 with self._lock 블록 마지막에 추가
self._inspect_passed = msg.passed
self._inspect_result_event.set()
```

---

### 3. `send_command` 수정

기존 메서드에 START/STOP 분기를 추가한다.

```python
def send_command(self, command: str):
    msg = String()
    msg.data = command
    self._cmd_pub.publish(msg)
    self.get_logger().info(f'HMI 명령: {command}')

    if command == 'START':
        self._start_sequence()
    elif command in ('STOP', 'ESTOP'):
        self._stop_sequence()
```

---

### 4. 시퀀스 메서드 3개 추가 (클래스 내부)

#### `_start_sequence`

```python
def _start_sequence(self):
    if self._seq_thread and self._seq_thread.is_alive():
        self.get_logger().warn('시퀀스 이미 실행 중')
        return
    self._seq_stop_event.clear()
    self._seq_thread = threading.Thread(
        target=self._run_sequence, daemon=True)
    self._seq_thread.start()
```

#### `_stop_sequence`

```python
def _stop_sequence(self):
    self._seq_stop_event.set()
```

#### `_run_sequence`

```python
def _run_sequence(self):
    from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncWrite

    # ── 하드웨어 상수 (test_sequence.py 동일) ──
    PORT     = '/dev/ttyFollower'
    BAUDRATE = 1_000_000
    PROTOCOL = 2.0
    MOTORS = {
        'shoulder_pan': 11, 'shoulder_lift': 12, 'elbow_flex': 13,
        'wrist_flex': 14,   'wrist_roll': 15,    'gripper': 16,
    }
    ADDR_TORQUE_ENABLE    = 64
    ADDR_PROFILE_ACCEL    = 108
    ADDR_PROFILE_VELOCITY = 112
    ADDR_GOAL_POSITION    = 116
    LEN_GOAL_POSITION     = 4
    PROFILE_VELOCITY      = 8
    PROFILE_ACCEL         = 3
    PROFILE_VELOCITY_GRIP = 20
    PROFILE_ACCEL_GRIP    = 5
    MOVE_WAIT             = 15.0
    GRIPPER_WAIT          = 3.0
    SETTLE                = 0.3
    RAIL_WAIT             = 5.0   # 레일 이동 완료 고정 대기

    # ── 웨이포인트 (test_sequence.py 동일) ──
    POSE_P1 = {'shoulder_pan':2054,'shoulder_lift':1258,'elbow_flex':2800,'wrist_flex':2981,'wrist_roll':2035,'gripper':2150}
    POSE_P2 = {'shoulder_pan':  12,'shoulder_lift':1843,'elbow_flex':2165,'wrist_flex':3123,'wrist_roll':2095,'gripper':2150}
    POSE_P3 = {'shoulder_pan':  16,'shoulder_lift':1736,'elbow_flex':2413,'wrist_flex':3018,'wrist_roll':2087,'gripper':2150}
    POSE_P4 = {'shoulder_pan':  16,'shoulder_lift':1841,'elbow_flex':2522,'wrist_flex':2759,'wrist_roll':2085,'gripper':2150}
    POSE_P5 = {'shoulder_pan':2047,'shoulder_lift':1854,'elbow_flex':2460,'wrist_flex':2909,'wrist_roll':2050,'gripper':2150}
    POSE_P6 = {'shoulder_pan':2039,'shoulder_lift':1076,'elbow_flex':2884,'wrist_flex':3094,'wrist_roll':1993,'gripper':2150}

    # ── 레일 위치 (RAIL_STATION_MAP 기준) ──
    RAIL_HOME = 12.5    # INSPECT (A) = 베드 = 검사 위치 = 복귀 홈
    RAIL_PASS = 25.0    # PASS (B)
    RAIL_FAIL = 125.0   # FAIL (C)

    # ── 내부 헬퍼 ──
    def open_bus():
        port = PortHandler(PORT)
        pkt  = PacketHandler(PROTOCOL)
        if not port.openPort():
            self.get_logger().error(f'포트 열기 실패: {PORT}')
            return None, None
        port.setBaudRate(BAUDRATE)
        return port, pkt

    def set_torque(port, pkt, enable):
        val = 1 if enable else 0
        for mid in MOTORS.values():
            pkt.write1ByteTxRx(port, mid, ADDR_TORQUE_ENABLE, val)

    def apply_profile(port, pkt, names, vel, acc):
        for n in names:
            mid = MOTORS[n]
            pkt.write4ByteTxRx(port, mid, ADDR_PROFILE_ACCEL, acc)
            pkt.write4ByteTxRx(port, mid, ADDR_PROFILE_VELOCITY, vel)

    def write_pose(port, pkt, pose):
        sw = GroupSyncWrite(port, pkt, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
        for name, val in pose.items():
            param = [val&0xFF,(val>>8)&0xFF,(val>>16)&0xFF,(val>>24)&0xFF]
            sw.addParam(MOTORS[name], param)
        sw.txPacket()
        sw.clearParam()

    def move_to(port, pkt, pose, label):
        if self._seq_stop_event.is_set():
            return False
        self.get_logger().info(f'[시퀀스] {label}')
        arm = [k for k in pose if k != 'gripper']
        if arm:
            apply_profile(port, pkt, arm, PROFILE_VELOCITY, PROFILE_ACCEL)
        write_pose(port, pkt, pose)
        end = time.time() + MOVE_WAIT + SETTLE
        while time.time() < end:
            if self._seq_stop_event.is_set():
                return False
            time.sleep(0.1)
        return True

    def rail_move(mm, label):
        if self._seq_stop_event.is_set():
            return False
        self.get_logger().info(f'[시퀀스] 레일 → {label} ({mm} mm)')
        self.send_rail_command(mm)
        time.sleep(RAIL_WAIT)
        return not self._seq_stop_event.is_set()

    def gripper_open(port, pkt):
        apply_profile(port, pkt, ['gripper'], PROFILE_VELOCITY_GRIP, PROFILE_ACCEL_GRIP)
        write_pose(port, pkt, {'gripper': 2300})
        time.sleep(GRIPPER_WAIT)

    def gripper_close(port, pkt):
        apply_profile(port, pkt, ['gripper'], PROFILE_VELOCITY_GRIP, PROFILE_ACCEL_GRIP)
        write_pose(port, pkt, {'gripper': 1800})
        time.sleep(GRIPPER_WAIT)

    # ── 시퀀스 본문 ──
    port, pkt = open_bus()
    if port is None:
        return
    set_torque(port, pkt, True)

    try:
        ok = True

        # 1. 베드(=검사 위치)에서 출력물 접근 및 내려놓기
        ok = ok and move_to(port, pkt, POSE_P1, 'P1 베드 위 대기')
        ok = ok and move_to(port, pkt, POSE_P2, 'P2 180도 회전')
        ok = ok and move_to(port, pkt, POSE_P3, 'P3 턴테이블 진입점')
        ok = ok and move_to(port, pkt, POSE_P4, 'P4 놓기 지점')
        if ok:
            gripper_open(port, pkt)

        ok = ok and move_to(port, pkt, POSE_P3, 'P3 퇴출 대기')

        # 2. 검사 결과 대기 (/inspection/result 수신 이벤트)
        if ok:
            self.get_logger().info('[시퀀스] 검사 결과 대기...')
            self._inspect_result_event.clear()
            signaled = self._inspect_result_event.wait(timeout=30.0)
            if not signaled:
                self.get_logger().warn('[시퀀스] 검사 타임아웃 — 양품으로 처리')
                self._inspect_passed = True
            passed = self._inspect_passed
            self.get_logger().info(f'[시퀀스] 판정: {"양품" if passed else "불량"}')

        # 3. 양불 레일 분기
        if ok:
            if passed:
                ok = rail_move(RAIL_PASS, 'PASS')
            else:
                ok = rail_move(RAIL_FAIL, 'FAIL')

        # 4. 재파지 후 배출
        ok = ok and move_to(port, pkt, POSE_P4, 'P4 출력물 재파지')
        if ok:
            gripper_close(port, pkt)
        ok = ok and move_to(port, pkt, POSE_P3, 'P3 퇴출')
        ok = ok and move_to(port, pkt, POSE_P5, 'P5 180도 반대 회전')
        ok = ok and move_to(port, pkt, POSE_P1, 'P1 베드 위 대기')
        ok = ok and move_to(port, pkt, POSE_P6, 'P6 배출 지점')
        if ok:
            gripper_open(port, pkt)

        # 5. 레일 홈 복귀 (베드=검사 위치)
        rail_move(RAIL_HOME, '홈/베드 복귀')
        self.get_logger().info('[시퀀스] 완료')

    except Exception as e:
        self.get_logger().error(f'[시퀀스] 예외: {e}')
    finally:
        port.closePort()
```

---

## 변경 금지 항목

- `test_sequence.py` — 수정 금지 (참고 자료)
- `RAIL_STATION_MAP` 상수 — mm 값 수정 시 이 파일 한 곳에서만 변경
- 기존 Flask 라우트, SocketIO 브로드캐스트, 카메라 콜백 — 건드리지 않음
- 한국어 주석 스타일 및 `# ─── Section ───` 구분자 형식 유지

---

## 점검 체크리스트

- [ ] `__init__`에 시퀀스 제어 변수 4개 추가 확인
- [ ] `_inspection_cb` 내 `_inspect_result_event.set()` 추가 확인
- [ ] `send_command`에 START/STOP 분기 추가 확인
- [ ] `_start_sequence`, `_stop_sequence`, `_run_sequence` 3개 메서드 추가 확인
- [ ] 레일 이동: 판정 분기 1회 + 복귀 1회, 총 2회만 발생하는지 확인
- [ ] STOP/ESTOP 버튼 누르면 `_seq_stop_event` 세트 → `move_to` 루프 즉시 탈출 확인
- [ ] `/inspection/result` 미수신 시 30초 후 양품 처리 확인
