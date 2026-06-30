# ACT/시퀀스 로봇 파손 — 원인 분석 및 해결 계획서

작성일: 2026-07-01
대상 브랜치: master (demo/sequence-no-act 병합 이후)
증상: ACT 모델 실행 시 로봇팔이 "쫙 펴지며" 폭주 → 파손. ACT 종료 후 시퀀스에서도 동일 폭주 발생. 시퀀스 타이밍 문제 동반.

---

## 1. 핵심 결론 (요약)

이번 폭주는 단일 버그가 아니라 **두 부류의 결함이 겹친 결과**다.

1. **구조적 결함 — 동일한 6개 서보를 두 개의 독립 컨트롤러가 같은 시리얼 포트로 동시에 제어한다.**
   `demo/sequence-no-act` 병합으로 들어온 HMI 자체 시퀀스(`_run_sequence`)와
   기존 `master`의 오케스트레이터→`robot_control_node` 경로가 **START 한 번에 동시에 작동**한다.
   둘 다 `/dev/ttyFollower`를 열고 같은 모터에 서로 다른 Goal_Position을 쓴다.
   → 패킷 충돌 + 모순된 목표값 → 예측 불가능한 폭주. **이것이 "ACT든 시퀀스든 똑같이 박살"의 직접 원인.**

2. **안전장치 결함 — ACT 출력에 대한 상대이동 제한·관절 제한이 모두 비활성/무효 상태다.**
   ACT가 이상치(특히 첫 추론·조명 변화 시) 하나만 뱉어도 모터가 그 위치로 **즉시 최대 프로파일 속도로 돌진**한다.
   다회전(EXTENDED_POSITION) 관절은 위치 제한조차 없어 "쫙 펴지는" 형태로 폭주한다.

아래에 각 원인을 코드 위치와 함께 정리하고, 우선순위별 해결 계획을 제시한다.

---

## 2. 근본 원인 상세

### C1. (치명적) 이중 제어 — 두 컨트롤러가 동일 포트/동일 모터를 동시 구동

**경로 A (오케스트레이터, master 원본):**
`full_system.launch.py:176-178` 가 `hmi_node` + `robot_control_node` + `main_orchestrator_node`를 **항상 함께** 기동.
`robot_control_node`는 `use_real_hardware=true`면 ACT 사용 여부와 무관하게 시작 시 `/dev/ttyFollower`를 연다
(`robot_control_node.py:233-234` → `_init_follower`). 게다가 10Hz로 `Present_Position`을 계속 읽는다
(`robot_control_node.py:246-248`, `_publish_joint_states`).

**경로 B (HMI 자체 시퀀스, demo 병합분):**
`hmi_node.send_command('START')` 이 두 가지를 **동시에** 한다 (`hmi_node.py:241-250`):
- `/hmi/command` 로 `START` 발행 → 오케스트레이터가 받아 경로 A 전체 사이클 구동
  (`main_orchestrator_node.py:207-222` → STARTUP → GRASPING → ...)
- 같은 함수에서 `self._start_sequence()` 호출 → `_run_sequence` 스레드가
  **raw `dynamixel_sdk`로 `/dev/ttyFollower`를 직접 열고**(`hmi_node.py:328,331,367-374`) P1~P6를 직접 구동.

결과:
- `/dev/ttyFollower` 1 Mbps 반이중 버스에 **두 개의 PortHandler**가 붙어 패킷이 섞임 → 깨진 Goal_Position 수신.
- 동시에 **두 안무(choreography)가 서로 다른 목표**를 보냄: 오케스트레이터는 grasp→place_chamber(lift/pan=0)→…,
  HMI 시퀀스는 P1→P2(pan 180°)→P3→P4… → 모터가 상반된 목표 사이에서 진동·폭주.
- `use_act=true`/`false` 모두에서 발생 → 사용자가 본 "ACT 종료 후 시퀀스도 동일" 증상과 정확히 일치.

> 참고: Linux에서 동일 tty를 두 번 open 하는 것은 기본적으로 막히지 않으므로 두 핸들러가 공존하며 충돌한다.
> 설령 open이 실패하더라도 `_run_sequence`는 조용히 종료될 뿐(경로 B만 죽음) 근본 설계 결함은 남는다.

### C2. (치명적, ACT) `max_relative_target=None` — 상대이동 안전캡이 꺼져 있음

`OmxFollower.send_action`은 `max_relative_target`가 설정된 경우에만 현재 위치 대비 이동량을 제한
(`ensure_safe_goal_position`)한다 (`omx_follower.py:257-260`). 그런데 기본값이 `None`
(`config_omx_follower.py:33`)이고 QUVI 코드는 이 값을 설정하지 않는다.
→ ACT가 내놓는 **한 스텝의 목표가 현재 위치에서 아무리 멀어도 그대로 전송**된다.
ACT는 첫 추론(액션 큐 미충전), 분포 밖 이미지(LED on/노출 변화) 등에서 이상치를 내기 쉽고,
그 한 번이면 모터가 극단 위치로 돌진한다.

### C3. ACT 경로의 관절 제한 `_clip_shoulder_lift`가 사실상 무효

`robot_control_node.py:905-938`. ACT 경로는 `is_raw=False`로 호출되는데(`:706`),
이때 한계값이 `-100.0` = 정규화 **최솟값**이라 `값 < -100.0` 조건이 거의 성립하지 않는다.
즉 클리핑이 **절대 동작하지 않는다**(코드 주석도 이 점을 자인: `:913-915`).
ACT 경로에는 실질적인 소프트웨어 관절 보호가 **전무**하다.

### C4. 다회전(EXTENDED_POSITION) 관절에 위치 제한 없음 → "쫙 펴짐"의 형태적 원인

`omx_follower.py:170-178`에서 `shoulder_pan`(ID11)·`wrist_roll`(ID15)은 **Operating_Mode 4 = EXTENDED_POSITION**(다회전).
`configure()`는 `shoulder_lift/elbow_flex/wrist_flex`에만 Min/Max_Position_Limit를 설정하고
(`omx_follower.py:197-204`), **shoulder_pan·wrist_roll에는 위치 제한을 두지 않는다.**
이 둘은 정규화 모드도 DEGREES(`omx_follower.py:56,60`)라, ACT가 분포 밖 값을 주면
베이스가 한 바퀴 이상 돌거나 손목이 끝없이 회전 → 팔 전체가 펼쳐지며 주변과 충돌·파손.

### C5. (검증 필요) ACT 이미지/관측 전처리가 학습 분포와 불일치 가능성

`robot_control_node.py:654-657`: 카메라 프레임을 640×480으로 resize 후 `/255.0`만 적용(평균/표준편차 정규화 없음),
관측 키는 `observation.images.camera1`(`:679`). 학습 당시 해상도·정규화·키 이름과 어긋나면
정책 입력이 OOD가 되어 이상치 액션을 유발한다. C2와 결합하면 곧바로 폭주로 이어진다.
모델 메타(`act_model_path`의 `config.json`/`normalize` 통계)와 대조해 **반드시 확인**해야 한다.

### C6. Profile_Velocity 전역 상태 누수 — 동작별 속도 불일치

`_apply_motor_profile`(`robot_control_node.py:943-953`)이 Profile_Velocity/Acceleration를 모터 EEPROM/RAM에
전역으로 쓰고 그대로 남는다. ACT 경로의 `send_action`은 프로파일을 세팅하지 않으므로
직전 동작이 남긴 값(초기 `configure()`=50, 또는 시퀀스가 남긴 1/2/8)을 그대로 물려받는다.
→ ACT 돌진 시 속도가 상황에 따라 달라지고, 빠른 프로파일이 남아 있으면 충격이 커진다.

### 타이밍 문제 (시퀀스)

- **T1. HMI 시퀀스는 전부 개루프 고정 대기.** `_run_sequence`의 `MOVE_WAIT=15s`,
  `RAIL_WAIT=5s`(`hmi_node.py:347-350`, `403-416`)는 **실제 완료 신호와 무관하게 sleep**만 한다.
  반면 오케스트레이터/`robot_control_node`는 done 토픽·`_wait_motion_done` 피드백 기반이라
  두 경로의 타이밍이 어긋나 서로의 동작 중간에 개입한다.
- **T2. 검사 트리거 이중화.** HMI 시퀀스가 P1~P4로 "안착"을 직접 하고 `_inspect_result_event`를 기다리는데
  (`hmi_node.py:441-459`), 동시에 오케스트레이터도 place_chamber→INSPECTING_*로 안착·턴테이블 회전을 구동한다.
  검사 대상 안착·턴테이블 회전 주체가 둘이라 검사 타이밍이 비결정적.
- **T3. `/motor/rail_done` 다중 구독.** 오케스트레이터가 동일 토픽에 콜백 2개를 단다
  (`main_orchestrator_node.py:176-177`). 상태 가드로 분리돼 치명적이진 않으나,
  ESP32 `/motor/rail_done`과 로봇 `/robot/rail_done`이 혼재해 추적이 어렵다(정리 필요).
- **T4. 검사 캡처 시점.** `inspect_node`는 `/motor/turntable_done` 수신 즉시 `_latest_frame`을 캡처
  (`inspect_node.py:181-210`)하므로, 기구 정지·LED 안정화 전 프레임이 잡힐 수 있다.
  (LED 3초 안정화는 회전 사이클 진입 전 1회뿐: `main_orchestrator_node.py:479-489`.)

---

## 3. 해결 계획 (우선순위 순)

### P0 — 이중 제어 제거 (C1) : 단일 제어 권한 확립  ※ 최우선·안전 직결

**결정 확정(2026-07-01):** 앞으로 ACT를 master에서 쓰므로 **오케스트레이터 + `robot_control_node` 경로(경로 A)를
단일 제어자로 채택**한다. HMI 자체 시퀀스(경로 B)는 **비활성화/제거**한다.

구체 작업:
1. `hmi_node.send_command`에서 `_start_sequence()`/`_stop_sequence()` 호출 제거
   (`hmi_node.py:247-250`). HMI는 `/hmi/command`만 발행하고 모터를 **직접 만지지 않는다.**
2. `hmi_node._run_sequence` 및 관련 멤버(`_seq_thread`, `_seq_stop_event`,
   `_inspect_result_event`, `_inspect_passed`) 제거 또는 데드코드 격리.
   → `/dev/ttyFollower`를 여는 주체를 `robot_control_node` **하나로** 만든다.
3. (보류된 대안) 데모용 HMI 직접 시퀀스는 채택하지 않음. 동시 사용은 어떤 경우에도 금지.
   추후 HMI 직접 시퀀스가 필요해지면 그때는 반대로 오케스트레이터/`robot_control_node`의
   팔 제어를 비활성화해 **둘 중 하나만** 포트를 열게 한다.

검증: START 후 `lsof /dev/ttyFollower`(또는 컨테이너 내 동등 수단)로 **연 프로세스가 정확히 하나**인지 확인.

### P1 — ACT 안전캡·관절 보호 복구 (C2·C3·C4)  ※ ACT 재가동 전 필수

1. **상대이동 안전캡 활성화 (C2).** `OmxFollowerConfig(max_relative_target=...)`에
   보수적 값을 준다. 관절별 raw step 기준(예: 한 스텝 최대 ~50~100 step ≈ 4~9°)으로 시작.
   `_init_follower`(`robot_control_node.py:312-327`)에서 config 생성 시 지정.
   → ACT 이상치 한 방으로 돌진하는 것을 하드웨어 직전 단계에서 차단.
2. **관절 소프트 리밋 실효화 (C3).** `_clip_shoulder_lift`를 폐기하고,
   **전 관절 raw 안전 범위 클램프**로 대체하거나, 최소한 ACT 출력을 정규화 범위
   `[-100,100]`(또는 DEGREES 관절은 학습 범위)로 **명시적으로 clip**한 뒤 `send_action`에 넘긴다.
3. **다회전 관절 위치 제한 (C4).** `shoulder_pan`·`wrist_roll`에도 Min/Max_Position_Limit를 설정하거나,
   ACT 운용 동안 이 두 관절을 POSITION 모드(±한도)로 운용하는 방안을 검토.
   최소한 소프트웨어 클램프(2번)로 두 관절의 목표를 안전 회전 범위로 제한.

### P2 — ACT 입력 파이프라인 학습 일치 검증 (C5)

1. `act_model_path`의 모델 메타(`config.json`, 정규화 통계)에서 **입력 이미지 해상도/정규화/관측 키**를 확인.
2. `robot_control_node.py:654-657,678-681`의 resize·`/255.0`·키 이름을 학습과 일치시킨다.
3. 첫 추론 워밍업: 본격 전송 전 N스텝은 현재 자세 유지/저속으로 시작해 큐를 채운다.

### P3 — 프로파일 일관화 (C6)

1. ACT 진입 시 `send_action` 직전에 의도한 ACT용 Profile_Velocity/Acceleration를 **명시적으로 1회 설정**.
2. 각 동작 모드(홈/시퀀스/ACT) 진입 시 프로파일을 항상 재설정해 누수 제거.

### P4 — 타이밍/검사 정합 (T1~T4)  ※ P0 적용 후 자연 해소 + 보강

1. P0로 HMI 개루프 시퀀스가 사라지면 T1·T2는 구조적으로 해소. 단일 경로(오케스트레이터)의
   피드백 기반(done 토픽·`_wait_motion_done`) 타이밍만 남긴다.
2. `inspect_node` 캡처(T4): `/motor/turntable_done` 수신 후 **정지·LED 안정화 지연(예: 0.3~0.5s)**을 두고 캡처,
   또는 각 각도 캡처 직전에도 짧은 안정화 대기를 추가.
3. `/motor/rail_done` 구독 정리(T3): 상태별 단일 콜백으로 통합.

---

## 4. 안전 점진 검증 절차 (하드웨어 파손 방지)

각 단계는 이전 단계 통과 후에만 진행한다.

1. **포트 단일화 확인:** P0 적용 → START 시 `/dev/ttyFollower` 개방 프로세스 1개 확인. (모터 토크 OFF 상태)
2. **수동 저속 점검:** 토크 ON, Profile_Velocity를 최저(1~2)로 고정하고 홈 복귀만 수행 → 정상 정지 확인.
3. **시퀀스 단독(ACT 미사용):** `use_act=false`로 오케스트레이터 전체 사이클 1회 → P1~P6 안무가
   **단일 경로로만** 매끄럽게 진행되는지, 폭주 없는지 확인.
4. **ACT 안전캡 검증:** `use_act=true`, **단 `max_relative_target` 적용 + 소프트 클램프 ON**,
   그리고 **물리적으로 팔 주변 여유 공간 확보 / 즉시 ESTOP 대기** 상태에서 첫 추론.
   첫 스텝이 돌진하지 않고 캡 한도 내로만 움직이는지 로그로 확인.
5. **속도 점증:** 안전 확인 후 Profile_Velocity·캡 한도를 단계적으로 상향.

---

## 5. 작업 체크리스트

- [ ] (P0) `hmi_node.send_command`의 시퀀스 직접 호출 제거 — 단일 제어자 확립
- [ ] (P0) `hmi_node._run_sequence`/raw dynamixel 포트 개방 제거 또는 격리
- [ ] (P0) START 후 `/dev/ttyFollower` 단일 개방 검증
- [ ] (P1) `OmxFollowerConfig.max_relative_target` 보수값 설정
- [ ] (P1) `_clip_shoulder_lift` 폐기 → 전 관절 안전 클램프로 대체
- [ ] (P1) `shoulder_pan`·`wrist_roll` 위치 제한/모드 검토 적용
- [ ] (P2) ACT 모델 메타 대조 — 이미지 해상도/정규화/관측 키 일치
- [ ] (P2) 첫 추론 워밍업 로직 추가
- [ ] (P3) ACT/시퀀스/홈 모드별 프로파일 명시 설정 — 누수 제거
- [ ] (P4) inspect 캡처 안정화 지연 추가
- [ ] (P4) `/motor/rail_done` 구독·done 토픽 정리
- [ ] (검증) 4장 점진 절차 1→5 순차 통과

---

## 부록 — 근거 코드 위치 요약

| 원인 | 파일:라인 |
|------|-----------|
| C1 이중 제어 (HMI START 분기) | `hmi_node.py:241-250` |
| C1 HMI raw 포트 직접 개방 | `hmi_node.py:328,331,367-374,428-486` |
| C1 robot_control 포트 상시 개방 | `robot_control_node.py:233-234,246-248` |
| C1 오케스트레이터 START 사이클 | `main_orchestrator_node.py:207-222,422-448` |
| C2 안전캡 비활성 (config 기본 None) | `config_omx_follower.py:33` |
| C2 캡 적용 조건부 | `omx_follower.py:257-260` |
| C3 무효 클리핑 | `robot_control_node.py:905-938` (특히 919,929) |
| C4 다회전 관절·제한 부재 | `omx_follower.py:56,60,170-178,197-204` |
| C5 ACT 이미지 전처리/관측 | `robot_control_node.py:654-657,678-681` |
| C6 프로파일 전역 누수 | `robot_control_node.py:943-953,193-194` |
| T1 개루프 고정 대기 | `hmi_node.py:347-350,403-416` |
| T4 캡처 즉시 트리거 | `inspect_node.py:181-210` |
</content>
</invoke>
