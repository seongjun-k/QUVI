# QUVI ROS 아키텍처 평가서

작성일: 2026-06-16
대상: `/home/ksj/QUVI`
참조 문서: `/home/ksj/.gemini/antigravity-cli/brain/ea019f7b-715a-42cf-bd72-77c348327d71/system_architecture_spec.md`

## 요약

QUVI 프로젝트는 ROS 2 Jazzy 기반으로 `quvi_msgs`, `quvi_yolo`, `quvi_inspect`, `quvi_robot_control`, `quvi_hmi`, `quvi_bringup`가 역할별로 분리되어 있다. 전체 구조는 시연용 통합 프로토타입으로는 방향이 잡혀 있지만, 실제 장비를 안정적으로 자동 운전하기에는 상태 피드백, 비상정지 전파, 하드웨어 완료 신호, 모델/파라미터 정합성에서 보완이 필요하다.

가장 중요한 문제는 다음 네 가지다.

- HMI/ROS E-STOP이 ESP32와 Dynamixel까지 전파되지 않는다.
- 턴테이블 명령 토픽을 검사 노드가 상태 피드백처럼 사용한다.
- 기본 launch 설정에서 ACT가 꺼져 있어 자율 파지 시퀀스가 실패할 가능성이 높다.
- YOLO 기본 모델과 기본 target class가 맞지 않아 탐지가 0개가 될 가능성이 높다.

## 주요 발견 사항

### 1. E-STOP 전파가 불완전함

파일:

- `src/quvi_robot_control/quvi_robot_control/main_orchestrator_node.py`
- `src/quvi_hmi/quvi_hmi/hmi_node.py`
- `firmware/quvi_esp32_firmware/quvi_esp32_firmware.ino`

HMI에서 `ESTOP` 명령을 보내면 오케스트레이터는 `ERROR` 상태로만 전이한다. 하지만 이 명령이 ESP32 레일/턴테이블 정지, Dynamixel torque off, 현재 실행 중인 robot_control_node 동작 중단으로 이어지지 않는다.

펌웨어에는 GPIO 17 기반 물리 E-STOP 인터럽트가 있지만, ROS 토픽 기반 소프트웨어 E-STOP subscriber는 없다. 실제 장비에서는 UI E-STOP을 눌러도 물리 구동계가 계속 움직일 수 있는 구조다.

개선 방향:

- `/system/estop` 또는 `/motor/estop` 같은 전역 E-STOP 토픽 추가
- `main_orchestrator_node`, `robot_control_node`, ESP32 펌웨어가 모두 해당 토픽을 구독
- `robot_control_node`는 E-STOP 수신 시 Dynamixel torque off 또는 안전 자세/정지 수행
- ESP32는 E-STOP 수신 시 레일/턴테이블 step generation 중단 및 LED relay OFF
- 복구는 명시적 reset 절차로만 허용

### 2. 턴테이블 명령과 상태 피드백이 분리되어 있지 않음

파일:

- `src/quvi_robot_control/quvi_robot_control/main_orchestrator_node.py`
- `src/quvi_inspect/quvi_inspect/inspect_node.py`
- `firmware/quvi_esp32_firmware/quvi_esp32_firmware.ino`

오케스트레이터는 `/motor/turntable`에 `0`, `90`, `180`, `270`도 명령을 발행한다. 검사 노드는 같은 `/motor/turntable`을 구독해서 현재 각도로 간주하고, `capture_delay_sec` 이후 이미지를 캡처한다.

이 구조에서는 실제 턴테이블이 목표 각도에 도착했는지 확인하지 않는다. 회전 시간이 지연되거나 미끄러짐, 스텝 누락, micro-ROS 지연이 발생하면 잘못된 각도에서 검사 이미지를 찍게 된다.

개선 방향:

- 명령 토픽과 상태 토픽 분리
- 예: `/motor/turntable_cmd`, `/motor/turntable_done`, `/motor/turntable_state`
- ESP32가 회전 완료 후 현재 각도와 done 상태를 발행
- 검사 노드는 명령 토픽이 아니라 완료 피드백을 기준으로 캡처

### 3. 기본 launch 설정에서 ACT 파지가 실패할 수 있음

파일:

- `src/quvi_bringup/launch/full_system.launch.py`
- `src/quvi_robot_control/quvi_robot_control/robot_control_node.py`

`full_system.launch.py`의 기본값은 `use_act=false`다. 반면 `robot_control_node`는 `/robot/grasp_command`를 받으면 ACT 파지를 실행한다. ACT가 비활성화되어 있으면 `_act_ready`가 false이고, 파지 완료 신호가 발행되지 않는다.

결과적으로 HMI에서 START를 누르면 오케스트레이터가 `GRASPING_WAIT`에서 timeout으로 빠질 가능성이 높다.

개선 방향:

- 자율 모드 기본값에서는 `use_act=true`로 맞추거나
- `use_act=false`일 때는 시뮬레이션 완료 신호를 명확히 발행하거나
- 오케스트레이터가 시작 전에 ACT readiness를 확인하고 사용자에게 상태를 노출

### 4. YOLO 기본 모델과 target class가 맞지 않음

파일:

- `src/quvi_yolo/quvi_yolo/yolo_node.py`
- `src/quvi_bringup/launch/vision_pipeline.launch.py`

YOLO 노드는 `model_path`가 비어 있으면 `yolov8n.pt`를 로드한다. 동시에 기본 target class는 `print_object`다. COCO 기반 `yolov8n.pt`에는 `print_object` 클래스가 없으므로, 별도 학습 모델을 지정하지 않으면 탐지 결과가 항상 0개일 가능성이 높다.

개선 방향:

- 프로젝트 전용 모델 경로를 launch argument로 명시
- 기본 `target_classes`를 모델 메타데이터와 일치시킴
- 모델 파일이 없거나 클래스가 불일치하면 노드 시작 시 fatal error 또는 명확한 warning 출력

### 5. 탐지 좌표가 실제 파지 제어에 사용되지 않음

파일:

- `src/quvi_robot_control/quvi_robot_control/main_orchestrator_node.py`
- `src/quvi_robot_control/quvi_robot_control/robot_control_node.py`

오케스트레이터는 `DetectedObject.x/y`를 mm 좌표로 변환해서 `GraspGoal`에 담는다. 하지만 `robot_control_node`는 이 좌표를 실제 제어에 사용하지 않고, ACT visuomotor 추론의 참고/로깅 값으로만 처리한다.

이 상태에서는 다중 객체를 순차 처리하는 FSM이 의도대로 동작한다는 보장이 약하다. ACT 정책이 어떤 객체를 집을지는 이미지와 학습 데이터에 의존하며, `object_index`나 목표 좌표가 행동에 반영되지 않는다.

개선 방향:

- 좌표 기반 pre-positioning 단계 추가
- ACT 입력에 target 좌표 또는 crop/attention 정보를 포함
- 다중 객체 처리 시 각 객체별 ROI를 명확히 지정
- `GraspGoal`이 실제 제어 계약인지, 단순 로깅 계약인지 문서화

### 6. 패키지 의존성 선언이 누락됨

파일:

- `src/quvi_yolo/package.xml`
- `src/quvi_inspect/package.xml`
- `src/quvi_yolo/quvi_yolo/yolo_node.py`
- `src/quvi_inspect/quvi_inspect/inspect_node.py`

`quvi_yolo`와 `quvi_inspect`는 `quvi_robot_control.utils`를 직접 import한다. 하지만 두 패키지의 `package.xml`에는 `quvi_robot_control` 의존성이 선언되어 있지 않다.

클린 빌드, 패키지 단독 실행, 배포 환경에서 import 순서 또는 런타임 의존성 문제가 발생할 수 있다.

개선 방향:

- 단기: `quvi_yolo`, `quvi_inspect`의 `package.xml`에 `quvi_robot_control` 의존성 추가
- 권장: 공용 유틸리티를 `quvi_common` 같은 별도 패키지로 분리

### 7. 레일 위치 파라미터가 실장비 스케일과 맞지 않아 보임

파일:

- `firmware/quvi_esp32_firmware/Config.h`
- `src/quvi_robot_control/quvi_robot_control/robot_control_node.py`

펌웨어 기준으로 레일은 420mm가 33600 steps다. 즉 80 steps/mm다. 그런데 ROS 기본 레일 위치는 `1000`, `1700`, `2400` steps로 설정되어 있다. 이는 약 12.5mm, 21.25mm, 30mm에 해당한다.

BED, INSPECT, PASS, FAIL 위치로는 placeholder에 가까워 보이며, 실제 장비에서는 캘리브레이션 값으로 교체해야 한다.

개선 방향:

- 레일 위치를 YAML 설정으로 분리
- mm 단위 파라미터를 받고 내부에서 steps로 변환
- 초기 homing 완료 후 각 zone 위치 캘리브레이션 절차 추가

### 8. micro-ROS 연결 상태와 모터 상태가 ROS graph에 충분히 노출되지 않음

현재 ESP32 펌웨어는 `/motor/rail_done`만 발행한다. 레일 현재 위치, 목표 위치, moving 상태, homing 상태, emergency 상태, micro-ROS agent 연결 상태는 ROS 측에서 충분히 관측하기 어렵다.

개선 방향:

- `/motor/status` 메시지 추가
- 필드 예: `rail_position`, `rail_target`, `rail_moving`, `turntable_angle`, `turntable_moving`, `homed`, `estop`, `error_code`
- HMI와 오케스트레이터는 완료 이벤트뿐 아니라 상태 메시지를 기준으로 판단

## 테스트 상태

실행 결과:

```bash
python3 -m pytest -q
```

루트 전체 테스트는 `lerobot` 서브모듈까지 수집하다가 `draccus` 미설치로 중단된다. 이는 QUVI 자체 테스트 실패라기보다 pytest 수집 범위 설정 문제다.

```bash
python3 -m pytest -q tests
```

결과는 `19 passed, 1 error`였다. 에러 원인은 `tests/test_alignment.py`의 `test_normal_rotation(angle_deg)`가 pytest fixture를 요구하는 테스트처럼 수집되기 때문이다. 해당 파일은 독립 실행 스크립트 형태로 작성되어 있어 pytest 규약에 맞게 정리해야 한다.

개선 방향:

- `pytest.ini` 또는 `pyproject.toml`에 `testpaths = tests` 지정
- `lerobot` 서브모듈 수집 제외
- `test_alignment.py`의 반환형 테스트 함수를 assert 기반 pytest 함수로 변환
- ROS 토픽 계약 테스트 추가

## 권장 개선 우선순위

1. E-STOP 전파 설계 및 구현
2. 턴테이블 명령/완료/상태 토픽 분리
3. ACT readiness와 launch 기본값 정합성 수정
4. YOLO 모델 경로와 class 설정 검증 추가
5. `quvi_common` 패키지 도입 또는 의존성 선언 보강
6. 레일 zone 위치 캘리브레이션 파라미터화
7. `/motor/status` 상태 메시지 추가
8. pytest 수집 범위와 테스트 규약 정리

## 결론

현재 QUVI는 기능별 패키지 분리, FSM 기반 오케스트레이션, HMI 통합, ESP32 펌웨어 설계가 갖춰진 시연형 시스템이다. 다만 실제 로봇 셀로 운용하려면 "명령을 보냈다"와 "하드웨어가 완료했다"를 명확히 분리해야 한다. 특히 E-STOP, 턴테이블 완료 피드백, ACT/YOLO readiness는 안전성과 성공률에 직접 영향을 주므로 우선적으로 보완해야 한다.
