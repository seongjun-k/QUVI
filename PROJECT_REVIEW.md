# QUVI 프로젝트 코드 리뷰 리포트

작성일: 2026-06-08  
대상: `/home/ksj/QUVI`

## 요약

QUVI는 ROS 2 Jazzy 기반으로 3D 프린터 출력물을 탐지, 파지, 검사, 분류하는 자동화 시스템입니다. 패키지 분리는 비교적 명확하며, `quvi_msgs`, `quvi_yolo`, `quvi_inspect`, `quvi_robot_control`, `quvi_hmi`, `quvi_bringup`가 각각의 역할을 가지고 있습니다.

현재 상태는 시연용 통합 프로토타입으로는 구조가 잘 잡혀 있지만, 실제 장비를 안정적으로 운용하기에는 상태 동기화, 하드웨어 완료 피드백, 비상정지 전파, 파라미터 적용, 의존성 선언, 저장소 정리가 부족합니다.

## 주요 발견 사항

### 1. FSM 무한 대기 가능성

파일: `src/quvi_robot_control/quvi_robot_control/main_orchestrator_node.py`

오케스트레이터의 여러 상태가 완료 플래그만 기다리고 timeout, retry, error transition을 갖고 있지 않습니다.

문제가 되는 상태:

- `DETECTING_WAIT`
- `GRASPING_WAIT`
- `INSPECTING_WAIT_RESULT`
- `SORTING_WAIT_RAIL`
- `RELEASING_WAIT`
- `HOMING_WAIT`

카메라, YOLO, 로봇 제어, 검사 노드, ESP32 중 하나라도 응답하지 않으면 FSM이 멈출 수 있습니다.

개선 방향:

- 상태별 timeout 파라미터 추가
- timeout 발생 시 `ERROR` 또는 복구 상태로 전이
- 실패 원인을 `/hmi/status.error_message`에 기록
- 재시도 가능한 단계와 즉시 정지해야 하는 단계를 구분

### 2. 레일 완료 신호가 실제 하드웨어 완료가 아님

파일: `src/quvi_robot_control/quvi_robot_control/robot_control_node.py`

`_execute_rail_move()`는 `/motor/rail` 명령을 발행한 뒤 `time.sleep(0.2)` 후 바로 `/robot/rail_done`을 발행합니다.

현재 문제:

- ESP32가 실제 목표 위치에 도착했는지 확인하지 않음
- 레일 이동 중인데 오케스트레이터가 다음 동작으로 넘어갈 수 있음
- 실물 장비에서는 충돌이나 파지 실패로 이어질 수 있음

개선 방향:

- ESP32 펌웨어에서 `/motor/rail_done` 또는 `/motor/status` 피드백 발행
- `robot_control_node`는 ESP32 피드백을 받은 뒤 `/robot/rail_done` 발행
- 목표 위치, 현재 위치, moving/error 상태를 별도 메시지로 관리

### 3. 턴테이블 명령 토픽을 현재 각도 피드백처럼 사용

파일: `src/quvi_inspect/quvi_inspect/inspect_node.py`

검사 노드는 `/motor/turntable`을 구독해서 현재 각도로 간주합니다. 하지만 이 토픽은 오케스트레이터가 턴테이블에 보내는 명령 토픽입니다.

현재 문제:

- 실제 회전 완료 전 캡처 타이머가 시작될 수 있음
- 0도, 90도, 180도, 270도 이미지가 실제 각도와 맞지 않을 수 있음
- 검사 결과 신뢰도가 크게 떨어질 수 있음

개선 방향:

- 명령 토픽과 상태 토픽 분리
- 예: `/motor/turntable_command`, `/motor/turntable_state`
- ESP32가 회전 완료 후 현재 각도와 done 상태 발행
- 검사 노드는 완료 피드백을 기준으로 캡처 수행

### 4. 비상정지 전파가 부족함

파일:

- `src/quvi_robot_control/quvi_robot_control/main_orchestrator_node.py`
- `src/quvi_hmi/quvi_hmi/hmi_node.py`
- `firmware/quvi_esp32_firmware/quvi_esp32_firmware.ino`

HMI에서 `ESTOP`을 보내면 오케스트레이터 FSM은 `ERROR`로 바뀌지만, Dynamixel torque off, 레일 정지, 턴테이블 정지, ESP32 emergency stop 명령으로 연결되어 있지 않습니다.

현재 문제:

- 소프트웨어 상태만 멈추고 하드웨어가 계속 움직일 수 있음
- 실제 장비 안전 요구사항을 만족하지 못함

개선 방향:

- `/system/estop` 같은 전역 토픽 추가
- `robot_control_node`에서 ESTOP 수신 시 Dynamixel torque disable
- ESP32에서 ESTOP 수신 시 모터 driver disable
- HMI, 오케스트레이터, 로봇, 펌웨어가 같은 ESTOP 상태를 공유
- ESTOP 해제는 자동 복구가 아니라 명시적 reset 절차로 제한

### 5. YOLO 클래스 필터 로직 오류

파일: `src/quvi_yolo/quvi_yolo/yolo_node.py`

기본값 `target_classes: ["print_object"]`일 때 모든 클래스가 통과하는 조건이 있습니다.

현재 문제:

- COCO 기본 모델 `yolov8n.pt` 사용 시 출력물이 아닌 객체도 탐지 결과로 발행될 수 있음
- 잘못된 객체를 파지 대상으로 선택할 수 있음

개선 방향:

- `target_classes`가 비어 있으면 전체 허용
- 값이 있으면 해당 class name만 허용
- 커스텀 모델 사용 시 class name과 YAML 설정을 일치시키기

권장 로직:

```python
if self._target_classes and cls_name not in self._target_classes:
    continue
```

### 6. 카메라 픽셀 좌표를 실제 로봇 좌표로 단순 변환

파일: `src/quvi_robot_control/quvi_robot_control/main_orchestrator_node.py`

현재 오케스트레이터는 YOLO 중심 좌표를 `px_to_mm_x`, `px_to_mm_y`, `offset_x`, `offset_y`로만 변환합니다.

현재 문제:

- 카메라 왜곡 미보정
- perspective 보정 없음
- 베드 평면 homography 없음
- hand-eye calibration 없음
- 실제 파지 좌표 오차가 커질 수 있음

개선 방향:

- 카메라 intrinsic calibration 수행
- 베드 평면 기준 homography 계산
- 픽셀 좌표를 베드 좌표계로 변환하는 별도 calibration 모듈 추가
- calibration 결과를 YAML로 저장하고 런치에서 로드

### 7. Launch argument와 YAML 설정이 실제로 연결되지 않음

파일: `src/quvi_bringup/launch/vision_pipeline.launch.py`

`yolo_config`, `inspect_config` launch argument가 선언되어 있지만 실제 `Node(parameters=...)`에는 사용되지 않습니다. 대신 런치 파일 내부에 파라미터가 하드코딩되어 있습니다.

현재 문제:

- `src/quvi_yolo/config/yolo_params.yaml`을 수정해도 런치 실행에는 반영되지 않을 수 있음
- 설정 변경 위치가 분산됨
- 실험 재현성이 떨어짐

개선 방향:

- 기본 YAML 경로를 `get_package_share_directory()`로 찾기
- launch argument가 비어 있으면 기본 YAML 사용
- 하드코딩 파라미터 최소화

### 8. Python 의존성 선언이 부족함

파일:

- `src/quvi_yolo/package.xml`
- `src/quvi_yolo/setup.py`
- `src/quvi_inspect/package.xml`
- `src/quvi_hmi/package.xml`
- `src/quvi_robot_control/package.xml`

Dockerfile에는 여러 Python 패키지가 설치되지만, ROS 패키지 메타데이터에는 런타임 의존성이 충분히 선언되어 있지 않습니다.

누락 또는 보강이 필요한 의존성 예:

- `ultralytics`
- `torch`
- `numpy`
- `opencv-python` 또는 시스템 `python3-opencv`
- `scikit-image`
- `flask`
- `flask-socketio`
- `trimesh`
- `dynamixel_sdk`
- `pyserial`

개선 방향:

- Python 의존성은 `requirements.txt` 또는 `pyproject.toml`로 통합
- ROS 패키지별 `package.xml`에는 실행에 필요한 시스템/ROS 의존성 명시
- Dockerfile은 이 의존성 파일을 기준으로 설치

### 9. HMI 제어 API 인증 없음

파일: `src/quvi_hmi/quvi_hmi/hmi_node.py`

HMI는 `0.0.0.0`으로 실행되고 `/api/command/<cmd>`에서 `start`, `stop`, `estop`, `reset` 명령을 받을 수 있습니다.

현재 문제:

- 같은 네트워크에서 누구나 장비 제어 API를 호출할 수 있음
- 실물 장비 제어 시스템으로는 위험함

개선 방향:

- 기본 host를 `127.0.0.1`로 제한하거나 launch argument로 명확히 선택
- 제어 API에 토큰 인증 추가
- HMI 읽기 API와 제어 API 권한 분리
- ESTOP은 별도 보호 로직 적용

### 10. 저장소에 빌드 산출물과 임시 파일이 섞여 있음

현재 저장소에는 다음과 같은 산출물이 보입니다.

- `build/`
- `install/`
- `docker/build/`
- `docker/install/`
- `docker/log/`
- `docker/yolov8n.pt`
- `docker/CACHED`
- `docker/ERROR`
- `docker/[internal]`
- `docker/reading`
- `docker/resolve`
- `docker/transferring`

`.gitignore`는 존재하지만, 이미 생성된 산출물과 Docker 출력 파일이 작업트리에 섞여 있습니다.

개선 방향:

- 빌드 산출물은 저장소에서 제거
- 모델 파일은 `data/models/` 또는 외부 다운로드 절차로 관리
- Docker build context를 정리
- `.gitignore`에 Docker 임시 출력과 모델 경로 보강

### 11. README와 실제 구현 간 차이

파일: `README.md`

README는 YOLO 노드가 "중심 3D 파지 좌표 계산 및 발행"을 한다고 설명하지만, 실제 메시지는 2D 픽셀 중심 좌표입니다.

현재 문제:

- 문서와 구현의 기대치가 다름
- 신규 개발자가 시스템 기능을 오해할 수 있음

개선 방향:

- 현재 구현 기준으로 README 수정
- 향후 목표 기능과 현재 구현 기능을 구분해서 작성
- calibration 미완료 상태를 명시

### 12. 검사 알고리즘의 기준 이미지 실패 처리

파일: `src/quvi_inspect/quvi_inspect/inspect_node.py`

기준 이미지가 없으면 CAD 비교는 실패 처리하지만, 표면 특징 분석에서는 기준 이미지가 없을 때 area ratio를 정상값 `1.0`으로 간주합니다.

현재 문제:

- 판정 근거가 일관되지 않음
- 기준 이미지 누락 상태에서 일부 지표가 정상처럼 보임

개선 방향:

- 기준 이미지 누락은 시스템 준비 실패로 별도 에러 처리
- 검사 시작 전 reference image readiness를 HMI에 표시
- 기준 이미지 없는 상태에서는 자동 검사 시작 차단

## 패키지별 평가

### `quvi_msgs`

장점:

- 패키지 간 메시지 계약을 별도 패키지로 분리한 점은 좋음
- `DetectedObject`, `ObjectArray`, `InspectionResult`, `SystemStatus`, `GraspGoal` 등 핵심 메시지가 명확함

개선점:

- 모터 상태/완료/에러 메시지가 없음
- ESTOP 관련 메시지 또는 상태 필드가 부족함
- detection 결과에 frame, timestamp, world coordinate, calibration status를 더 명확히 담을 필요가 있음

### `quvi_yolo`

장점:

- 트리거 기반 탐지라 불필요한 상시 추론을 줄일 수 있음
- debug image 발행 구조가 있음
- proximity warning 개념이 있음

개선점:

- class filtering 버그 수정 필요
- stale frame 방지 필요
- model path 검증 필요
- YAML 설정 적용 방식 정리 필요
- 실제 좌표 변환은 별도 calibration 모듈과 연동 필요

### `quvi_inspect`

장점:

- CAD 비교와 표면 특징 분석을 분리한 구조는 확장성이 좋음
- SSIM fallback 구현이 있음
- inspection history와 debug image 연동이 가능함

개선점:

- 턴테이블 완료 피드백 없이 캡처함
- 기준 이미지 readiness 체크가 약함
- ROI 관련 파라미터가 선언되어 있지만 핵심 전처리에 적극적으로 쓰이지 않음
- threshold 기반 판정은 조명 변화에 취약하므로 조명 보정/마스킹/캘리브레이션 필요

### `quvi_robot_control`

장점:

- 로봇팔, 레일, 턴테이블, ACT, 텔레옵을 한 노드에서 통합하려는 구조가 있음
- Dynamixel I/O lock을 둔 점은 적절함
- 시뮬레이션 모드와 실제 하드웨어 모드를 구분함

개선점:

- 레일/턴테이블 완료 피드백이 실제 하드웨어 상태와 연결되지 않음
- ACT 실패 시 오케스트레이터에 실패 완료/에러를 명확히 전달하지 않음
- ESTOP 시 torque off와 motor disable이 필요함
- topic command보다 action/service 기반 제어가 더 적합한 부분이 있음

### `quvi_hmi`

장점:

- ROS 상태, 카메라 스트림, 검사 결과를 한 UI에 모으는 방향이 좋음
- 수동 트리거를 FSM 상태에 따라 제한하려는 코드가 있음
- pass/fail 카운트 source of truth 중복 문제를 의식한 주석이 있음

개선점:

- 제어 API 인증이 없음
- 서버가 운영 배포용으로는 약함
- WebSocket broadcast에서 예외를 삼켜 디버깅이 어려움
- teleop 상태를 HMI 내부에서 직접 덮어써 실제 로봇 상태와 불일치할 수 있음

### `quvi_bringup`

장점:

- 전체 시스템 런치와 비전 파이프라인 런치를 분리한 점은 좋음
- 주요 장치 경로를 launch argument로 받을 수 있음

개선점:

- YAML config argument가 실제 사용되지 않음
- full system에서 vision launch로 data/config 관련 argument를 충분히 전달하지 않음
- 하드코딩 파라미터가 많음

### `firmware/quvi_esp32_firmware`

장점:

- FreeRTOS task를 motor/communication으로 분리함
- limit switch와 emergency interrupt 개념이 있음
- micro-ROS와 serial CLI 양쪽을 고려함

개선점:

- ROS 쪽으로 이동 완료/현재 위치/에러 상태를 발행하지 않음
- ESTOP 복구 정책과 상위 ROS 상태 연동이 부족함
- 명령 수신 후 soft limit 실패 시 상위 시스템에 실패를 알리지 않음

## 권장 개선 우선순위

### 1순위: 안전성과 상태 동기화

- ESTOP 전역 토픽 추가
- Dynamixel torque off 구현
- ESP32 motor disable 연동
- 레일/턴테이블 실제 완료 피드백 추가
- FSM timeout/error transition 추가

### 2순위: 런타임 신뢰성

- YOLO class filter 수정
- stale frame 방지
- YAML 설정 실제 적용
- 모델 경로 검증
- 기준 이미지 readiness 체크

### 3순위: 좌표 정확도

- 카메라 intrinsic calibration
- 베드 homography
- 픽셀 좌표에서 베드 좌표계 변환
- 변환 결과 검증용 calibration script 추가

### 4순위: 배포/협업 품질

- build/install/log 산출물 제거
- Docker context 정리
- requirements 파일 추가
- package.xml 의존성 보강
- README를 현재 구현 기준으로 수정

### 5순위: 테스트

- FSM 단위 테스트
- YOLO parsing/filter 테스트
- 검사 알고리즘 synthetic image 테스트
- HMI API 테스트
- 하드웨어 없이 실행 가능한 simulation test 추가

## 제안하는 다음 작업 목록

1. `quvi_msgs`에 모터 상태 메시지 추가
2. ESP32 펌웨어에서 현재 위치와 완료 상태 publish
3. `robot_control_node`가 실제 피드백 기반으로 done 발행하도록 수정
4. `main_orchestrator_node`에 상태별 timeout 추가
5. YOLO class filter 수정
6. `vision_pipeline.launch.py`에서 YAML config 실제 로드
7. HMI command API에 token 인증 추가
8. 저장소 산출물 정리 및 `.gitignore` 보강
9. README와 실제 구현 차이 수정
10. 주요 FSM 흐름에 대한 테스트 추가

## 결론

QUVI는 기능별 패키지 구조와 전체 시스템 흐름이 이미 잡혀 있어 확장 가능성이 있습니다. 다만 현재 구현은 "명령을 보냈다"를 "동작이 완료됐다"로 간주하는 부분이 많아 실제 하드웨어 운용에서는 위험합니다.

가장 먼저 해야 할 일은 알고리즘 고도화보다 하드웨어 상태 피드백, ESTOP 전파, FSM timeout을 넣는 것입니다. 이 세 가지가 해결되면 YOLO, 검사, ACT 성능 개선을 더 안정적으로 진행할 수 있습니다.
