# 텔레오퍼레이션(Teleop) 오작동 및 모터 폭주 이슈 해결 보고서

본 문서는 리더-팔로워 텔레오퍼레이션(실시간 조종) 구동 시 발생했던 관절 각도 불일치, 동작 정지, 모터 폭주 및 배선 꼬임 이슈의 분석 과정과 최종 해결 방안을 기록합니다.

---

## 1. 이슈 요약 및 증상

| 번호 | 대상 모터 | 증상 | 영향 |
| :--- | :--- | :--- | :--- |
| **이슈 1** | **2번 & 12번**<br>(shoulder_lift) | 조종기(리더)와 로봇(팔로워)의 각도가 1:1로 맞지 않고 둔하게 움직임. | 텔레옵 조작 시 원점이 맞지 않고 조작감이 매우 불량함. |
| **이슈 2** | **5번 & 15번**<br>(wrist_roll) | 조종기를 돌려도 로봇 손목 회전 모터가 구동되지 않고 먹통이 됨. | 그리퍼 회전 조종 불가. |
| **이슈 3** | **15번**<br>(wrist_roll) | 텔레옵 활성화 시 15번 모터가 급격히 1.5바퀴 이상 폭주 회전하여 그리퍼 배선이 감기고 모터가 과부하로 다운됨. | 하드웨어 셧다운 및 배선 손상 위험 발생. |

---

## 2. 근본 원인 분석 (Root Cause Analysis)

### 2.1. 12번 모터 각도 불일치 (이슈 1)
* **원인**: LeRobot의 `[-100, 100]` 값 정규화(Normalization) 과정에서 양쪽 모터의 가동 범위 설정이 불일치했습니다.
  * 로봇(팔로워 12번): 안전 가드를 위해 `[1024, 4095]` (최소 90도 제한)로 범위가 한정됨.
  * 조종기(리더 2번): 가동 범위가 `[0, 4095]`로 열려 있음.
* **결과**: 가동 범위 크기가 달라 스케일링 비율(기울기)과 오프셋이 달라지면서, 조종기를 움직이는 각도의 약 75%만 로봇에 전달되어 1:1 매핑이 깨졌습니다.

### 2.2. 15번 모터 구동 불능 및 먹통 현상 (이슈 2)
* **원인**: 동작 모드(Operating Mode) 불일치 및 가동 한계값 제한 때문이었습니다.
  * 로봇(팔로워 15번)은 단일 회전 전용 **Position Control Mode (3)**로 구동 중이었으며, `Min/Max_Position_Limit` 또한 `[0, 4095]`로 고정되어 있었습니다.
  * 하지만 리더 5번은 전류 제어 모드(0)로 동작하여 다회전(Multi-turn) 틱이 무제한 누적되는 상태였습니다.
* **결과**: 리더의 틱 누적치가 `4095` 한계를 초과하면, 팔로워가 처리할 수 없는 값(Out of Range)이 되어 명령을 무시하고 정지했습니다.

### 2.3. 15번 모터 텔레옵 시작 시 1.5바퀴 폭주 및 배선 꼬임 (이슈 3)
* **원인 1 (LeRobot 32-bit Signed 디코딩 버그)**:  
  LeRobot Dynamixel 버스 라이브러리의 `tables.py` 내에 `Present_Position`과 `Goal_Position` 레지스터가 부호 있는 정수(Two's Complement)로 매핑되어 있지 않았습니다. 
  그 결과, 다회전 상태에서 리더 관절이 음수 각도로 진입하여 레지스터가 음수(예: `-1`)가 되었을 때, 이를 부호 없는 32비트 정수 `4294967295`로 오해독하여 캘리브레이션 Homing Offset에 엄청나게 큰 값이 쓰여 모터가 비정상 제어되었습니다.
* **원인 2 (초기 정렬 동기화 차이)**:  
  동작 모드를 다회전으로 변경한 직후, 초기 기동 시 리더의 다회전 누적값(예: 10330 틱)과 팔로워의 초기 기동값(예: 4047 틱) 사이에 1.5바퀴(약 550도) 가량의 틱 격차가 그대로 전달되어, 정렬 보간(Smooth Alignment) 시 팔로워가 격차를 따라잡기 위해 제자리에서 급회전하면서 배선이 꼬였습니다.

---

## 3. 해결 방안 및 조치 내역

### 3.1. Homing/Normalization 범위 동기화
* **수정 파일**:  
  * [quvi_leader.json](file:///home/ksj/QUVI/lerobot/src/lerobot/teleoperators/omx_leader/calibration/quvi_leader.json)  
  * [omx_leader_arm.json](file:///home/ksj/QUVI/lerobot/src/lerobot/teleoperators/omx_leader/calibration/omx_leader_arm.json)
* **조치 내용**:  
  리더 2번 모터의 `range_min`을 팔로워와 동일한 `1024`로 수정하여 각도 매핑을 1:1로 일치시켰습니다. 조종기를 1024 미만으로 내려도 안전하게 1024로 자동 클램핑되어 팔로워 로봇을 보호합니다.

### 3.2. 동작 모드 다회전 모드로 변경 및 리밋 해제
* **수정 파일**: [omx_follower.py](file:///home/ksj/QUVI/lerobot/src/lerobot/robots/omx_follower/omx_follower.py)
* **조치 내용**:  
  `wrist_roll` (15번 모터)의 `Operating_Mode`를 `3` (POSITION)에서 **`4` (EXTENDED_POSITION)**로 수정하여 다회전 입력을 수용하도록 했고, `[0, 4095]` 범위로 묶여 있던 가동 범위 제한 코드를 제거하여 무한 회전이 가능하도록 했습니다.

### 3.3. 32-bit Signed 디코딩 버그 패치
* **수정 파일**: [tables.py](file:///home/ksj/QUVI/lerobot/src/lerobot/motors/dynamixel/tables.py)
* **조치 내용**:  
  `X_SERIES_ENCODINGS_TABLE`에 `Goal_Position`과 `Present_Position` 레지스터 항목을 추가 등록하여 4바퀴 이상의 음수 각도 구역에서도 부호 있는 32비트 정수(Two's Complement)로 정확히 디코딩(`decode_twos_complement`)되도록 라이브러리를 보완했습니다.

### 3.4. 최단경로 동기화 및 누적회전 보정 오프셋 적용
* **수정 파일**: [robot_control_node.py](file:///home/ksj/QUVI/src/quvi_robot_control/quvi_robot_control/robot_control_node.py)
* **조치 내용**:  
  * 텔레옵 기동 전 정렬 시점(`_start_teleop`)에 다회전 조인트(`shoulder_pan`, `wrist_roll`)의 각도 차이를 최단경로(`[-180도, +180도]`)로 랩핑(`wrapping`)하고, 초과된 바퀴 수는 보정 오프셋(`_teleop_offsets`)에 저장합니다.
  * 실시간 조종 루프(`_teleop_loop`)에서도 리더가 보내는 각도 값에서 이 오프셋을 실시간 감산하여, 팔로워가 배선 꼬임 한계를 넘어 팽글그르르 도는 불필요한 과회전 현상을 완전히 차단하고 1:1 조종 각도를 유지시킵니다.

---

## 4. 최종 검증 결과
* **빌드 결과**: Docker 내부의 ROS 2 환경에서 `colcon build --symlink-install` 명령어로 전체 패키지 빌드가 오류 없이 통과하는 것을 검증하였습니다.
* **실시간 조작 테스트**: 리더 조종 시 12번 및 15번 모터 모두 폭주나 통신 에러, 배선 꼬임 없이 1:1로 일치하여 정교하게 동작하는 것이 확인되었습니다.

---
**기록일**: 2026-06-22  
**담당 엔지니어**: Antigravity (Advanced Agentic Coding Team, Google DeepMind)
