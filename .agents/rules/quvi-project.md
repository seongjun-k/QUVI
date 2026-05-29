---
activation: alwaysOn
---
# QUVI 프로젝트 규칙

## 커뮤니케이션
- 한국어로 답변
- 이모티콘 사용 금지
- 완료 표현 과장 금지
- 불필요한 배경 설명 생략, 코드/핵심만
- 코드 제공 시 실행 경로(또는 작업 디렉터리) 항상 명시

## 환경
- OS: Ubuntu 24.04 (Noble Numbat) / ROS 2 Jazzy Jalisco
- 개발 환경: Docker 컨테이너 (`quvi-dev`) 사용 (NVIDIA GPU 연동 지원)
- 작업 경로: 
  - 호스트: `~/QUVI/`
  - 컨테이너 내부: `/workspace/` (호스트의 소스 코드, 데이터, 스크립트가 볼륨 마운트됨)
- 실행 전 필수: `source /opt/ros/jazzy/setup.bash` 및 `source /workspace/install/setup.bash`

## 빌드 및 실행
- 빌드: Docker 컨테이너 내부 `/workspace`에서 수행
  ```bash
  cd /workspace && colcon build --symlink-install
  ```
- 소스: `source /workspace/install/setup.bash`
- 전체 시스템 실행: `ros2 launch quvi_bringup full_system.launch.py`
- 비전 파이프라인 단독 실행: `ros2 launch quvi_bringup vision_pipeline.launch.py`
- 코드 수정 후 반드시 `colcon build` 안내

## 하드웨어 및 센서
- 로봇팔: OMX 5DOF (Dynamixel XL430/XL330), LeRobot ACT 모방학습 파지 모델 연동
- 레일: NEMA17 + TB6600, GT2 벨트, 0.2mm/스텝 (ESP32-S3를 통해 제어)
- ESP32-S3: micro-ROS over USB (`/dev/ttyUSB0`, 921600 baud)
- 핸드캠 (Zone 1): `/dev/video0` (1920x1080, MJPEG, AutoFocus OFF)
- 사이드캠 (Zone 2): `/dev/video2` (1920x1080, MJPEG, AutoFocus OFF, Focus=80)
- 턴테이블: NEMA17 #2, 90°=400스텝 (1/8 마이크로스텝)

## ROS 2 패키지 및 노드 구조
- `quvi_yolo` / `yolo_node.py` (`yolo_node`) → YOLOv8n 객체 감지 및 베드 위 출력물 좌표 발행
- `quvi_inspect` / `inspect_node.py` (`inspect_node`) → OpenCV/SSIM 양불 판정 (CAD STL 렌더링 비교 + 표면 분석)
- `quvi_robot_control` / `robot_control_node.py` (`robot_control_node`) → OMX 로봇팔 + 리니어 레일 + 턴테이블 통합 제어 및 ESP32 통신
- `quvi_hmi` / `hmi_node.py` (`hmi_node`) → Flask + WebSocket 기반 Web GUI 대시보드 (기본 포트 5000)
- `quvi_bringup` → Launch 파일 및 전체 시스템 기동 관리
- `quvi_msgs` → 커스텀 메시지 및 서비스 정의 (`SystemStatus`, `InspectionResult`, `ObjectArray`, `GraspGoal` 등)
- *참고: 전체 상태머신을 담당할 메인 오케스트레이터 노드(`main_orchestrator_node.py`)는 설계상 정의되어 있으나 실구현 또는 추가 통합이 필요할 수 있습니다.*

## 코딩 규칙
- Python 3 우선 (ROS 2 노드 개발)
- MCU 코드는 Arduino C++ (ESP32-S3)
- ROS 2 토픽/서비스 이름은 `snake_case` 사용
- 불필요한 주석 없이 핵심 로직 중심의 간결한 코드 작성

## 계획 수립
- 복잡한 아키텍처나 주요 로직 변경 전 `implementation_plan.md` 작성 후 승인 요청
- 단순 수정, 명령어 실행, 질문 답변은 즉시 처리
