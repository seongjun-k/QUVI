# QUVI (QUality VIsion)

**AI 비전 로봇을 활용한 3D 프린터 출력물 자동 양불 판정 시스템**

> "보는 것이 곧 품질이다 (Seeing is Quality)"

## 개요

3D 프린터에서 출력된 제품을 로봇이 자동으로 픽업하여 머신비전으로 품질을 검사하고, 양품과 불량품으로 분류하는 완전 자동화 셀.

## 시스템 구성

| 구성 | 설명 |
|------|------|
| **메인 제어** | Ubuntu 24.04 + ROS 2 Jazzy (Docker) |
| **하위 제어** | ESP32 S3 |
| **로봇팔** | OMX AI 매니퓰레이터 (팔로워) |
| **카메라** | InnoMaker USB U20CAM-1080P × 2 |
| **AI** | YOLOv8n + ACT + SSIM + 표면 특징 분석 |

## ROS 2 노드 및 패키지 구성

| 패키지명 | 실행 노드명 | 역할 |
|----------|------------|------|
| `quvi_robot_control` | `main_orchestrator_node` | 전체 자율 제어 및 유한상태머신(FSM) 시퀀스 제어 |
| `quvi_robot_control` | `robot_control_node` | OMX 로봇팔 다이나믹셀 제어, LeRobot ACT 모방학습 파지 구동 및 ESP32 통신 중계 |
| `quvi_yolo` | `yolo_node` | YOLOv8n 기반 3D 프린터 베드 위 출력물 감지, 좌표 목록 및 근접 충돌 방지 경고 발행 |
| `quvi_inspect` | `inspect_node` | 양불 판정 (CAD 정사영 실루엣 비교 및 solidity/hole 등 표면 특징 데이터 이중 검증) |
| `quvi_hmi` | `hmi_node` | Flask + SocketIO 기반 실시간 웹 대시보드(모니터링, 다중 MJPEG 스트리밍 및 수동 제어 브리지) |

## 빠른 시작 (Docker)

```bash
# 1. 리포지토리 클론
git clone https://github.com/seongjun-k/QUVI.git
cd QUVI

# 2. Docker 이미지 빌드
cd docker
docker compose build

# 3. 컨테이너 실행
docker compose up -d

# 4. 컨테이너 접속
docker exec -it quvi-dev bash

# 5. ROS 2 워크스페이스 빌드
cd /workspace
colcon build --symlink-install
source install/setup.bash
```

## 프로젝트 구조

```
QUVI/
├── docker/                  # Docker 환경
│   ├── Dockerfile
│   └── docker-compose.yml
├── src/
│   ├── quvi_msgs/           # 커스텀 메시지/서비스
│   ├── quvi_bringup/        # Launch 파일, 설정
│   ├── quvi_yolo/           # YOLO 객체 탐지 노드
│   └── quvi_inspect/        # 양불 판정 노드
├── data/
│   ├── datasets/            # 학습 데이터
│   └── reference_stl/       # CAD 기준 STL 파일
├── scripts/                 # 유틸리티 스크립트
└── docs/                    # 문서
```

## 기술 스택

- **OS**: Ubuntu 24.04 (Noble Numbat)
- **ROS**: ROS 2 Jazzy Jalisco
- **비전**: OpenCV 4.9, scikit-image
- **AI**: PyTorch 2.x, Ultralytics YOLOv8, Hugging Face LeRobot
- **HMI**: PyQt5 / Flask

## 팀

- 서울로봇고등학교 졸업작품 돼지껍데기 팀

## 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다.
