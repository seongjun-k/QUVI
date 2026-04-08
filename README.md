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

## ROS 2 노드

| 노드 | 역할 |
|------|------|
| `MAIN_CONTROLLER` | 전체 시퀀스 관리, 상태 머신 |
| `YOLO_NODE` | 출력물 감지, 좌표 목록 발행 |
| `GRASP_NODE` | ACT 파지 모델 실행, 로봇팔 제어 |
| `INSPECT_NODE` | 양불 판정 (CAD 비교 + 표면 특징) |
| `SORT_NODE` | 분류 명령 발행 |
| `MOTOR_CONTROL` | STM32 통신, 레일/턴테이블 제어 |
| `HMI_NODE` | 대시보드 표시 |

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
