# QUVI (QUality VIsion)

**AI 비전 로봇을 활용한 3D 프린터 출력물 자동 양불 판정 및 분류 시스템**

> "보는 것이 곧 품질이다 (Seeing is Quality)"

---

## 📌 개요

3D 프린터에서 출력을 완료한 제품을 로봇팔이 자동으로 인식 및 파지(Pickup)하여, 검사 챔버에서 머신비전을 통해 품질을 정밀 분석(양불 판정)하고 결과에 따라 합격(PASS)과 불량(FAIL) 분류함으로 적재하는 3D 프린팅 후처리 자동화 시스템(Smart Cell)입니다.

---

## 🛠️ 시스템 구성

| 구성 요소 | 기술 사양 | 역할 및 특징 |
| :--- | :--- | :--- |
| **메인 제어** | Ubuntu 24.04 + ROS 2 Jazzy (Docker) | 전체 노드 오케스트레이션 및 상태 머신(FSM) 제어 |
| **하위 제어** | ESP32-S3 (TB6600 구동) | 리니어 레일(스텝 모터) 및 턴테이블 각도 구동 제어 |
| **로봇팔** | OMX AI 매니퓰레이터 (팔로워) | 6자유도 다이나믹셀 기반 매니퓰레이터 (리더-팔로워 텔레옵 지원) |
| **카메라** | InnoMaker USB U20CAM × 2 | Zone 1 (핸드캠: 조종 및 YOLO) 및 Zone 2 (고정캠: 품질 검사용) |
| **AI 및 알고리즘** | YOLOv8n + LeRobot ACT + SSIM | 객체 탐지, 모방 학습 파지 제어, 정사영 형상비교 검사 |

---

## 📦 ROS 2 패키지 및 노드 구조

| 패키지명 | 실행 노드명 | 주요 역할 |
| :--- | :--- | :--- |
| **`quvi_robot_control`** | `main_orchestrator_node` | 전체 자율 시퀀스 유한상태머신(FSM) 제어 및 오케스트레이션 |
| | `robot_control_node` | 로봇팔 다이나믹셀(XL430, XL330) 제어, LeRobot ACT 모방학습 파지 구동, ESP32 레일/턴테이블 통신 중계 |
| **`quvi_yolo`** | `yolo_node` | YOLOv8n 기반 3D 프린터 베드 위 출력물 감지, 중심 3D 파지 좌표 계산 및 발행 |
| **`quvi_inspect`** | `inspect_node` | 4방향 카메라 정사영 정적 이미지 분석 (SSIM 유사도, Solidity 워핑 검사, Hole 개수 검사, 표면 텍스처 검증) |
| **`quvi_hmi`** | `hmi_node` | **Flask + SocketIO 기반 실시간 웹 대시보드** (시스템 상태 모니터링, 다중 MJPEG 비전 스트리밍 및 수동 제어) |
| **`quvi_msgs`** | - | 패키지 간 데이터 통신을 위한 커스텀 ROS 2 메시지 (`SystemStatus`, `InspectionResult` 등) |

---

## 🖥️ Web HMI 주요 기능 (대시보드)

* **실시간 시스템 상태 모니터링 (System Status)**
  * **로봇 6축 관절 각도 시각화**: `/robot/joint_states` 피드백을 실시간 도($^\circ$) 단위 및 게이지바로 표시.
  * **리니어 레일 트랙 모션**: 구역 위치에 맞춰 HMI 내 로봇 캐리지가 실시간으로 이동하는 CSS 애니메이션.
  * **턴테이블 나침반**: ESP32 구동 각도(0~360도)에 매핑되어 회전하는 원형 다이얼 그래픽.
  * **FSM 흐름도**: 초기화부터 탐지, 파지, 검사, 투하, 홈 복귀 단계가 활성화될 때마다 네온 글로우 효과로 단계 시각화.
* **실시간 MJPEG 비디오 스트리밍**
  * Zone 1 핸드캠 피드, Zone 2 고정캠 피드, YOLO 객체 탐지 결과 바운딩 박스, 품질 비전 정사영 디버깅 뷰.
* **수동 제어 및 텔레오퍼레이션 (Leader-Follower)**
  * 리더 로봇암을 이용한 1:1 팔로워 실시간 추종 조작 토글 브리지, 자율 시퀀스 시작/정지/비상정지(E-STOP) 제어.

---

## 📂 프로젝트 폴더 구조

```
QUVI/
├── docker/                  # Docker 가상화 개발 환경 (Dockerfile, compose)
├── firmware/                # ESP32-S3 기반 레일 및 턴테이블 구동용 펌웨어 소스
├── src/                     # ROS 2 소스 코드 디렉토리
│   ├── quvi_msgs/           # 시스템 공용 커스텀 메시지 정의
│   ├── quvi_bringup/        # 시스템 런처 파일 및 YAML 설정 파일
│   ├── quvi_robot_control/  # 로봇팔 및 메인 시퀀스(FSM) 제어 패키지
│   ├── quvi_yolo/           # YOLOv8 기반 출력물 탐지 패키지
│   ├── quvi_inspect/        # 이미지 비교 알고리즘 기반 품질 검사 패키지
│   └── quvi_hmi/            # Flask + SocketIO 웹 대시보드 HMI 패키지
├── data/                    # 모델 파일 및 CAD 기준 형상 데이터
├── scripts/                 # 유틸리티 및 캘리브레이션 헬퍼 스크립트
└── docs/                    # 기술 설계 문서 및 이미지 자료
```

---

## 🚀 빠른 시작 (Docker Environment)

### 1. 리포지토리 클론 및 폴더 이동
```bash
git clone https://github.com/seongjun-k/QUVI.git
cd QUVI
```

### 2. Docker 환경 구성 및 실행
```bash
cd docker
docker compose build
docker compose up -d
```

### 3. 개발 컨테이너 내부 진입 및 빌드
```bash
docker exec -it quvi-dev bash
cd /workspace
colcon build --symlink-install
source install/setup.bash
```

### 4. 전체 시스템 런칭 (HMI 웹 포함)
```bash
ros2 launch quvi_bringup quvi_system.launch.py
```
* 웹 브라우저를 열고 `http://localhost:5000`에 접속하여 HMI 대시보드를 모니터링합니다.

---

## 🛠️ 개발 기술 스택

* **Operating System**: Ubuntu 24.04 LTS (Noble Numbat)
* **Middleware**: ROS 2 Jazzy Jalisco
* **Vision & AI**: OpenCV 4.9, scikit-image, Ultralytics YOLOv8, PyTorch 2.x, Hugging Face LeRobot
* **Web HMI**: Python Flask, Flask-SocketIO, Vanilla JS (WebSocket Client), HTML5/CSS3 (Industrial Dark Theme)
* **Embedded Hardware**: ESP32-S3, TB6600, Dynamixel SDK (Protocol 2.0)

---

## 👥 팀 정보

- **서울로봇고등학교 졸업작품 돼지껍데기 팀**

---

## 📄 라이선스

이 프로젝트는 [MIT License](LICENSE)에 따라 라이선스가 부여됩니다.
