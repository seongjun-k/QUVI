# QUVI (QUality VIsion)

**AI 비전 로봇을 활용한 3D 프린터 출력물 자동 양불 판정 및 분류 시스템**

> "보는 것이 곧 품질이다 (Seeing is Quality)"

---

## 개요

3D 프린터에서 출력을 완료한 제품을 로봇팔이 자동으로 파지(Pickup)하여, 검사 챔버의 턴테이블에서 4방향 촬영 후 머신비전으로 품질을 분석(양불 판정)하고, 결과에 따라 합격(PASS)과 불량(FAIL) 스테이션으로 분류 적재하는 3D 프린팅 후처리 자동화 시스템(Smart Cell)입니다.

전 과정은 유한상태머신(FSM) 오케스트레이터가 자율 제어하며, 파지는 LeRobot ACT 모방학습, 검사는 표면 특징 룰 판정 + PatchCore 이상탐지(섀도우 모드)로 수행됩니다.

---

## 시스템 구성

| 구성 요소 | 기술 사양 | 역할 및 특징 |
| :--- | :--- | :--- |
| **메인 제어** | Ubuntu 24.04 + ROS 2 Jazzy (Docker) | 전체 노드 오케스트레이션 및 상태 머신(FSM) 제어 |
| **하위 제어** | ESP32-S3 + TB6600 (micro-ROS) | 리니어 레일(스텝 모터)·턴테이블 각도·조명 LED 구동 제어 |
| **로봇팔** | ROBOTIS OMX 매니퓰레이터 (팔로워) | 다이나믹셀(XL430/XL330) 기반, 리더-팔로워 텔레옵 지원 |
| **카메라** | USB UVC 카메라 × 2 | 사이드캠(Zone 1: 파지 영역), 검사캠(Zone 2: 품질 검사 챔버) |
| **AI 및 알고리즘** | LeRobot ACT + OpenCV + PatchCore | 모방학습 파지 제어, 표면 특징 룰 판정, ML 이상탐지(섀도우 모드) |

---

## 검사 방식

검사캠이 턴테이블 4방향(0°/90°/180°/270°) 이미지를 캡처하여 두 갈래로 분석합니다.

1. **표면 특징 룰 판정 (판정 주체)** — 각도별 worst-case로 PASS/FAIL 결정
   * Solidity (컨벡스 헐 대비 윤곽 면적 — 워핑 감지)
   * 면적비 (기준 이미지 대비 — 미출력/과출력 감지). 턴테이블 편심에 의한 물체-카메라 거리 변화를 상쇄하기 위해 **면적/폭² 거리 불변 정규화**로 비교
   * 구멍 개수·구멍 면적비 (레이어 분리 감지) — 구멍 1개부터 FAIL
   * 텍스처 분산 (라플라시안 — 스트링잉 감지)

   판정 임계값은 `src/quvi_inspect/config/inspect_params.yaml`에서 관리하며, HMI 표시 기준(`dashboard.js` THRESHOLDS)과 동기화를 유지합니다.
2. **PatchCore 이상탐지 (섀도우 모드)** — WideResNet50 백본 기반 각도별 메모리뱅크로 이상점수를 계산해 로그에만 기록. 판정에는 반영하지 않으며, 룰 판정과의 일치율을 축적해 컷오버 여부를 검증하는 단계입니다 (`scripts/shadow_report.py`).

기준 이미지는 HMI의 기준 이미지 캡처 모드로 정상품을 실촬영하여 생성합니다.

---

## ROS 2 패키지 및 노드 구조

| 패키지명 | 실행 노드명 | 주요 역할 |
| :--- | :--- | :--- |
| **`quvi_robot_control`** | `main_orchestrator_node` | 전체 자율 시퀀스 FSM 제어 (파지 → 챔버 안착 → 검사 → 분류 → 홈 복귀) |
| | `robot_control_node` | 로봇팔 다이나믹셀 제어, LeRobot ACT 파지 추론, 레일/턴테이블 명령 중계, E-STOP 처리 |
| **`quvi_inspect`** | `inspect_node` | 4방향 표면 특징 분석 양불 판정 + PatchCore 이상탐지(섀도우), 검사 로그 저장, 기준 이미지·ML 데이터셋 캡처 모드 |
| **`quvi_hmi`** | `hmi_node` | **Flask + SocketIO 기반 실시간 웹 대시보드** (상태 모니터링, MJPEG 스트리밍, 수동 제어) |
| **`quvi_msgs`** | - | 커스텀 메시지 (`SystemStatus`, `InspectionResult`, `GraspGoal`, `MotorStatus` 등) |
| **`quvi_bringup`** | - | 시스템 런치 파일 (`full_system.launch.py`, `vision_pipeline.launch.py`) |

토픽 이름은 `quvi_robot_control/topics.py`에서 일원 관리합니다.

---

## Web HMI 주요 기능 (대시보드)

* **실시간 시스템 상태 모니터링**
  * 로봇 관절 각도 시각화 (`/robot/joint_states` 실시간 게이지)
  * 리니어 레일 트랙 모션 (스테이션 맵: INSPECT / PASS / FAIL / BED)
  * 턴테이블 나침반 다이얼, FSM 단계별 흐름도 시각화
  * 검사 이력·통계 (PASS/FAIL 카운트)
* **실시간 MJPEG 비디오 스트리밍** — 사이드캠, 검사캠, 검사 디버그 뷰(4방향 타일 + 판정 오버레이)
* **수동 제어**
  * 자율 시퀀스 시작/정지, **비상정지(E-STOP)** 및 리셋
  * 리더-팔로워 텔레오퍼레이션 토글, 레일/턴테이블/LED 수동 구동
  * ACT 모델 스캔·선택(핫스왑), 장치 매핑(카메라·시리얼 포트) 설정 및 재시작
  * 기준 이미지 캡처 모드, ML 정상품 데이터셋 촬영 모드

---

## 프로젝트 폴더 구조

```
QUVI/
├── docker/                  # Docker 개발 환경 (Dockerfile, compose)
├── firmware/                # ESP32-S3 레일·턴테이블 펌웨어 (PlatformIO, micro-ROS)
├── lerobot/                 # LeRobot 서브모듈 (OMX 지원 브랜치)
├── src/                     # ROS 2 소스
│   ├── quvi_msgs/           # 커스텀 메시지 정의
│   ├── quvi_bringup/        # 런치 파일
│   ├── quvi_robot_control/  # 로봇팔·FSM 오케스트레이터·공용 유틸/토픽
│   ├── quvi_inspect/        # 양불 판정 + PatchCore 이상탐지 패키지
│   └── quvi_hmi/            # Flask + SocketIO 웹 대시보드
├── data/                    # 기준 이미지, 검사 로그, ML 데이터셋·모델, 장치 설정
├── scripts/                 # ACT 녹화/학습, 이상탐지 학습, 캘리브레이션·진단 스크립트
├── tests/                   # pytest 로직 테스트
└── docs/                    # 기술 설계 문서
```

---

## 빠른 시작 (Docker Environment)

### 1. 리포지토리 클론 및 서브모듈 초기화
```bash
git clone https://github.com/seongjun-k/QUVI.git
cd QUVI
git submodule update --init --recursive
```

### 2. Docker 환경 구성
```bash
cd docker
docker compose build
docker compose up -d
```

### 3. 빌드 및 실행 (호스트에서)
```bash
./build.sh   # 컨테이너 기동 + colcon build --symlink-install
./run.sh     # full_system.launch.py 실행
```
* 웹 브라우저에서 `http://localhost:5000` 접속 → HMI 대시보드.
* 수동 실행 시: 컨테이너 내부에서 `ros2 launch quvi_bringup full_system.launch.py`

### 4. 테스트
```bash
docker exec quvi-dev bash -c "cd /workspace && python3 -m pytest tests/ -q"
```

---

## 개발 기술 스택

* **Operating System**: Ubuntu 24.04 LTS
* **Middleware**: ROS 2 Jazzy + micro-ROS (ESP32-S3)
* **Vision & AI**: OpenCV, PyTorch (numpy <2 고정), Hugging Face LeRobot (ACT), PatchCore(WideResNet50)
* **Web HMI**: Flask, Flask-SocketIO (threading 모드), Vanilla JS, HTML5/CSS3 (Industrial Dark Theme)
* **Embedded**: ESP32-S3, TB6600, Dynamixel SDK (Protocol 2.0), PlatformIO

---

## LeRobot ACT 모방학습 가이드

로봇팔 파지(Zone 1)는 LeRobot ACT(Action Chunking with Transformers) 모방학습 기반 비주오모터 제어로 수행됩니다.

### 1. 텔레오퍼레이션 데이터 수집
리더-팔로워 시연 데이터를 헬퍼 스크립트로 녹화합니다 (호스트/컨테이너 어디서든 실행 가능).
```bash
./scripts/act_record.sh <HF_USER> <에피소드수> <에피소드시간(초)>
```

### 2. ACT 모델 학습
```bash
./scripts/act_train.sh <HF_USER>
```
CUDA 미가용 시 CPU로 폴백하며 경고를 출력합니다.

### 3. 추론 및 배포
HMI 대시보드에서 모델 스캔·선택으로 런타임 교체가 가능하며, **마지막으로 선택한 모델은 `data/act_last_model.json`에 저장되어 재기동 시 자동 로드·활성화**됩니다. 저장된 선택이 없으면 `act_model_path` 파라미터 기본 경로를 사용합니다.

---

## PatchCore 이상탐지 학습 파이프라인 (섀도우 모드)

```bash
# 0. HMI 데이터셋 촬영 모드 또는 기존 PASS 검사 로그로 정상품 이미지 수집
python3 scripts/build_anomaly_dataset.py     # PASS 로그 → 각도별 raw/ 정리 + 검수 시트 생성
# (사람이 review_sheet_{angle}.png 를 보고 불량 혼입 이미지를 raw/에서 삭제)

# 1. 각도별 메모리뱅크 학습 + 임계값 산정
python3 scripts/train_anomaly_bank.py        # → data/models/bank_{angle}.pt, thresholds.json

# 2. 섀도우 운영 후 룰 vs ML 일치율 리포트 (컷오버 게이트)
python3 scripts/shadow_report.py
```

런치 인자 `anomaly_enabled`(기본 true)로 켜고 끄며, 모델 파일이 없거나 로드에 실패하면 자동 비활성되어 룰 판정만 사용합니다.

---

## ESP32-S3 펌웨어 빌드·플래시

리니어 레일·턴테이블·LED를 담당하는 ESP32-S3 펌웨어는 PlatformIO 프로젝트입니다 (`firmware/quvi_esp32_firmware/`). ESP32-S3는 CH340 브리지 경유(`/dev/ttyESP32` udev 심링크, `scripts/99-esp32.rules`)로 연결되며, 부트 버튼 조작 없이 자동 리셋 업로드됩니다.

```bash
# 호스트에서 실행. micro-ROS agent가 포트를 잡고 있으면 먼저 종료할 것.
cd firmware/quvi_esp32_firmware
pio run                                        # 컴파일
pio run -t upload --upload-port /dev/ttyESP32  # 플래시
```

호밍(3단계: 코스 탐색 → 백오프 → 정밀 탐색)·레일 좌표계·소프트 리밋 등 하드웨어 캘리브레이션 상수는 `Config.h`에서 관리합니다.

---

## 팀 정보

- **서울로봇고등학교 졸업작품 돼지껍데기 팀**

---

## 라이선스

이 프로젝트는 [MIT License](LICENSE)에 따라 라이선스가 부여됩니다.
