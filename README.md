# QUVI (QUality VIsion)

[![한국어](https://img.shields.io/badge/lang-한국어-blue)](README.md)
[![English](https://img.shields.io/badge/lang-English-red)](README.en.md)
[![日本語](https://img.shields.io/badge/lang-日本語-white)](README.ja.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
![ROS 2](https://img.shields.io/badge/ROS_2-Jazzy-22314E?logo=ros)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/MCU-ESP32--S3-E7352C)
![Docker](https://img.shields.io/badge/Docker-quvi--dev-2496ED?logo=docker&logoColor=white)
[![Web Demo](https://img.shields.io/badge/Web_Demo-Live-brightgreen)](https://seongjun-k.github.io/QUVI/)

**AI 비전 로봇팔 기반 3D프린터 출력물 무인 검사·분류 시스템**

> "보는 것이 곧 품질이다 (Seeing is Quality)"

▶ **[웹 데모 바로가기](https://seongjun-k.github.io/QUVI/)** — 하드웨어 없이 정상품·불량품 검사 사이클을 실기 녹화 그대로 체험할 수 있습니다. 모바일 화면도 지원합니다.

---

## 미션과 성과 지표

**미션** — 우리는 장시간·야간에 3D프린터를 돌리는 메이커와 소규모 공방·출력 사업자가 출력물을 계속 지켜볼 수 없어 불량을 뒤늦게 발견하고 재료와 시간을 버리는 문제를, 출력 직후 로봇이 자동 촬영하고 AI가 불량을 판정하는 방식으로 없앱니다.

**북극성 지표** — 출력이 끝난 순간부터 판정·분류가 끝나기까지 걸리는 시간입니다. 이전 심사 피드백에 따라 북극성을 측정 가능한 구간으로 재정의하고 실측했습니다. 사람이 확인하러 올 때까지 수십 분에서 수 시간 방치되던 것을 **현재 중앙값 64초**(실기 23회, 9월 71초 대비 -10%)까지 줄였습니다. 다음 목표는 가장 긴 검사 구간(24초)을 줄여 60초 이하로 만드는 것입니다.

큰 실패(필라멘트 소진·충돌)는 프린터가 스스로 멈추고, 끝까지 출력된 불량(층 밀림·뒤틀림·실 날림)은 QUVI가 잡습니다. QUVI는 Klipper 상태가 `complete`일 때만 검사를 시작합니다.

---

## 본선 기간 개선 (2026-10-21 ~ 11-10)

실기 로그(ROS 로그 4,356개 파일, 파지 궤적 203개, 검사 기록 268건)와 측정 스크립트로 다시 센 값입니다. 수정 전후 기준 시점은 파지·학습이 새 VLA 모델로 바꾼 11-08 19:10, 검사가 품종 자동 선택을 적용한 11-09 18:55입니다. 상세 수치와 원자료 목록은 [`docs/metrics_20261110.md`](docs/metrics_20261110.md)에 있습니다.

| 지표 | 전 | 후 | 변화 |
| :--- | ---: | ---: | ---: |
| **북극성: 출력 완료 → 분류 완료 시간(중앙값)** | 71초 (66회) | **64초** (23회) | **-7초 (-10%)** |
| 파지 동작 완료율 | 53% (37/70) | **77%** (34/44) | **+24%p** |
| 오류·비상정지 비율 | 26% (18/70) | **5%** (2/44) | **-21%p** |
| 빈손 운반 비율 | 20% (1/5) | **0%** (0/16) | **-20%p** |
| VLA 학습 예측 오차(관절 평균 절대 오차) | 0.1457 | **0.0409** | **-72%** |
| 검사 판정 완료율 | 30% (43/143) | **100%** (7/7) | **+70%p** |
| 검사 판정 불가 건수 | 100건 | **0건** | **-100건** |
| 정상품 오탐률(밝기 1.3배, box) | 46% | **6%** (2/32) | **-40%p** |
| 새 품목 검사 준비 시간 | 측정 없음 | **약 15분** | 처음 측정 |
| 실사용자 테스트 | 0명 | **1명** (긱블 메이커 수드래곤, 11-10) | +1명 |

**실사용자 테스트 — 긱블 메이커 수드래곤 (2026-11-10)**

QUVI 프로토타입을 직접 사용해 보고 인터뷰했습니다. 상용화되면 사용할 의향이 있다고 답했습니다.

* "기존 솔루션들은 대부분 오픈형 챔버에서 베드를 교체하는 접근이었는데, 챔버든 노출형이든 적용할 수 있는 솔루션을 고민했다는 점이 굉장히 인상적입니다. 특히 같은 부품을 여러 번 뽑아야 하는 부분에서 기대가 됩니다."
* "실제로 솔루션이 상용화되었으면 한다. 매우 편리할 것 같다."
* "메이커 관점에서 보면 출력물에서 생기는 문제들을 신속하게 해결할 수 있는 점이 좋았다."

| 피드백 | 반영 계획 |
| :--- | :--- |
| 챔버형·노출형 모두 적용 가능 | 베드 교체 방식은 챔버형에 쓸 수 없다는 차별점으로 확인, 비교 근거로 제시 |
| 같은 부품 반복 출력에 기대 | 반복 생산 현장을 핵심 사용처로, 연속 사이클 64초·새 품목 준비 약 15분을 근거로 제시 |
| 출력물 문제를 신속하게 해결 | 북극성 지표(출력 완료 → 분류 완료 64초)로 계속 측정 |

**무엇을 바꿨나**

* **파지 모델 재학습** — 시연 115개를 새로 수집해 112개로 정제(9,296프레임), SmolVLA를 15,000 step 학습(`smolvla_1108_box2c_new15k`, loss 0.542 → 0.079). 홈 자세를 학습 시작 자세에 맞추고 VLA 실행 시간을 시연 길이와 같은 5.5초로 맞춤
* **자체 수집·학습 파이프라인** — 외부 Cyclo 스택 의존 없이 웹 수집 도구(녹화·재생·검수) → LeRobot v3.0 변환 → 학습 스크립트로 일원화. 수집과 추론이 같은 카메라 노출·화이트밸런스 프로필을 씀
* **안정성** — 물리 비상정지 버튼(누르는 동안 정지, 떼면 복귀), 로봇팔 USB 끊김 자동 재연결, 검사캠 노드 자동 재시작, 카메라 3대 USB 대역폭 시분할과 허브 재연결 시 자동 복구
* **검사** — 다품종 검사 자산(box·cube) 등록과 시작 시 품종 자동 선택으로 판정 불가(품종 미선택 70건·기준 이미지 부족 30건)를 0건으로. 새 품목은 정상품 160장 촬영(13분)과 학습(1~2분)으로 준비하며 불량 샘플·라벨링이 필요 없음
* **하드웨어 보정** — 턴테이블 놓기 자세 +8mm 상향·전방 +10mm 이동, 그리퍼 파지 힘 50% 감소

**시험했지만 채택하지 않은 것** — 검사 밝기 정규화(11-02)는 box 밝기 1.3배 오탐을 46% → 4%로 줄였지만 긁힘 탐지 92% → 50%, 모서리 결손 탐지 54% → 8%로 놓침이 늘고 cube 잘림 오탐이 0% → 54%로 늘어 기각했습니다(합성 결함, 품종당 24장). 재생 주기 30→15Hz(9월 실험)는 떨림이 2배로 늘어 기각했습니다. 떨림의 원인은 재생 주기가 아니라 정책 출력 자체의 진동으로 확인했고, 다음 단계는 액션 스무딩입니다.

**측정 한계**

* 파지 동작 완료율은 파지 시퀀스를 끝까지 실행한 비율입니다. 그리퍼 빈손 판정은 아직 shadow 모드라 반영되지 않았고, 실제 운반 여부는 검사 결과로 따로 확인했습니다(판정 기록이 있는 16사이클 모두 출력물 있음).
* 학습 예측 오차는 현재 모델의 학습 데이터 기준 개루프 평가 값입니다.
* 검사 판정 완료율의 수정 후 표본은 7건입니다. 북극성의 9월 값은 이전 집계이고, 시연에서는 HMI 시작 버튼으로 출력 완료 신호를 대신합니다.
* 검사 오탐률은 조명 밝기에 영향을 받으므로 검사 챔버 조명을 일정하게 유지해야 합니다.

---

## 문제 정의

3D프린터 값이 내려가면서 출력 대행·시제품·소량 생산으로 사업을 하는 소규모 사업자가 늘었습니다. 그런데 FDM 출력에서 실패는 없앨 수 있는 사고가 아니라 확률로 남는 상수입니다. 공용 출력 환경의 데스크톱 FDM 실패율은 41.1%로 조사됐고(Song & Telenko, *Material and energy loss due to human and machine error in commercial FDM printers*, Journal of Cleaner Production, 2017), 숙련자와 프린트 팜도 2~8%까지 낮출 뿐 0으로 만들지는 못합니다.

진짜 손실은 실패 자체가 아니라 **출력이 끝난 뒤 아무도 확인하지 않는 시간**입니다. 끝까지 출력된 불량은 사람이 봐야 알 수 있고, 밤새 프린터를 돌리는 사업자에게 새벽에 끝난 출력은 아침까지 베드 점유로 이어지며, 그 몇 시간이 하루치 사이클을 통째로 날립니다. 손실은 검사 인건비가 아니라 프린터 가동률에서 납니다.

산업용 비전 검사 장비는 수천만 원대이고 제품이 바뀔 때마다 도면과 라벨링 데이터를 다시 준비해야 해서, 주문마다 다른 것을 뽑는 현장에는 맞지 않습니다. QUVI는 **파지(모방학습 로봇팔) → 검사(머신비전) → 분류(자동 적재)** 전 과정을 60만 원대 부품으로 무인화한 시스템입니다. 출력이 끝난 베드가 들어오면 로봇팔이 출력물을 파지해 검사 챔버의 턴테이블에서 4방향 촬영 후 양불을 판정하고, 결과에 따라 합격(PASS)·불량(FAIL) 스테이션으로 분류 적재합니다. 그 사이 사람 손은 닿지 않습니다.

전 과정은 유한상태머신(FSM) 오케스트레이터가 자율 제어하며, 파지는 SmolVLA(비전-언어-행동 모델) 모방학습, 검사는 표면 특징 룰 판정과 PatchCore 이상탐지를 결합한 하이브리드 방식으로 수행됩니다. Klipper 프린터의 출력 완료 신호를 받아 시퀀스를 시작할 수 있습니다.

---

## 주요 수치 (1차 집계, 2026-07-30 기준)

본선 기간 개선 수치는 위 [본선 기간 개선](#본선-기간-개선-2026-10-21--11-10) 표를 참조하십시오. 아래 파지 성공률은 이전 ACT 정책·다른 집계 기준이라 위 표의 파지 동작 완료율과 직접 비교할 수 없습니다.

| 항목 | 값 |
| :--- | :--- |
| 전체 부품비 | 약 60만 원 — 수천만 원대 산업용 비전 검사 장비의 1/50 이하 |
| 사이클 타임 | 파지 시작부터 분류 적재까지 약 1분 10초 (실기 녹화 기준, 정상품 67초·불량품 64초) |
| 실기 검사 사이클 | 누적 133회 완료 (검사 로그 기준, 2026-06-26~07-30 20:43 KST) |
| 파지 성공률 | 실기 13회 시도 중 11회 성공, 84.6% (2026-07-30 실측, 사람이 시험을 수동 중단한 4건 제외 — 포함 시 11/17 = 64.7%) |
| 룰-ML 판정 일치율 | 최근 13건(2026-07-15 이후) 100% — 검증 초기를 포함한 전체 103건 기준 88.3% (ML 판정이 없는 07-08 이전 30건 제외) |

집계 범위 이후 기록을 포함한 원자료 전체는 137회, 룰-ML 95/107 = 88.8%, 파지 15/17 = 88.2%입니다. 원자료는 [`docs/metrics/`](docs/metrics/)에 있고 `python3 scripts/recount_metrics.py`로 다시 계산할 수 있습니다. 웹 데모 지표는 [W6 GA4 리포트](docs/metrics/QUVI_W6_실측지표.pdf)(2026-08-31~09-27)를 따릅니다.

---

## 아키텍처

### 시스템 구성

| 구성 요소 | 기술 사양 | 역할 및 특징 |
| :--- | :--- | :--- |
| **메인 제어** | Ubuntu 24.04 + ROS 2 Jazzy (Docker) | 전체 노드 오케스트레이션 및 상태 머신(FSM) 제어 |
| **하위 제어** | ESP32-S3 + TB6600 (micro-ROS) | 리니어 레일(스텝 모터)·턴테이블 각도·조명 LED 구동 제어, 물리 비상정지 버튼 |
| **로봇팔** | ROBOTIS OMX 매니퓰레이터 (팔로워) | 다이나믹셀(XL430/XL330) 기반, 리더-팔로워 텔레옵 지원 |
| **카메라** | USB UVC 카메라 × 3 | 사이드캠·탑뷰캠(Zone 1: 파지 영역, VLA 입력), 검사캠(Zone 2: 품질 검사 챔버). 같은 USB 2.0 버스라 대역폭 시분할 |
| **프린터 연동** | Klipper + Moonraker (KP3S) | 출력 완료 감지 → 자동 시작(옵션), 출력 종료 매크로로 베드 배출 |
| **AI 및 알고리즘** | LeRobot SmolVLA + OpenCV + PatchCore | 모방학습 파지 제어, 표면 특징 룰 판정, ML 이상탐지(하이브리드 결합) |

### 검사 방식

검사캠이 턴테이블 4방향(0°/90°/180°/270°) 이미지를 캡처하여 두 갈래로 분석하고, 두 결과를 **하이브리드로 결합**해 최종 판정합니다. 검사 기준(기준 이미지·ML 모델)은 품종별로 등록하며, 시작 시 품종이 자동 선택됩니다(현재 box·cube 2종).

1. **표면 특징 룰 판정** — 각도별 worst-case로 PASS/FAIL 결정
   * Solidity (컨벡스 헐 대비 윤곽 면적 — 워핑 감지)
   * 면적비 (기준 이미지 대비 — 미출력/과출력 감지). 턴테이블 편심에 의한 물체-카메라 거리 변화를 상쇄하기 위해 **면적/폭² 거리 불변 정규화**로 비교
   * 구멍 개수·구멍 면적비 (레이어 분리 감지) — 구멍 1개부터 FAIL
   * 텍스처 분산 (라플라시안 — 스트링잉 감지)

   판정 임계값은 `src/quvi_inspect/config/inspect_params.yaml`에서 관리하며, HMI 표시 기준(`dashboard.js` THRESHOLDS)과 동기화를 유지합니다.
2. **PatchCore 이상탐지** — WideResNet50 백본 기반 각도별 메모리뱅크로 이상점수를 계산. 정상품 이미지만으로 학습하므로 불량 샘플 수집·라벨링이 필요 없습니다.

**하이브리드 판정 규칙** — 룰과 ML이 모두 PASS일 때만 최종 PASS입니다. 룰이 PASS로 본 제품이라도 ML이 명시적으로 FAIL을 내면 최종 FAIL로 뒤집습니다. 반대로 ML이 룰의 FAIL을 PASS로 되돌리지는 않습니다. 즉 ML은 판정을 보수적으로 강화하는 방향으로만 작동하며, 불량품이 양품으로 새는 경우(false-accept)를 늘리지 않습니다. ML 모델이 로드되지 않은 환경에서는 룰 단독 판정으로 자동 폴백합니다.

판정 결과에는 `anomaly_score_worst`(4방향 중 최악 이상점수)와 `ml_passed`(-1=미사용 / 0=FAIL / 1=PASS)가 함께 발행되며, HMI 대시보드에 ML 이상 점수로 표시됩니다.

기준 이미지는 HMI의 기준 이미지 캡처 모드로 정상품을 실촬영하여 생성합니다. 룰 판정의 면적비 항목이 기준 이미지를 사용하므로 현재는 여전히 필요합니다. 다만 같은 제품을 반복 출력하는 소량 생산에서는 그 이미지가 곧 정답이라 제약이 되지 않으며, 기준 이미지 의존을 완전히 걷어내는 것은 향후 과제입니다.

### ROS 2 패키지 및 노드 구조

| 패키지명 | 실행 노드명 | 주요 역할 |
| :--- | :--- | :--- |
| **`quvi_robot_control`** | `main_orchestrator_node` | 전체 자율 시퀀스 FSM 제어 (파지 → 챔버 안착 → 검사 → 분류 → 홈 복귀) |
| | `robot_control_node` | 로봇팔 다이나믹셀 제어, SmolVLA 파지 추론(별도 사이드카 프로세스), 레일/턴테이블 명령 중계, E-STOP 처리 |
| | `printer_monitor_node` | Klipper/Moonraker 출력 상태 감시, 출력 완료 신호 발행 (아무것도 움직이지 않음) |
| **`quvi_inspect`** | `inspect_node` | 4방향 표면 특징 분석 + PatchCore 이상탐지 하이브리드 양불 판정, 품종별 검사 자산 관리, 검사 로그 저장, 기준 이미지·ML 데이터셋 캡처 모드 |
| **`quvi_hmi`** | `hmi_node` | **Flask + SocketIO 기반 실시간 웹 대시보드** (상태 모니터링, MJPEG 스트리밍, 수동 제어) |
| **`quvi_msgs`** | - | 커스텀 메시지 (`SystemStatus`, `InspectionResult`, `GraspGoal`, `MotorStatus` 등) |
| **`quvi_bringup`** | - | 시스템 런치 파일 (`full_system.launch.py`, `vision_pipeline.launch.py`) |

토픽 이름은 `quvi_robot_control/topics.py`에서 일원 관리합니다.

### Web HMI 주요 기능 (대시보드)

* **실시간 시스템 상태 모니터링**
  * 로봇 관절 각도 시각화 (`/robot/joint_states` 실시간 게이지)
  * 리니어 레일 트랙 모션 (스테이션 맵: INSPECT / PASS / FAIL / BED)
  * 턴테이블 나침반 다이얼, FSM 단계별 흐름도 시각화
  * 검사 이력·통계 (PASS/FAIL 카운트)
* **실시간 MJPEG 비디오 스트리밍** — 사이드캠, 탑뷰캠, 검사캠, 검사 디버그 뷰(4방향 타일 + 판정 오버레이)
* **수동 제어**
  * 자율 시퀀스 시작/정지, **비상정지(E-STOP)** 및 리셋 (물리 버튼은 누르는 동안 정지, 떼면 자동 복귀)
  * 리더-팔로워 텔레오퍼레이션 토글, 레일/턴테이블/LED 수동 구동
  * VLA 모델 스캔·선택(핫스왑), 장치 매핑(카메라·시리얼 포트) 설정 및 재시작
  * 검사 품종 선택·등록, 기준 이미지 캡처 모드, ML 정상품 데이터셋 촬영 모드

### 프로젝트 폴더 구조

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
├── scripts/                 # VLA 학습·평가, 이상탐지 학습, 지표 집계, 캘리브레이션·진단 스크립트
│   ├── collect/             # VLA 시연 수집 웹 앱 (녹화·재생·검수, 포트 5100)
│   └── vla_sidecar/         # SmolVLA 추론 사이드카 (lerobot 0.6.1 별도 venv)
├── tests/                   # pytest 로직 테스트
└── docs/                    # 기술 설계 문서
```

---

## 사용 스택

* **Operating System**: Ubuntu 24.04 LTS
* **Middleware**: ROS 2 Jazzy + micro-ROS (ESP32-S3)
* **Vision & AI**: OpenCV, PyTorch (numpy <2 고정), Hugging Face LeRobot (SmolVLA), PatchCore(WideResNet50)
* **Web HMI**: Flask, Flask-SocketIO (threading 모드), Vanilla JS, HTML5/CSS3 (Industrial Dark Theme)
* **Embedded**: ESP32-S3, TB6600, Dynamixel SDK (Protocol 2.0), PlatformIO

---

## 실행 방법 (Docker Environment)

> **하드웨어가 없으시면** 아래 절차 대신 [웹 데모](https://seongjun-k.github.io/QUVI/)를 이용해 주세요. 설치 없이 브라우저에서 실기 녹화 그대로 검사 사이클 전 과정을 보실 수 있습니다.
> 아래 절차는 로봇팔·레일·카메라 실물을 갖춘 환경 기준이며, 호스트는 Ubuntu 24.04 + Docker를 전제로 합니다.

### 1. 리포지토리 클론 및 서브모듈 초기화
```bash
git clone https://github.com/seongjun-k/QUVI.git
cd QUVI
git submodule update --init --recursive
```

### 2. 장치 udev 규칙 설치 (실물 연결 시 1회)
ESP32-S3와 카메라 3대는 고정 심링크(`/dev/ttyESP32`, `/dev/sidecam`, `/dev/topcam`, `/dev/fixed_cam`)로 잡아야 런치 파일의 기본 장치 경로와 맞습니다.
```bash
sudo cp scripts/99-esp32.rules scripts/99-uvc-cameras.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```
> 카메라 심링크는 USB 허브의 물리 포트 위치를 기준으로 하므로, 허브를 다른 포트에 꽂으면 생성되지 않습니다. 그 경우 HMI 장치 설정에서 raw 노드를 직접 지정하십시오.

### 3. Docker 환경 구성
```bash
cd docker
docker compose build
docker compose up -d
```

### 4. 빌드 및 실행 (호스트에서)
```bash
./build.sh   # 컨테이너 기동 + colcon build --symlink-install
./run.sh     # full_system.launch.py 실행
```
* 웹 브라우저에서 `http://localhost:5000` 접속 → HMI 대시보드.
* 수동 실행 시: 컨테이너 내부에서 `ros2 launch quvi_bringup full_system.launch.py`

### 5. 테스트
```bash
docker exec quvi-dev bash -c "cd /workspace && python3 -m pytest tests/ -q"
```

---

## SmolVLA 모방학습 가이드

로봇팔 파지(Zone 1)는 LeRobot SmolVLA 모방학습 기반 비주오모터 제어로 수행됩니다. 사이드캠과 탑뷰캠 2장과 관절값을 입력으로 받습니다.

> **모방학습이란?** 동작을 좌표·규칙으로 프로그래밍하는 대신, 사람이 리더암을 잡고 파지 동작을 여러 번 시연하면 그때의 카메라 영상과 로봇 관절값 쌍을 신경망이 학습하는 방식입니다. 학습을 마친 모델은 카메라 영상만 보고 다음 관절 동작을 스스로 생성하므로, 출력물의 위치가 조금씩 달라져도 별도 티칭 없이 파지할 수 있습니다. SmolVLA는 사전학습된 비전-언어 모델 위에 행동 출력부를 얹은 소형 VLA 정책으로, 적은 시연 데이터로 파인튜닝할 수 있습니다.

quvi-dev의 메인 파이썬은 lerobot 0.3.4/numpy<2 환경이라, SmolVLA(lerobot 0.6.1)는 별도 venv의 사이드카 프로세스(`scripts/vla_sidecar/`)에서 추론합니다.

### 1. 시연 데이터 수집 (웹 수집 도구)
HMI에서 텔레옵을 켠 뒤, 수집 웹 앱(`http://localhost:5100`)에서 에피소드를 녹화·재생하고 사용 여부를 검수합니다. 녹화 시작 자세는 학습 데이터 평균 자세로 맞춥니다.
```bash
# 호스트에서: 변환 도구 venv 생성 (최초 1회)
docker exec quvi-dev bash /workspace/scripts/collect/setup_tools_venv.sh
# 컨테이너 quvi-dev 안에서: 수집 앱 실행
source /opt/ros/jazzy/setup.bash
PYTHONPATH=/workspace/src/quvi_robot_control:$PYTHONPATH python3 /workspace/scripts/collect/app.py
```

### 2. LeRobot v3.0 데이터셋 변환
수집 원본(`data/episodes_raw/<세션>`)을 15fps로 리샘플해 LeRobot v3.0 데이터셋으로 변환하고 품질을 점검합니다. 수집 앱에서 실행하거나 직접 실행합니다.
```bash
# 컨테이너 quvi-dev 안에서 실행
/workspace/data/collect_tools/venv/bin/python /workspace/scripts/build_lerobot_dataset.py \
    /workspace/data/episodes_raw/<세션> --out /workspace/data/datasets/<이름>
```

### 3. SmolVLA 학습
```bash
# 호스트에서 실행 (8GB VRAM을 거의 다 쓰므로 QUVI를 먼저 내릴 것)
./scripts/smolvla_train.sh <데이터셋 이름> <출력 이름> [steps] [batch]
# 학습된 모델에서 이어서 학습: BASE_MODEL=<data/models 아래 경로> ./scripts/smolvla_train.sh ...
```
이미지 증강이 기본으로 켜져 있으며, `scripts/eval_vla_open_loop.py`로 녹화 데이터 대비 예측 오차와 떨림(반전율)을 평가합니다.

### 4. 추론 및 배포
HMI 대시보드에서 모델 스캔·선택으로 런타임 교체가 가능하며, **마지막으로 선택한 모델은 `data/act_last_model.json`에 저장되어 재기동 시 자동 로드·활성화**됩니다. 저장된 선택이 없으면 `act_model_path` 파라미터 기본 경로를 사용합니다. 이전 ACT 정책용 스크립트(`act_record.sh`, `act_train.sh`)도 남아 있습니다.

---

## PatchCore 이상탐지 학습 파이프라인

```bash
# 0. HMI 데이터셋 촬영 모드 또는 기존 PASS 검사 로그로 정상품 이미지 수집
python3 scripts/build_anomaly_dataset.py     # PASS 로그 → 각도별 raw/ 정리 + 검수 시트 생성
# (사람이 review_sheet_{angle}.png 를 보고 불량 혼입 이미지를 raw/에서 삭제)

# 1. 각도별 메모리뱅크 학습 + 임계값 산정
python3 scripts/train_anomaly_bank.py --product <품종>   # → data/inspection_products/<품종>/models/bank_{angle}.pt, thresholds.json

# 2. 룰 vs ML 일치율 리포트 (판정 신뢰도 점검)
python3 scripts/shadow_report.py
```

런치 인자 `anomaly_enabled`로 켜고 끄며, 모델 파일이 없거나 로드에 실패하면 자동 비활성되어 룰 판정만 사용합니다.

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

## 향후 계획

다음 목표는 검사 자동화가 아니라 **출력–회수–검사–재출력이 사람 없이 도는 무인 루프**입니다. 프린터 연동은 코드로 들어와 있으며, 실제 프린터를 움직이는 동작이라 기본값은 꺼 두었습니다.

| 단계 | 상태 |
| :--- | :--- |
| 출력 완료 감지 → 자동 시작 | 구현 (`printer_monitor_node` + Moonraker, `auto_start_on_print_done` 기본 false) |
| 재출력 지시 | 구현 — 불량 판정 후 마지막 gcode 재출력 요청 (`auto_reprint_on_fail` 기본 false) |
| 베드 착탈 | 계획 — 같은 자리·같은 궤적의 반복이라 로봇팔이 아니라 레일 수준의 전담 기구로 처리 |
| 불량품 회수 | 계획 — 걸러낸 것을 분쇄해 필라멘트로 되돌림 |

파지 쪽 다음 과제는 정책 출력 떨림을 줄이는 액션 스무딩과 그리퍼 빈손 판정의 실기 투입(현재 shadow 모드)입니다. 하드웨어 확장으로는 자석식 툴체인저(그리퍼 자동 교체), 접근 감지 정지(비상정지에 더한 안전층)를 계획하고 있습니다.

---

## 팀 정보

- **서울로봇고등학교 올라운더** — 대회를 목표로 활동하는 동아리. QUVI는 졸업작품으로 시작해 전국마이스터고 스타프로젝트 출품까지 이어진 프로젝트입니다.

---

## AI 사용 내역 (외부 사용 내역 공개)

투명성 원칙에 따라 본 프로젝트 제작에 활용한 AI, 오픈소스, 외부 도움을 공개합니다.

- **개발 지원 AI**: Anthropic **Claude Code** (Claude Fable 5·Opus·Sonnet·Haiku 모델), OpenAI **Codex**, Google **Antigravity** — 코드 작성·리뷰·디버깅 보조. 최종 설계 판단과 실기 검증은 팀이 직접 수행
- **제품에 탑재된 AI 모델**: **SmolVLA** 파지 정책 (사전학습 모델 `lerobot/smolvla_base`를 자체 텔레오퍼레이션 시연 데이터로 파인튜닝, 이전 버전은 **ACT** 정책을 직접 학습), **PatchCore** 이상탐지 (WideResNet50 백본, 자체 정상품 이미지로 메모리뱅크 구축)
- **오픈소스** (라이선스 종류 포함): ROS 2 Jazzy (Apache-2.0), micro-ROS (Apache-2.0), LeRobot (Apache-2.0), OpenCV (Apache-2.0), PyTorch (BSD-3-Clause), Flask (BSD-3-Clause), Flask-SocketIO (MIT), NumPy (BSD-3-Clause), DYNAMIXEL SDK (Apache-2.0), PlatformIO (Apache-2.0), ROBOTIS cyclo_intelligence (Apache-2.0, 일부 모듈 포함) — 상세 출처는 아래 [레퍼런스](#레퍼런스) 참조
- **발표·데모 영상 음성(AI TTS)**: 8주차 3분 데모 영상은 **edge-tts**(Microsoft Edge 신경망 음성, `ko-KR-HyunsuMultilingualNeural`), 피칭 영상과 4주차 3분 데모 영상(1차 데모데이)은 **Typecast**(typecast.ai)로 합성한 AI 음성을 사용
- **웹 데모 계측**: Google Analytics 4 — 공개 웹 데모의 방문·검사 사이클 완주 수를 집계. 개인 식별 정보는 수집하지 않음
- **외부 자문**: 없음 (지도교사 지도 외 외부 기관·업체 자문 없음)

---

## 레퍼런스

- **LeRobot** — Hugging Face의 로봇 모방학습 프레임워크. 본 프로젝트는 OMX 지원 브랜치인 [ROBOTIS-GIT/lerobot](https://github.com/ROBOTIS-GIT/lerobot) 포크를 서브모듈로 사용 (원본: [huggingface/lerobot](https://github.com/huggingface/lerobot))
- **SmolVLA** — Shukor et al., ["SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics"](https://arxiv.org/abs/2506.01844) (2025) — 현재 파지 정책, 베이스 모델 [lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base)
- **ACT** — Zhao et al., ["Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"](https://arxiv.org/abs/2304.13705) (RSS 2023) — 이전 파지 모방학습 정책
- **ROBOTIS OpenMANIPULATOR-X (OMX)** — [ROBOTIS-GIT/open_manipulator](https://github.com/ROBOTIS-GIT/open_manipulator) — 로봇팔 하드웨어 및 [DYNAMIXEL SDK](https://github.com/ROBOTIS-GIT/DynamixelSDK)
- **PatchCore** — Roth et al., ["Towards Total Recall in Industrial Anomaly Detection"](https://arxiv.org/abs/2106.08265) (CVPR 2022) — 이상탐지 알고리즘
- **ROBOTIS cyclo_intelligence** — [ROBOTIS-GIT/cyclo_intelligence](https://github.com/ROBOTIS-GIT/cyclo_intelligence) (Apache-2.0). `cyclo_brain/policy/lerobot/lerobot_engine/`의 추론용 모듈 4개(constants, image_preprocessing, loading, prediction)를 2026-09-08 기준 원본 그대로 `scripts/vla_sidecar/lerobot_engine/`에 포함. 각 파일의 ROBOTIS 저작권·Apache-2.0 헤더 유지. 이후 `image_preprocessing.py`만 중복 상수 정의를 `constants.py` import로 바꾸는 2줄 수정(2026-09-17, 커밋 5465f84)
- **micro-ROS** — [micro-ROS-Agent](https://github.com/micro-ROS/micro-ROS-Agent), [micro_ros_platformio](https://github.com/micro-ROS/micro_ros_platformio) — ESP32-S3 ↔ ROS 2 통신

---

## 라이선스

이 프로젝트는 [MIT License](LICENSE)에 따라 라이선스가 부여됩니다. 단, `scripts/vla_sidecar/lerobot_engine/`의 ROBOTIS 원본 파일은 [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)을 따릅니다.
