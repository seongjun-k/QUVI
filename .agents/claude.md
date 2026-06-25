# QUVI — Claude Space Instructions

## Project

자동화 3D 프린팅 불량 검사 시스템 (OMX 로봇팔 + 리니어 레일 + AI 비전)

- OS: Ubuntu 24.04 / ROS 2 Jazzy
- HW: ROBOTIS OMX, XL430/XL330, ESP32-S3, Linear Rail
- Stack: YOLOv8, LeRobot ACT, micro-ROS, Flask + SocketIO

---

## Response Rules

- 한국어로만 답변
- 이모티콘, 인사말, 칭찬, 팀명 언급 금지
- 코드/명령 전 반드시 실행 경로 명시
- 다단계 작업은 계획 먼저 제시 후 실행

---

## Key Files

| 파일 | 역할 |
|------|------|
| `hmi_node.py` | ROS 2 HMI 노드, FSM 로직, `RAIL_STATION_MAP` 상수 |
| `web/dashboard.html` | UI 마크업, JS id 레퍼런스의 권위 소스 |
| `web/dashboard.js` | SocketIO 이벤트, UI 상태 렌더링 |
| `robot_control_node.py` | OMX 로봇 제어 노드 |

실제 경로는 `github.com/seongjun-k/QUVI` (master) 기준. 파일명 변경 전 반드시 확인.

---

## ROS 2 Interface

| Topic/Service | Name | Direction |
|---|---|---|
| HMI command | `/hmi/command` | publish |
| HMI status | `/hmi/status` | subscribe |

- 토픽/서비스명 변경 전 확인 필수
- FSM 상태 변경 시 `hmi_node.py`, `dashboard.js`, `dashboard.html` 동시 수정

---

## FSM

- 단일 진실 소스: `hmi_node.py` 상수
- 신규 상태 추가 시 3개 파일 동시 수정
- JS UI 상태값: `ok` / `warn` / `bad` 패턴만 사용

---

## Coding Rules

1. **코딩 전 사고**
   - 가정을 명시적으로 서술
   - 해석이 모호하면 코드 전 질문
   - `RAIL_STATION_MAP` step 값: `hmi_node.py` 상수만 수정

2. **단순성 우선**
   - 요청하지 않은 기능/추상화/예외처리 추가 금지
   - Flask route는 `create_flask_app()` 내부에만 추가
   - `threading` async_mode 사용 중 — `time.sleep` 사용, `socketio.sleep` 금지

3. **외과적 수정**
   - 무관한 코드/주석/포매팅 수정 금지
   - 스타일 유지: 한국어 주석, `# ─── Section ───` 구분선
   - 본인이 추가한 orphan 코드만 제거
   - `dashboard.html` 수정 시 JS id 레퍼런스 유지
   - `_system_status` dict 변경은 `get_status()` 출력에 직접 영향

4. **목표 중심 실행**
   - 다단계 작업은 계획 먼저
