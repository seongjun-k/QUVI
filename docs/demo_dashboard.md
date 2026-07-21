# 데모 대시보드 — 전체 뷰 영상 패널 테스트 가이드

`demo/dashboard` 브랜치 전용. 대시보드 카메라 그리드 아래 줄에 [로봇 전체 뷰 (1/3) | ACT 추론 rerun (2/3)] 패널이 추가됐다.

## 동작 방식

- `src/quvi_hmi/quvi_hmi/static/demo/robot_overview.mp4` 파일이 **있으면**: 전체 뷰 패널 표시(자동재생·무음·루프), rerun 패널은 오른쪽 2칸으로 축소. 첫 접속 시 데모 안내 모달 표시(패널 설명 + "▶ 시작" 버튼 안내, 세션당 1회 — sessionStorage `demoNoticeSeen`).
- 파일이 **없으면**: 패널 숨김, rerun 전체 폭 — 기존 레이아웃 그대로 (평시 실기 운용에 영향 없음).
- 영상은 **세로(9:16) 촬영 기준**으로 배치돼 있다. 가로 영상을 넣으면 위아래 검은 여백이 생긴다.

## 테스트 절차 (호스트에서)

```bash
# 1. 영상 배치 (세로 촬영 mp4)
cp <촬영본>.mp4 src/quvi_hmi/quvi_hmi/static/demo/robot_overview.mp4

# 2. 빌드 — static/demo/ 나 demo.launch.py 가 처음 생긴 경우에만 필요
./build.sh

# 3. 데모 실행(UI 전용) 후 http://localhost:5000 접속
./demo.sh
```

`demo.sh` → `demo.launch.py` 는 **hmi_node만** 기동한다. 로봇 구동 계열(robot_control,
orchestrator, micro-ROS, vision, inspect)은 띄우지 않는다 — 카메라 패널은 No Signal이
정상이고, 추후 bag 재생이 해당 토픽을 공급한다. 실기 전체 기동은 기존 `./run.sh`.

확인 항목:

1. 전체 뷰 패널이 보이고 영상이 루프 재생되는가
2. rerun 패널이 오른쪽 2칸에 하단(log_time)까지 채워지는가
3. 첫 접속 시 안내 모달이 뜨고, "둘러보기 시작"으로 닫은 뒤 새로고침하면 다시 뜨지 않는가 (새 시크릿 창에서는 다시 떠야 정상)
4. mp4를 지우고 새로고침하면 패널·모달이 사라지고 rerun이 전체 폭으로 복귀하는가

## 주의

- **HTML(`dashboard.html`) 수정 후에는 HMI 재시작 필수** — Flask가 시작 시점 템플릿을 캐시한다. JS/CSS는 새로고침만으로 반영.
- 이전 런치가 살아 있으면 포트 5000/9877 충돌로 새 hmi_node가 조용히 죽는다. 재실행 전 기존 `run.sh` 세션을 Ctrl-C로 종료할 것.
- 테스트 mp4는 커밋하지 않는다 (플레이스홀더는 `ffmpeg -f lavfi -i testsrc2=size=360x640:rate=15:duration=3 -pix_fmt yuv420p <경로>`로 생성 가능).

## 남은 작업 (데모 완성까지)

1. 실기 테이크 2개 녹화: **양품 1사이클 + 불량품 1사이클** 각각 `ros2 bag record`
   (카메라·`/hmi/status` 등 대시보드 구독 토픽) + rerun `.rrd` 저장 — 같은 런을 동시에 기록
2. 데모 컨트롤러: HMI "▶ 시작" 토픽 구독 → 누를 때마다 PASS/FAIL bag 을 교대로
   `ros2 bag play` (재생 중 재클릭 무시). demo.launch.py 에 추가
3. rerun rrd 재서빙(설치 버전의 CLI 옵션 확인 필요)을 demo.launch.py 에 추가
4. 폰 **세로** 촬영본(양품+불량 두 사이클 이어붙인 한 편)으로 `robot_overview.mp4` 교체
