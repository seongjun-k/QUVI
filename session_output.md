╭─── Claude Code v2.1.181 ─────────────────────────────────────────────────────╮
│                                                    │ Tips for getting        │
│                    Welcome back!                   │ started                 │
│                                                    │ Run /init to create a … │
│                       ▐▛███▜▌                      │ ─────────────────────── │
│                      ▝▜█████▛▘                     │ What's new              │
│                        ▘▘ ▝▝                       │ Check the Claude Code … │
│                                                    │                         │
│   deepseek-ai/deepseek-v4-pro · API Usage Billing  │                         │
│                       ~/QUVI                       │                         │
╰──────────────────────────────────────────────────────────────────────────────╯


❯ QUVI 디렉토리에 있는 코드를 리뷰해줘 절대 수정 금지                           

● 전체 코드베이스를 리뷰하기 위해 프로젝트 구조를 먼저

  Searched for 2 patterns (ctrl+o to expand)

● 이제 주요 소스 파일들을 병렬로 읽

  Read 12 files (ctrl+o to expand)

● 이

  Read 1 file (ctrl+o to expand)

● 이제 launch 파일들과 설정 파일들을 확인하겠

  Read 3 files (ctrl+o to expand)

● 이제 나머지 파일들도 확인하겠

  Read 7 files (ctrl+o to expand)

● 모든 주요 소스 파일을 검토했습니다. 이제 종합적인 코드 리뷰를 제공하겠습니다.

  ---
  QUVI 프로젝트 코드 리뷰

  프로젝트 개요

  QUVI는 3D 프린터 출력물 자동 양불 판정 시스템으로, YOLO 기반 객체 탐지 →
  로봇팔 파지(ACT 모방학습) → 4방향 턴테이블 검사(SSIM+표면분석) → 분류 레일
  적재를 FSM으로 자동화하는 ROS 2 기반 풀스택 프로젝트입니다.

  ---
  전체 평가: ★★★★☆ (4/5)

  전반적으로 매우 잘 구조화된 프로덕션급 로봇 소프트웨어입니다. 아래에서 강점과
  개선 포인트를 상세히 다룹니다.

  ---
  1. 아키텍처 설계 — 강점

  ✅ 명확한 관심사 분리 (Separation of Concerns)

  각 노드의 역할이 명확히 분리되어 있습니다:

  ┌────────────────────────┬─────────────────────┬───────────────────────────┐
  │          노드          │        책임         │           파일            │
  ├────────────────────────┼─────────────────────┼───────────────────────────┤
  │ main_orchestrator_node │ FSM 기반 전체       │ main_orchestrator_node.py │
  │                        │ 시퀀스 조율         │                           │
  ├────────────────────────┼─────────────────────┼───────────────────────────┤
  │                        │ 로봇팔 + 레일 +     │                           │
  │ robot_control_node     │ 턴테이블 하드웨어   │ robot_control_node.py     │
  │                        │ 제어                │                           │
  ├────────────────────────┼─────────────────────┼───────────────────────────┤
  │ yolo_node              │ YOLO 객체 탐지      │ yolo_node.py              │
  ├────────────────────────┼─────────────────────┼───────────────────────────┤
  │ inspect_node           │ SSIM + 표면특징     │ inspect_node.py           │
  │                        │ 품질 검사           │                           │
  ├────────────────────────┼─────────────────────┼───────────────────────────┤
  │ hmi_node               │ Flask Web HMI       │ hmi_node.py               │
  │                        │ 대시보드            │                           │
  └────────────────────────┴─────────────────────┴───────────────────────────┘

  ✅ FSM 패턴이 깔끔하게 구현됨

  FsmState Enum을 통한 명시적 상태 정의, _fsm_loop()의 직관적인 if-elif 체인,
  상태 전이 로깅이 잘 되어 있습니다. 상태 전이 시 디버깅이 용이합니다.

  ✅ 캐싱을 통한 성능 최적화

  BinaryCache 클래스(utils.py:65-236)의 설계가 훌륭합니다. 이진화/윤곽 검출
  결과를 캐싱하고 holes()는 lazy 초기화로 필요할 때만 RETR_TREE를 실행합니다.
  __slots__ 사용으로 메모리 효율도 챙겼습니다.

  ✅ 퇴화 케이스(fallback) 처리

  - SSIM 계산 시 scikit-image → OpenCV 폴백 (inspect_node.py:418-425)
  - 정렬 후 SSIM이 오히려 악화되면 원본 폴백 (inspect_node.py:396-405)
  - YOLO 모델 없으면 기본 yolov8n.pt 사용 (yolo_node.py:95)
  - 각종 타임아웃으로 무한 대기 방지 (FSM 각 상태에 timeout 적용)

  ---
  2. 코드 품질 — 강점

  ✅ 일관된 한국어 주석 스타일

  전체 코드베이스가 한국어로 일관성 있게 문서화되어 있습니다. 특히
  robot_control_node.py의 docstring(1-49행)은 ROS 인터페이스 전체를 한눈에
  파악할 수 있게 해줍니다.

  ✅ Thread-safety 고려

  - robot_control_node.py의 _state_lock, _handcam_lock, _dxl_io_lock으로
  멀티스레드 환경에서의 데이터 경쟁 방지
  - hmi_node.py의 _lock으로 공유 상태 보호
  - MultiThreadedExecutor 사용으로 블로킹 서비스와 타이머의 상호 간섭 방지

  ✅ 파라미터 관리

  declare_and_get() 유틸리티(utils.py:50-53)가 파라미터 선언과 로드를 한 줄로
  줄여줍니다. Launch 파일에서 override 가능한 구조도 잘 갖춰져 있습니다.

  ✅ 정적 무결성 테스트 (test_config_integrity.py)

  패키지 빌드 설정(ament_python 일관성), launch 파일 절대경로 하드코딩 금지,
  micro-ROS 보드레이트 일관성 등을 정적으로 검증하는 테스트가 인상적입니다.
  "실수로 잘못된 설정이 배포되는 것"을 코드 레벨에서 방지합니다.

  ✅ 소프트웨어 정렬(Software Alignment)

  BinaryCache.get_aligned_roi()의 minAreaRect 기반 역회전 정렬이 잘 구현되어
  있습니다. 검사 대상 출력물이 약간 비뚤어져 배치되어도 SSIM 비교 정확도를 높일
  수 있습니다. 이진 이미지가 아닌 원본 그레이스케일에 warpAffine을 적용하는
  선택도 올바릅니다(경계 아티팩트 방지).

  ---
  3. 개선 포인트 (중요도 순)

  🔴 HIGH — inspect_params.yaml 누락된 alignment_enabled 파라미터

  inspect_node.py:120에서 'alignment_enabled' 파라미터를 선언하지만,
  inspect_params.yaml과 vision_pipeline.launch.py의 파라미터 목록에는 이 항목이
  없습니다. 기본값 True로 동작하겠지만, 명시적 설정이 불가능한 상태입니다.

  🔴 HIGH — 컨테이너 시작 실패 핸들링 누락

  build.sh와 run.sh에서 docker compose up -d 실패 시 exit 1로 종료하지만, docker
  exec 실패 시에는 별도 핸들링이 없습니다. docker exec가 실패해도 스크립트가
  성공 종료 코드를 반환할 수 있습니다.

  🟡 MEDIUM — destroy_node의 순서 문제

  robot_control_node.py:844-856에서 destroy_node가 super().destroy_node()를
  호출하기 전에 텔레옵 정지와 follower 해제를 수행하지만, destroy_node가 여러 번
  호출될 경우(SIGINT → rclpy.shutdown → destroy_node) follower.disconnect가 두
  번 호출될 수 있습니다. idempotency 가드가 있으면 좋겠습니다.

  🟡 MEDIUM — build.sh가 대화형에 의존

  build.sh:31의 docker exec -it가 -it 플래그를 사용하기 때문에 CI/CD 환경(TTY
  없음)에서는 실패합니다. CI 환경 감지 후 -i만 사용하는 분기가 필요할 수
  있습니다.

  🟡 MEDIUM — hmi_node.py의 time.sleep(0.1) WebSocket 루프

  hmi_node.py:457의 _ws_broadcast()가 100ms 주기로 폴링하는데, 이벤트
  기반(eventlet/gevent의 sleep(0))이나 socketio.sleep(0)으로 변경하면 CPU
  사용량을 줄일 수 있습니다. 또한 _ws_broadcast 루프가 rclpy.ok() 체크 없이 무한
  루프를 돌기 때문에, ROS 2 shutdown 후에도 계속 실행될 수 있습니다.

  🟡 MEDIUM — inspect_node.py의 _current_object_index 동기화

  inspect_node.py:171-173에서 _grasp_cmd_callback으로 object_index를
  동기화하지만, inspect_node는 턴테이블 회전 후 검사를 수행하므로 이 값이 실제
  검사 대상과 불일치할 가능성이 있습니다. 오케스트레이터가 검사 트리거와 함께
  현재 인덱스를 전달하는 것이 더 견고합니다.

  🟡 MEDIUM — 로깅 포맷 불일치

  - main_orchestrator_node.py:167: self.get_logger().info(f'HMI 명령 수신: 
  {command}') — f-string
  - robot_control_node.py:201-206: f-string 사용
  - inspect_node.py:257-263: f-string 내장

  일부 로그에서는 f-string, 일부에서는 % 포맷을 사용하고 있어 일관성이 약간
  떨어집니다. (ROS 2 logger는 % 포맷을 권장하지만 심각한 문제는 아닙니다.)

  🟢 LOW — train_yolo.py의 input() 사용

  train_yolo.py:70-72에서 input()으로 대화형 확인을 받습니다. CI/CD
  파이프라인에서 실행할 수 없습니다. --yes 또는 --no-interactive 플래그를
  추가하는 것이 좋습니다.

  🟢 LOW — POSE_BACK == POSE_FRONT

  robot_control_node.py:124-127에서 POSE_BACK과 POSE_FRONT가 완전히 동일한
  값입니다. 노드 자체에서도 _execute_pose에서 이 점을 감지해 경고를
  출력하지만(:716-717), 실제 회전 동작을 위해선 다른 값으로 설정되어야 합니다.
  주석에 "캘리브레이션이 필요합니다"라고 명시된 점은 좋습니다.

  🟢 LOW — YOLO 탐지 후 _detection_enabled 자동 비활성화

  yolo_node.py:228에서 한 번 탐지를 수행한 후 _detection_enabled = False로
  설정하는데, 이는 FSM이 매번 trigger를 다시 보내도록 설계된 의도입니다. 이
  동작이 명시적으로 문서화되면 좋겠습니다.

  🟢 LOW — ESP32 펌웨어 home()의 watchdog feed

  StepperMotor.cpp:102-137의 home() 함수에서 yield()를 호출해 ESP32 watchdog을
  피드하고 있지만, yield()만으로는 충분하지 않을 수 있습니다. 공식적으로
  esp_task_wdt_reset() 또는 vTaskDelay(1)을 사용하는 것이 더 안전합니다.

  ---
  4. 잠재적 버그

  🐛 inspect_node.py:493 — 정렬 후 area_ratio 계산 시 참조 이미지 크기 불일치

  # inspect_node.py:488-492
  if ref is not None:
      ref_resized = cv2.resize(ref, (gray.shape[1], gray.shape[0]))  # gray 
  기준으로 리사이즈
      ref_area = BinaryCache(ref_resized,
  self._bin_thresh).largest_external_area()
      cap_area = cache.largest_external_area()  # cache는 정렬된 이미지 기준

  정렬이 적용된 경우 cache는 aligned 이미지에서 생성되므로 크기가 gray와
  다릅니다. 하지만 ref_resized는 여전히 gray.shape 기준으로 리사이즈되어
  area_ratio 계산에 크기 불일치가 발생할 수 있습니다. aligned.shape에 맞춰 ref를
  리사이즈해야 합니다.

  🐛 main_orchestrator_node.py:471-475 — count_publishers 사용

  _publish_hmi_status()에서 count_publishers()를 사용해 노드 온라인 상태를
  감지하는데, 이는 현재 노드가 구독 중인 토픽의 publisher 수만 반환합니다.
  main_orchestrator_node는 /robot/joint_states를 구독하지 않으므로 항상 0이
  반환됩니다. 백업으로 self._grasp_online / self._motor_online 플래그를
  사용하지만, or 조건(or self._grasp_online)이 count_publishers > 0 결과를
  덮어쓰기 때문에 실질적으로는 플래그만 유효합니다.

  ---
  5. 테스트 커버리지

  테스트 파일이 3개 존재하며, 목적이 분명합니다:

  ┌────────────────────────────┬─────────────────────────────────────────────┐
  │        테스트 파일         │                  검증 대상                  │
  ├────────────────────────────┼─────────────────────────────────────────────┤
  │ test_config_integrity.py   │ 빌드 설정, launch 파일, 보드레이트 일관성   │
  ├────────────────────────────┼─────────────────────────────────────────────┤
  │ test_orchestrator_logic.py │ 픽셀→로봇좌표 변환, HMI 트리거 안전 가드    │
  ├────────────────────────────┼─────────────────────────────────────────────┤
  │ test_alignment.py          │ BinaryCache 정렬 파이프라인 (7개 테스트     │
  │                            │ 케이스)                                     │
  └────────────────────────────┴─────────────────────────────────────────────┘

  test_alignment.py는 ROS 없이 순수 OpenCV+numpy+pytest로
  BinaryCache.get_aligned_roi()의 다양한 시나리오(정상 회전, 빈 프레임, 경계
  객체, 다중 윤곽, 종횡비, kill switch, 캐시 일관성)를 검증합니다. ROS 2 없이
  핵심 알고리즘을 독립 테스트하는 접근법이 매우 훌륭합니다.

  부족한 점: FSM 상태 전이, ACT 파지, 검사 판정 로직에 대한 통합 테스트가
  없습니다. 실제 하드웨어 없이는 어렵지만, 모의 토픽을 사용한 ROS 2 통합
  테스트가 추가되면 좋겠습니다.

  ---
  6. 요약

  ┌───────────────┬────────────────────────────────────────────────┐
  │     항목      │                      평가                      │
  ├───────────────┼────────────────────────────────────────────────┤
  │ 아키텍처 설계 │ ★★★★★ 명확한 관심사 분리, FSM 패턴, 모듈성     │
  ├───────────────┼────────────────────────────────────────────────┤
  │ 코드 가독성   │ ★★★★☆ 일관된 한국어 주석, 직관적 변수명        │
  ├───────────────┼────────────────────────────────────────────────┤
  │ 에러 처리     │ ★★★★☆ 타임아웃, 폴백, ESTOP 체계적 적용        │
  ├───────────────┼────────────────────────────────────────────────┤
  │ 성능 최적화   │ ★★★★☆ BinaryCache 캐싱, lazy 초기화, sqrt 회피 │
  ├───────────────┼────────────────────────────────────────────────┤
  │ 테스트        │ ★★★☆☆ 단위 테스트 우수, 통합 테스트 부재       │
  ├───────────────┼────────────────────────────────────────────────┤
  │ 설정 관리     │ ★★★★☆ 파라미터/launch 구조화, 정적 무결성 검증 │
  ├───────────────┼────────────────────────────────────────────────┤
  │ Thread-safety │ ★★★★☆ Lock 기반 보호, MultiThreadedExecutor    │
  ├───────────────┼────────────────────────────────────────────────┤
  │ 배포 용이성   │ ★★★★☆ Docker Compose, 심볼릭 링크 빌드         │
  └───────────────┴────────────────────────────────────────────────┘

  결론: QUVI는 전반적으로 높은 품질의 ROS 2 로봇 소프트웨어입니다. FSM 기반
  오케스트레이션, 소프트웨어 정렬, 캐싱 최적화, 정적 무결성 테스트 등 모범
  사례가 잘 적용되어 있습니다. 위에서 지적한 개선 포인트는 대부분 엣지 케이스나
  마이너한 이슈로, 시스템의 핵심 기능에는 문제가 없습니다. 특히
  inspect_node.py:493의 정렬 후 area_ratio 계산과 inspect_params.yaml의 누락된
  alignment_enabled 파라미터는 조속히 수정하는 것이 좋겠습니다.

✻ Cogitated for 7m 10s

※ recap: QUVI 코드베이스 전체 리뷰를 완료했습니다. 전반적으로 잘 설계된 프로젝트이며, 주요 발견사항으로 `inspect_node.py`의 정렬 후 area_ratio 계산 버그(493행)와 `inspect_params.yaml`에 `alignment_enabled` 파라미터 누락이 확인되었습니다. 계속 진행하려면 말씀해주세요. (disable recaps in /config)

❯ /export session_output.md'                                                    
  ⎿  Conversation exported to: /home/ksj/QUVI/session_output.md'