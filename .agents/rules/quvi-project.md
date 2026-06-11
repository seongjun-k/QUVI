---
activation: alwaysOn
---
# QUVI AI 에이전트 규칙

## 응답 스타일
- 한국어, 이모티콘 금지, 인사말/서론/결론 생략
- 코드와 핵심 사실만, 실행 경로 항상 명시

## 코딩 원칙
- 불명확한 사양은 가정하지 않고 질문
- 여러 구현 방향이 있으면 트레이드오프 제시 후 동의 요청
- 불필요한 추상화, 과잉 설계, 투기적 예외 처리 금지
- 최소 블록 단위 수정, 관련 없는 코드 손대지 않음
- 수정으로 미사용된 import/변수/함수 즉시 제거

## 검증
- 코드 변경 후 반드시 빌드 확인: `cd /workspace && colcon build --symlink-install`
- 실패 시 추측 금지, 실제 에러 로그 확인

## QUVI 개발 환경
| 항목 | 값 |
|---|---|
| OS / 미들웨어 | Ubuntu 24.04 / ROS 2 Jazzy Jalisco |
| 실행 환경 | Docker quvi-dev `/workspace` |
| ESP32-S3 | `/dev/ttyUSB0`, micro-ROS, 921600 baud |
| 핸드캠 (Zone 1) | `/dev/video0`, 1920x1080 MJPEG, ACT 입력 640x480, AutoFocus OFF |
| 사이드캠 (Zone 2) | `/dev/video2`, 1920x1080 MJPEG, AutoFocus OFF, Focus=80 |
| 리니어 레일 | ESP32-S3 제어, 0.2mm/스텝 |
| 턴테이블 | NEMA17, 90° = 400스텝 (1/8 마이크로스텝) |
| 로봇팔 | OMX 5DOF, Dynamixel XL430/XL330, LeRobot ACT 연동 |
| 토픽/서비스 | snake_case |

## 실행 명령어
```bash
# 전체 시스템
ros2 launch quvi_bringup full_system.launch.py

# 비전 파이프라인
ros2 launch quvi_bringup vision_pipeline.launch.py
```

