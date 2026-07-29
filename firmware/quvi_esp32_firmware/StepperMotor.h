#ifndef STEPPER_MOTOR_H
#define STEPPER_MOTOR_H

#include <Arduino.h>
#include <AccelStepper.h>

class StepperMotor {
public:
    // invertDir=true 시 DIR 핀 극성 반전 (호밍/주행 모두 적용)
    StepperMotor(int8_t pulPin, int8_t dirPin, int8_t enaPin, int8_t limitPin = -1, bool invertDir = false);

    // 초기화
    void begin();

    // 모터 제어 명령
    void setTargetPosition(long target);
    void setCurrentPosition(long current);
    void setMaxSpeed(float speed);
    void setAcceleration(float accel);

    // 모터 활성화/비활성화
    void enable();
    void disable();
    bool isEnabled() const;

    // 이동 업데이트
    // 가능한 한 자주(예: 고속 루프에서) 호출해야 한다
    // 목표까지 아직 이동 중이면 true 반환
    bool update();

    // 호밍 시퀀스
    // 절대 캘리브레이션 보장을 위해 동기적으로 수행된다
    // accel: 코스 탐색 가속도 (steps/s²) — 낮으면 코스 속도 도달까지 레일
    // 대부분을 가속 구간으로 소모해 호밍이 느리게 출발한다.
    bool home(bool homingDir, float coarseSpeed, float fineSpeed, long backoffSteps, float accel);

    // 리밋 스위치 조회
    bool isLimitPressed();

    // 위치 접근자
    long getCurrentPosition() const;
    long getTargetPosition() const;
    float getCurrentSpeed() const;
    bool isMoving() const;

private:
    mutable AccelStepper _stepper; // mutable: AccelStepper 액세서가 non-const이므로 const 멤버에서 호출 가능하도록 선언
    int8_t _pulPin;
    int8_t _dirPin;
    int8_t _enaPin;
    int8_t _limitPin;
    bool   _invertDir;
    bool   _enabled;
};

#endif // STEPPER_MOTOR_H
