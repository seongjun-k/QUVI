#include "StepperMotor.h"

// 비상정지 플래그 (quvi_esp32_firmware.ino 정의). 호밍 무한루프 탈출용.
extern volatile bool isEmergencyStopped;

// 호밍 단계별 최대 대기 시간 (ms). 리밋 스위치 미검출 시 무한루프 방지.
#define HOMING_STAGE_TIMEOUT_MS 30000UL

StepperMotor::StepperMotor(int8_t pulPin, int8_t dirPin, int8_t enaPin, int8_t limitPin, bool invertDir)
    : _stepper(AccelStepper::DRIVER, pulPin, dirPin),
      _pulPin(pulPin), _dirPin(dirPin), _enaPin(enaPin), _limitPin(limitPin),
      _invertDir(invertDir), _enabled(false) {
}

void StepperMotor::begin() {
    if (_limitPin >= 0) {
        pinMode(_limitPin, INPUT_PULLUP);
    }
    if (_enaPin >= 0) {
        pinMode(_enaPin, OUTPUT);
        digitalWrite(_enaPin, HIGH); // 기본값 비활성화 (Active LOW)
    }
    // DIR 핀 극성 반전 (invertDir=true 시 호밍/주행 모두 적용)
    if (_invertDir) {
        _stepper.setPinsInverted(true, true);
    }
    _enabled = false;
}

void StepperMotor::enable() {
    if (_enaPin >= 0) {
        digitalWrite(_enaPin, LOW); // Active LOW (로우 신호로 활성화)
    }
    _enabled = true;
}

void StepperMotor::disable() {
    if (_enaPin >= 0) {
        digitalWrite(_enaPin, HIGH); // Active LOW (하이 신호로 비활성화)
    }
    _enabled = false;
    _stepper.stop();
}

bool StepperMotor::isEnabled() const {
    return _enabled;
}

// 목표 위치 설정 (steps 단위)
void StepperMotor::setTargetPosition(long target) {
    _stepper.moveTo(target);
}

// 현재 위치 강제 재설정
void StepperMotor::setCurrentPosition(long current) {
    _stepper.setCurrentPosition(current);
}

void StepperMotor::setMaxSpeed(float speed) {
    _stepper.setMaxSpeed(speed);
}

void StepperMotor::setAcceleration(float accel) {
    _stepper.setAcceleration(accel);
}

bool StepperMotor::isLimitPressed() {
    if (_limitPin < 0) return false;
    // Active LOW (GND에 연결되면 감지됨)
    return (digitalRead(_limitPin) == LOW);
}

// 논블로킹 업데이트 루프
bool StepperMotor::update() {
    if (!_enabled) {
        return false;
    }

    // 안전 체크: 리밋 스위치가 눌렸고 그 방향으로 이동 중이면 정지
    if (isLimitPressed()) {
        if (_stepper.distanceToGo() < 0) {
            _stepper.stop();
            _stepper.setCurrentPosition(0);
            return false;
        }
    }

    return _stepper.run();
}

// AccelStepper 기반 3단계 동기식 호밍 시퀀스
bool StepperMotor::home(bool homingDir, float coarseSpeed, float fineSpeed, long backoffSteps, float accel) {
    if (_limitPin < 0) return false;

    enable();

    // 호밍 방향: homingDir 이 true(HIGH) 면 양의 방향, false(LOW) 면 음의 방향으로 이동
    long directionMultiplier = homingDir ? 1 : -1;

    // ==========================================
    // 1단계: 코스 탐색 (스위치 방향으로 고속 이동)
    // ==========================================
    _stepper.setMaxSpeed(coarseSpeed);
    _stepper.setAcceleration(accel);
    _stepper.move(directionMultiplier * 100000); // 매우 큰 거리로 설정 (스위치에 닿을 때까지 이동)

    unsigned long lastFeed = millis();
    unsigned long stageStart = millis();
    while (!isLimitPressed()) {
        if (isEmergencyStopped || (millis() - stageStart >= HOMING_STAGE_TIMEOUT_MS)) {
            _stepper.stop();
            return false;  // 비상정지 또는 타임아웃 — 호밍 실패
        }
        _stepper.run();
        if (millis() - lastFeed >= 10) {
            delay(1);
            lastFeed = millis();
        } else {
            yield();
        }
    }
    _stepper.stop();
    _stepper.setCurrentPosition(0);
    delay(100); // 정정(settle) 대기

    // ==========================================
    // 2단계: 백오프 (스위치에서 멀어지는 방향으로 이동)
    // ==========================================
    _stepper.setMaxSpeed(fineSpeed);
    _stepper.move(-directionMultiplier * backoffSteps);
    lastFeed = millis();
    while (_stepper.distanceToGo() != 0) {
        if (isEmergencyStopped) {
            _stepper.stop();
            return false;  // 비상정지 — 호밍 실패
        }
        _stepper.run();
        if (millis() - lastFeed >= 10) {
            delay(1);
            lastFeed = millis();
        } else {
            yield();
        }
    }
    delay(100); // 정정(settle) 대기

    // ==========================================
    // 3단계: 파인 탐색 (스위치 방향으로 저속 이동)
    // ==========================================
    _stepper.move(directionMultiplier * backoffSteps * 2);
    lastFeed = millis();
    stageStart = millis();
    while (!isLimitPressed()) {
        if (isEmergencyStopped || (millis() - stageStart >= HOMING_STAGE_TIMEOUT_MS)) {
            _stepper.stop();
            return false;  // 비상정지 또는 타임아웃 — 호밍 실패
        }
        _stepper.run();
        if (millis() - lastFeed >= 10) {
            delay(1);
            lastFeed = millis();
        } else {
            yield();
        }
    }
    _stepper.stop();
    _stepper.setCurrentPosition(0);
    delay(50);

    // 리밋 스위치가 계속 눌린 상태로 남지 않도록 살짝 후퇴
    _stepper.move(-directionMultiplier * 50);
    lastFeed = millis();
    while (_stepper.distanceToGo() != 0) {
        if (isEmergencyStopped) {
            _stepper.stop();
            return false;  // 비상정지 — 호밍 실패
        }
        _stepper.run();
        if (millis() - lastFeed >= 10) {
            delay(1);
            lastFeed = millis();
        } else {
            yield();
        }
    }
    _stepper.setCurrentPosition(50); // 실제 위치 오프셋 반영

    return true;
}

long StepperMotor::getCurrentPosition() const {
    return _stepper.currentPosition();
}

long StepperMotor::getTargetPosition() const {
    return _stepper.targetPosition();
}

float StepperMotor::getCurrentSpeed() const {
    return _stepper.speed();
}

bool StepperMotor::isMoving() const {
    return _stepper.distanceToGo() != 0;
}
