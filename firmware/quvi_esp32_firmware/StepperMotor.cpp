#include "StepperMotor.h"

// Constructor
// dirPin/pulPin 순서 교환으로 AccelStepper 하드웨어 방향 반전
StepperMotor::StepperMotor(int8_t pulPin, int8_t dirPin, int8_t enaPin, int8_t limitPin)
    : _stepper(AccelStepper::DRIVER, pulPin, dirPin),
      _pulPin(pulPin), _dirPin(dirPin), _enaPin(enaPin), _limitPin(limitPin), _enabled(false) {
}

// Initialize GPIOs
void StepperMotor::begin() {
    if (_limitPin >= 0) {
        pinMode(_limitPin, INPUT_PULLUP);
    }
    if (_enaPin >= 0) {
        pinMode(_enaPin, OUTPUT);
        digitalWrite(_enaPin, HIGH); // Disable by default (Active LOW)
    }
    _enabled = false;
}

// Enable Motor
void StepperMotor::enable() {
    if (_enaPin >= 0) {
        digitalWrite(_enaPin, LOW); // Active LOW
    }
    _enabled = true;
}

// Disable Motor
void StepperMotor::disable() {
    if (_enaPin >= 0) {
        digitalWrite(_enaPin, HIGH); // Active LOW
    }
    _enabled = false;
    _stepper.stop();
}

bool StepperMotor::isEnabled() const {
    return _enabled;
}

// Set target position (steps)
void StepperMotor::setTargetPosition(long target) {
    _stepper.moveTo(target);
}

// Force-override current position
void StepperMotor::setCurrentPosition(long current) {
    _stepper.setCurrentPosition(current);
}

void StepperMotor::setMaxSpeed(float speed) {
    _stepper.setMaxSpeed(speed);
}

void StepperMotor::setAcceleration(float accel) {
    _stepper.setAcceleration(accel);
}

// Read limit switch status
bool StepperMotor::isLimitPressed() {
    if (_limitPin < 0) return false;
    // Active LOW (triggered when connected to GND)
    return (digitalRead(_limitPin) == LOW);
}

// Non-blocking update loop
bool StepperMotor::update() {
    if (!_enabled) {
        return false;
    }

    // Safety check: if limit switch is pressed and we are moving towards it, stop.
    if (isLimitPressed()) {
        if (_stepper.distanceToGo() < 0) {
            _stepper.stop();
            _stepper.setCurrentPosition(0);
            return false;
        }
    }

    return _stepper.run();
}

// Robust 3-stage synchronous Homing Sequence using AccelStepper
bool StepperMotor::home(bool homingDir, float coarseSpeed, float fineSpeed, long backoffSteps) {
    if (_limitPin < 0) return false;

    enable();
    
    // Homing direction: if homingDir is true (HIGH), we move in positive direction.
    // If false (LOW), we move in negative direction.
    long directionMultiplier = homingDir ? 1 : -1;

    // ==========================================
    // STAGE 1: Coarse search (Fast towards switch)
    // ==========================================
    _stepper.setMaxSpeed(coarseSpeed);
    _stepper.setAcceleration(500.0);
    _stepper.move(directionMultiplier * 100000); // Move a very large distance

    unsigned long lastFeed = millis();
    while (!isLimitPressed()) {
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
    delay(100); // Settle

    // ==========================================
    // STAGE 2: Back-off (Move away from switch)
    // ==========================================
    _stepper.setMaxSpeed(fineSpeed);
    _stepper.move(-directionMultiplier * backoffSteps);
    lastFeed = millis();
    while (_stepper.distanceToGo() != 0) {
        _stepper.run();
        if (millis() - lastFeed >= 10) {
            delay(1);
            lastFeed = millis();
        } else {
            yield();
        }
    }
    delay(100); // Settle

    // ==========================================
    // STAGE 3: Fine search (Slow towards switch)
    // ==========================================
    _stepper.move(directionMultiplier * backoffSteps * 2);
    lastFeed = millis();
    while (!isLimitPressed()) {
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

    // Back off slightly from limit switch to prevent constant triggering
    _stepper.move(-directionMultiplier * 50);
    lastFeed = millis();
    while (_stepper.distanceToGo() != 0) {
        _stepper.run();
        if (millis() - lastFeed >= 10) {
            delay(1);
            lastFeed = millis();
        } else {
            yield();
        }
    }
    _stepper.setCurrentPosition(50); // Reflect actual position offset

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
