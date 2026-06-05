#include "StepperMotor.h"

// Constructor
StepperMotor::StepperMotor(uint8_t pulPin, uint8_t dirPin, uint8_t enaPin, int8_t limitPin) {
    _pulPin = pulPin;
    _dirPin = dirPin;
    _enaPin = enaPin;
    _limitPin = limitPin;

    _currentPos = 0;
    _targetPos = 0;
    _maxSpeed = 1000.0;
    _accel = 1000.0;
    _speed = 0.0;
    _c0 = 0;
    _cn = 0;
    _minDelay = 0;
    _n = 0;
    _decelSteps = 0;
    _totalSteps = 0;
    _stepsMoved = 0;
    _lastStepTime = 0;
    _enabled = false;
}

// Initialize GPIOs
void StepperMotor::begin() {
    pinMode(_pulPin, OUTPUT);
    pinMode(_dirPin, OUTPUT);
    pinMode(_enaPin, OUTPUT);
    
    digitalWrite(_pulPin, HIGH); // Common Anode: Active LOW, so write HIGH for idle (inactive)
    digitalWrite(_dirPin, LOW);
    
    // Disable by default (Active LOW for TB6600, so write HIGH to disable)
    disable();

    if (_limitPin >= 0) {
        pinMode(_limitPin, INPUT_PULLUP);
    }
}

// Enable Motor (Active LOW on TB6600)
void StepperMotor::enable() {
    digitalWrite(_enaPin, LOW);
    _enabled = true;
    delayMicroseconds(5); // Driver turn-on stabilization delay
}

// Disable Motor (Active LOW on TB6600)
void StepperMotor::disable() {
    digitalWrite(_enaPin, HIGH);
    _enabled = false;
    _speed = 0.0;
    _n = 0;
}

bool StepperMotor::isEnabled() const {
    return _enabled;
}

// Set target position (steps)
void StepperMotor::setTargetPosition(long target) {
    if (_targetPos == target) return;

    _targetPos = target;
    resetAcceleration();
}

// Force-override current position
void StepperMotor::setCurrentPosition(long current) {
    _currentPos = current;
    _targetPos = current;
    _speed = 0.0;
    _n = 0;
}

void StepperMotor::setMaxSpeed(float speed) {
    if (speed > 0.0) {
        _maxSpeed = speed;
        _minDelay = (unsigned long)(1000000.0 / _maxSpeed);
    }
}

void StepperMotor::setAcceleration(float accel) {
    if (accel > 0.0) {
        _accel = accel;
    }
}

// Reset motion parameters for a new target
void StepperMotor::resetAcceleration() {
    long stepsToMove = _targetPos - _currentPos;
    _totalSteps = abs(stepsToMove);
    _stepsMoved = 0;
    
    if (_totalSteps == 0) {
        _speed = 0.0;
        _n = 0;
        return;
    }

    // Set Direction Pin
    digitalWrite(_dirPin, (stepsToMove > 0) ? HIGH : LOW);

    // Initial step delay calculated from acceleration:
    // c0 = 1,000,000 * 0.676 * sqrt(2 / accel)
    _c0 = (unsigned long)(676000.0 * sqrt(2.0 / _accel));
    _cn = _c0;
    
    // Calculate the number of steps required to decelerate to a stop
    // decelSteps = speed_max^2 / (2 * accel)
    _decelSteps = (long)((_maxSpeed * _maxSpeed) / (2.0 * _accel));

    // If the movement distance is too short to reach maximum speed:
    // limit deceleration steps to half of the total steps
    if (_decelSteps > _totalSteps / 2) {
        _decelSteps = _totalSteps / 2;
    }

    _n = 0;
    _speed = 0.0;
    _lastStepTime = micros();
}

// Read limit switch status
bool StepperMotor::isLimitPressed() {
    if (_limitPin < 0) return false;
    // Active LOW (triggered when connected to GND)
    return (digitalRead(_limitPin) == LOW);
}

// Non-blocking update loop
bool StepperMotor::update() {
    if (!_enabled || _currentPos == _targetPos) {
        return false;
    }

    unsigned long currentTime = micros();
    unsigned long timeElapsed = currentTime - _lastStepTime;

    // Wait until the step interval has passed
    if (timeElapsed >= _cn) {
        // Double check direction safety and limit switch
        if (isLimitPressed()) {
            // Read direction: if we are moving towards the limit switch, STOP!
            // LOW represents homing direction (towards motor/left)
            if (digitalRead(_dirPin) == LOW) {
                _targetPos = _currentPos;
                _speed = 0.0;
                _n = 0;
                return false;
            }
        }

        // Generate Step Pulse (TB6600 requires at least 2.2us pulse width)
        // Common Anode: LOW = Active, HIGH = Inactive
        digitalWrite(_pulPin, LOW);
        delayMicroseconds(5);
        digitalWrite(_pulPin, HIGH);

        // Update Position
        if (_targetPos > _currentPos) {
            _currentPos++;
        } else {
            _currentPos--;
        }

        _stepsMoved++;
        _lastStepTime = currentTime;

        // Plan the next step delay
        if (_currentPos == _targetPos) {
            _speed = 0.0;
            _n = 0;
            return false;
        }

        calculateNextStep();
    }

    return true;
}

// Speed calculations using constant acceleration/deceleration approximations
void StepperMotor::calculateNextStep() {
    long stepsRemaining = _totalSteps - _stepsMoved;

    if (stepsRemaining <= 0) {
        _cn = _c0;
        return;
    }

    // 1. Deceleration Phase
    if (stepsRemaining <= _decelSteps) {
        _n--;
        if (_n < 1) _n = 1;
        // cn = cn-1 * (1 + 2 / (4n - 9))  [deceleration expansion]
        _cn = (unsigned long)((float)_cn * (1.0 + 2.0 / (4.0 * _n - 9.0)));
    }
    // 2. Acceleration Phase
    else if (_n < _decelSteps && _cn > _minDelay) {
        _n++;
        // cn = cn-1 * (1 - 2 / (4n + 1))  [acceleration contraction]
        _cn = (unsigned long)((float)_cn * (1.0 - 2.0 / (4.0 * _n + 1.0)));
        if (_cn < _minDelay) {
            _cn = _minDelay;
        }
    }
    // 3. Constant Speed Phase
    else {
        _cn = _minDelay;
    }

    // Protect against zero division or overflow
    if (_cn < 5) _cn = 5; // Enforce a hardware speed safety barrier
}

// Robust 3-stage synchronous Homing Sequence (Coarse -> Back-off -> Fine)
bool StepperMotor::home(bool homingDir, float coarseSpeed, float fineSpeed, long backoffSteps) {
    if (_limitPin < 0) return false;

    enable();
    
    // Determine stepping direction
    digitalWrite(_dirPin, homingDir ? HIGH : LOW);
    bool exitSwitchState = homingDir ? HIGH : LOW; // opposite direction for backing off

    // Step interval calculation for homing (constant speeds, no ramp necessary)
    unsigned long coarseDelay = (unsigned long)(1000000.0 / coarseSpeed);
    unsigned long fineDelay = (unsigned long)(1000000.0 / fineSpeed);

    // ==========================================
    // STAGE 1: Coarse search (Fast towards switch)
    // ==========================================
    digitalWrite(_dirPin, homingDir ? HIGH : LOW);
    while (!isLimitPressed()) {
        digitalWrite(_pulPin, LOW);
        delayMicroseconds(5);
        digitalWrite(_pulPin, HIGH);
        delayMicroseconds(coarseDelay);
        yield(); // Feed ESP32 watchdog
    }

    // ==========================================
    // STAGE 2: Back-off (Move away from switch)
    // ==========================================
    digitalWrite(_dirPin, !homingDir ? HIGH : LOW); // Reverse direction
    for (long i = 0; i < backoffSteps; i++) {
        digitalWrite(_pulPin, LOW);
        delayMicroseconds(5);
        digitalWrite(_pulPin, HIGH);
        delayMicroseconds(fineDelay);
        yield();
    }
    delay(100); // Allow physical vibration to settle

    // ==========================================
    // STAGE 3: Fine search (Slow towards switch)
    // ==========================================
    digitalWrite(_dirPin, homingDir ? HIGH : LOW);
    while (!isLimitPressed()) {
        digitalWrite(_pulPin, LOW);
        delayMicroseconds(5);
        digitalWrite(_pulPin, HIGH);
        delayMicroseconds(fineDelay);
        yield();
    }

    // Reset coordinates to 0 (origin calibrated successfully)
    setCurrentPosition(0);
    
    // Move slightly off the limit switch so it isn't constantly pressed
    digitalWrite(_dirPin, !homingDir ? HIGH : LOW); // Reverse
    for (int i = 0; i < 50; i++) {
        digitalWrite(_pulPin, LOW);
        delayMicroseconds(5);
        digitalWrite(_pulPin, HIGH);
        delayMicroseconds(fineDelay);
        yield();
    }
    _currentPos = 50; // Reflect actual physical shift
    _targetPos = 50;

    return true;
}
