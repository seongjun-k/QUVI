#ifndef STEPPER_MOTOR_H
#define STEPPER_MOTOR_H

#include <Arduino.h>
#include <AccelStepper.h>

class StepperMotor {
public:
    // Constructor
    StepperMotor(int8_t pulPin, int8_t dirPin, int8_t enaPin, int8_t limitPin = -1);

    // Initialization
    void begin();

    // Motor control commands
    void setTargetPosition(long target);
    void setCurrentPosition(long current);
    void setMaxSpeed(float speed);
    void setAcceleration(float accel);
    
    // Motor enable / disable (no-op or simple status tracking since EN pin is removed)
    void enable();
    void disable();
    bool isEnabled() const;

    // Movement updates
    // Must be called as frequently as possible (e.g., in a high-speed loop)
    // Returns true if the motor is still moving to target
    bool update();

    // Homing Sequence
    // Performed synchronously to ensure absolute calibration
    bool home(bool homingDir, float coarseSpeed, float fineSpeed, long backoffSteps);

    // Limit switch query
    bool isLimitPressed();

    // Position accessors
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
    bool _enabled;
};

#endif // STEPPER_MOTOR_H
