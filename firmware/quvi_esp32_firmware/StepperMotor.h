#ifndef STEPPER_MOTOR_H
#define STEPPER_MOTOR_H

#include <Arduino.h>

class StepperMotor {
public:
    // Constructor
    StepperMotor(uint8_t pulPin, uint8_t dirPin, uint8_t enaPin, int8_t limitPin = -1);

    // Initialization
    void begin();

    // Motor control commands
    void setTargetPosition(long target);
    void setCurrentPosition(long current);
    void setMaxSpeed(float speed);
    void setAcceleration(float accel);
    
    // Motor enable / disable (Active LOW on TB6600)
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
    long getCurrentPosition() const { return _currentPos; }
    long getTargetPosition() const { return _targetPos; }
    float getCurrentSpeed() const { return _speed; }
    bool isMoving() const { return _currentPos != _targetPos; }

private:
    // Pin definitions
    uint8_t _pulPin;
    uint8_t _dirPin;
    uint8_t _enaPin;
    int8_t _limitPin;

    // Motion parameters
    long _currentPos;    // Current absolute step position
    long _targetPos;     // Target absolute step position
    float _maxSpeed;     // Max speed in steps/sec
    float _accel;        // Acceleration rate in steps/sec^2

    // Internal motion planning states
    float _speed;          // Current speed in steps/sec
    unsigned long _c0;     // Initial step interval in microseconds
    unsigned long _cn;     // Current step interval in microseconds
    unsigned long _minDelay; // Minimum delay corresponding to max speed
    long _n;               // Step counter for acceleration profile
    long _decelSteps;      // Number of steps required to decelerate to stop
    long _totalSteps;      // Total steps in current move
    long _stepsMoved;      // Steps completed in current move
    
    unsigned long _lastStepTime; // Timestamp of the last step pulse (microseconds)
    bool _enabled;         // Active state

    // Private helper: calculates the next step delay
    void calculateNextStep();
    void resetAcceleration();
};

#endif // STEPPER_MOTOR_H
