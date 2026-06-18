/**
 * =============================================================================
 * QUVI ESP32-S3 Firmware - Linear Rail and Turntable Controller
 * Target Board: Wemos LOLIN S3 V1.0.0 (ESP32-S3)
 * Motor Drivers: 2 x TB6600 Stepper Drivers
 * =============================================================================
 */

#include "Config.h"
#include "StepperMotor.h"
#include <Adafruit_NeoPixel.h>

#ifdef USE_MICRO_ROS
  #include <micro_ros_arduino.h>
  #include <rcl/rcl.h>
  #include <rcl/error_handling.h>
  #include <rclc/rclc.h>
  #include <rclc/executor.h>
  #include <std_msgs/msg/int32.h>
  #include <std_msgs/msg/bool.h>
  #include <quvi_msgs/msg/motor_status.h>
#endif

// =============================================================================
// GLOBAL OBJECTS
// =============================================================================
// 1. Motors
// Linear Rail: GPIO 5 (PUL), GPIO 6 (DIR), -1 (ENA), GPIO 7 (LIMIT Switch at left/motor side)
StepperMotor railMotor(RAIL_PUL_PIN, RAIL_DIR_PIN, RAIL_ENA_PIN, RAIL_LIMIT_PIN);

// Turntable: GPIO 15 (PUL), GPIO 16 (DIR), -1 (ENA), -1 (Optional Zero switch)
StepperMotor turnMotor(TURN_PUL_PIN, TURN_DIR_PIN, TURN_ENA_PIN, TURN_LIMIT_PIN);

// 2. Onboard Status LED (WS2812B RGB LED)
Adafruit_NeoPixel statusLed(1, ONBOARD_LED_PIN, NEO_GRB + NEO_KHZ800);

// =============================================================================
// STATE & SAFETY VARIABLES
// =============================================================================
volatile bool isEmergencyStopped = false;
volatile bool isHomingCompleted = false;

// Color presets for status visualization
const uint32_t COLOR_OFF    = statusLed.Color(0, 0, 0);
const uint32_t COLOR_BLUE   = statusLed.Color(0, 0, 100);    // Idle / Safe
const uint32_t COLOR_YELLOW = statusLed.Color(100, 100, 0);  // Homing in progress
const uint32_t COLOR_GREEN  = statusLed.Color(0, 100, 0);    // Operation complete / Calibration OK
const uint32_t COLOR_RED    = statusLed.Color(120, 0, 0);    // Emergency Stop / Error
const uint32_t COLOR_PURPLE = statusLed.Color(80, 0, 80);    // Connection pending

// =============================================================================
// MICRO-ROS SETUP
// =============================================================================
#ifdef USE_MICRO_ROS
  rcl_subscription_t rail_sub;
  rcl_subscription_t turn_sub;
  rcl_subscription_t turn_led_sub;
  rcl_subscription_t estop_sub;
  std_msgs__msg__Int32 rail_msg;
  std_msgs__msg__Int32 turn_msg;
  std_msgs__msg__Bool turn_led_msg;
  std_msgs__msg__Bool estop_msg;
  rcl_publisher_t rail_done_pub;
  std_msgs__msg__Bool rail_done_msg;
  rcl_publisher_t turn_done_pub;
  std_msgs__msg__Bool turn_done_msg;
  rcl_publisher_t status_pub;
  quvi_msgs__msg__MotorStatus status_msg;
  rclc_executor_t executor;
  rclc_support_t support;
  rcl_allocator_t allocator;
  rcl_node_t node;

  volatile bool rail_done_pending = false;
  volatile bool turn_done_pending = false;

  #define RCCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){ error_loop(); }}
  #define RCSOFTCHECK(fn) { rcl_ret_t temp_rc = fn; if((temp_rc != RCL_RET_OK)){ }}
#endif

// =============================================================================
// FREERTOS TASKS
// =============================================================================
TaskHandle_t MotorTaskHandle = NULL;
TaskHandle_t CommTaskHandle = NULL;

void vMotorTask(void *pvParameters);
void vCommTask(void *pvParameters);

// Homing Calibration routine
void performHomingCalibration();
void handleEmergencyStop();
void setLedColor(uint32_t color);

// =============================================================================
// MAIN SETUP
// =============================================================================
void setup() {
    // Initialize Status LED
    statusLed.begin();
    setLedColor(COLOR_PURPLE);

    // Initialize Motor Drivers
    railMotor.begin();
    turnMotor.begin();

    // Initialize LED Relay Pin
    pinMode(TURN_LED_RELAY_PIN, OUTPUT);
    digitalWrite(TURN_LED_RELAY_PIN, LOW); // Default to OFF

    // Max Speed & Acceleration Profiles
    railMotor.setMaxSpeed(RAIL_MAX_SPEED);
    railMotor.setAcceleration(RAIL_ACCELERATION);
    turnMotor.setMaxSpeed(TURN_MAX_SPEED);
    turnMotor.setAcceleration(TURN_ACCELERATION);

    // Initialize Emergency Stop Hardware interrupt
    pinMode(ESTOP_PIN, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(ESTOP_PIN), handleEmergencyStop, FALLING);

    // If micro-ROS mode, initialize micro-ROS communication transport
    #ifdef USE_MICRO_ROS
        set_microros_transports();
    #else
        Serial.begin(SERIAL_BAUDRATE);
    #endif

    // Pin setup confirmation delay
    delay(500);

    // Create FreeRTOS Tasks
    // 1. Motor control task on Core 0 (Extreme scheduling priority)
    xTaskCreatePinnedToCore(
        vMotorTask,         // Task function
        "MotorTask",        // Task name
        4096,               // Stack size (bytes)
        NULL,               // Parameter
        3,                  // Priority (higher = more critical)
        &MotorTaskHandle,   // Task handle
        0                   // Pinned to Core 0
    );

    // 2. Communication and parser task on Core 1
    xTaskCreatePinnedToCore(
        vCommTask,
        "CommTask",
        8192,
        NULL,
        1,
        &CommTaskHandle,
        1                   // Pinned to Core 1
    );
}

// Keep main Arduino loop empty (FreeRTOS runs tasks asynchronously)
void loop() {
    vTaskDelay(pdMS_TO_TICKS(1000));
}

// =============================================================================
// TASK 1: HIGH-SPEED STEP PULSING (CORE 0)
// =============================================================================
void vMotorTask(void *pvParameters) {
    (void) pvParameters;
    
    // Safety check loop frequency: 100 kHz (10 microseconds sleep)
    TickType_t xLastWakeTime = xTaskGetTickCount();

    for (;;) {
        if (isEmergencyStopped) {
            railMotor.disable();
            turnMotor.disable();
            digitalWrite(TURN_LED_RELAY_PIN, LOW); // E-STOP safety action
            setLedColor(COLOR_RED);
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        // Handle step calculations for both motors in parallel
        bool railMoving = railMotor.update();
        bool turnMoving = turnMotor.update();

        // Dynamic Status visual feedback
        if (!isHomingCompleted) {
            // Blinking yellow if not calibrated
            static unsigned long lastBlink = 0;
            static bool blinkState = false;
            if (millis() - lastBlink > 250) {
                blinkState = !blinkState;
                setLedColor(blinkState ? COLOR_YELLOW : COLOR_OFF);
                lastBlink = millis();
            }
        } else if (railMoving || turnMoving) {
            // Green when moving and healthy
            setLedColor(COLOR_GREEN);
        } else {
            // Static blue when idle and calibrated
            setLedColor(COLOR_BLUE);
        }

        // Microsecond precision scheduling
        // Yield execution to prevent ESP32 Core 0 watchdog triggers
        delayMicroseconds(10);
    }
}

// =============================================================================
// CALIBRATION: 3-STAGE HOMING ROUTINE
// =============================================================================
void performHomingCalibration() {
    if (isEmergencyStopped) return;

    isHomingCompleted = false;
    setLedColor(COLOR_YELLOW);

    #ifndef USE_MICRO_ROS
        Serial.println("[INFO] Homing calibration started...");
    #endif

    // Calibrate linear rail (towards motor on left)
    railMotor.home(RAIL_HOMING_DIR, RAIL_HOME_COARSE_SPD, RAIL_HOME_FINE_SPD, RAIL_HOME_BACKOFF);

    // Calibrate turntable (optional - reset absolute zero position on boot)
    turnMotor.setCurrentPosition(0);
    
    railMotor.enable();
    turnMotor.enable();

    isHomingCompleted = true;
    setLedColor(COLOR_BLUE);

    #ifndef USE_MICRO_ROS
        Serial.println("[SUCCESS] Homing complete. Position set to absolute 0.");
        Serial.print("[STATUS] Current Rail Steps: ");
        Serial.println(railMotor.getCurrentPosition());
    #endif
}

// =============================================================================
// TASK 2: COMMUNICATION & COMMAND PARSING (CORE 1)
// =============================================================================
#ifdef USE_MICRO_ROS
// micro-ROS Subscriptions callback functions
void rail_subscription_callback(const void * msin) {
    const std_msgs__msg__Int32 * msg = (const std_msgs__msg__Int32 *)msin;
    if (isEmergencyStopped || !isHomingCompleted) return;

    long target_steps = msg->data;
    
    // Soft Limit check for safety
    if (target_steps >= RAIL_MIN_LIMIT && target_steps <= RAIL_MAX_LIMIT) {
        railMotor.setTargetPosition(target_steps);
        rail_done_pending = true;
    }
}

void turn_subscription_callback(const void * msin) {
    const std_msgs__msg__Int32 * msg = (const std_msgs__msg__Int32 *)msin;
    if (isEmergencyStopped || !isHomingCompleted) return;

    double target_angle = msg->data;

    // Shortest-Path Angular Translation algorithm
    float currentAngle = turnMotor.getCurrentPosition() / TURN_STEPS_PER_DEGREE;
    double normCurrent = fmod(currentAngle, 360.0);
    if (normCurrent < 0) normCurrent += 360.0;

    double delta = target_angle - normCurrent;
    
    // Resolve absolute shortest angular path
    while (delta < -180.0) delta += 360.0;
    while (delta > 180.0)  delta -= 360.0;

    long deltaSteps = round(delta * TURN_STEPS_PER_DEGREE);
    long targetSteps = turnMotor.getCurrentPosition() + deltaSteps;

    turnMotor.setTargetPosition(targetSteps);
    turn_done_pending = true;
}

void turn_led_subscription_callback(const void * msin) {
    const std_msgs__msg__Bool * msg = (const std_msgs__msg__Bool *)msin;
    if (isEmergencyStopped) return;
    digitalWrite(TURN_LED_RELAY_PIN, msg->data ? HIGH : LOW);
}

void estop_subscription_callback(const void * msin) {
    const std_msgs__msg__Bool * msg = (const std_msgs__msg__Bool *)msin;
    if (msg->data) {
        handleEmergencyStop();
    }
}

void error_loop() {
    while (1) {
        setLedColor(COLOR_RED);
        delay(100);
        setLedColor(COLOR_OFF);
        delay(100);
    }
}
#endif

void vCommTask(void *pvParameters) {
    (void) pvParameters;

    #ifdef USE_MICRO_ROS
        // Initialize micro-ROS Allocator
        allocator = rcl_get_default_allocator();

        // Create micro-ROS Support
        RCCHECK(rclc_support_init(&support, 0, NULL, &allocator));

        // Create Node
        RCCHECK(rclc_node_init_default(&node, "quvi_esp32_sub", "", &support));

        // Create Subscribers
        RCCHECK(rclc_subscription_init_default(
            &rail_sub,
            &node,
            ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32),
            TOPIC_RAIL_CMD
        ));

        RCCHECK(rclc_subscription_init_default(
            &turn_sub,
            &node,
            ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Int32),
            TOPIC_TURN_CMD
        ));

        RCCHECK(rclc_subscription_init_default(
            &turn_led_sub,
            &node,
            ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Bool),
            TOPIC_TURN_LED
        ));

        RCCHECK(rclc_subscription_init_default(
            &estop_sub,
            &node,
            ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Bool),
            TOPIC_ESTOP
        ));

        // Create Publishers
        RCCHECK(rclc_publisher_init_default(
            &rail_done_pub,
            &node,
            ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Bool),
            TOPIC_RAIL_DONE
        ));

        RCCHECK(rclc_publisher_init_default(
            &turn_done_pub,
            &node,
            ROSIDL_GET_MSG_TYPE_SUPPORT(std_msgs, msg, Bool),
            TOPIC_TURN_DONE
        ));

        RCCHECK(rclc_publisher_init_default(
            &status_pub,
            &node,
            ROSIDL_GET_MSG_TYPE_SUPPORT(quvi_msgs, msg, MotorStatus),
            TOPIC_STATUS
        ));

        // Create Executor (4 subscriptions)
        RCCHECK(rclc_executor_init(&executor, &support.context, 4, &allocator));
        RCCHECK(rclc_executor_add_subscription(&executor, &rail_sub, &rail_msg, &rail_subscription_callback, ON_NEW_DATA));
        RCCHECK(rclc_executor_add_subscription(&executor, &turn_sub, &turn_msg, &turn_subscription_callback, ON_NEW_DATA));
        RCCHECK(rclc_executor_add_subscription(&executor, &turn_led_sub, &turn_led_msg, &turn_led_subscription_callback, ON_NEW_DATA));
        RCCHECK(rclc_executor_add_subscription(&executor, &estop_sub, &estop_msg, &estop_subscription_callback, ON_NEW_DATA));

        // Trigger Auto Homing Calibration upon connection
        performHomingCalibration();

        // micro-ROS Loop Executor
        for (;;) {
            if (isEmergencyStopped) {
                vTaskDelay(pdMS_TO_TICKS(100));
                continue;
            }
            RCSOFTCHECK(rclc_executor_spin_some(&executor, RCL_MS_TO_NS(10)));
            
            if (rail_done_pending && !railMotor.isMoving()) {
                rail_done_msg.data = true;
                RCSOFTCHECK(rcl_publish(&rail_done_pub, &rail_done_msg, NULL));
                rail_done_pending = false;
            }
            
            if (turn_done_pending && !turnMotor.isMoving()) {
                turn_done_msg.data = true;
                RCSOFTCHECK(rcl_publish(&turn_done_pub, &turn_done_msg, NULL));
                turn_done_pending = false;
            }
            
            static unsigned long last_status_pub = 0;
            if (millis() - last_status_pub > 100) {
                status_msg.rail_position = railMotor.getCurrentPosition();
                status_msg.rail_target = railMotor.getTargetPosition();
                status_msg.turntable_angle = turnMotor.getCurrentPosition() / (float)TURN_STEPS_PER_DEGREE;
                status_msg.is_moving = railMotor.isMoving() || turnMotor.isMoving();
                status_msg.homed = isHomingCompleted;
                status_msg.estop = isEmergencyStopped;
                RCSOFTCHECK(rcl_publish(&status_pub, &status_msg, NULL));
                last_status_pub = millis();
            }
            
            vTaskDelay(pdMS_TO_TICKS(10));
        }

    #else
        // Standard Serial Mode Command Line Interface (CLI)
        Serial.println("=================================================");
        Serial.println("QUVI ESP32 Stepper Controller (LOLIN S3)");
        Serial.println("Communication Mode: UART Serial");
        Serial.println("Commands:");
        Serial.println("  H            - Trigger Homing Calibration");
        Serial.println("  R <steps>    - Move Linear Rail to absolute step");
        Serial.println("  T <degrees>  - Rotate Turntable to absolute angle");
        Serial.println("  S            - Show System Position & Limit Switch Status");
        Serial.println("  E            - EMERGENCY STOP");
        Serial.println("  L <0 or 1>   - Turntable LED Ring Relay (0:OFF, 1:ON)");
        Serial.println("=================================================");

        // Automatically trigger homing calibration on boot in serial mode
        performHomingCalibration();

        String inputBuffer = "";

        for (;;) {
            if (isEmergencyStopped) {
                if (Serial.available()) {
                    String clr = Serial.readStringUntil('\n');
                    Serial.println("[EMERGENCY] Hardware locked. Reset board to recover.");
                }
                vTaskDelay(pdMS_TO_TICKS(200));
                continue;
            }

            while (Serial.available() > 0) {
                char ch = Serial.read();
                if (ch == '\n' || ch == '\r') {
                    inputBuffer.trim();
                    if (inputBuffer.length() > 0) {
                        // Parse CLI command
                        char cmd = inputBuffer.charAt(0);
                        cmd = toupper(cmd);

                        if (cmd == 'H') {
                            performHomingCalibration();
                        }
                        else if (cmd == 'R') {
                            long steps = inputBuffer.substring(2).toInt();
                            if (steps >= RAIL_MIN_LIMIT && steps <= RAIL_MAX_LIMIT) {
                                Serial.print("[MOVE] Linear Rail set to target steps: ");
                                Serial.println(steps);
                                railMotor.setTargetPosition(steps);
                            } else {
                                Serial.print("[ERROR] Step value out of soft-limits (0 - ");
                                Serial.print(RAIL_MAX_LIMIT);
                                Serial.println(")");
                            }
                        }
                        else if (cmd == 'T') {
                            double angle = inputBuffer.substring(2).toFloat();
                            Serial.print("[MOVE] Turntable set to target degrees: ");
                            Serial.println(angle);

                            // Shortest path logic
                            float currentAngle = turnMotor.getCurrentPosition() / TURN_STEPS_PER_DEGREE;
                            double normCurrent = fmod(currentAngle, 360.0);
                            if (normCurrent < 0) normCurrent += 360.0;

                            double delta = angle - normCurrent;
                            while (delta < -180.0) delta += 360.0;
                            while (delta > 180.0)  delta -= 360.0;

                            long deltaSteps = round(delta * TURN_STEPS_PER_DEGREE);
                            long targetSteps = turnMotor.getCurrentPosition() + deltaSteps;

                            turnMotor.setTargetPosition(targetSteps);
                        }
                        else if (cmd == 'S') {
                            Serial.println("----------------------------------------");
                            Serial.print("Rail Position:   "); Serial.print(railMotor.getCurrentPosition()); Serial.println(" steps");
                            Serial.print("Rail Target:     "); Serial.print(railMotor.getTargetPosition()); Serial.println(" steps");
                            Serial.print("Rail Limit Switch: "); Serial.println(railMotor.isLimitPressed() ? "ACTIVE (PRESSED)" : "INACTIVE");
                            Serial.print("Turntable Steps: "); Serial.print(turnMotor.getCurrentPosition()); Serial.println(" steps");
                            Serial.print("Turntable Angle: "); Serial.print(turnMotor.getCurrentPosition() / TURN_STEPS_PER_DEGREE); Serial.println(" deg");
                            Serial.println("----------------------------------------");
                        }
                        else if (cmd == 'E') {
                            handleEmergencyStop();
                        }
                        else if (cmd == 'L') {
                            int state = inputBuffer.substring(2).toInt();
                            if (state == 1) {
                                digitalWrite(TURN_LED_RELAY_PIN, HIGH);
                                Serial.println("[LED] Turntable LED Relay ON");
                            } else {
                                digitalWrite(TURN_LED_RELAY_PIN, LOW);
                                Serial.println("[LED] Turntable LED Relay OFF");
                            }
                        }
                        else {
                            Serial.println("[ERROR] Command syntax unrecognized.");
                        }
                    }
                    inputBuffer = "";
                } else {
                    inputBuffer += ch;
                }
            }
            vTaskDelay(pdMS_TO_TICKS(50));
        }
    #endif
}

// =============================================================================
// EMERGENCY STOP HANDLER (INTERRUPT DRIVEN)
// =============================================================================
void IRAM_ATTR handleEmergencyStop() {
    isEmergencyStopped = true;
    // Hard-disable the motor signals inside the ISR instantly
    if (RAIL_ENA_PIN >= 0) digitalWrite(RAIL_ENA_PIN, HIGH); // Disable TB6600
    if (TURN_ENA_PIN >= 0) digitalWrite(TURN_ENA_PIN, HIGH); // Disable TB6600
    digitalWrite(TURN_LED_RELAY_PIN, LOW); // Turn off LED relay for safety
}

// Set WS2812B Color
void setLedColor(uint32_t color) {
    statusLed.setPixelColor(0, color);
    statusLed.show();
}
