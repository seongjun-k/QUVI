"""
QUVI 시스템 전체 통신 토픽 명세 상수.
여러 노드에서 동일한 토픽 이름을 보장하기 위해 한 곳에서 관리합니다.
"""

# HMI <-> Orchestrator
TOPIC_HMI_STATUS = '/hmi/status'
TOPIC_HMI_COMMAND = '/hmi/command'

# Orchestrator <-> Robot Control
TOPIC_ROBOT_RAIL_CMD = '/robot/rail_command'
TOPIC_ROBOT_ROTATE_CMD = '/robot/rotate_command'
TOPIC_ROBOT_RELEASE_CMD = '/robot/release_command'
TOPIC_ROBOT_HOME_CMD = '/robot/home_command'
TOPIC_ROBOT_GRASP_CMD = '/robot/grasp_command'
TOPIC_ROBOT_RESET_CMD = '/robot/reset_command'

TOPIC_ROBOT_STATUS = '/robot/status'
TOPIC_ROBOT_ACT_DONE = '/robot/act_done'
TOPIC_ROBOT_GRASP_DONE = '/robot/grasp_done'
TOPIC_ROBOT_RELEASE_DONE = '/robot/release_done'
TOPIC_ROBOT_HOME_DONE = '/robot/home_done'
TOPIC_ROBOT_RAIL_DONE = '/robot/rail_done'

# Robot Control / Orchestrator <-> ESP32 Firmware
TOPIC_MOTOR_RAIL_CMD = '/motor/rail'
TOPIC_MOTOR_TURNTABLE_CMD = '/motor/turntable_cmd'
TOPIC_MOTOR_RAIL_DONE = '/motor/rail_done'
TOPIC_MOTOR_TURNTABLE_DONE = '/motor/turntable_done'
TOPIC_MOTOR_STATUS = '/motor/status'

# Orchestrator <-> Vision/Inspection
TOPIC_DETECTION_TRIGGER = '/detection/trigger'
TOPIC_INSPECTION_TRIGGER = '/inspection/trigger'

# Global Status
TOPIC_ESTOP = '/system/estop'

TOPIC_ROBOT_ROTATE_DONE = '/robot/rotate_done'

TOPIC_ROBOT_PLACE_CHAMBER_CMD = '/robot/place_in_chamber'
TOPIC_ROBOT_PLACE_CHAMBER_DONE = '/robot/place_in_chamber_done'
TOPIC_ROBOT_PICK_CHAMBER_CMD = '/robot/pick_in_chamber'
TOPIC_ROBOT_PICK_CHAMBER_DONE = '/robot/pick_in_chamber_done'
