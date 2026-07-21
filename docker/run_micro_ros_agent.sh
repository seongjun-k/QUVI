#!/bin/bash
source /opt/ros/jazzy/setup.bash
source /uros_ws/install/setup.bash
exec /uros_ws/install/micro_ros_agent/lib/micro_ros_agent/micro_ros_agent "$@"
