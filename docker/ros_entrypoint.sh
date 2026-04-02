#!/bin/bash
set -e

# ROS 2 환경 설정
source /opt/ros/${ROS_DISTRO}/setup.bash

# 워크스페이스 빌드가 있으면 source
if [ -f /workspace/install/setup.bash ]; then
    source /workspace/install/setup.bash
fi

exec "$@"
