#!/bin/bash
# ------------------------------------------------------------------
# QUVI Docker Build Shortcut
# 호스트 PC에서 실행 시 quvi-dev 컨테이너 내부로 접속하여 
# ROS 2 워크스페이스를 빌드한 후 컨테이너 셸(bash) 접속을 유지합니다.
# ------------------------------------------------------------------

CONTAINER_NAME="quvi-dev"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
COMPOSE_FILE="${SCRIPT_DIR}/docker/docker-compose.yml"

# 컨테이너가 실행 중이 아니면 자동으로 docker compose up -d 실행
if [ ! "$(docker ps -q -f name=^/${CONTAINER_NAME}$)" ]; then
    echo "🔄 [QUVI] '${CONTAINER_NAME}' 컨테이너가 꺼져 있습니다. 자동으로 실행을 시도합니다..."
    
    if [ -f "${COMPOSE_FILE}" ]; then
        docker compose -f "${COMPOSE_FILE}" up -d
        if [ $? -ne 0 ]; then
            echo "❌ 오류: docker compose 구동 실패."
            exit 1
        fi
        echo "⏳ 컨테이너가 부팅될 때까지 잠시 대기합니다..."
        sleep 2
    else
        echo "❌ 오류: '${COMPOSE_FILE}' 설정 파일을 찾을 수 없습니다."
        exit 1
    fi
fi

echo "🔄 [QUVI] '${CONTAINER_NAME}' 컨테이너 내부 ROS 2 워크스페이스 빌드 중..."
if [ -t 0 ]; then
    docker exec -it "${CONTAINER_NAME}" bash -c "source /opt/ros/jazzy/setup.bash && cd /workspace && colcon build --symlink-install && source install/setup.bash && exec bash"
    if [ $? -ne 0 ]; then
        echo "❌ 오류: 컨테이너 빌드 세션 실행 실패."
        exit 1
    fi
else
    docker exec -i "${CONTAINER_NAME}" bash -c "source /opt/ros/jazzy/setup.bash && cd /workspace && colcon build --symlink-install"
    if [ $? -ne 0 ]; then
        echo "❌ 오류: colcon 빌드 실패."
        exit 1
    fi
fi
