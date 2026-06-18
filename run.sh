#!/bin/bash
# ------------------------------------------------------------------
# QUVI System Launch Shortcut
# 호스트 PC에서 실행 시 quvi-dev 컨테이너 내부로 접속하여
# full_system.launch.py 메인 런치 파일을 실행합니다.
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

echo "🚀 [QUVI] '${CONTAINER_NAME}' 컨테이너 내부에서 메인 프로그램 실행 중..."
if [ -t 0 ]; then
    docker exec -it "${CONTAINER_NAME}" bash -c "source /workspace/install/setup.bash && ros2 launch quvi_bringup full_system.launch.py"
else
    docker exec -i "${CONTAINER_NAME}" bash -c "source /workspace/install/setup.bash && ros2 launch quvi_bringup full_system.launch.py"
fi

if [ $? -ne 0 ]; then
    echo "❌ 오류: 메인 프로그램 실행 실패."
    exit 1
fi
