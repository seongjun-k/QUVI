#!/bin/bash
# ------------------------------------------------------------------
# QUVI System Launch Shortcut
# 호스트 PC에서 실행 시 quvi-dev 컨테이너 내부로 접속하여
# full_system.launch.py 메인 런치 파일을 실행합니다.
# ------------------------------------------------------------------

CONTAINER_NAME="quvi-dev"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
COMPOSE_FILE="${SCRIPT_DIR}/docker/docker-compose.yml"

# docker compose 커맨드 자동 확인
if docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker-compose"
else
    echo "❌ 오류: docker compose 또는 docker-compose 명령을 찾을 수 없습니다."
    exit 1
fi

# 1. 먼저 실행 중인 컨테이너가 있는지 확인
TARGET_CONTAINER=$(docker ps -q --filter "name=${CONTAINER_NAME}" | head -n 1)

if [ -z "${TARGET_CONTAINER}" ]; then
    echo "🔄 [QUVI] '${CONTAINER_NAME}' 컨테이너가 실행 중이 아닙니다."
    
    # 2. 존재하지만 멈춰있는 컨테이너가 있는지 확인
    EXISTING_CONTAINER_ID=$(docker ps -aq --filter "name=${CONTAINER_NAME}" | head -n 1)
    
    if [ -n "${EXISTING_CONTAINER_ID}" ]; then
        echo "🔄 [QUVI] 기존에 생성된 컨테이너(${EXISTING_CONTAINER_ID})를 시작합니다..."
        docker start "${EXISTING_CONTAINER_ID}"
        if [ $? -ne 0 ]; then
            echo "❌ 오류: 컨테이너 시작 실패."
            exit 1
        fi
        echo "⏳ 컨테이너가 준비될 때까지 잠시 대기합니다..."
        sleep 2
        TARGET_CONTAINER="${EXISTING_CONTAINER_ID}"
    else
        # 3. 아예 존재하지 않는 경우에만 docker compose 구동
        echo "🔄 [QUVI] 새 컨테이너를 구동합니다..."
        if [ -f "${COMPOSE_FILE}" ]; then
            $DOCKER_COMPOSE -f "${COMPOSE_FILE}" up -d
            if [ $? -ne 0 ]; then
                echo "❌ 오류: docker compose 구동 실패."
                exit 1
            fi
            echo "⏳ 컨테이너가 부팅될 때까지 잠시 대기합니다..."
            sleep 2
            TARGET_CONTAINER=$(docker ps -q --filter "name=${CONTAINER_NAME}" | head -n 1)
        else
            echo "❌ 오류: '${COMPOSE_FILE}' 설정 파일을 찾을 수 없습니다."
            exit 1
        fi
    fi
fi

echo "🚀 [QUVI] 컨테이너(${TARGET_CONTAINER}) 내부에서 메인 프로그램 실행 중..."
if [ -t 0 ]; then
    docker exec -it "${TARGET_CONTAINER}" bash -c "source /opt/ros/jazzy/setup.bash && [ -f /uros_ws/install/setup.bash ] && source /uros_ws/install/setup.bash; source /workspace/install/setup.bash && ros2 launch quvi_bringup full_system.launch.py"
else
    docker exec -i "${TARGET_CONTAINER}" bash -c "source /opt/ros/jazzy/setup.bash && [ -f /uros_ws/install/setup.bash ] && source /uros_ws/install/setup.bash; source /workspace/install/setup.bash && ros2 launch quvi_bringup full_system.launch.py"
fi

if [ $? -ne 0 ]; then
    echo "❌ 오류: 메인 프로그램 실행 실패."
    exit 1
fi
