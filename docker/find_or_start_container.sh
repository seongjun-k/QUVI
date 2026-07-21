# ------------------------------------------------------------------
# QUVI 컨테이너 탐색·기동 공통 로직 (source 전용, shebang 없음)
# 호출측에서 CONTAINER_NAME, COMPOSE_FILE 변수를 미리 설정해야 함.
# 완료 후 TARGET_CONTAINER 변수에 컨테이너 ID가 채워짐.
# ------------------------------------------------------------------

# docker compose 커맨드 자동 확인
if docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker-compose"
else
    echo "오류: docker compose 또는 docker-compose 명령을 찾을 수 없습니다."
    exit 1
fi

# 1. 먼저 실행 중인 컨테이너가 있는지 확인
TARGET_CONTAINER=$(docker ps -q --filter "name=${CONTAINER_NAME}" | head -n 1)

if [ -z "${TARGET_CONTAINER}" ]; then
    echo "[QUVI] '${CONTAINER_NAME}' 컨테이너가 실행 중이 아닙니다."

    # 2. 존재하지만 멈춰있는 컨테이너가 있는지 확인
    EXISTING_CONTAINER_ID=$(docker ps -aq --filter "name=${CONTAINER_NAME}" | head -n 1)

    if [ -n "${EXISTING_CONTAINER_ID}" ]; then
        echo "[QUVI] 기존에 생성된 컨테이너(${EXISTING_CONTAINER_ID})를 시작합니다..."
        docker start "${EXISTING_CONTAINER_ID}"
        if [ $? -ne 0 ]; then
            echo "오류: 컨테이너 시작 실패."
            exit 1
        fi
        echo "컨테이너가 준비될 때까지 잠시 대기합니다..."
        sleep 2
        TARGET_CONTAINER="${EXISTING_CONTAINER_ID}"
    else
        # 3. 아예 존재하지 않는 경우에만 docker compose 구동
        echo "[QUVI] 새 컨테이너를 구동합니다..."
        if [ -f "${COMPOSE_FILE}" ]; then
            $DOCKER_COMPOSE -f "${COMPOSE_FILE}" up -d
            if [ $? -ne 0 ]; then
                echo "오류: docker compose 구동 실패."
                exit 1
            fi
            echo "컨테이너가 부팅될 때까지 잠시 대기합니다..."
            sleep 2
            TARGET_CONTAINER=$(docker ps -q --filter "name=${CONTAINER_NAME}" | head -n 1)
        else
            echo "오류: '${COMPOSE_FILE}' 설정 파일을 찾을 수 없습니다."
            exit 1
        fi
    fi
fi
