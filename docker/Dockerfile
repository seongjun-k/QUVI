########################################
# QUVI Development Environment
# ROS 2 Jazzy + OpenCV + PyTorch
########################################
FROM ros:jazzy-ros-base-noble

LABEL maintainer="QUVI Team"
LABEL description="QUVI: AI Vision Robot QA System"

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=jazzy
ENV PIP_ROOT_USER_ACTION=ignore

SHELL ["/bin/bash", "-c"]

# ─── 시스템 패키지 ───
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    python3-numpy \
    python3-colcon-common-extensions \
    python3-rosdep \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-image-transport \
    ros-${ROS_DISTRO}-compressed-image-transport \
    ros-${ROS_DISTRO}-usb-cam \
    ros-${ROS_DISTRO}-rosbridge-suite \
    libopencv-dev \
    v4l-utils \
    nano \
    git \
    wget \
    curl \
    usbutils \
    # OpenGL 및 비전 렌더링용 의존 라이브러리 추가
    libglib2.0-0 \
    mesa-utils \
    libgl1 \
    libglu1-mesa-dev \
    libosmesa6-dev \
    && rm -rf /var/lib/apt/lists/*

# ─── Python 패키지 ───
# 1단계: 대용량 PyTorch 및 Torchvision을 타임아웃 방지 옵션 및 아시아 미러 서버를 지정하여 분리 설치
RUN pip3 install --break-system-packages --ignore-installed --no-cache-dir --default-timeout=1000 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    torch>=2.1.0 \
    torchvision>=0.16.0

# 2단계: 나머지 파이썬 의존성 패키지 설치
RUN pip3 install --break-system-packages --ignore-installed --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    opencv-python-headless>=4.9.0 \
    scikit-image>=0.22.0 \
    pyserial>=3.5 \
    Pillow>=10.0.0 \
    scipy>=1.12.0 \
    matplotlib>=3.8.0 \
    flask>=3.0.0 \
    flask-socketio>=5.3.0 \
    PyYAML>=6.0.0 \
    trimesh>=4.0.0

# 3단계: colcon-core와의 호환성을 위해 setuptools 버전을 <80으로 명시적 강제 다운그레이드
RUN pip3 install --break-system-packages --ignore-installed --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    "setuptools<80"

# ─── micro-ROS agent 빌드 ───
# micro_ros_agent는 apt 패키지가 없으므로 소스에서 직접 빌드
RUN apt-get update && apt-get install -y --no-install-recommends \
    libasio-dev \
    libtinyxml2-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /uros_ws/src && \
    cd /uros_ws/src && \
    git clone -b jazzy https://github.com/micro-ROS/micro-ROS-Agent.git micro_ros_agent && \
    git clone -b jazzy https://github.com/micro-ROS/micro_ros_msgs.git && \
    cd /uros_ws && \
    bash -c "source /opt/ros/jazzy/setup.bash && \
    apt-get update && \
    rosdep init 2>/dev/null || true && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -y && \
    colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release && \
    rm -rf build log src" && \
    rm -rf /var/lib/apt/lists/*

# ─── 워크스페이스 설정 ───
WORKDIR /workspace

# ROS 2 환경 자동 source
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc && \
    echo "if [ -f /uros_ws/install/setup.bash ]; then source /uros_ws/install/setup.bash; fi" >> /root/.bashrc && \
    echo "if [ -f /workspace/install/setup.bash ]; then source /workspace/install/setup.bash; fi" >> /root/.bashrc

# ─── 엔트리포인트 ───
COPY ros_entrypoint.sh /ros_entrypoint.sh
COPY run_micro_ros_agent.sh /usr/local/bin/run_micro_ros_agent.sh
RUN chmod +x /ros_entrypoint.sh /usr/local/bin/run_micro_ros_agent.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
