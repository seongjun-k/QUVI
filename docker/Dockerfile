########################################
# QUVI Development Environment
# ROS 2 Jazzy + OpenCV + PyTorch + YOLO
########################################
FROM ros:jazzy-ros-base-noble

LABEL maintainer="QUVI Team"
LABEL description="QUVI: AI Vision Robot QA System"

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=jazzy

SHELL ["/bin/bash", "-c"]

# ─── 시스템 패키지 ───
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
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
    libgl1-mesa-glx \
    libglib2.0-0 \
    mesa-utils \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libosmesa6-dev \
    && rm -rf /var/lib/apt/lists/*

# ─── Python 패키지 ───
# 1단계: 대용량 PyTorch 및 Torchvision을 타임아웃 방지 옵션을 주어 먼저 분리 설치
RUN pip3 install --break-system-packages --no-cache-dir --default-timeout=1000 \
    torch>=2.1.0 \
    torchvision>=0.16.0

# 2단계: 나머지 파이썬 의존성 패키지 설치
RUN pip3 install --break-system-packages --no-cache-dir \
    opencv-python-headless>=4.9.0 \
    numpy>=1.26.0 \
    scikit-image>=0.22.0 \
    ultralytics>=8.1.0 \
    pyserial>=3.5 \
    Pillow>=10.0.0 \
    scipy>=1.12.0 \
    matplotlib>=3.8.0 \
    flask>=3.0.0 \
    flask-socketio>=5.3.0 \
    PyYAML>=6.0.0 \
    trimesh>=4.0.0

# ─── 워크스페이스 설정 ───
WORKDIR /workspace

# ROS 2 환경 자동 source
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc && \
    echo "if [ -f /workspace/install/setup.bash ]; then source /workspace/install/setup.bash; fi" >> /root/.bashrc

# ─── 엔트리포인트 ───
COPY ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
