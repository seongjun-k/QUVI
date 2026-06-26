"""
QUVI 비전 파이프라인 Launch 파일
────────────────────────────────
INSPECT_NODE + 카메라 노드를 한 번에 실행.

사용법:
  ros2 launch quvi_bringup vision_pipeline.launch.py

  # 카메라 장치 변경:
  ros2 launch quvi_bringup vision_pipeline.launch.py \
    sidecam_device:=/dev/video0 \
    fixed_cam_device:=/dev/video2
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _default_data_dir() -> str:
    """데이터 디렉토리 기본값 계산.

    절대경로 하드코딩 대신 환경변수 QUVI_DATA_DIR > 컨테이너 표준 경로
    /workspace/data 순으로 결정한다. launch arg(data_dir)로 override 가능.
    """
    return os.environ.get('QUVI_DATA_DIR', '/workspace/data')


def generate_launch_description():

    default_data_dir = _default_data_dir()

    # ─── Launch Arguments ───
    sidecam_device_arg = DeclareLaunchArgument(
        'sidecam_device', default_value='/dev/sidecam',
        description='사이드캠(Zone 1) USB 장치 경로')

    fixed_cam_device_arg = DeclareLaunchArgument(
        'fixed_cam_device', default_value='/dev/fixed_cam',
        description='고정 카메라(Zone 2 검사 챔버) USB 장치 경로')

    data_dir_arg = DeclareLaunchArgument(
        'data_dir', default_value=default_data_dir,
        description='기준/로그 데이터 루트 (기본: $QUVI_DATA_DIR 또는 /workspace/data)')

    reference_dir_arg = DeclareLaunchArgument(
        'reference_image_dir',
        default_value=[LaunchConfiguration('data_dir'), '/reference_images'],
        description='기준 이미지(STL 렌더링) 디렉토리')

    inspection_log_dir_arg = DeclareLaunchArgument(
        'inspection_log_dir',
        default_value=[LaunchConfiguration('data_dir'), '/inspection_logs'],
        description='검사 로그 저장 디렉토리')

    sidecam_topic_arg = DeclareLaunchArgument(
        'sidecam_topic', default_value='/camera1/image_raw/compressed',
        description='로봇이 구독할 사이드캠 압축 이미지 토픽')

    inspect_topic_arg = DeclareLaunchArgument(
        'inspect_topic', default_value='/camera2/image_raw/compressed',
        description='검사 노드가 구독할 검사챔버 압축 이미지 토픽')

    inspect_config_arg = DeclareLaunchArgument(
        'inspect_config',
        default_value=os.path.join(
            get_package_share_directory('quvi_inspect'), 'config', 'inspect_params.yaml'),
        description='검사 파라미터 YAML 경로')

    sidecam_autoexposure_arg = DeclareLaunchArgument(
        'sidecam_autoexposure', default_value='false',
        description='사이드캠 자동 노출 활성화 여부 (true/false)')

    sidecam_exposure_arg = DeclareLaunchArgument(
        'sidecam_exposure', default_value='150',
        description='사이드캠 수동 노출값 (autoexposure가 false일 때 적용)')

    fixed_cam_autoexposure_arg = DeclareLaunchArgument(
        'fixed_cam_autoexposure', default_value='false',
        description='고정캠 자동 노출 활성화 여부 (true/false)')

    fixed_cam_exposure_arg = DeclareLaunchArgument(
        'fixed_cam_exposure', default_value='150',
        description='고정캠 수동 노출값 (autoexposure가 false일 때 적용)')

    # ─── 카메라 1: 사이드캠 (Zone 1 - 베드 위 출력물 촬영) ───
    camera1_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='camera1',
        namespace='camera1',
        parameters=[{
            'video_device': LaunchConfiguration('sidecam_device'),
            'image_width': 1920,
            'image_height': 1080,
            'pixel_format': 'raw_mjpeg',
            'framerate': 30.0,
            'camera_name': 'sidecam',
            'autoexposure': ParameterValue(LaunchConfiguration('sidecam_autoexposure'), value_type=bool),
            'exposure': ParameterValue(LaunchConfiguration('sidecam_exposure'), value_type=int),
        }],
        remappings=[
            ('image_raw', 'image_raw'),
            ('image_raw/compressed', 'image_raw/compressed'),
        ],
    )

    # ─── 카메라 2: 고정 카메라 (Zone 2 - 검사 챔버) ───
    camera2_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='camera2',
        namespace='camera2',
        parameters=[{
            'video_device': LaunchConfiguration('fixed_cam_device'),
            'image_width': 1920,
            'image_height': 1080,
            'pixel_format': 'mjpeg2rgb',
            'framerate': 30.0,
            'camera_name': 'inspection_cam',
            'autoexposure': ParameterValue(LaunchConfiguration('fixed_cam_autoexposure'), value_type=bool),
            'exposure': ParameterValue(LaunchConfiguration('fixed_cam_exposure'), value_type=int),
        }],
        remappings=[
            ('image_raw', 'image_raw'),
            ('image_raw/compressed', 'image_raw/compressed'),
        ],
    )

    # ─── INSPECT_NODE ───
    inspect_node = Node(
        package='quvi_inspect',
        executable='inspect_node',
        name='inspect_node',
        parameters=[
            LaunchConfiguration('inspect_config'),
            {
                'camera_topic': LaunchConfiguration('inspect_topic'),
                'reference_image_dir': LaunchConfiguration('reference_image_dir'),
                'inspection_log_dir': LaunchConfiguration('inspection_log_dir'),
            }
        ],
        output='screen',
    )

    return LaunchDescription([
        # Arguments
        sidecam_device_arg,
        fixed_cam_device_arg,
        data_dir_arg,
        reference_dir_arg,
        inspection_log_dir_arg,
        sidecam_topic_arg,
        inspect_topic_arg,
        inspect_config_arg,
        sidecam_autoexposure_arg,
        sidecam_exposure_arg,
        fixed_cam_autoexposure_arg,
        fixed_cam_exposure_arg,

        # 로그
        LogInfo(msg='====== QUVI Vision Pipeline 시작 ======'),

        # 노드
        camera1_node,
        camera2_node,
        inspect_node,
    ])
