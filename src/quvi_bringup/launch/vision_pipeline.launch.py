"""
QUVI 비전 파이프라인 Launch 파일
────────────────────────────────
YOLO_NODE + INSPECT_NODE + 카메라 노드를 한 번에 실행.

사용법:
  ros2 launch quvi_bringup vision_pipeline.launch.py

  # 카메라 장치 변경:
  ros2 launch quvi_bringup vision_pipeline.launch.py \
    handcam_device:=/dev/video0 \
    fixed_cam_device:=/dev/video2
"""

import os

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
    handcam_device_arg = DeclareLaunchArgument(
        'handcam_device', default_value='/dev/handcam',
        description='핸드캠(Zone 1) USB 장치 경로')

    fixed_cam_device_arg = DeclareLaunchArgument(
        'fixed_cam_device', default_value='/dev/fixed_cam',
        description='고정 카메라(Zone 2 검사 챔버) USB 장치 경로')

    data_dir_arg = DeclareLaunchArgument(
        'data_dir', default_value=default_data_dir,
        description='기준/로그 데이터 루트 (기본: $QUVI_DATA_DIR 또는 /workspace/data)')

    yolo_model_path_arg = DeclareLaunchArgument(
        'yolo_model_path', default_value=[LaunchConfiguration('data_dir'), '/models/best.pt'],
        description='YOLO 모델 파일 경로')

    reference_dir_arg = DeclareLaunchArgument(
        'reference_image_dir',
        default_value=[LaunchConfiguration('data_dir'), '/reference_images'],
        description='기준 이미지(STL 렌더링) 디렉토리')

    inspection_log_dir_arg = DeclareLaunchArgument(
        'inspection_log_dir',
        default_value=[LaunchConfiguration('data_dir'), '/inspection_logs'],
        description='검사 로그 저장 디렉토리')

    handcam_topic_arg = DeclareLaunchArgument(
        'handcam_topic', default_value='/camera1/image_raw/compressed',
        description='YOLO/로봇이 구독할 핸드캠 압축 이미지 토픽')

    inspect_topic_arg = DeclareLaunchArgument(
        'inspect_topic', default_value='/camera2/image_raw/compressed',
        description='검사 노드가 구독할 검사챔버 압축 이미지 토픽')

    yolo_config_arg = DeclareLaunchArgument(
        'yolo_config', default_value='',
        description='YOLO 파라미터 YAML 경로 (빈 문자열이면 기본값)')

    inspect_config_arg = DeclareLaunchArgument(
        'inspect_config', default_value='',
        description='검사 파라미터 YAML 경로 (빈 문자열이면 기본값)')

    handcam_autoexposure_arg = DeclareLaunchArgument(
        'handcam_autoexposure', default_value='false',
        description='핸드캠 자동 노출 활성화 여부 (true/false)')

    handcam_exposure_arg = DeclareLaunchArgument(
        'handcam_exposure', default_value='150',
        description='핸드캠 수동 노출값 (autoexposure가 false일 때 적용)')

    fixed_cam_autoexposure_arg = DeclareLaunchArgument(
        'fixed_cam_autoexposure', default_value='false',
        description='고정캠 자동 노출 활성화 여부 (true/false)')

    fixed_cam_exposure_arg = DeclareLaunchArgument(
        'fixed_cam_exposure', default_value='150',
        description='고정캠 수동 노출값 (autoexposure가 false일 때 적용)')

    # ─── 카메라 1: 핸드캠 (Zone 1 - 베드 위 출력물 촬영) ───
    camera1_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='camera1',
        namespace='camera1',
        parameters=[{
            'video_device': LaunchConfiguration('handcam_device'),
            'image_width': 1920,
            'image_height': 1080,
            'pixel_format': 'mjpeg2rgb',
            'framerate': 30.0,
            'camera_name': 'handcam',
            'autoexposure': ParameterValue(LaunchConfiguration('handcam_autoexposure'), value_type=bool),
            'exposure': ParameterValue(LaunchConfiguration('handcam_exposure'), value_type=int),
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

    # ─── YOLO_NODE ───
    yolo_node = Node(
        package='quvi_yolo',
        executable='yolo_node',
        name='yolo_node',
        parameters=[{
            'camera_topic': LaunchConfiguration('handcam_topic'),
            'use_compressed': True,
            'model_path': LaunchConfiguration('yolo_model_path'),
            'confidence_threshold': 0.5,
            'iou_threshold': 0.45,
            'max_detections': 20,
            'min_bbox_area': 500,
            'proximity_warning_px': 60,
            'sort_direction': 'left_to_right',
            'input_width': 640,
            'input_height': 480,
            'publish_debug_image': True,
        }],
        output='screen',
    )

    # ─── INSPECT_NODE ───
    inspect_node = Node(
        package='quvi_inspect',
        executable='inspect_node',
        name='inspect_node',
        parameters=[{
            'camera_topic': LaunchConfiguration('inspect_topic'),
            'use_compressed': True,
            'reference_image_dir': LaunchConfiguration('reference_image_dir'),
            'ssim_threshold': 0.85,
            'area_ratio_min': 0.90,
            'area_ratio_max': 1.10,
            'pixel_diff_threshold': 0.10,
            'solidity_min': 0.85,
            'solidity_max': 1.00,
            'feature_area_ratio_min': 0.90,
            'feature_area_ratio_max': 1.10,
            'hole_count_max': 2,
            'hole_area_ratio_max': 0.05,
            'texture_variance_max': 500.0,
            'min_hole_area_px': 50,
            'turntable_angles': [0, 90, 180, 270],
            'capture_delay_sec': 0.5,
            'save_inspection_images': True,
            'inspection_log_dir': LaunchConfiguration('inspection_log_dir'),
            'publish_debug_image': True,
            'alignment_enabled': True,
            'align_max_dimension': 200,
            'align_padding_pct': 0.15,
            'align_min_bbox_area': 500,
        }],
        output='screen',
    )

    return LaunchDescription([
        # Arguments
        handcam_device_arg,
        fixed_cam_device_arg,
        data_dir_arg,
        yolo_model_path_arg,
        reference_dir_arg,
        inspection_log_dir_arg,
        handcam_topic_arg,
        inspect_topic_arg,
        yolo_config_arg,
        inspect_config_arg,
        handcam_autoexposure_arg,
        handcam_exposure_arg,
        fixed_cam_autoexposure_arg,
        fixed_cam_exposure_arg,

        # 로그
        LogInfo(msg='====== QUVI Vision Pipeline 시작 ======'),

        # 노드
        camera1_node,
        camera2_node,
        yolo_node,
        inspect_node,
    ])
