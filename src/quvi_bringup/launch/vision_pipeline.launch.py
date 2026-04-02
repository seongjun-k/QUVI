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


def generate_launch_description():

    # ─── Launch Arguments ───
    handcam_device_arg = DeclareLaunchArgument(
        'handcam_device', default_value='/dev/video0',
        description='핸드캠(Zone 1) USB 장치 경로')

    fixed_cam_device_arg = DeclareLaunchArgument(
        'fixed_cam_device', default_value='/dev/video2',
        description='고정 카메라(Zone 2 검사 챔버) USB 장치 경로')

    yolo_config_arg = DeclareLaunchArgument(
        'yolo_config', default_value='',
        description='YOLO 파라미터 YAML 경로 (빈 문자열이면 기본값)')

    inspect_config_arg = DeclareLaunchArgument(
        'inspect_config', default_value='',
        description='검사 파라미터 YAML 경로 (빈 문자열이면 기본값)')

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
            'camera_topic': '/camera1/image_raw/compressed',
            'use_compressed': True,
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
            'camera_topic': '/camera2/image_raw/compressed',
            'use_compressed': True,
            'reference_image_dir': '/workspace/data/reference_images',
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
            'inspection_log_dir': '/workspace/data/inspection_logs',
            'publish_debug_image': True,
        }],
        output='screen',
    )

    return LaunchDescription([
        # Arguments
        handcam_device_arg,
        fixed_cam_device_arg,
        yolo_config_arg,
        inspect_config_arg,

        # 로그
        LogInfo(msg='====== QUVI Vision Pipeline 시작 ======'),

        # 노드
        camera1_node,
        camera2_node,
        yolo_node,
        inspect_node,
    ])
