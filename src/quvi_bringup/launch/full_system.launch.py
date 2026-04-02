"""
QUVI 전체 시스템 Launch 파일
───────────────────────────────
카메라 + YOLO + 검사 + HMI Web UI 한 번에 실행.

사용법:
  ros2 launch quvi_bringup full_system.launch.py
  ros2 launch quvi_bringup full_system.launch.py hmi_port:=8080
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    # ─── Launch Arguments ───
    hmi_port_arg = DeclareLaunchArgument(
        'hmi_port', default_value='5000',
        description='HMI Web UI 포트')

    handcam_arg = DeclareLaunchArgument(
        'handcam_device', default_value='/dev/video0',
        description='핸드캠 장치 경로')

    fixed_cam_arg = DeclareLaunchArgument(
        'fixed_cam_device', default_value='/dev/video2',
        description='고정 카메라 장치 경로')

    # ─── Vision Pipeline 포함 ───
    bringup_dir = get_package_share_directory('quvi_bringup')
    vision_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_dir, 'launch', 'vision_pipeline.launch.py')),
        launch_arguments={
            'handcam_device': LaunchConfiguration('handcam_device'),
            'fixed_cam_device': LaunchConfiguration('fixed_cam_device'),
        }.items(),
    )

    # ─── HMI_NODE ───
    hmi_node = Node(
        package='quvi_hmi',
        executable='hmi_node',
        name='hmi_node',
        parameters=[{
            'host': '0.0.0.0',
            'port': 5000,
            'debug': False,
            'camera1_topic': '/camera1/image_raw/compressed',
            'camera2_topic': '/camera2/image_raw/compressed',
            'yolo_debug_topic': '/yolo/debug_image',
            'inspect_debug_topic': '/inspect/debug_image',
            'jpeg_quality': 70,
            'stream_fps': 15,
        }],
        output='screen',
    )

    return LaunchDescription([
        hmi_port_arg,
        handcam_arg,
        fixed_cam_arg,

        LogInfo(msg='====== QUVI Full System 시작 ======'),
        LogInfo(msg='  Web HMI: http://localhost:5000'),

        vision_launch,
        hmi_node,
    ])
