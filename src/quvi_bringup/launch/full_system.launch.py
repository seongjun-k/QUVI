"""
QUVI 전체 시스템 Launch 파일
───────────────────────────────
카메라 + YOLO + 검사 + HMI Web UI 한 번에 실행.

사용법:
  ros2 launch quvi_bringup full_system.launch.py
  ros2 launch quvi_bringup full_system.launch.py hmi_port:=8080
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    # ─── Launch Arguments ───
    hmi_port_arg = DeclareLaunchArgument(
        'hmi_port', default_value='5000',
        description='HMI Web UI 포트')

    handcam_arg = DeclareLaunchArgument(
        'handcam_device', default_value='/dev/handcam',
        description='핸드캠 장치 경로')

    fixed_cam_arg = DeclareLaunchArgument(
        'fixed_cam_device', default_value='/dev/fixed_cam',
        description='고정 카메라 장치 경로')

    use_real_hardware_arg = DeclareLaunchArgument(
        'use_real_hardware', default_value='true',
        description='실제 Dynamixel 하드웨어 사용 여부 (false=시뮬레이션)')

    use_act_arg = DeclareLaunchArgument(
        'use_act', default_value='true',
        description='ACT 모방학습 정책 로드 여부')

    yolo_model_path_arg = DeclareLaunchArgument(
        'yolo_model_path', default_value='/workspace/data/models/best.pt',
        description='YOLO 모델 파일 경로')

    follower_port_arg = DeclareLaunchArgument(
        'dxl_port', default_value='/dev/ttyFollower',
        description='팔로워 암 Dynamixel 포트')

    leader_port_arg = DeclareLaunchArgument(
        'leader_dxl_port', default_value='/dev/ttyLeader',
        description='리더 암 Dynamixel 포트 (텔레오퍼레이션)')

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

    micro_ros_port_arg = DeclareLaunchArgument(
        'micro_ros_port', default_value='/dev/ttyESP32',
        description='micro-ROS agent 시리얼 포트')

    micro_ros_baud_arg = DeclareLaunchArgument(
        'micro_ros_baud', default_value='921600',
        description='micro-ROS agent 보 레이트')

    # ─── Vision Pipeline 포함 ───
    bringup_dir = get_package_share_directory('quvi_bringup')
    vision_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_dir, 'launch', 'vision_pipeline.launch.py')),
        launch_arguments={
            'handcam_device': LaunchConfiguration('handcam_device'),
            'fixed_cam_device': LaunchConfiguration('fixed_cam_device'),
            'handcam_autoexposure': LaunchConfiguration('handcam_autoexposure'),
            'handcam_exposure': LaunchConfiguration('handcam_exposure'),
            'fixed_cam_autoexposure': LaunchConfiguration('fixed_cam_autoexposure'),
            'fixed_cam_exposure': LaunchConfiguration('fixed_cam_exposure'),
            'yolo_model_path': LaunchConfiguration('yolo_model_path'),
        }.items(),
    )

    # ─── HMI_NODE ───
    hmi_node = Node(
        package='quvi_hmi',
        executable='hmi_node',
        name='hmi_node',
        parameters=[{
            'host': '0.0.0.0',
            'port': ParameterValue(LaunchConfiguration('hmi_port'), value_type=int),
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

    # ─── ROBOT_CONTROL_NODE (로봇팔 + 텔레오퍼레이션) ───
    robot_control_node = Node(
        package='quvi_robot_control',
        executable='robot_control_node',
        name='robot_control_node',
        parameters=[{
            'use_real_hardware': LaunchConfiguration('use_real_hardware'),
            'use_act': LaunchConfiguration('use_act'),
            'dxl_port': LaunchConfiguration('dxl_port'),
            'leader_dxl_port': LaunchConfiguration('leader_dxl_port'),
            'dxl_baudrate': 1000000,
            'act_device': 'cpu',
            'handcam_topic': '/camera1/image_raw/compressed',
            'use_compressed': True,
        }],
        output='screen',
    )

    # ─── MAIN_ORCHESTRATOR_NODE ───
    main_orchestrator_node = Node(
        package='quvi_robot_control',
        executable='main_orchestrator_node',
        name='main_orchestrator_node',
        parameters=[{
            'use_act': LaunchConfiguration('use_act'),
            'px_to_mm_x': 0.5,
            'px_to_mm_y': 0.5,
            'offset_x': 100.0,
            'offset_y': 100.0,
            'target_z': 15.0,
            'step_delay_sec': 2.0,
            'loop_rate_hz': 10.0,
        }],
        output='screen',
    )

    micro_ros_agent = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'micro_ros_agent', 'micro_ros_agent',
            'serial', '--dev',
            LaunchConfiguration('micro_ros_port'),
            '-b',
            LaunchConfiguration('micro_ros_baud'),
        ],
        output='screen',
    )

    return LaunchDescription([
        hmi_port_arg,
        handcam_arg,
        fixed_cam_arg,
        use_real_hardware_arg,
        use_act_arg,
        yolo_model_path_arg,
        follower_port_arg,
        leader_port_arg,
        handcam_autoexposure_arg,
        handcam_exposure_arg,
        fixed_cam_autoexposure_arg,
        fixed_cam_exposure_arg,
        micro_ros_port_arg,
        micro_ros_baud_arg,

        LogInfo(msg='====== QUVI Full System 시작 ======'),
        LogInfo(msg='  Web HMI: http://localhost:5000'),

        micro_ros_agent,
        vision_launch,
        hmi_node,
        robot_control_node,
        main_orchestrator_node,
    ])
