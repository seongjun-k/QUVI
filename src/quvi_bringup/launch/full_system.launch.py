"""
QUVI 전체 시스템 Launch 파일
───────────────────────────────
카메라 + 검사 + HMI Web UI 한 번에 실행.

사용법:
  ros2 launch quvi_bringup full_system.launch.py
  ros2 launch quvi_bringup full_system.launch.py hmi_port:=8080
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os


def _load_device_config():
    """대시보드에서 저장한 장치 매핑을 읽어 기본값으로 사용한다.
    파일이 없거나 손상되면 빈 dict → 기존 하드코딩 기본값을 사용한다.
    """
    import json
    for path in ('/workspace/data/device_config.json',
                 os.path.join(os.path.dirname(__file__), '..', '..', '..',
                              'data', 'device_config.json')):
        try:
            with open(path, encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            continue
    return {}


def generate_launch_description():

    # 대시보드 장치 설정(있으면) 로드 — 하드코딩 기본값을 덮어쓴다.
    _dev = _load_device_config()

    # ─── Launch Arguments ───
    hmi_port_arg = DeclareLaunchArgument(
        'hmi_port', default_value='5000',
        description='HMI Web UI 포트')

    sidecam_arg = DeclareLaunchArgument(
        'sidecam_device', default_value=_dev.get('sidecam_device', '/dev/sidecam'),
        description='사이드캠 장치 경로')

    fixed_cam_arg = DeclareLaunchArgument(
        'fixed_cam_device', default_value=_dev.get('fixed_cam_device', '/dev/fixed_cam'),
        description='고정 카메라 장치 경로')

    use_real_hardware_arg = DeclareLaunchArgument(
        'use_real_hardware', default_value='true',
        description='실제 Dynamixel 하드웨어 사용 여부 (false=시뮬레이션)')

    use_act_arg = DeclareLaunchArgument(
        'use_act', default_value='false',
        description='ACT 모방학습 정책 로드 여부 (demo/sequence-no-act 에서는 false)')

    follower_port_arg = DeclareLaunchArgument(
        'dxl_port', default_value=_dev.get('dxl_port', '/dev/ttyFollower'),
        description='팔로워 암 Dynamixel 포트')

    leader_port_arg = DeclareLaunchArgument(
        'leader_dxl_port', default_value=_dev.get('leader_dxl_port', '/dev/ttyLeader'),
        description='리더 암 Dynamixel 포트 (텔레오퍼레이션)')

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

    micro_ros_port_arg = DeclareLaunchArgument(
        'micro_ros_port', default_value=_dev.get('micro_ros_port', '/dev/ttyESP32'),
        description='micro-ROS agent 시리얼 포트')

    micro_ros_baud_arg = DeclareLaunchArgument(
        'micro_ros_baud', default_value='115200',
        description='micro-ROS agent 보 레이트')

    # ─── Vision Pipeline 포함 ───
    bringup_dir = get_package_share_directory('quvi_bringup')
    vision_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_dir, 'launch', 'vision_pipeline.launch.py')),
        launch_arguments={
            'sidecam_device': LaunchConfiguration('sidecam_device'),
            'fixed_cam_device': LaunchConfiguration('fixed_cam_device'),
            'sidecam_autoexposure': LaunchConfiguration('sidecam_autoexposure'),
            'sidecam_exposure': LaunchConfiguration('sidecam_exposure'),
            'fixed_cam_autoexposure': LaunchConfiguration('fixed_cam_autoexposure'),
            'fixed_cam_exposure': LaunchConfiguration('fixed_cam_exposure'),
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
            'sidecam_topic': '/camera1/image_raw/compressed',
            'camera2_topic': '/camera2/image_raw/compressed',
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
            'sidecam_topic': '/camera1/image_raw/compressed',
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
            '/usr/local/bin/run_micro_ros_agent.sh',
            'serial', '--dev',
            LaunchConfiguration('micro_ros_port'),
            '-b',
            LaunchConfiguration('micro_ros_baud'),
        ],
        output='screen',
        # HMI 리셋 시 오케스트레이터가 agent 를 종료 → ESP32 하드 리셋 순으로
        # USB 링크를 재수립한다. 종료된 agent 는 여기서 자동 재기동된다.
        respawn=True,
        respawn_delay=2.0,
    )

    # ESP32 는 agent 재기동에 재협상하지 못하므로, agent 기동 전에 DTR/RTS로
    # 하드 리셋해 프레시 부팅시킨다 (부팅 시 재시도 루프가 새 agent를 잡음).
    esp32_reset = ExecuteProcess(
        cmd=[
            'python3', '/workspace/scripts/reset_esp32.py',
            '--port', LaunchConfiguration('micro_ros_port'),
        ],
        name='esp32_reset',
        output='screen',
    )

    return LaunchDescription([
        hmi_port_arg,
        sidecam_arg,
        fixed_cam_arg,
        use_real_hardware_arg,
        use_act_arg,
        follower_port_arg,
        leader_port_arg,
        sidecam_autoexposure_arg,
        sidecam_exposure_arg,
        fixed_cam_autoexposure_arg,
        fixed_cam_exposure_arg,
        micro_ros_port_arg,
        micro_ros_baud_arg,

        LogInfo(msg='====== QUVI Full System 시작 ======'),
        LogInfo(msg='  Web HMI: http://localhost:5000'),

        esp32_reset,
        RegisterEventHandler(
            OnProcessExit(target_action=esp32_reset, on_exit=[micro_ros_agent])
        ),
        vision_launch,
        hmi_node,
        robot_control_node,
        main_orchestrator_node,
    ])
