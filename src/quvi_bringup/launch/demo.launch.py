"""데모 전용 런치 — UI(hmi_node)만 기동한다.

로봇 구동 계열(robot_control, orchestrator, micro-ROS, vision, inspect)은 의도적으로
띄우지 않는다. 카메라·FSM·판정 토픽은 추후 ros2 bag 재생이 대신 공급한다
(docs/demo_dashboard.md 참고). 데모 판단(전체 뷰 패널·안내 모달)은 hmi_node가
static/demo/robot_overview.mp4 존재 여부로 스스로 한다.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

DEMO_BAG_DIR = '/workspace/data/demo_bags'
ACT_RRD_PATH = os.path.join(DEMO_BAG_DIR, 'act.rrd')


def generate_launch_description():
    hmi_port_arg = DeclareLaunchArgument('hmi_port', default_value='5000')

    # 토픽 이름은 full_system.launch.py 의 hmi_node 설정과 동일해야
    # 같은 토픽을 녹화한 bag 재생이 그대로 붙는다.
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

    demo_controller = Node(
        package='quvi_hmi',
        executable='demo_controller',
        name='demo_controller',
        parameters=[{'bag_dir': DEMO_BAG_DIR}],
        output='screen',
    )

    actions = [hmi_port_arg, hmi_node, demo_controller]

    # act.rrd 가 있을 때만 rerun 웹 뷰어로 재서빙 (없으면 조용히 생략 — 평시 무영향)
    if os.path.exists(ACT_RRD_PATH):
        actions.append(ExecuteProcess(
            cmd=['rerun', '--serve-web', ACT_RRD_PATH,
                 '--web-viewer-port', '9090', '--ws-server-port', '9877'],
            output='screen',
        ))

    return LaunchDescription(actions)
