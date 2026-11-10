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
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from quvi_robot_control import camera_profile

# 탑뷰 프로필이 없을 때의 고정값 — 자동노출 + brightness 50 이면 베드 반사가 포화된다(2026-10-06 실측 포화 14%)
TOPCAM_FALLBACK_CTRLS = 'auto_exposure=1,exposure_time_absolute=80,brightness=0'


def _vla_cam_ctrls(context):
    """수집 앱이 저장한 data/camera_profile.json 이 있으면 그 값을 건다 — 수집과 추론의 영상 조건을 맞춘다.
    없으면 기존 동작(탑뷰 고정값, 사이드캠은 손대지 않음)."""
    profile = camera_profile.load(LaunchConfiguration('data_dir').perform(context))
    actions = []
    for cam, dev_arg, fallback in (('camera1', 'sidecam_device', ''),
                                   ('camera3', 'topcam_device', TOPCAM_FALLBACK_CTRLS)):
        arg = camera_profile.set_ctrl_arg(profile[cam]) if cam in profile else fallback
        if not arg:
            continue
        src = '프로필' if cam in profile else '기본값'
        actions += [LogInfo(msg=f'{cam} v4l2 설정({src}): {arg}'),
                    ExecuteProcess(cmd=['v4l2-ctl', '-d', LaunchConfiguration(dev_arg), '--set-ctrl', arg],
                                   output='screen')]
    return actions


def _default_data_dir() -> str:
    """데이터 디렉토리 기본값 계산.

    절대경로 하드코딩 대신 환경변수 QUVI_DATA_DIR > 컨테이너 표준 경로
    /workspace/data 순으로 결정한다. launch arg(data_dir)로 override 가능.
    """
    return os.environ.get('QUVI_DATA_DIR', '/workspace/data')


# usb_cam 실행 직전에 by-path 장치를 실제 /dev/videoN 으로 푸는 래퍼(scripts/usb_cam_resolve.sh)
_CAM_RESOLVE = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..', 'scripts', 'usb_cam_resolve.sh')


def generate_launch_description():

    default_data_dir = _default_data_dir()

    # ─── 런치 인자 ───
    sidecam_device_arg = DeclareLaunchArgument(
        'sidecam_device', default_value='/dev/sidecam',
        description='사이드캠(Zone 1) USB 장치 경로')

    fixed_cam_device_arg = DeclareLaunchArgument(
        'fixed_cam_device', default_value='/dev/fixed_cam',
        description='고정 카메라(Zone 2 검사 챔버) USB 장치 경로')

    topcam_device_arg = DeclareLaunchArgument(
        'topcam_device', default_value='/dev/topcam',
        description='탑뷰 카메라(Zone 3 3D 프린터 상단) USB 장치 경로')

    data_dir_arg = DeclareLaunchArgument(
        'data_dir', default_value=default_data_dir,
        description='기준/로그 데이터 루트 (기본: $QUVI_DATA_DIR 또는 /workspace/data)')

    reference_dir_arg = DeclareLaunchArgument(
        'reference_image_dir',
        default_value=[LaunchConfiguration('data_dir'), '/reference_images'],
        description='기준 이미지 디렉토리')

    inspection_log_dir_arg = DeclareLaunchArgument(
        'inspection_log_dir',
        default_value=[LaunchConfiguration('data_dir'), '/inspection_logs'],
        description='검사 로그 저장 디렉토리')

    # 다품종 검사 자산 루트 (reference_image_dir/anomaly_model_dir 는 레거시로 유지 —
    # inspect_node 는 기동 시 UNSELECTED 로 시작해 이 디렉토리 하위 품종을 선택받는다)
    inspection_products_dir_arg = DeclareLaunchArgument(
        'inspection_products_dir',
        default_value=[LaunchConfiguration('data_dir'), '/inspection_products'],
        description='다품종 검사 자산 루트 (품종별 reference_images/models/anomaly_dataset/params.json)')

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
        'fixed_cam_exposure', default_value='70',
        description='고정캠(검사챔버) 수동 노출값 (autoexposure가 false일 때 적용) — 조명 ON 시 과노출 방지 실측 튜닝')

    fixed_cam_rotate_180_arg = DeclareLaunchArgument(
        'fixed_cam_rotate_180', default_value='false',
        description='검사캠을 거꾸로 장착했을 때 true — 검사 프레임을 180° 회전')

    # 재기동마다 HMI에서 품종을 다시 골라야 했다 — 미선택이면 판정 불가로 FAIL 처리된다(2026-11-09)
    inspection_product_arg = DeclareLaunchArgument(
        'inspection_product', default_value='box',
        description="시작 시 자동 선택할 검사 품종 ('' = 미선택으로 시작)")

    anomaly_enabled_arg = DeclareLaunchArgument(
        'anomaly_enabled', default_value='false',
        description='PatchCore 이상탐지 하이브리드 판정 활성화 여부 (룰 PASS 라도 ML FAIL 이면 최종 FAIL)')

    area_ratio_enabled_arg = DeclareLaunchArgument(
        'area_ratio_enabled', default_value='false',
        description='면적비 축(기준이미지 필요, 화각 가림에 취약) 판정 사용 여부. false면 표면 축+ML로만 판정')

    # ─── 카메라 1: 사이드캠 (Zone 1 - 픽업 영역) ───
    camera1_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='camera1',
        namespace='camera1',
        prefix=[_CAM_RESOLVE + ' ', LaunchConfiguration('sidecam_device')],
        respawn=True, respawn_delay=3.0,   # 허브 재연결로 죽으면 다시 켬(번호는 래퍼가 새로 풂)
        parameters=[{
            'video_device': LaunchConfiguration('sidecam_device'),
            'image_width': 640,
            'image_height': 480,
            'pixel_format': 'mjpeg2rgb',
            'framerate': 30.0,
            'camera_name': 'sidecam',
            'autoexposure': ParameterValue(LaunchConfiguration('sidecam_autoexposure'), value_type=bool),
            'exposure': ParameterValue(LaunchConfiguration('sidecam_exposure'), value_type=int),
            # ACT 학습 데이터가 brightness 0 기준 — 장치 잔존값(50 등)이 관측 분포를 흔들지 않게 고정
            'brightness': 0,
        }],
        remappings=[
            ('image_raw', 'image_raw'),
            ('image_raw/compressed', 'image_raw/compressed'),
        ],
    )

    # ─── 카메라 2: 검사캠 (Zone 2 - 검사 챔버) ───
    camera2_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='camera2',
        namespace='camera2',
        prefix=[_CAM_RESOLVE + ' ', LaunchConfiguration('fixed_cam_device')],
        # 2026-11-08 검사 중 exit -6 으로 죽어 검사 타임아웃. 다시 켜지면 inspect_node 가 전환 상태를 다시 맞춘다
        respawn=True, respawn_delay=3.0,
        parameters=[{
            'video_device': LaunchConfiguration('fixed_cam_device'),
            # 2026-11-08 640x480 로 낮춰 3대 동시를 시도했으나 탑뷰가 죽었다(해상도 무관, 실측 재확인)
            'image_width': 1920,
            'image_height': 1080,
            'pixel_format': 'mjpeg2rgb',
            'framerate': 30.0,
            'camera_name': 'inspection_cam',
            'autoexposure': ParameterValue(LaunchConfiguration('fixed_cam_autoexposure'), value_type=bool),
            'exposure': ParameterValue(LaunchConfiguration('fixed_cam_exposure'), value_type=int),
            # 조명 ON 시 잔존값(50)에 의한 과노출 방지
            'brightness': 0,
        }],
        remappings=[
            ('image_raw', 'image_raw'),
            ('image_raw/compressed', 'image_raw/compressed'),
        ],
    )

    # ─── 카메라 3: 탑뷰캠 (Zone 3 - 3D 프린터 상단) ───
    camera3_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='camera3',
        namespace='camera3',
        prefix=[_CAM_RESOLVE + ' ', LaunchConfiguration('topcam_device')],
        respawn=True, respawn_delay=3.0,   # 허브 재연결로 죽으면 다시 켬(번호는 래퍼가 새로 풂)
        parameters=[{
            'video_device': LaunchConfiguration('topcam_device'),
            'image_width': 640,
            'image_height': 480,
            'pixel_format': 'mjpeg2rgb',
            'framerate': 30.0,
            'camera_name': 'topcam',
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
                # 노출 v4l2-ctl 대상 — usb_cam 과 같은 장치여야 한다(HMI 장치 재매핑 시 /dev/fixed_cam 고정이면 노출 설정 실패)
                'inspection_cam_device': LaunchConfiguration('fixed_cam_device'),
                'reference_image_dir': LaunchConfiguration('reference_image_dir'),
                'inspection_log_dir': LaunchConfiguration('inspection_log_dir'),
                'inspection_products_dir': LaunchConfiguration('inspection_products_dir'),
                'anomaly_enabled': ParameterValue(LaunchConfiguration('anomaly_enabled'), value_type=bool),
                'default_product': ParameterValue(LaunchConfiguration('inspection_product'), value_type=str),
                'area_ratio_enabled': ParameterValue(LaunchConfiguration('area_ratio_enabled'), value_type=bool),
                'fixed_cam_rotate_180': ParameterValue(LaunchConfiguration('fixed_cam_rotate_180'), value_type=bool),
                'anomaly_model_dir': [LaunchConfiguration('data_dir'), '/models'],
                'anomaly_device': 'cuda',
            }
        ],
        output='screen',
    )

    # ─── VLA 입력 카메라(사이드캠·탑뷰) v4l2 설정 ───
    # usb_cam 은 구형 컨트롤명(exposure_auto/exposure_absolute)을 써서 이 커널에선 노출 설정이 무시된다.
    # usb_cam 초기화 후 v4l2-ctl 로 덮어쓴다. 정책 입력 영상이 바뀌므로 값 변경 시 파지 성능 재확인 필요.
    # ─── 카메라 시분할 기동 순서 (USB 대역폭) ───
    # 카메라 3대를 동시에 켜면 마지막 카메라가 대역폭이 없어 죽는다. 검사캠을 먼저 켜서 멈춰 두고
    # (검사 때 inspect_node 가 켬), 사이드캠·탑뷰를 그 뒤에 켠다.
    pause_inspect_cam = TimerAction(period=2.0, actions=[ExecuteProcess(
        cmd=['ros2', 'service', 'call', '/camera2/set_capture', 'std_srvs/srv/SetBool', '{data: false}'],
        output='screen')])
    vla_cams = TimerAction(period=7.0, actions=[camera1_node, camera3_node])
    # usb_cam 이 기동 때 자기 컨트롤을 걸므로 그 뒤에 덮어쓴다(카메라 기동 + 4초)
    vla_cam_ctrls = TimerAction(period=11.0, actions=[OpaqueFunction(function=_vla_cam_ctrls)])

    return LaunchDescription([
        # 인자
        sidecam_device_arg,
        fixed_cam_device_arg,
        topcam_device_arg,
        data_dir_arg,
        reference_dir_arg,
        inspection_log_dir_arg,
        inspection_products_dir_arg,
        inspect_topic_arg,
        inspect_config_arg,
        sidecam_autoexposure_arg,
        sidecam_exposure_arg,
        fixed_cam_autoexposure_arg,
        fixed_cam_exposure_arg,
        fixed_cam_rotate_180_arg,
        inspection_product_arg,
        anomaly_enabled_arg,
        area_ratio_enabled_arg,

        # 로그
        LogInfo(msg='====== QUVI Vision Pipeline 시작 ======'),

        # 노드
        camera2_node,
        pause_inspect_cam,
        vla_cams,
        vla_cam_ctrls,
        inspect_node,
    ])
