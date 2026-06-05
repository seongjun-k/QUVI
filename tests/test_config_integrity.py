"""
QUVI 설정/패키지 무결성 회귀 테스트 (ROS 불필요, 순수 pytest).

이전 분석 리포트 4.2 의 결함이 재발하지 않도록 정적으로 검증한다:
  - quvi_robot_control 빌드 설정 (ament_python 일관성, console_scripts, setup.cfg)
  - launch 파일에 절대경로(/home/...) 하드코딩 없음
  - micro-ROS 보드레이트 일관성 (Config.h / agent 문서 / 스크립트)
실행: cd <repo> && pytest tests/
"""

import os
import re
import xml.etree.ElementTree as ET

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, 'src')
FIRMWARE = os.path.join(REPO_ROOT, 'firmware', 'quvi_esp32_firmware')


def _read(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


# ─────────────────────────────────────────────
# 항목 1: quvi_robot_control 빌드 설정
# ─────────────────────────────────────────────
def test_robot_control_package_xml_is_ament_python_only():
    pkg = os.path.join(SRC, 'quvi_robot_control', 'package.xml')
    root = ET.parse(pkg).getroot()
    buildtools = [e.text for e in root.findall('buildtool_depend')]
    assert 'ament_python' in buildtools
    assert 'ament_cmake' not in buildtools, 'ament_cmake 와 ament_python 혼용 금지'
    build_types = [e.text for e in root.findall('./export/build_type')]
    assert build_types == ['ament_python']


def test_robot_control_setup_cfg_installs_scripts():
    cfg = os.path.join(SRC, 'quvi_robot_control', 'setup.cfg')
    assert os.path.isfile(cfg), 'setup.cfg 없으면 console_scripts 설치 경로 어긋남'
    text = _read(cfg)
    assert 'install_scripts=$base/lib/quvi_robot_control' in text


def test_robot_control_entry_points_present():
    setup_py = _read(os.path.join(SRC, 'quvi_robot_control', 'setup.py'))
    assert 'robot_control_node = quvi_robot_control.robot_control_node:main' in setup_py
    assert 'main_orchestrator_node = quvi_robot_control.main_orchestrator_node:main' in setup_py


# ─────────────────────────────────────────────
# 항목 2: launch 절대경로 하드코딩 금지
# ─────────────────────────────────────────────
def test_no_hardcoded_home_paths_in_launch():
    launch_dir = os.path.join(SRC, 'quvi_bringup', 'launch')
    offenders = []
    for name in os.listdir(launch_dir):
        if not name.endswith('.py'):
            continue
        text = _read(os.path.join(launch_dir, name))
        if re.search(r'/home/\w+/', text):
            offenders.append(name)
    assert not offenders, f'launch 에 절대 홈 경로 하드코딩 발견: {offenders}'


def test_vision_launch_uses_data_dir_arg():
    text = _read(os.path.join(SRC, 'quvi_bringup', 'launch',
                              'vision_pipeline.launch.py'))
    assert "'data_dir'" in text
    assert 'reference_image_dir' in text
    assert 'inspection_log_dir' in text


# ─────────────────────────────────────────────
# 항목 3: micro-ROS 보드레이트 일관성
# ─────────────────────────────────────────────
def test_microros_baudrate_consistent_921600():
    config_h = _read(os.path.join(FIRMWARE, 'Config.h'))
    m = re.search(r'#define\s+MICRO_ROS_BAUDRATE\s+(\d+)', config_h)
    assert m, 'Config.h 에 MICRO_ROS_BAUDRATE 정의 없음'
    assert m.group(1) == '921600'

    readme = _read(os.path.join(FIRMWARE, 'README.md'))
    assert '-b 921600' in readme, 'firmware README agent 명령 보드레이트 불일치'

    agent_script = os.path.join(REPO_ROOT, 'scripts', 'run_microros_agent.sh')
    assert os.path.isfile(agent_script), 'micro-ROS agent 실행 스크립트 누락'
    assert '921600' in _read(agent_script)


# ─────────────────────────────────────────────
# 항목 5: 핸드캠 토픽 일관성
# ─────────────────────────────────────────────
def test_handcam_topic_default_matches_camera1():
    rc = _read(os.path.join(SRC, 'quvi_robot_control', 'quvi_robot_control',
                            'robot_control_node.py'))
    assert "'handcam_topic', '/camera1/image_raw/compressed'" in rc
