"""VLA 카메라 프로필: v4l2-ctl 출력 해석, 적용 순서, 저장·불러오기."""
import json

from quvi_robot_control import camera_profile as cp

# 노트북 내장 카메라 실제 출력 일부(2026-11-02) + 게인 줄
LIST_CTRLS = """
User Controls

                     brightness 0x00980900 (int)    : min=0 max=255 step=1 default=128 value=128 flags=0x00001000
                       contrast 0x00980901 (int)    : min=0 max=100 step=1 default=32 value=32 flags=0x00001000
        white_balance_automatic 0x0098090c (bool)   : default=1 value=1
      white_balance_temperature 0x0098091a (int)    : min=2800 max=6500 step=10 default=4600 value=4600 flags=inactive, 0x00001000
                           gain 0x00980913 (int)    : min=0 max=100 step=1 default=0 value=7

Camera Controls

                  auto_exposure 0x009a0901 (menu)   : min=0 max=3 default=3 value=3 (Aperture Priority Mode)
         exposure_time_absolute 0x009a0902 (int)    : min=2 max=1250 step=1 default=156 value=156 flags=inactive, 0x00001000
     exposure_dynamic_framerate 0x009a0903 (bool)   : default=0 value=1
"""


def test_parse_list_ctrls():
    c = cp.parse_list_ctrls(LIST_CTRLS)
    assert 'contrast' not in c  # 프로필 대상만
    assert c['brightness'] == {'type': 'int', 'inactive': False, 'min': 0, 'max': 255, 'step': 1,
                               'default': 128, 'value': 128}
    assert c['white_balance_temperature']['inactive'] is True
    assert c['auto_exposure']['value'] == 3 and c['auto_exposure']['type'] == 'menu'
    assert c['white_balance_automatic']['value'] == 1
    assert c['gain']['value'] == 7


def test_set_ctrl_arg_puts_auto_modes_first():
    arg = cp.set_ctrl_arg({'white_balance_temperature': 4000, 'exposure_time_absolute': 150, 'bogus': 1,
                           'white_balance_automatic': 0, 'auto_exposure': 1})
    assert arg == 'auto_exposure=1,white_balance_automatic=0,exposure_time_absolute=150,white_balance_temperature=4000'
    assert cp.set_ctrl_arg({}) == ''


def test_from_ctrls_drops_values_ignored_by_auto_modes():
    auto = cp.from_ctrls(cp.parse_list_ctrls(LIST_CTRLS))
    assert 'exposure_time_absolute' not in auto and 'white_balance_temperature' not in auto
    manual_text = LIST_CTRLS.replace('default=3 value=3', 'default=3 value=1').replace(
        'automatic 0x0098090c (bool)   : default=1 value=1', 'automatic 0x0098090c (bool)   : default=1 value=0')
    manual = cp.from_ctrls(cp.parse_list_ctrls(manual_text))
    assert manual['exposure_time_absolute'] == 156 and manual['white_balance_temperature'] == 4600


def test_mismatches():
    assert cp.mismatches({'brightness': 0}, {'brightness': 0}) == []
    assert cp.mismatches({'brightness': 0, 'gain': 5}, {'brightness': 3}) == ['gain: 설정 5 / 실제 없음',
                                                                             'brightness: 설정 0 / 실제 3']


def test_save_load_roundtrip_and_bad_files(tmp_path):
    assert cp.load(str(tmp_path)) == {}  # 파일 없음 = 적용 안 함
    cp.save(str(tmp_path), {'camera1': {'auto_exposure': 1, 'bogus': 9}, 'camera9': {'gain': 1}})
    assert cp.load(str(tmp_path)) == {'camera1': {'auto_exposure': 1}}
    (tmp_path / cp.PROFILE_FILE).write_text('{"camera1": {"gain": "x"}}')
    assert cp.load(str(tmp_path)) == {}
    (tmp_path / cp.PROFILE_FILE).write_text('[1]')
    assert cp.load(str(tmp_path)) == {}
    (tmp_path / cp.PROFILE_FILE).write_text(json.dumps({'camera3': {'brightness': 0}}))
    assert cp.load(str(tmp_path)) == {'camera3': {'brightness': 0}}
