"""VLA 입력 카메라(사이드캠 camera1·탑뷰 camera3)의 v4l2 설정 프로필.

수집 앱이 현장에서 맞춘 값을 data/camera_profile.json 에 저장하고, 수집 세션 시작과
로봇 시스템 기동(vision_pipeline.launch.py) 때 같은 값을 적용한다 — 수집과 추론의 영상 조건을 맞춘다.
ROS import 금지(launch 파일·수집 앱에서 import 된다).
"""
import json
import os
import re

PROFILE_FILE = 'camera_profile.json'  # data_dir 아래. 장치마다 값이 달라 git 에 올리지 않는다
CAMERAS = ('camera1', 'camera3')
# 적용 순서가 의미를 가진다: 자동 모드를 먼저 꺼야 수동 값이 먹는다(--set-ctrl 콤마 인자는 왼쪽부터 적용).
# exposure_dynamic_framerate=1 이면 노출이 길 때 30fps 아래로 떨어지므로 함께 고정한다.
CTRL_ORDER = ('auto_exposure', 'white_balance_automatic', 'exposure_dynamic_framerate',
              'exposure_time_absolute', 'gain', 'brightness', 'white_balance_temperature')
AUTO_EXPOSURE_MANUAL = 1

_LINE_RE = re.compile(r'^\s*(\w+) 0x[0-9a-f]+ \((\w+)\)\s*:\s*(.*)$')
_KV_RE = re.compile(r'\b(min|max|step|default|value)=(-?\d+)')


def parse_list_ctrls(text: str) -> dict:
    """`v4l2-ctl --list-ctrls` 출력 → {이름: {type, value, min, max, step, default, inactive}}. CTRL_ORDER 만."""
    out = {}
    for line in text.splitlines():
        m = _LINE_RE.match(line)
        if not m or m.group(1) not in CTRL_ORDER:
            continue
        name, typ, rest = m.groups()
        info = {'type': typ, 'inactive': 'inactive' in rest}
        info.update({k: int(v) for k, v in _KV_RE.findall(rest)})
        if 'value' in info:
            out[name] = info
    return out


def set_ctrl_arg(ctrls: dict) -> str:
    """{이름: 값} → `v4l2-ctl --set-ctrl` 인자. CTRL_ORDER 순서로, 모르는 이름은 버린다."""
    return ','.join(f'{k}={int(ctrls[k])}' for k in CTRL_ORDER if k in ctrls)


def from_ctrls(ctrls: dict) -> dict:
    """parse_list_ctrls 결과 → 프로필 값. 자동 모드면 그 모드가 무시하는 수동 값은 넣지 않는다
    (넣으면 적용 때마다 장치가 거부해 불일치 경고만 생긴다)."""
    vals = {k: c['value'] for k, c in ctrls.items() if k in CTRL_ORDER}
    if vals.get('auto_exposure', AUTO_EXPOSURE_MANUAL) != AUTO_EXPOSURE_MANUAL:
        vals.pop('exposure_time_absolute', None)
    if vals.get('white_balance_automatic', 0):
        vals.pop('white_balance_temperature', None)
    return vals


def mismatches(want: dict, got: dict) -> list:
    """적용하려던 값과 읽은 값이 다른 항목 설명(범위 밖 값이 잘리거나 장치가 거부한 경우)."""
    return [f'{k}: 설정 {want[k]} / 실제 {got.get(k, "없음")}'
            for k in CTRL_ORDER if k in want and got.get(k) != want[k]]


def load(data_dir: str) -> dict:
    """프로필 {camera: {이름: 값}}. 파일이 없거나 깨졌으면 {}(= 적용 안 함)."""
    try:
        with open(os.path.join(data_dir, PROFILE_FILE), encoding='utf-8') as f:
            raw = json.load(f)
        return {cam: {k: int(v) for k, v in raw[cam].items() if k in CTRL_ORDER}
                for cam in CAMERAS if isinstance(raw.get(cam), dict)}
    except (OSError, ValueError, TypeError, AttributeError):
        return {}


def save(data_dir: str, profile: dict) -> None:
    clean = {cam: {k: int(profile[cam][k]) for k in CTRL_ORDER if k in profile[cam]}
             for cam in CAMERAS if cam in profile}
    path = os.path.join(data_dir, PROFILE_FILE)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
