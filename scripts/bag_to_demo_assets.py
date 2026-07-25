#!/usr/bin/env python3
"""데모 bag → 정적 데모 사이트(gh-pages) 자산 추출.

카메라 토픽은 이미 JPEG(CompressedImage)라 디코드 없이 ffmpeg 에 그대로 흘려보낸다.
검사 오버레이(/inspect/debug_image)는 raw Image 라 디코드해서 png 로 저장한다.

사용법(컨테이너 안):
    python3 bag_to_demo_assets.py /workspace/data/demo_bags/pass /workspace/data/demo_assets pass
"""

import os
import subprocess
import sys

import numpy as np
import cv2
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

CAM_TOPICS = {
    '/camera1/image_raw/compressed': 'sidecam',
    '/camera2/image_raw/compressed': 'camera2',
}
# 검사캠(camera2)은 거꾸로 장착돼 있어 hmi_node 가 표시 직전 cv2.flip(frame, -1) 로
# 180° 돌린다. bag 은 원본 토픽이라 여기서 같은 보정을 걸어야 화면과 일치한다.
EXTRA_VF = {'camera2': 'hflip,vflip'}
DEBUG_TOPIC = '/inspect/debug_image'
# 대시보드 패널이 작아 원본 해상도가 불필요하다 — 폭 기준 640으로 줄여 용량을 억제한다.
TARGET_WIDTH = 640
CRF = '28'


def read_bag(bag_path):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap'),
        rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}

    frames = {name: [] for name in CAM_TOPICS.values()}
    stamps = {name: [] for name in CAM_TOPICS.values()}
    debug_img = None

    while reader.has_next():
        topic, data, ts = reader.read_next()
        if topic in CAM_TOPICS:
            msg = deserialize_message(data, get_message(types[topic]))
            key = CAM_TOPICS[topic]
            frames[key].append(bytes(msg.data))
            stamps[key].append(ts)
        elif topic == DEBUG_TOPIC and debug_img is None:
            msg = deserialize_message(data, get_message(types[topic]))
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, -1)
            # inspect_node 는 bgr8 로 발행한다 — cv2.imwrite 가 기대하는 순서와 동일.
            debug_img = arr
    return frames, stamps, debug_img


def encode_mp4(jpegs, fps, out_path, extra_vf=None):
    # 브라우저 재생을 위해 H.264 + yuv420p. faststart 로 메타데이터를 앞으로 옮겨
    # 다운로드 완료 전에도 재생이 시작되게 한다.
    vf = f'scale={TARGET_WIDTH}:-2'
    if extra_vf:
        vf = f'{extra_vf},{vf}'
    cmd = [
        'ffmpeg', '-y', '-f', 'image2pipe', '-framerate', f'{fps:.3f}', '-i', '-',
        '-vf', vf, '-c:v', 'libx264', '-preset', 'slow',
        '-crf', CRF, '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for jpg in jpegs:
        proc.stdin.write(jpg)
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(proc.stderr.read().decode()[-2000:])


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 1
    bag_path, out_dir, tag = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)

    frames, stamps, debug_img = read_bag(bag_path)

    for name, jpegs in frames.items():
        if not jpegs:
            print(f'  {name}: 프레임 없음 — 건너뜀')
            continue
        span_sec = (stamps[name][-1] - stamps[name][0]) / 1e9
        fps = (len(jpegs) - 1) / span_sec if span_sec > 0 else 30.0
        out_path = os.path.join(out_dir, f'{name}_{tag}.mp4')
        encode_mp4(jpegs, fps, out_path, EXTRA_VF.get(name))
        print(f'  {name}_{tag}.mp4: {len(jpegs)}프레임 {span_sec:.1f}s @{fps:.1f}fps '
              f'→ {os.path.getsize(out_path)/1e6:.1f}MB')

    if debug_img is not None:
        png_path = os.path.join(out_dir, f'captured_{tag}.png')
        cv2.imwrite(png_path, debug_img)
        print(f'  captured_{tag}.png: {debug_img.shape} '
              f'→ {os.path.getsize(png_path)/1e6:.1f}MB')
    else:
        print(f'  captured_{tag}.png: {DEBUG_TOPIC} 없음 — 건너뜀')
    return 0


if __name__ == '__main__':
    sys.exit(main())
