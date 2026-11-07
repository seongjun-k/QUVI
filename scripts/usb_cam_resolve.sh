#!/bin/bash
# usb_cam 실행 직전에 카메라 장치를 실제 /dev/videoN 으로 풀어 넘긴다.
# 사용: usb_cam_resolve.sh <장치 경로> <usb_cam 실행 파일> [인자...]
# - /dev/v4l/by-path/...(꽂힌 포트 고정 이름)를 쓰면 허브가 재연결돼 번호가 바뀌어도 맞는 카메라를 연다.
# - usb_cam 은 by-path 상대 링크를 잘못 풀어서 직접 넘기면 장치를 못 연다.
# - launch 의 respawn 으로 다시 켜질 때마다 새로 푼다.
dev=$(readlink -f "$1")
shift
exec "$@" -p "video_device:=${dev}"
