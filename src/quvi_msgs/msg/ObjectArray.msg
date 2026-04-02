# 탐지된 객체 목록 (/detection/objects 토픽)
std_msgs/Header header
DetectedObject[] objects    # 탐지된 출력물 배열
int32 total_count           # 총 감지 개수
