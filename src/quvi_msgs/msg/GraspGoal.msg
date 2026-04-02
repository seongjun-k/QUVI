# 파지 목표 좌표 (/grasp/command 토픽)
std_msgs/Header header
float32 target_x        # 목표 x 좌표 (px → mm 변환 후)
float32 target_y        # 목표 y 좌표 (px → mm 변환 후)
float32 target_z        # 목표 z 좌표 (높이)
int32 object_index      # 출력물 인덱스
