# 시스템 전체 상태 (/hmi/status 토픽)
std_msgs/Header header

# 상태 머신
string current_state        # IDLE, DETECTING, GRASPING, INSPECTING, SORTING, DONE, ERROR
int32 total_objects          # 총 출력물 개수
int32 processed_count        # 처리 완료 개수
int32 pass_count             # 합격 개수
int32 fail_count             # 불합격 개수

# 하위 시스템 상태
bool yolo_ready
bool grasp_ready
bool inspect_ready
bool motor_ready

# 에러 정보
string error_message
