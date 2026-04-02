# 양불 판정 결과 (/inspection/result 토픽)
std_msgs/Header header

# 판정 결과
bool passed                         # true=PASS, false=FAIL
string fail_reason                  # 불량 사유 (e.g. "warping", "stringing", "hole", "layer_sep")

# CAD 비교 결과 (4방향)
float32[] ssim_scores               # 4방향 SSIM 점수 [0°, 90°, 180°, 270°]
float32[] area_ratios               # 4방향 면적 비율
float32[] pixel_diff_ratios         # 4방향 픽셀 차이 비율

# 표면 특징 분석 결과
float32 solidity                    # 윤곽 면적 / 컨벡스 헐 면적
float32 area_ratio                  # 실제 면적 / 기준 면적
int32 hole_count                    # 내부 공동 개수
float32 hole_area_ratio             # 구멍 총 면적 / 전체 면적
float32 texture_variance            # 라플라시안 분산

# 메타 정보
int32 object_index                  # 현재 검사 중인 출력물 번호
float32 inspection_time_sec         # 검사 소요 시간 (초)
