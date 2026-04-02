"""
QUVI INSPECT_NODE
─────────────────
검사 챔버(Zone 2)에서 턴테이블 4방향 촬영 이미지를 받아
양불 판정을 수행하고 결과를 발행한다.

듀얼 검사 방식:
  1. CAD 기준 형상 비교 (SSIM, 면적 비율, 픽셀 차이)
  2. 표면 특징 기반 검사 (Solidity, Area Ratio, Hole Count, Hole Area, Texture)

판정 로직:
  PASS = (CAD 비교 ALL PASS) AND (표면 특징 ALL 정상)
  FAIL = otherwise
"""

import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, Int32

from quvi_msgs.msg import InspectionResult


class InspectNode(Node):
    """양불 판정 노드 — CAD 비교 + 표면 특징 분석."""

    def __init__(self):
        super().__init__('inspect_node')

        # ─── 파라미터 선언 ───
        self._declare_all_params()

        # ─── 파라미터 로드 ───
        self._load_params()

        # ─── 기준 이미지 로드 ───
        self._reference_images: Dict[int, np.ndarray] = {}
        self._load_reference_images()

        # ─── ROS 통신 ───
        self._bridge = CvBridge()

        # Subscriber: 검사 챔버 카메라
        if self._use_compressed:
            self._img_sub = self.create_subscription(
                CompressedImage, self._camera_topic,
                self._image_callback, 10)
        else:
            self._img_sub = self.create_subscription(
                Image, self._camera_topic,
                self._image_callback_raw, 10)

        # Subscriber: 턴테이블 현재 각도
        self._turntable_sub = self.create_subscription(
            Int32, '/motor/turntable',
            self._turntable_callback, 10)

        # Subscriber: 검사 트리거
        self._trigger_sub = self.create_subscription(
            Bool, '/inspection/trigger',
            self._trigger_callback, 10)

        # Publisher: 양불 판정 결과
        self._result_pub = self.create_publisher(
            InspectionResult, '/inspection/result', 10)

        # Publisher: 디버그 이미지
        if self._pub_debug:
            self._debug_pub = self.create_publisher(
                Image, self._debug_topic, 5)

        # 내부 상태
        self._latest_frame: Optional[np.ndarray] = None
        self._current_angle: int = 0
        self._captured_images: Dict[int, np.ndarray] = {}  # {angle: image}
        self._inspection_active = False
        self._current_object_index = 0

        self.get_logger().info(
            f'INSPECT_NODE 초기화 완료 | '
            f'SSIM 임계값: {self._ssim_thresh} | '
            f'촬영 각도: {self._angles}')

    # ─────────────────────────────────────────────
    # 파라미터
    # ─────────────────────────────────────────────
    def _declare_all_params(self):
        """모든 파라미터를 선언한다."""
        self.declare_parameter('camera_topic', '/camera2/image_raw/compressed')
        self.declare_parameter('use_compressed', True)
        self.declare_parameter('reference_image_dir', '/workspace/data/reference_images')
        # CAD 비교
        self.declare_parameter('ssim_threshold', 0.85)
        self.declare_parameter('area_ratio_min', 0.90)
        self.declare_parameter('area_ratio_max', 1.10)
        self.declare_parameter('pixel_diff_threshold', 0.10)
        # 표면 특징
        self.declare_parameter('solidity_min', 0.85)
        self.declare_parameter('solidity_max', 1.00)
        self.declare_parameter('feature_area_ratio_min', 0.90)
        self.declare_parameter('feature_area_ratio_max', 1.10)
        self.declare_parameter('hole_count_max', 2)
        self.declare_parameter('hole_area_ratio_max', 0.05)
        self.declare_parameter('texture_variance_max', 500.0)
        self.declare_parameter('min_hole_area_px', 50)
        # 턴테이블
        self.declare_parameter('turntable_angles', [0, 90, 180, 270])
        self.declare_parameter('capture_delay_sec', 0.5)
        # 전처리
        self.declare_parameter('roi_margin', 20)
        self.declare_parameter('gaussian_blur_ksize', 5)
        self.declare_parameter('binary_threshold', 127)
        # 디버그
        self.declare_parameter('save_inspection_images', True)
        self.declare_parameter('inspection_log_dir', '/workspace/data/inspection_logs')
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('debug_image_topic', '/inspect/debug_image')

    def _load_params(self):
        """파라미터를 멤버 변수로 로드한다."""
        self._camera_topic = self.get_parameter('camera_topic').value
        self._use_compressed = self.get_parameter('use_compressed').value
        self._ref_dir = self.get_parameter('reference_image_dir').value
        self._ssim_thresh = self.get_parameter('ssim_threshold').value
        self._area_min = self.get_parameter('area_ratio_min').value
        self._area_max = self.get_parameter('area_ratio_max').value
        self._px_diff_thresh = self.get_parameter('pixel_diff_threshold').value
        self._sol_min = self.get_parameter('solidity_min').value
        self._sol_max = self.get_parameter('solidity_max').value
        self._f_area_min = self.get_parameter('feature_area_ratio_min').value
        self._f_area_max = self.get_parameter('feature_area_ratio_max').value
        self._hole_max = self.get_parameter('hole_count_max').value
        self._hole_area_max = self.get_parameter('hole_area_ratio_max').value
        self._tex_var_max = self.get_parameter('texture_variance_max').value
        self._min_hole_px = self.get_parameter('min_hole_area_px').value
        self._angles = self.get_parameter('turntable_angles').value
        self._cap_delay = self.get_parameter('capture_delay_sec').value
        self._roi_margin = self.get_parameter('roi_margin').value
        self._blur_k = self.get_parameter('gaussian_blur_ksize').value
        self._bin_thresh = self.get_parameter('binary_threshold').value
        self._save_images = self.get_parameter('save_inspection_images').value
        self._log_dir = self.get_parameter('inspection_log_dir').value
        self._pub_debug = self.get_parameter('publish_debug_image').value
        self._debug_topic = self.get_parameter('debug_image_topic').value

    # ─────────────────────────────────────────────
    # 기준 이미지 로드
    # ─────────────────────────────────────────────
    def _load_reference_images(self):
        """기준 이미지(STL 렌더링 결과)를 로드한다.
        파일 네이밍: ref_0.png, ref_90.png, ref_180.png, ref_270.png
        """
        if not os.path.isdir(self._ref_dir):
            self.get_logger().warn(
                f'기준 이미지 디렉토리 없음: {self._ref_dir} — '
                f'stl_renderer로 먼저 생성하세요.')
            return

        for angle in self._angles:
            path = os.path.join(self._ref_dir, f'ref_{angle}.png')
            if os.path.isfile(path):
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self._reference_images[angle] = img
                    self.get_logger().info(f'기준 이미지 로드: {path}')
                else:
                    self.get_logger().warn(f'기준 이미지 읽기 실패: {path}')
            else:
                self.get_logger().warn(f'기준 이미지 없음: {path}')

        self.get_logger().info(
            f'기준 이미지 {len(self._reference_images)}/{len(self._angles)}개 로드됨')

    # ─────────────────────────────────────────────
    # 콜백
    # ─────────────────────────────────────────────
    def _image_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is not None:
            self._latest_frame = frame

    def _image_callback_raw(self, msg: Image):
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if frame is not None:
            self._latest_frame = frame

    def _turntable_callback(self, msg: Int32):
        """턴테이블 현재 각도 수신 → 해당 각도 이미지 캡처."""
        self._current_angle = msg.data
        if self._inspection_active and msg.data in self._angles:
            # 안정화 대기 후 캡처
            time.sleep(self._cap_delay)
            if self._latest_frame is not None:
                self._captured_images[msg.data] = self._latest_frame.copy()
                self.get_logger().info(f'캡처 완료: {msg.data}°')

                # 4방향 모두 캡처 완료 시 검사 실행
                if len(self._captured_images) == len(self._angles):
                    self._run_inspection()

    def _trigger_callback(self, msg: Bool):
        """검사 트리거 수신."""
        if msg.data:
            self._inspection_active = True
            self._captured_images.clear()
            self.get_logger().info('검사 모드 활성화 — 턴테이블 회전 대기 중')
        else:
            self._inspection_active = False

    # ─────────────────────────────────────────────
    # 메인 검사 로직
    # ─────────────────────────────────────────────
    def _run_inspection(self):
        """4방향 이미지로 듀얼 검사 실행."""
        start_time = time.time()
        self.get_logger().info('='*50)
        self.get_logger().info('양불 판정 시작')

        # 1) CAD 기준 형상 비교
        cad_results = self._cad_comparison()

        # 2) 표면 특징 분석
        surface_results = self._surface_analysis()

        # 3) 최종 판정
        cad_pass = cad_results['passed']
        surface_pass = surface_results['passed']
        final_pass = cad_pass and surface_pass

        # 불량 사유 결정
        fail_reason = ''
        if not final_pass:
            reasons = []
            if not cad_pass:
                reasons.append(f"CAD비교실패({cad_results['fail_detail']})")
            if not surface_pass:
                reasons.append(f"표면특징({surface_results['fail_detail']})")
            fail_reason = ', '.join(reasons)

        elapsed = time.time() - start_time

        # ─── 결과 메시지 발행 ───
        result = InspectionResult()
        result.header.stamp = self.get_clock().now().to_msg()
        result.header.frame_id = 'inspection_chamber'
        result.passed = final_pass
        result.fail_reason = fail_reason

        # CAD 비교 결과
        result.ssim_scores = cad_results['ssim_scores']
        result.area_ratios = cad_results['area_ratios']
        result.pixel_diff_ratios = cad_results['pixel_diff_ratios']

        # 표면 특징
        result.solidity = surface_results['solidity']
        result.area_ratio = surface_results['area_ratio']
        result.hole_count = surface_results['hole_count']
        result.hole_area_ratio = surface_results['hole_area_ratio']
        result.texture_variance = surface_results['texture_variance']

        result.object_index = self._current_object_index
        result.inspection_time_sec = elapsed

        self._result_pub.publish(result)

        # 로그
        status = 'PASS ✓' if final_pass else f'FAIL ✗ ({fail_reason})'
        self.get_logger().info(f'판정: {status} | 소요: {elapsed:.2f}s')
        self.get_logger().info(
            f'  SSIM: {cad_results["ssim_scores"]} | '
            f'면적비: {cad_results["area_ratios"]}')
        self.get_logger().info(
            f'  Solidity: {surface_results["solidity"]:.3f} | '
            f'구멍: {surface_results["hole_count"]}개 | '
            f'텍스처: {surface_results["texture_variance"]:.1f}')
        self.get_logger().info('='*50)

        # ─── 디버그 이미지 ───
        if self._pub_debug:
            self._publish_debug_image(final_pass, cad_results, surface_results)

        # ─── 검사 이미지 저장 ───
        if self._save_images:
            self._save_inspection_log(final_pass, cad_results, surface_results)

        # 상태 초기화
        self._inspection_active = False
        self._captured_images.clear()
        self._current_object_index += 1

    # ─────────────────────────────────────────────
    # 1) CAD 기준 형상 비교
    # ─────────────────────────────────────────────
    def _cad_comparison(self) -> Dict:
        """4방향 CAD 기준 이미지와 SSIM, 면적 비율, 픽셀 차이를 비교한다."""
        ssim_scores = []
        area_ratios = []
        pixel_diffs = []
        all_pass = True
        fail_angles = []

        for angle in self._angles:
            captured = self._captured_images.get(angle)
            reference = self._reference_images.get(angle)

            if captured is None or reference is None:
                self.get_logger().warn(f'{angle}° 이미지 또는 기준 이미지 없음')
                ssim_scores.append(0.0)
                area_ratios.append(0.0)
                pixel_diffs.append(1.0)
                all_pass = False
                fail_angles.append(f'{angle}°:이미지없음')
                continue

            # 그레이스케일 변환 + 전처리
            gray = self._preprocess(captured)

            # 기준 이미지를 캡처 이미지 크기로 리사이즈
            ref_resized = cv2.resize(reference, (gray.shape[1], gray.shape[0]))

            # (a) SSIM 계산
            ssim_val = self._compute_ssim(gray, ref_resized)
            ssim_scores.append(float(ssim_val))

            # (b) 면적 비율
            a_ratio = self._compute_area_ratio(gray, ref_resized)
            area_ratios.append(float(a_ratio))

            # (c) 픽셀 차이 비율
            px_diff = self._compute_pixel_diff(gray, ref_resized)
            pixel_diffs.append(float(px_diff))

            # 개별 판정
            angle_pass = (
                ssim_val >= self._ssim_thresh and
                self._area_min <= a_ratio <= self._area_max and
                px_diff <= self._px_diff_thresh
            )

            if not angle_pass:
                all_pass = False
                details = []
                if ssim_val < self._ssim_thresh:
                    details.append(f'SSIM={ssim_val:.3f}')
                if not (self._area_min <= a_ratio <= self._area_max):
                    details.append(f'면적비={a_ratio:.3f}')
                if px_diff > self._px_diff_thresh:
                    details.append(f'픽셀차={px_diff:.3f}')
                fail_angles.append(f'{angle}°:{"/".join(details)}')

        return {
            'passed': all_pass,
            'ssim_scores': ssim_scores,
            'area_ratios': area_ratios,
            'pixel_diff_ratios': pixel_diffs,
            'fail_detail': '; '.join(fail_angles) if fail_angles else '',
        }

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """이미지 전처리: 그레이스케일 + 블러."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        gray = cv2.GaussianBlur(gray, (self._blur_k, self._blur_k), 0)
        return gray

    def _compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """SSIM (Structural Similarity Index) 계산.
        scikit-image 사용 → OpenCV 순수 구현 폴백.
        """
        try:
            from skimage.metrics import structural_similarity
            score, _ = structural_similarity(img1, img2, full=True)
            return float(score)
        except ImportError:
            # OpenCV 기반 SSIM 구현
            return self._ssim_opencv(img1, img2)

    def _ssim_opencv(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """OpenCV로 SSIM 직접 계산 (scikit-image 없을 때 폴백)."""
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

        ssim_map = numerator / denominator
        return float(np.mean(ssim_map))

    def _compute_area_ratio(self, captured: np.ndarray, reference: np.ndarray) -> float:
        """실제 윤곽 면적 / 기준 면적."""
        cap_area = self._get_object_area(captured)
        ref_area = self._get_object_area(reference)

        if ref_area == 0:
            return 0.0
        return cap_area / ref_area

    def _get_object_area(self, gray: np.ndarray) -> float:
        """이진화 후 가장 큰 윤곽선의 면적을 반환."""
        _, binary = cv2.threshold(gray, self._bin_thresh, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.0
        largest = max(contours, key=cv2.contourArea)
        return float(cv2.contourArea(largest))

    def _compute_pixel_diff(self, captured: np.ndarray, reference: np.ndarray) -> float:
        """차이 픽셀 개수 / 전체 픽셀."""
        diff = cv2.absdiff(captured, reference)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        diff_pixels = np.count_nonzero(thresh)
        total_pixels = captured.shape[0] * captured.shape[1]
        return diff_pixels / total_pixels if total_pixels > 0 else 1.0

    # ─────────────────────────────────────────────
    # 2) 표면 특징 분석
    # ─────────────────────────────────────────────
    def _surface_analysis(self) -> Dict:
        """4방향 이미지의 표면 특징을 추출하고 평균으로 판정한다.

        특징 5가지:
          - Solidity: 윤곽 면적 / 컨벡스 헐 면적 (워핑 감지)
          - Area Ratio: 실제 면적 / 기준 면적 (미출력 감지)
          - Hole Count: 내부 공동 개수 (레이어 분리)
          - Hole Area Ratio: 구멍 총 면적 / 전체 면적 (레이어 분리)
          - Texture Variance: 라플라시안 분산 (스트링잉)
        """
        all_solidity = []
        all_area_ratio = []
        all_hole_count = []
        all_hole_area = []
        all_texture_var = []

        for angle in self._angles:
            captured = self._captured_images.get(angle)
            if captured is None:
                continue

            gray = self._preprocess(captured)

            # Solidity
            solidity = self._compute_solidity(gray)
            all_solidity.append(solidity)

            # Area Ratio (표면 특징용 — 기준 이미지 대비)
            ref = self._reference_images.get(angle)
            if ref is not None:
                ref_resized = cv2.resize(ref, (gray.shape[1], gray.shape[0]))
                a_ratio = self._compute_area_ratio(gray, ref_resized)
            else:
                a_ratio = 1.0  # 기준 없으면 정상으로 간주
            all_area_ratio.append(a_ratio)

            # Hole Count & Hole Area Ratio
            h_count, h_area_ratio = self._compute_holes(gray)
            all_hole_count.append(h_count)
            all_hole_area.append(h_area_ratio)

            # Texture Variance
            tex_var = self._compute_texture_variance(gray)
            all_texture_var.append(tex_var)

        # 4방향 평균
        avg_sol = float(np.mean(all_solidity)) if all_solidity else 0.0
        avg_area = float(np.mean(all_area_ratio)) if all_area_ratio else 0.0
        avg_holes = int(np.mean(all_hole_count)) if all_hole_count else 0
        avg_hole_area = float(np.mean(all_hole_area)) if all_hole_area else 0.0
        avg_tex = float(np.mean(all_texture_var)) if all_texture_var else 0.0

        # 판정
        all_pass = True
        fail_details = []

        if not (self._sol_min <= avg_sol <= self._sol_max):
            all_pass = False
            fail_details.append(f'워핑:Solidity={avg_sol:.3f}')

        if not (self._f_area_min <= avg_area <= self._f_area_max):
            all_pass = False
            fail_details.append(f'미출력:면적비={avg_area:.3f}')

        if avg_holes > self._hole_max:
            all_pass = False
            fail_details.append(f'레이어분리:구멍={avg_holes}개')

        if avg_hole_area > self._hole_area_max:
            all_pass = False
            fail_details.append(f'레이어분리:구멍면적={avg_hole_area:.3f}')

        if avg_tex > self._tex_var_max:
            all_pass = False
            fail_details.append(f'스트링잉:텍스처={avg_tex:.1f}')

        return {
            'passed': all_pass,
            'solidity': avg_sol,
            'area_ratio': avg_area,
            'hole_count': avg_holes,
            'hole_area_ratio': avg_hole_area,
            'texture_variance': avg_tex,
            'fail_detail': '; '.join(fail_details) if fail_details else '',
        }

    def _compute_solidity(self, gray: np.ndarray) -> float:
        """윤곽 면적 / 컨벡스 헐 면적."""
        _, binary = cv2.threshold(gray, self._bin_thresh, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.0

        largest = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest)
        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)

        if hull_area == 0:
            return 0.0
        return contour_area / hull_area

    def _compute_holes(self, gray: np.ndarray) -> Tuple[int, float]:
        """내부 공동(구멍) 개수와 면적 비율.
        Returns: (hole_count, hole_area_ratio)
        """
        _, binary = cv2.threshold(gray, self._bin_thresh, 255, cv2.THRESH_BINARY)

        # 외부 + 내부 윤곽 모두 검출
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if hierarchy is None:
            return 0, 0.0

        hole_count = 0
        hole_total_area = 0.0
        total_area = gray.shape[0] * gray.shape[1]

        for i, h in enumerate(hierarchy[0]):
            # h = [next, prev, child, parent]
            # parent가 있고 (내부 윤곽) 면적이 min_hole_px 이상인 것만 카운트
            if h[3] != -1:
                area = cv2.contourArea(contours[i])
                if area >= self._min_hole_px:
                    hole_count += 1
                    hole_total_area += area

        hole_area_ratio = hole_total_area / total_area if total_area > 0 else 0.0
        return hole_count, hole_area_ratio

    def _compute_texture_variance(self, gray: np.ndarray) -> float:
        """라플라시안 분산 — 표면 거칠기/스트링잉 감지."""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    # ─────────────────────────────────────────────
    # 디버그 / 로깅
    # ─────────────────────────────────────────────
    def _publish_debug_image(self, passed: bool, cad: Dict, surface: Dict):
        """4방향 이미지를 타일링하고 판정 결과를 오버레이."""
        tiles = []
        for angle in self._angles:
            img = self._captured_images.get(angle)
            if img is not None:
                tile = cv2.resize(img, (320, 240))
            else:
                tile = np.zeros((240, 320, 3), dtype=np.uint8)

            # 각도 텍스트
            cv2.putText(tile, f'{angle}deg', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            tiles.append(tile)

        # 2×2 타일링
        top = np.hstack([tiles[0], tiles[1]])
        bottom = np.hstack([tiles[2], tiles[3]])
        debug_img = np.vstack([top, bottom])

        # 판정 결과 오버레이
        color = (0, 255, 0) if passed else (0, 0, 255)
        label = 'PASS' if passed else 'FAIL'
        cv2.putText(debug_img, label, (270, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        # SSIM 값 표시
        y_offset = 80
        for i, angle in enumerate(self._angles):
            ssim_val = cad['ssim_scores'][i] if i < len(cad['ssim_scores']) else 0.0
            text = f'{angle}deg SSIM:{ssim_val:.3f}'
            cv2.putText(debug_img, text, (10, y_offset + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        debug_msg = self._bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
        self._debug_pub.publish(debug_msg)

    def _save_inspection_log(self, passed: bool, cad: Dict, surface: Dict):
        """검사 이미지와 결과를 로그 디렉토리에 저장."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_str = 'PASS' if passed else 'FAIL'
        log_subdir = os.path.join(
            self._log_dir,
            f'{timestamp}_obj{self._current_object_index}_{result_str}')

        os.makedirs(log_subdir, exist_ok=True)

        # 캡처 이미지 저장
        for angle, img in self._captured_images.items():
            path = os.path.join(log_subdir, f'captured_{angle}.png')
            cv2.imwrite(path, img)

        # 판정 결과 텍스트 저장
        result_path = os.path.join(log_subdir, 'result.txt')
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(f'판정: {result_str}\n')
            f.write(f'SSIM: {cad["ssim_scores"]}\n')
            f.write(f'면적비: {cad["area_ratios"]}\n')
            f.write(f'픽셀차이: {cad["pixel_diff_ratios"]}\n')
            f.write(f'Solidity: {surface["solidity"]:.4f}\n')
            f.write(f'면적비(표면): {surface["area_ratio"]:.4f}\n')
            f.write(f'구멍수: {surface["hole_count"]}\n')
            f.write(f'구멍면적비: {surface["hole_area_ratio"]:.4f}\n')
            f.write(f'텍스처분산: {surface["texture_variance"]:.2f}\n')

        self.get_logger().info(f'검사 로그 저장: {log_subdir}')


def main(args=None):
    rclpy.init(args=args)
    node = InspectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
