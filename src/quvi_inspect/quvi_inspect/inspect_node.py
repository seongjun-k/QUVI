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

from quvi_msgs.msg import GraspGoal, InspectionResult
from quvi_robot_control.utils import decode_compressed, decode_raw, declare_and_get, BinaryCache


class InspectNode(Node):
    """양불 판정 노드 — CAD 비교 + 표면 특징 분석."""

    def __init__(self):
        super().__init__('inspect_node')

        # ─── 파라미터 선언 + 로드 (1단계) ───
        self._load_params()

        # ─── 기준 이미지 로드 ───
        self._reference_images: Dict[int, np.ndarray] = {}
        self._load_reference_images()

        # ─── ROS 통신 ───
        self._bridge = CvBridge()

        if self._use_compressed:
            self._img_sub = self.create_subscription(
                CompressedImage, self._camera_topic,
                self._image_callback, 10)
        else:
            self._img_sub = self.create_subscription(
                Image, self._camera_topic,
                self._image_callback_raw, 10)

        self._turntable_sub = self.create_subscription(
            Int32, '/motor/turntable',
            self._turntable_callback, 10)

        self._trigger_sub = self.create_subscription(
            Bool, '/inspection/trigger',
            self._trigger_callback, 10)

        self._grasp_cmd_sub = self.create_subscription(
            GraspGoal, '/robot/grasp_command',
            self._grasp_cmd_callback, 10)

        self._result_pub = self.create_publisher(
            InspectionResult, '/inspection/result', 10)

        if self._pub_debug:
            self._debug_pub = self.create_publisher(
                Image, self._debug_topic, 5)

        # 내부 상태
        self._latest_frame: Optional[np.ndarray] = None
        self._current_angle: int = 0
        self._captured_images: Dict[int, np.ndarray] = {}
        self._inspection_active = False
        self._current_object_index = 0
        self._capture_timers: set = set()
        self._scheduled_angles: Dict[int, rclpy.timer.Timer] = {}

        self.get_logger().info(
            f'INSPECT_NODE 초기화 완료 | '
            f'SSIM 임계값: {self._ssim_thresh} | '
            f'촬영 각도: {self._angles}')

    # ─────────────────────────────────────────────
    # 파라미터 (선언 + 로드 통합)
    # ─────────────────────────────────────────────
    def _load_params(self):
        """declare_and_get 으로 선언과 로드를 1단계로 처리한다."""
        g = lambda name, default: declare_and_get(self, name, default)

        self._camera_topic   = g('camera_topic',            '/camera2/image_raw/compressed')
        self._use_compressed = g('use_compressed',          True)
        self._ref_dir        = g('reference_image_dir',     '/workspace/data/reference_images')
        # CAD 비교
        self._ssim_thresh    = g('ssim_threshold',          0.85)
        self._area_min       = g('area_ratio_min',          0.90)
        self._area_max       = g('area_ratio_max',          1.10)
        self._px_diff_thresh = g('pixel_diff_threshold',    0.10)
        # 표면 특징
        self._sol_min        = g('solidity_min',            0.85)
        self._sol_max        = g('solidity_max',            1.00)
        self._f_area_min     = g('feature_area_ratio_min',  0.90)
        self._f_area_max     = g('feature_area_ratio_max',  1.10)
        self._hole_max       = g('hole_count_max',          2)
        self._hole_area_max  = g('hole_area_ratio_max',     0.05)
        self._tex_var_max    = g('texture_variance_max',    500.0)
        self._min_hole_px    = g('min_hole_area_px',        50)
        # 턴테이블
        self._angles         = g('turntable_angles',        [0, 90, 180, 270])
        self._cap_delay      = g('capture_delay_sec',       0.5)
        # 전처리
        self._roi_margin     = g('roi_margin',              20)
        self._blur_k         = g('gaussian_blur_ksize',     5)
        self._bin_thresh     = g('binary_threshold',        127)
        # 디버그
        self._save_images    = g('save_inspection_images',  True)
        self._log_dir        = g('inspection_log_dir',      '/workspace/data/inspection_logs')
        self._pub_debug      = g('publish_debug_image',     True)
        self._debug_topic    = g('debug_image_topic',       '/inspect/debug_image')

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
        frame = decode_compressed(msg)
        if frame is not None:
            self._latest_frame = frame

    def _image_callback_raw(self, msg: Image):
        frame = decode_raw(msg)
        if frame is not None:
            self._latest_frame = frame

    def _grasp_cmd_callback(self, msg: GraspGoal):
        self._current_object_index = msg.object_index
        self.get_logger().info(f'Object index 동기화: {self._current_object_index}')

    def _turntable_callback(self, msg: Int32):
        """턴테이블 현재 각도 수신 → 안정화 후 비동기 캡처 예약."""
        self._current_angle = msg.data
        if self._inspection_active and msg.data in self._angles:
            if msg.data in self._captured_images or msg.data in self._scheduled_angles:
                return
            self._schedule_capture(msg.data)

    def _schedule_capture(self, angle: int):
        """capture_delay_sec 후 한 번만 실행되는 캡처 타이머 등록."""
        def _do_capture():
            timer.cancel()
            self._capture_timers.discard(timer)
            self._scheduled_angles.pop(angle, None)
            self._capture_angle(angle)

        timer = self.create_timer(self._cap_delay, _do_capture)
        self._capture_timers.add(timer)
        self._scheduled_angles[angle] = timer

    def _capture_angle(self, angle: int):
        """현재 프레임을 해당 각도로 캡처."""
        if not self._inspection_active:
            return
        if self._latest_frame is not None:
            self._captured_images[angle] = self._latest_frame.copy()
            self.get_logger().info(f'캡처 완료: {angle}°')
            if len(self._captured_images) == len(self._angles):
                self._run_inspection()
        else:
            self.get_logger().warn(f'{angle}° 캡처 실패: 카메라 프레임 없음')

    def _trigger_callback(self, msg: Bool):
        """검사 트리거 수신."""
        if msg.data:
            self._inspection_active = True
            self._captured_images.clear()
            for timer in self._capture_timers:
                timer.cancel()
            self._capture_timers.clear()
            self._scheduled_angles.clear()
            self.get_logger().info('검사 모드 활성화 — 턴테이블 회전 대기 중')
        else:
            self._inspection_active = False
            for timer in self._capture_timers:
                timer.cancel()
            self._capture_timers.clear()
            self._scheduled_angles.clear()

    # ─────────────────────────────────────────────
    # 메인 검사 로직
    # ─────────────────────────────────────────────
    def _run_inspection(self):
        """4방향 이미지로 듀얼 검사 실행."""
        start_time = time.time()
        self.get_logger().info('=' * 50)
        self.get_logger().info('양불 판정 시작')

        cad_results     = self._cad_comparison()
        surface_results = self._surface_analysis()

        cad_pass    = cad_results['passed']
        surface_pass = surface_results['passed']
        final_pass  = cad_pass and surface_pass

        fail_reason = ''
        if not final_pass:
            reasons = []
            if not cad_pass:
                reasons.append(f"CAD비교실패({cad_results['fail_detail']})")
            if not surface_pass:
                reasons.append(f"표면특징({surface_results['fail_detail']})")
            fail_reason = ', '.join(reasons)

        elapsed = time.time() - start_time

        result = InspectionResult()
        result.header.stamp    = self.get_clock().now().to_msg()
        result.header.frame_id = 'inspection_chamber'
        result.passed          = final_pass
        result.fail_reason     = fail_reason
        result.ssim_scores        = cad_results['ssim_scores']
        result.area_ratios        = cad_results['area_ratios']
        result.pixel_diff_ratios  = cad_results['pixel_diff_ratios']
        result.solidity           = surface_results['solidity']
        result.area_ratio         = surface_results['area_ratio']
        result.hole_count         = surface_results['hole_count']
        result.hole_area_ratio    = surface_results['hole_area_ratio']
        result.texture_variance   = surface_results['texture_variance']
        result.object_index       = self._current_object_index
        result.inspection_time_sec = elapsed
        self._result_pub.publish(result)

        status = 'PASS ✓' if final_pass else f'FAIL ✗ ({fail_reason})'
        self.get_logger().info(f'판정: {status} | 소요: {elapsed:.2f}s')
        self.get_logger().info(
            f'  SSIM: {cad_results["ssim_scores"]} | '
            f'면적비: {cad_results["area_ratios"]}')
        self.get_logger().info(
            f'  Solidity: {surface_results["solidity"]:.3f} | '
            f'구멍: {surface_results["hole_count"]}개 | '
            f'텍스처: {surface_results["texture_variance"]:.1f}')
        self.get_logger().info('=' * 50)

        if self._pub_debug:
            self._publish_debug_image(final_pass, cad_results, surface_results)
        if self._save_images:
            self._save_inspection_log(final_pass, cad_results, surface_results)

        self._inspection_active = False
        self._captured_images.clear()
        for timer in self._capture_timers:
            timer.cancel()
        self._capture_timers.clear()
        self._scheduled_angles.clear()
        self._current_object_index += 1

    # ─────────────────────────────────────────────
    # 1) CAD 기준 형상 비교
    # ─────────────────────────────────────────────
    def _cad_comparison(self) -> Dict:
        """4방향 CAD 기준 이미지와 SSIM, 면적 비율, 픽셀 차이를 비교한다."""
        ssim_scores  = []
        area_ratios  = []
        pixel_diffs  = []
        all_pass     = True
        fail_angles  = []

        for angle in self._angles:
            captured  = self._captured_images.get(angle)
            reference = self._reference_images.get(angle)

            if captured is None or reference is None:
                self.get_logger().warn(f'{angle}° 이미지 또는 기준 이미지 없음')
                ssim_scores.append(0.0)
                area_ratios.append(0.0)
                pixel_diffs.append(1.0)
                all_pass = False
                fail_angles.append(f'{angle}°:이미지없음')
                continue

            gray        = self._preprocess(captured)
            ref_resized = cv2.resize(reference, (gray.shape[1], gray.shape[0]))

            ssim_val = self._compute_ssim(gray, ref_resized)
            ssim_scores.append(float(ssim_val))

            a_ratio  = self._compute_area_ratio(gray, ref_resized)
            area_ratios.append(float(a_ratio))

            px_diff  = self._compute_pixel_diff(gray, ref_resized)
            pixel_diffs.append(float(px_diff))

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
            'passed':           all_pass,
            'ssim_scores':      ssim_scores,
            'area_ratios':      area_ratios,
            'pixel_diff_ratios': pixel_diffs,
            'fail_detail':      '; '.join(fail_angles) if fail_angles else '',
        }

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """이미지 전처리: 그레이스케일 + 가우시안 블러."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        return cv2.GaussianBlur(gray, (self._blur_k, self._blur_k), 0)

    def _compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """SSIM 계산. scikit-image 우선, 없으면 OpenCV 폴백."""
        try:
            from skimage.metrics import structural_similarity
            score, _ = structural_similarity(img1, img2, full=True)
            return float(score)
        except ImportError:
            return self._ssim_opencv(img1, img2)

    def _ssim_opencv(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """OpenCV 기반 SSIM 직접 계산 (scikit-image 없을 때 폴백)."""
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        i1 = img1.astype(np.float64)
        i2 = img2.astype(np.float64)
        mu1 = cv2.GaussianBlur(i1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(i2, (11, 11), 1.5)
        mu1_sq  = mu1 ** 2
        mu2_sq  = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        s1_sq   = cv2.GaussianBlur(i1 ** 2, (11, 11), 1.5) - mu1_sq
        s2_sq   = cv2.GaussianBlur(i2 ** 2, (11, 11), 1.5) - mu2_sq
        s12     = cv2.GaussianBlur(i1 * i2, (11, 11), 1.5) - mu1_mu2
        num     = (2 * mu1_mu2 + C1) * (2 * s12 + C2)
        den     = (mu1_sq + mu2_sq + C1) * (s1_sq + s2_sq + C2)
        return float(np.mean(num / den))

    def _compute_area_ratio(self, captured: np.ndarray, reference: np.ndarray) -> float:
        """실제 윤곽 면적 / 기준 면적."""
        cap_area = BinaryCache(captured,  self._bin_thresh).largest_external_area()
        ref_area = BinaryCache(reference, self._bin_thresh).largest_external_area()
        return cap_area / ref_area if ref_area > 0 else 0.0

    def _compute_pixel_diff(self, captured: np.ndarray, reference: np.ndarray) -> float:
        """차이 픽셀 개수 / 전체 픽셀."""
        diff        = cv2.absdiff(captured, reference)
        _, thresh   = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        total       = captured.shape[0] * captured.shape[1]
        return np.count_nonzero(thresh) / total if total > 0 else 1.0

    # ─────────────────────────────────────────────
    # 2) 표면 특징 분석
    # ─────────────────────────────────────────────
    def _surface_analysis(self) -> Dict:
        """4방향 이미지의 표면 특징을 추출하고 worst-case 로 판정한다.

        BinaryCache 를 각 이미지당 1 회만 생성하여
        Solidity / Area / Holes 를 공유 계산한다.
        """
        angle_features: Dict[int, Dict] = {}

        for angle in self._angles:
            captured = self._captured_images.get(angle)
            if captured is None:
                continue

            gray  = self._preprocess(captured)
            cache = BinaryCache(gray, self._bin_thresh)  # 이진화 1회

            solidity = cache.solidity()

            ref = self._reference_images.get(angle)
            if ref is not None:
                ref_resized = cv2.resize(ref, (gray.shape[1], gray.shape[0]))
                ref_area = BinaryCache(ref_resized, self._bin_thresh).largest_external_area()
                cap_area = cache.largest_external_area()
                a_ratio  = cap_area / ref_area if ref_area > 0 else 0.0
            else:
                a_ratio = 1.0

            h_count, h_area_ratio = cache.holes(self._min_hole_px)

            lap     = cv2.Laplacian(gray, cv2.CV_64F)
            tex_var = float(lap.var())

            angle_features[angle] = {
                'solidity':        solidity,
                'area_ratio':      a_ratio,
                'hole_count':      h_count,
                'hole_area_ratio': h_area_ratio,
                'texture_variance': tex_var,
            }

        all_pass    = True
        fail_details: List[str] = []

        for angle, feats in angle_features.items():
            sol   = feats['solidity']
            area  = feats['area_ratio']
            holes = feats['hole_count']
            h_ar  = feats['hole_area_ratio']
            tex   = feats['texture_variance']

            if not (self._sol_min <= sol <= self._sol_max):
                all_pass = False
                fail_details.append(f'{angle}°워핑:Solidity={sol:.3f}')
            if not (self._f_area_min <= area <= self._f_area_max):
                all_pass = False
                fail_details.append(f'{angle}°미출력:면적비={area:.3f}')
            if holes > self._hole_max:
                all_pass = False
                fail_details.append(f'{angle}°레이어분리:구멍={holes}개')
            if h_ar > self._hole_area_max:
                all_pass = False
                fail_details.append(f'{angle}°레이어분리:구멍면적={h_ar:.3f}')
            if tex > self._tex_var_max:
                all_pass = False
                fail_details.append(f'{angle}°스트링잉:텍스처={tex:.1f}')

        vals = list(angle_features.values())
        all_sol      = [f['solidity']         for f in vals]
        all_area     = [f['area_ratio']        for f in vals]
        all_holes    = [f['hole_count']        for f in vals]
        all_hole_ar  = [f['hole_area_ratio']   for f in vals]
        all_tex      = [f['texture_variance']  for f in vals]

        # worst_area: 1.0 에서 가장 멀리 벗어난 값 (max by key, sqrt 불필요)
        worst_area = max(all_area, key=lambda a: abs(1.0 - a)) if all_area else 1.0

        return {
            'passed':           all_pass,
            'solidity':         min(all_sol)     if all_sol     else 1.0,
            'area_ratio':       worst_area,
            'hole_count':       max(all_holes)   if all_holes   else 0,
            'hole_area_ratio':  max(all_hole_ar) if all_hole_ar else 0.0,
            'texture_variance': max(all_tex)     if all_tex     else 0.0,
            'fail_detail':      '; '.join(fail_details) if fail_details else '',
        }

    # ─────────────────────────────────────────────
    # 디버그 / 로깅
    # ─────────────────────────────────────────────
    def _publish_debug_image(self, passed: bool, cad: Dict, surface: Dict):
        """촬영 이미지를 동적으로 타일링하고 판정 결과를 오버레이한다."""
        tiles = []
        for angle in self._angles:
            img  = self._captured_images.get(angle)
            tile = cv2.resize(
                img if img is not None else np.zeros((240, 320, 3), np.uint8),
                (320, 240))
            cv2.putText(tile, f'{angle}deg', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            tiles.append(tile)

        # 동적 타일링: _angles 길이에 맞게 열 수 자동 결정
        n    = len(tiles)
        cols = 2
        rows = (n + cols - 1) // cols
        while len(tiles) < rows * cols:
            tiles.append(np.zeros((240, 320, 3), np.uint8))
        row_imgs   = [np.hstack(tiles[i * cols:(i + 1) * cols]) for i in range(rows)]
        debug_img  = np.vstack(row_imgs)

        color = (0, 255, 0) if passed else (0, 0, 255)
        cv2.putText(debug_img, 'PASS' if passed else 'FAIL',
                    (270, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        for i, angle in enumerate(self._angles):
            ssim_val = cad['ssim_scores'][i] if i < len(cad['ssim_scores']) else 0.0
            cv2.putText(debug_img, f'{angle}deg SSIM:{ssim_val:.3f}',
                        (10, 80 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        self._debug_pub.publish(
            self._bridge.cv2_to_imgmsg(debug_img, encoding='bgr8'))

    def _save_inspection_log(self, passed: bool, cad: Dict, surface: Dict):
        """검사 이미지와 결과를 로그 디렉토리에 저장."""
        timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_str  = 'PASS' if passed else 'FAIL'
        log_subdir  = os.path.join(
            self._log_dir,
            f'{timestamp}_obj{self._current_object_index}_{result_str}')
        os.makedirs(log_subdir, exist_ok=True)

        for angle, img in self._captured_images.items():
            cv2.imwrite(os.path.join(log_subdir, f'captured_{angle}.png'), img)

        with open(os.path.join(log_subdir, 'result.txt'), 'w', encoding='utf-8') as f:
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
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
