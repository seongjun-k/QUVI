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
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, Int32

from quvi_msgs.msg import GraspGoal, InspectionResult
from quvi_robot_control.utils import decode_compressed, decode_raw, BinaryCache, encode_bgr


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
        if self._use_compressed:
            self._img_sub = self.create_subscription(
                CompressedImage, self._camera_topic,
                self._image_callback, 10)
        else:
            self._img_sub = self.create_subscription(
                Image, self._camera_topic,
                self._image_callback_raw, 10)

        self._turntable_done_sub = self.create_subscription(
            Bool, '/motor/turntable_done',
            self._turntable_done_callback, 10)

        self._trigger_sub = self.create_subscription(
            Bool, '/inspection/trigger',
            self._trigger_callback, 10)

        self._ref_capture_sub = self.create_subscription(
            Bool, '/inspection/capture_reference',
            self._ref_capture_trigger_callback, 10)

        self._grasp_cmd_sub = self.create_subscription(
            GraspGoal, '/robot/grasp_command',
            self._grasp_cmd_callback, 10)

        self._result_pub = self.create_publisher(
            InspectionResult, '/inspection/result', 10)

        if self._pub_debug:
            self._debug_pub = self.create_publisher(
                Image, self._debug_topic, 5)

        self._latest_frame: Optional[np.ndarray] = None
        self._captured_images: Dict[int, np.ndarray] = {}
        self._inspection_active = False
        self._ref_capture_active = False
        self._current_object_index = 0
        self._last_align_info: Dict[int, Dict] = {}

        self.get_logger().info(
            f'INSPECT_NODE 초기화 완료 | '
            f'SSIM 임계값: {self._ssim_thresh} | '
            f'촬영 각도: {self._angles}')

    # ─────────────────────────────────────────────
    # 파라미터 (선언 + 로드 통합)
    # ─────────────────────────────────────────────
    def _load_params(self):
        """모든 파라미터를 선언하고 로컬 멤버 변수로 로드합니다."""
        params = [
            ('camera_topic',            '/camera2/image_raw/compressed',    '_camera_topic'),
            ('use_compressed',          True,                               '_use_compressed'),
            ('reference_image_dir',     '/workspace/data/reference_images',  '_ref_dir'),
            ('ssim_threshold',          0.85,                               '_ssim_thresh'),
            ('area_ratio_min',          0.90,                               '_area_min'),
            ('area_ratio_max',          1.10,                               '_area_max'),
            ('pixel_diff_threshold',    0.10,                               '_px_diff_thresh'),
            ('solidity_min',            0.85,                               '_sol_min'),
            ('solidity_max',            1.00,                               '_sol_max'),
            ('feature_area_ratio_min',  0.90,                               '_f_area_min'),
            ('feature_area_ratio_max',  1.10,                               '_f_area_max'),
            ('hole_count_max',          2,                                  '_hole_max'),
            ('hole_area_ratio_max',     0.05,                               '_hole_area_max'),
            ('texture_variance_max',    500.0,                              '_tex_var_max'),
            ('min_hole_area_px',        50,                                 '_min_hole_px'),
            ('turntable_angles',        [0, 90, 180, 270],                  '_angles'),
            ('roi_margin',              20,                                 '_roi_margin'),
            ('gaussian_blur_ksize',     5,                                  '_blur_k'),
            ('binary_threshold',        127,                                '_bin_thresh'),
            ('alignment_enabled',       True,                               '_align_enabled'),
            ('align_max_dimension',     200,                                '_align_max_dim'),
            ('align_padding_pct',       0.15,                               '_align_padding'),
            ('align_min_bbox_area',     500,                                '_align_min_area'),
            ('save_inspection_images',  True,                               '_save_images'),
            ('inspection_log_dir',      '/workspace/data/inspection_logs',  '_log_dir'),
            ('publish_debug_image',     True,                               '_pub_debug'),
            ('debug_image_topic',       '/inspect/debug_image',             '_debug_topic'),
        ]

        for name, default, attr_name in params:
            self.declare_parameter(name, default)
            setattr(self, attr_name, self.get_parameter(name).value)

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

        loaded = len(self._reference_images)
        expected = len(self._angles)
        self.get_logger().info(f'기준 이미지 {loaded}/{expected}개 로드됨')

        # 완전성 검증: 일부만 로드된 경우 에러 로그
        if 0 < loaded < expected:
            missing = [a for a in self._angles if a not in self._reference_images]
            self.get_logger().error(
                f'기준 이미지가 불완전합니다! '
                f'누락된 각도: {missing}. '
                f'검사 결과의 신뢰도가 떨어집니다.')

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

    def _turntable_done_callback(self, msg: Bool):
        """턴테이블 이동 완료 시 즉시 캡처."""
        if not msg.data:
            return

        if self._ref_capture_active:
            for angle in self._angles:
                if angle not in self._captured_images:
                    self._capture_reference_angle(angle)
                    break
            return

        if not self._inspection_active:
            return

        # 오케스트레이터가 순차적으로 명령을 보내므로, done 순서 = 각도 순서로 가정.
        # 이미 캡처된 각도는 건너뛰고, 아직 캡처되지 않은 가장 빠른 각도를 캡처.
        for angle in self._angles:
            if angle not in self._captured_images:
                self._capture_angle(angle)
                break

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
            self.get_logger().info('검사 모드 활성화 — 턴테이블 회전 대기 중')
        else:
            self._inspection_active = False

    def _ref_capture_trigger_callback(self, msg: Bool):
        """기준 이미지 캡처 트리거 수신 (정상품을 챔버에 올려둔 상태에서 발행)."""
        if msg.data:
            if self._inspection_active:
                self.get_logger().warn('검사 진행 중 — 기준 캡처 무시')
                return
            self._ref_capture_active = True
            self._captured_images.clear()
            self.get_logger().info(
                f'기준 이미지 캡처 모드 활성화 | '
                f'저장 경로: {self._ref_dir} | '
                f'턴테이블 {self._angles}° 순서로 회전시키세요')
        else:
            self._ref_capture_active = False

    def _capture_reference_angle(self, angle: int):
        """현재 프레임을 기준 이미지로 캡처 후 파일 저장."""
        if self._latest_frame is None:
            self.get_logger().warn(f'{angle}° 기준 캡처 실패: 카메라 프레임 없음')
            return

        gray = self._preprocess(self._latest_frame)
        self._captured_images[angle] = gray

        os.makedirs(self._ref_dir, exist_ok=True)
        path = os.path.join(self._ref_dir, f'ref_{angle}.png')
        cv2.imwrite(path, gray)
        self.get_logger().info(f'기준 이미지 저장: {path}')

        if len(self._captured_images) == len(self._angles):
            self._reference_images = dict(self._captured_images)
            self._captured_images.clear()
            self._ref_capture_active = False
            self.get_logger().info(
                f'기준 이미지 {len(self._angles)}장 캡처 완료 — 즉시 적용됨')

    # ─────────────────────────────────────────────
    # 메인 검사 로직
    # ─────────────────────────────────────────────
    def _run_inspection(self):
        """4방향 이미지로 듀얼 검사 실행."""
        start_time = time.time()
        self.get_logger().info('=' * 50)
        self.get_logger().info('양불 판정 시작')
        self._last_align_info.clear()

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

            gray = self._preprocess(captured)

            # ── 소프트웨어 정렬 (캡처만 정렬, Reference는 리사이즈) ──
            final_cap, final_ref = self._get_aligned_pair(
                gray, reference, angle)

            ssim_val = self._compute_ssim(final_cap, final_ref)
            ssim_scores.append(float(ssim_val))

            a_ratio  = self._compute_area_ratio(final_cap, final_ref)
            area_ratios.append(float(a_ratio))

            px_diff  = self._compute_pixel_diff(final_cap, final_ref)
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

    def _get_aligned_pair(
        self, gray: np.ndarray, reference: np.ndarray, angle: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """캡처 이미지 정렬 + Reference 리사이즈 쌍을 반환한다.

        정렬 비활성화, 정렬 실패, 또는 정렬 후 SSIM 악화 시 원본 폴백.
        Reference(STL 렌더링)는 이미 정렬된 이미지이므로 역회전하지 않고
        캡처 정렬 결과 크기에 맞춰 리사이즈만 수행한다.

        Args:
            gray: 전처리된 캡처 이미지 (그레이스케일).
            reference: 기준 이미지 (그레이스케일).
            angle: 현재 턴테이블 각도 (로깅용).

        Returns:
            (비교용 캡처, 비교용 레퍼런스) 튜플.
        """
        ref_orig = cv2.resize(reference, (gray.shape[1], gray.shape[0]))

        if not self._align_enabled:
            self._last_align_info[angle] = {
                'aligned': None, 'rotation_deg': None,
                'ssim_before': 0.0, 'ssim_after': 0.0,
                'used_alignment': False, 'reason': 'disabled',
            }
            return gray, ref_orig

        cap_cache = BinaryCache(gray, self._bin_thresh)
        aligned_cap = cap_cache.get_aligned_roi(
            max_dim=self._align_max_dim,
            padding_pct=self._align_padding,
            min_area=self._align_min_area)

        if aligned_cap is None:
            self._last_align_info[angle] = {
                'aligned': None, 'rotation_deg': None,
                'ssim_before': 0.0, 'ssim_after': 0.0,
                'used_alignment': False, 'reason': 'no_contour',
            }
            return gray, ref_orig

        aligned_ref = cv2.resize(
            reference, (aligned_cap.shape[1], aligned_cap.shape[0]))

        # 정렬 신뢰도 검증: 정렬 후 SSIM이 오히려 악화되면 원본 사용
        ssim_before = self._compute_ssim(gray, ref_orig)
        ssim_after  = self._compute_ssim(aligned_cap, aligned_ref)

        rot_info = cap_cache.get_rotation_info()
        rot_deg  = rot_info[0] if rot_info else None

        if ssim_after < ssim_before:
            self.get_logger().warn(
                f'{angle}° 정렬 폴백: '
                f'SSIM {ssim_after:.3f} < {ssim_before:.3f}')
            self._last_align_info[angle] = {
                'aligned': aligned_cap, 'rotation_deg': rot_deg,
                'ssim_before': ssim_before, 'ssim_after': ssim_after,
                'used_alignment': False, 'reason': 'ssim_worse',
            }
            return gray, ref_orig

        if rot_info:
            self.get_logger().debug(
                f'{angle}° 정렬 적용: 보정각도={rot_info[0]:.1f}°')

        self._last_align_info[angle] = {
            'aligned': aligned_cap, 'rotation_deg': rot_deg,
            'ssim_before': ssim_before, 'ssim_after': ssim_after,
            'used_alignment': True, 'reason': 'ok',
        }
        return aligned_cap, aligned_ref

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

            # ── 소프트웨어 정렬 (정렬된 이미지로 표면 분석) ──
            if self._align_enabled:
                aligned = cache.get_aligned_roi(
                    max_dim=self._align_max_dim,
                    padding_pct=self._align_padding,
                    min_area=self._align_min_area)
                if aligned is not None:
                    cache = BinaryCache(aligned, self._bin_thresh)

            solidity = cache.solidity()

            ref = self._reference_images.get(angle)
            if ref is not None:
                ref_resized = cv2.resize(ref, (cache.gray.shape[1], cache.gray.shape[0]))
                ref_area = BinaryCache(ref_resized, self._bin_thresh).largest_external_area()
                cap_area = cache.largest_external_area()
                a_ratio  = cap_area / ref_area if ref_area > 0 else 0.0
            else:
                a_ratio = float('nan')

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
            if not math.isnan(area):
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

        # worst_area: 1.0 에서 가장 멀리 벗어난 값 (NaN 제외)
        valid_areas = [a for a in all_area if not math.isnan(a)]
        worst_area = max(valid_areas, key=lambda a: abs(1.0 - a)) if valid_areas else float('nan')

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
        """원본 + 정렬 이미지를 2행 타일링하고 판정 결과를 오버레이한다."""
        TILE_W, TILE_H = 320, 240
        n = len(self._angles)
        cols = n if n <= 4 else 4

        # ── Row 1: 원본 촬영 이미지 ──
        orig_tiles = []
        for i, angle in enumerate(self._angles):
            img = self._captured_images.get(angle)
            tile = cv2.resize(
                img if img is not None else np.zeros((TILE_H, TILE_W, 3), np.uint8),
                (TILE_W, TILE_H))
            cv2.putText(tile, f'{angle}deg [RAW]', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # SSIM 오버레이
            ssim_val = cad['ssim_scores'][i] if i < len(cad['ssim_scores']) else 0.0
            cv2.putText(tile, f'SSIM:{ssim_val:.3f}', (10, TILE_H - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            orig_tiles.append(tile)

        # ── Row 2: 정렬된 ROI 이미지 ──
        align_tiles = []
        for angle in self._angles:
            info = self._last_align_info.get(angle, {})
            aligned = info.get('aligned')

            if aligned is not None:
                # 그레이스케일 → BGR 변환 후 리사이즈
                if len(aligned.shape) == 2:
                    aligned_bgr = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)
                else:
                    aligned_bgr = aligned
                tile = cv2.resize(aligned_bgr, (TILE_W, TILE_H))
            else:
                tile = np.zeros((TILE_H, TILE_W, 3), np.uint8)
                cv2.putText(tile, 'N/A', (TILE_W // 2 - 30, TILE_H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)

            # 정렬 상태 오버레이
            used = info.get('used_alignment', False)
            rot_deg = info.get('rotation_deg')
            reason = info.get('reason', '')

            status_color = (0, 255, 0) if used else (0, 120, 255)
            status_text = 'ALIGNED' if used else f'FALLBACK({reason})'
            cv2.putText(tile, f'{angle}deg [{status_text}]', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)

            if rot_deg is not None:
                cv2.putText(tile, f'rot:{rot_deg:+.1f}deg', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

            ssim_b = info.get('ssim_before', 0.0)
            ssim_a = info.get('ssim_after', 0.0)
            if ssim_b > 0 or ssim_a > 0:
                cv2.putText(tile, f'SSIM:{ssim_b:.3f}->{ssim_a:.3f}',
                            (10, TILE_H - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            align_tiles.append(tile)

        # 빈 슬롯 채우기
        rows_needed = 2
        while len(orig_tiles) < cols:
            orig_tiles.append(np.zeros((TILE_H, TILE_W, 3), np.uint8))
        while len(align_tiles) < cols:
            align_tiles.append(np.zeros((TILE_H, TILE_W, 3), np.uint8))

        row1 = np.hstack(orig_tiles[:cols])
        row2 = np.hstack(align_tiles[:cols])
        debug_img = np.vstack([row1, row2])

        # 전체 판정 배너
        color = (0, 255, 0) if passed else (0, 0, 255)
        label = 'PASS' if passed else 'FAIL'
        cv2.putText(debug_img, label,
                    (debug_img.shape[1] - 150, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        self._debug_pub.publish(encode_bgr(debug_img))

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

        # 정렬 이미지 저장
        for angle, info in self._last_align_info.items():
            aligned = info.get('aligned')
            if aligned is not None:
                cv2.imwrite(
                    os.path.join(log_subdir, f'aligned_{angle}.png'), aligned)

        with open(os.path.join(log_subdir, 'result.txt'), 'w', encoding='utf-8') as f:
            f.write(f'판정: {result_str}\n')
            f.write(f'SSIM: {cad["ssim_scores"]}\n')
            f.write(f'면적비: {cad["area_ratios"]}\n')
            f.write(f'픽셀차이: {cad["pixel_diff_ratios"]}\n')
            f.write(f'Solidity: {surface["solidity"]:.4f}\n')
            f.write(f'면적비(표면): {"N/A" if math.isnan(surface["area_ratio"]) else f"{surface["area_ratio"]:.4f}"}\n')
            f.write(f'구멍수: {surface["hole_count"]}\n')
            f.write(f'구멍면적비: {surface["hole_area_ratio"]:.4f}\n')
            f.write(f'텍스처분산: {surface["texture_variance"]:.2f}\n')
            f.write(f'\n--- 정렬 정보 ---\n')
            for angle, info in self._last_align_info.items():
                f.write(
                    f'{angle}°: used={info["used_alignment"]} '
                    f'rot={info["rotation_deg"]} '
                    f'ssim={info["ssim_before"]:.3f}->{info["ssim_after"]:.3f} '
                    f'reason={info["reason"]}\n')

        self.get_logger().info(f'검사 로그 저장: {log_subdir}')


def main(args=None):
    rclpy.init(args=args)
    node = InspectNode()
    # 캡처 지연 타이머와 이미지/트리거 콜백이 서로를 막지 않도록 멀티스레드 실행.
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
