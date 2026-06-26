"""
QUVI INSPECT_NODE
─────────────────
검사 챔버(Zone 2)에서 턴테이블 4방향 촬영 이미지를 받아
표면 특징 분석으로 양불 판정을 수행하고 결과를 발행한다.

검사 방식:
  표면 특징 기반 검사 (Solidity, Area Ratio, Hole Count, Hole Area, Texture)

판정 로직:
  PASS = 표면 특징 ALL 정상
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
            # ─── 표면 특징 분석 임계값 ───
            ('solidity_min',            0.85,                               '_sol_min'),
            ('solidity_max',            1.00,                               '_sol_max'),
            ('feature_area_ratio_min',  0.90,                               '_f_area_min'),
            ('feature_area_ratio_max',  1.10,                               '_f_area_max'),
            ('hole_count_max',          2,                                  '_hole_max'),
            ('hole_area_ratio_max',     0.05,                               '_hole_area_max'),
            ('texture_variance_max',    500.0,                              '_tex_var_max'),
            ('min_hole_area_px',        50,                                 '_min_hole_px'),
            # ─── 턴테이블 / 전처리 ───
            ('turntable_angles',        [0, 90, 180, 270],                  '_angles'),
            ('roi_margin',              20,                                 '_roi_margin'),
            ('gaussian_blur_ksize',     5,                                  '_blur_k'),
            ('binary_threshold',        127,                                '_bin_thresh'),
            ('alignment_enabled',       True,                               '_align_enabled'),
            ('align_max_dimension',     200,                                '_align_max_dim'),
            ('align_padding_pct',       0.15,                               '_align_padding'),
            ('align_min_bbox_area',     500,                                '_align_min_area'),
            # ─── 디버그 / 로그 ───
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
        """4방향 이미지로 표면 특징 분석 검사 실행."""
        start_time = time.time()
        self.get_logger().info('=' * 50)
        self.get_logger().info('양불 판정 시작 (표면 특징 분석)')
        self._last_align_info.clear()

        surface_results = self._surface_analysis()
        final_pass      = surface_results['passed']
        fail_reason     = surface_results['fail_detail'] if not final_pass else ''

        elapsed = time.time() - start_time

        result = InspectionResult()
        result.header.stamp        = self.get_clock().now().to_msg()
        result.header.frame_id     = 'inspection_chamber'
        result.passed              = final_pass
        result.fail_reason         = fail_reason
        result.ssim_scores         = []   # CAD 비교 미사용 — 빈 배열
        result.area_ratios         = []   # CAD 비교 미사용 — 빈 배열
        result.pixel_diff_ratios   = []   # CAD 비교 미사용 — 빈 배열
        result.solidity            = surface_results['solidity']
        result.area_ratio          = surface_results['area_ratio']
        result.hole_count          = surface_results['hole_count']
        result.hole_area_ratio     = surface_results['hole_area_ratio']
        result.texture_variance    = surface_results['texture_variance']
        result.object_index        = self._current_object_index
        result.inspection_time_sec = elapsed
        self._result_pub.publish(result)

        status = 'PASS ✓' if final_pass else f'FAIL ✗ ({fail_reason})'
        self.get_logger().info(f'판정: {status} | 소요: {elapsed:.2f}s')
        self.get_logger().info(
            f'  Solidity: {surface_results["solidity"]:.3f} | '
            f'구멍: {surface_results["hole_count"]}개 | '
            f'텍스처: {surface_results["texture_variance"]:.1f}')
        self.get_logger().info('=' * 50)

        if self._pub_debug:
            self._publish_debug_image(final_pass, surface_results)
        if self._save_images:
            self._save_inspection_log(final_pass, surface_results)

        self._inspection_active = False
        self._captured_images.clear()

    # ─────────────────────────────────────────────
    # 이미지 전처리
    # ─────────────────────────────────────────────
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """이미지 전처리: 그레이스케일 + 가우시안 블러."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        return cv2.GaussianBlur(gray, (self._blur_k, self._blur_k), 0)

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
    def _publish_debug_image(self, passed: bool, surface: Dict):
        """원본 이미지를 타일링하고 표면 특징 분석 결과를 오버레이한다."""
        TILE_W, TILE_H = 320, 240
        n = len(self._angles)
        cols = n if n <= 4 else 4

        tiles = []
        for angle in self._angles:
            img = self._captured_images.get(angle)
            tile = cv2.resize(
                img if img is not None else np.zeros((TILE_H, TILE_W, 3), np.uint8),
                (TILE_W, TILE_H))
            cv2.putText(tile, f'{angle}deg', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            tiles.append(tile)

        while len(tiles) < cols:
            tiles.append(np.zeros((TILE_H, TILE_W, 3), np.uint8))

        debug_img = np.hstack(tiles[:cols])

        # 표면 특징 요약 오버레이
        summary = (
            f'Sol:{surface["solidity"]:.3f} '
            f'Holes:{surface["hole_count"]} '
            f'Tex:{surface["texture_variance"]:.0f}'
        )
        cv2.putText(debug_img, summary, (10, debug_img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        color = (0, 255, 0) if passed else (0, 0, 255)
        label = 'PASS' if passed else 'FAIL'
        cv2.putText(debug_img, label,
                    (debug_img.shape[1] - 150, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        self._debug_pub.publish(encode_bgr(debug_img))

    def _save_inspection_log(self, passed: bool, surface: Dict):
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
            f.write(f'Solidity: {surface["solidity"]:.4f}\n')
            f.write(f'면적비(표면): {"N/A" if math.isnan(surface["area_ratio"]) else f"{surface["area_ratio"]:.4f}"}\n')
            f.write(f'구멍수: {surface["hole_count"]}\n')
            f.write(f'구멍면적비: {surface["hole_area_ratio"]:.4f}\n')
            f.write(f'텍스처분산: {surface["texture_variance"]:.2f}\n')

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
