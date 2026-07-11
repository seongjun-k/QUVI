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
import json
import threading
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool

from quvi_msgs.msg import GraspGoal, InspectionResult
from quvi_robot_control.utils import decode_compressed, decode_raw, BinaryCache, encode_bgr
from quvi_inspect.ml_preprocess import preprocess_for_ml


class InspectNode(Node):
    """양불 판정 노드 — 표면 특징 분석 (CAD 비교는 제거됨)."""

    def __init__(self):
        super().__init__('inspect_node')

        # ─── 파라미터 선언 + 로드 (1단계) ───
        self._load_params()

        # ─── 기준 이미지 로드 ───
        self._reference_images: Dict[int, np.ndarray] = {}
        self._load_reference_images()

        # ─── ML 이상탐지 (섀도우 모드, Phase 2/3) ───
        self._anomaly_detectors: Dict[int, object] = {}
        self._anomaly_thresholds: Dict[int, float] = {}
        self._init_anomaly()

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

        # turntable_done 누락(0도->0도 무이동 등) 시 검사 모드 캡처가 밀리는 문제를
        # 방지하기 위해 오케스트레이터가 각도별로 명시 발행하는 캡처 명령.
        self._capture_now_sub = self.create_subscription(
            Bool, '/inspection/capture_now',
            self._capture_now_callback, 10)

        self._ref_capture_sub = self.create_subscription(
            Bool, '/inspection/capture_reference',
            self._ref_capture_trigger_callback, 10)

        self._dataset_capture_sub = self.create_subscription(
            Bool, '/inspection/capture_dataset',
            self._dataset_capture_trigger_callback, 10)

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
        # MultiThreadedExecutor에서 _capture_angle과 _watchdog_cb가 동시에
        # _run_inspection에 진입할 수 있어 이중 판정을 막는 락 (#12).
        self._inspection_lock = threading.Lock()
        self._ref_capture_active = False
        self._dataset_capture_active = False
        self._current_object_index = 0

        # ─── 캡처 안정화 타이머 (T4) ───
        # 턴테이블 done 직후 기구 진동·카메라 노출이 안정될 때까지 잠깐 대기 후 캡처한다.
        # 콜백을 blocking sleep 하면 이미지 갱신도 멈춰 오래된 프레임을 잡으므로,
        # 비차단 일회성 타이머로 지연시킨다. 재사용을 위해 생성 후 즉시 취소해 둔다.
        self._pending_ref = False
        self._settle_timer = self.create_timer(
            max(0.05, float(self._capture_settle)), self._on_settle_elapsed)
        self._settle_timer.cancel()

        # ─── 데이터셋 촬영 모드 전용 안정화 타이머 ───
        # ML 정상품 데이터셋 수집용 별도 병렬 모드. 기존 _settle_timer 경로는 건드리지 않고
        # 전용 타이머로 분리해 노출 안정 대기 시간(dataset_capture_settle_sec)을 독립 적용한다.
        self._ds_settle_timer = self.create_timer(
            max(0.05, float(self._ds_settle_sec)), self._on_dataset_settle_elapsed)
        self._ds_settle_timer.cancel()

        # ─── 검사 워치독 (#4) ───
        # turntable_done 누락(예: 0°에서 0°로 '이동' 시 done 미발행)으로 캡처가
        # 4장을 못 채우면 판정이 영영 실행되지 않아 오케스트레이터가 타임아웃/ERROR
        # 로 빠진다. 검사 활성 후 일정 시간 내 미완료면 확보된 캡처로 마무리한다.
        self._inspection_start = 0.0
        self._inspection_watchdog = self.create_timer(1.0, self._watchdog_cb)

        self.get_logger().info(
            f'INSPECT_NODE 초기화 완료 | '
            f'촬영 각도: {self._angles} | 판정 타임아웃: {self._finalize_sec}s')

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
            ('feature_area_ratio_max',  1.50,                               '_f_area_max'),
            ('hole_count_max',          2,                                  '_hole_max'),
            ('hole_area_ratio_max',     0.05,                               '_hole_area_max'),
            ('texture_variance_max',    500.0,                              '_tex_var_max'),
            ('min_hole_area_px',        50,                                 '_min_hole_px'),
            # ─── 턴테이블 / 전처리 ───
            ('turntable_angles',        [0, 90, 180, 270],                  '_angles'),
            # 데이터셋 촬영(1.5s 노출 안정)과 실검사 노출 조건 정합 — train/infer skew 방지 (계획서 Phase 2)
            ('capture_settle_sec',      1.5,                                '_capture_settle'),
            # 한 바퀴 캡처 ≈ 20s(각도당 ~5s) + LED 안정화 5s + 여유 — 12s는 270° 캡처 전에 판정을 강행했다
            ('inspection_finalize_sec', 45.0,                               '_finalize_sec'),
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
            # ─── 데이터셋 촬영 모드 (ML 정상품 수집) ───
            ('dataset_capture_settle_sec', 1.5,                             '_ds_settle_sec'),
            ('dataset_capture_dir',     '/workspace/data/anomaly_dataset/raw', '_ds_dir'),
            # ─── ML 이상탐지 (섀도우 모드 — passed 판정에는 반영하지 않음) ───
            ('anomaly_enabled',         False,                              '_anomaly_enabled'),
            ('anomaly_model_dir',       '/workspace/data/models',           '_anomaly_model_dir'),
            ('anomaly_device',          'cuda',                             '_anomaly_device'),
        ]

        for name, default, attr_name in params:
            self.declare_parameter(name, default)
            setattr(self, attr_name, self.get_parameter(name).value)

    # ─────────────────────────────────────────────
    # 기준 이미지 로드
    # ─────────────────────────────────────────────
    def _load_reference_images(self):
        """기준 이미지(HMI에서 정상품을 챔버에 올려두고 캡처한 결과)를 로드한다.
        파일 네이밍: ref_0.png, ref_90.png, ref_180.png, ref_270.png
        """
        if not os.path.isdir(self._ref_dir):
            self.get_logger().warn(
                f'기준 이미지 디렉토리 없음: {self._ref_dir} — '
                f'HMI 기준 이미지 캡처로 먼저 생성하세요.')
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
    # ML 이상탐지 초기화 (섀도우 모드, Phase 2)
    # ─────────────────────────────────────────────
    def _init_anomaly(self):
        """각도별 PatchCore 뱅크를 로드한다. 실패해도 노드는 계속 동작한다(자동 비활성)."""
        if not self._anomaly_enabled:
            self.get_logger().info('ML 이상탐지 비활성화(anomaly_enabled=False) — 룰 판정만 사용')
            return

        try:
            # torch 는 이 모듈 내부에서만 import 되므로 비활성 시 로드 비용이 없다.
            from quvi_inspect.anomaly_detector import PatchCoreDetector

            thresholds_path = os.path.join(self._anomaly_model_dir, 'thresholds.json')
            with open(thresholds_path, encoding='utf-8') as f:
                thresholds = json.load(f)

            backbone_path = os.path.join(self._anomaly_model_dir, 'wide_resnet50.pth')
            shared_backbone = None  # 첫 로드에서 채워 이후 각도는 백본을 재사용(GPU 메모리 절약)

            for angle in self._angles:
                bank_path = os.path.join(self._anomaly_model_dir, f'bank_{angle}.pt')
                if not os.path.isfile(bank_path):
                    continue
                detector = PatchCoreDetector.load(
                    bank_path,
                    device=self._anomaly_device,
                    backbone_weights_path=backbone_path,
                    backbone=shared_backbone)
                if shared_backbone is None:
                    shared_backbone = detector.backbone
                self._anomaly_detectors[angle] = detector
                self._anomaly_thresholds[angle] = float(thresholds[str(angle)]['threshold'])

            if self._anomaly_detectors:
                self.get_logger().info(
                    f'ML 이상탐지 로드 완료 | 각도: {sorted(self._anomaly_detectors.keys())} | '
                    f'디바이스: {self._anomaly_device} | 임계값: {self._anomaly_thresholds}')
            else:
                self.get_logger().warn('ML 이상탐지: 뱅크 파일 없음 — 비활성 상태로 진행')
        except Exception as exc:  # noqa: BLE001 — 로드 실패는 절대 노드를 죽이지 않는다
            self.get_logger().warn(f'ML 이상탐지 로드 실패({exc}) — 자동 비활성, 룰 판정만 사용')
            self._anomaly_detectors = {}

    # ─────────────────────────────────────────────
    # 콜백
    # ─────────────────────────────────────────────
    def _image_callback(self, msg: CompressedImage):
        frame = decode_compressed(msg)
        if frame is not None:
            # 검사캠이 거꾸로 장착되어 상하 반전 + 좌우 반전(2026-07-10) → 동시 -1
            self._latest_frame = cv2.flip(frame, -1)

    def _image_callback_raw(self, msg: Image):
        frame = decode_raw(msg)
        if frame is not None:
            # 검사캠이 거꾸로 장착되어 상하 반전 + 좌우 반전(2026-07-10) → 동시 -1
            self._latest_frame = cv2.flip(frame, -1)

    def _grasp_cmd_callback(self, msg: GraspGoal):
        self._current_object_index = msg.object_index
        self.get_logger().info(f'Object index 동기화: {self._current_object_index}')

    def _turntable_done_callback(self, msg: Bool):
        """턴테이블 이동 완료 시 안정화 지연 후 캡처를 예약 (T4).

        검사 모드(_inspection_active) 캡처는 capture_now 콜백으로 이관됨 —
        0도->0도 무이동 시 done 미발행으로 캡처가 밀리는 문제 방지 목적.
        기준 이미지/데이터셋 촬영 모드는 오케스트레이터 경로 밖이라 기존대로 done 기반 유지.
        """
        if not msg.data:
            return
        if self._dataset_capture_active:
            # 데이터셋 촬영 모드는 전용 타이머로 분리 처리 (기존 경로 무변경).
            if not self._ds_settle_timer.is_canceled():
                return
            self._ds_settle_timer.reset()   # dataset_capture_settle_sec 후 _on_dataset_settle_elapsed 발화
            return
        if not self._ref_capture_active:
            return
        # 이미 안정화 대기 중이면 중복 done 무시 (done 재발행 방어).
        if not self._settle_timer.is_canceled():
            return
        self._pending_ref = True
        self._settle_timer.reset()   # capture_settle_sec 후 _on_settle_elapsed 발화

    def _capture_now_callback(self, msg: Bool):
        """오케스트레이터의 명시 캡처 명령 수신 (검사 모드 전용, T4 이관분)."""
        if not msg.data:
            return
        if not self._inspection_active:
            return
        if not self._settle_timer.is_canceled():
            return
        self._pending_ref = False
        self._settle_timer.reset()   # capture_settle_sec 후 _on_settle_elapsed 발화

    def _on_settle_elapsed(self):
        """안정화 지연 경과 후 실제 캡처 수행 (일회성)."""
        self._settle_timer.cancel()

        if self._pending_ref:
            if not self._ref_capture_active:
                return
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
            if self._ref_capture_active:
                # 기준 캡처 중 동시 진입 방지 — _captured_images 공유로 인한 교차 오염 방지 (#15)
                self.get_logger().warn('기준 이미지 캡처 진행 중 — 검사 트리거 무시')
                return
            self._inspection_active = True
            self._captured_images.clear()
            self._inspection_start = time.time()   # 워치독 기준 (#4)
            self.get_logger().info('검사 모드 활성화 — 턴테이블 회전 대기 중')
        else:
            self._inspection_active = False

    def _watchdog_cb(self):
        """검사 완료 워치독 (#4). turntable_done 누락 등으로 캡처가 부족해도
        일정 시간 후 확보된 이미지로 판정을 마무리해 오케스트레이터 정지를 막는다."""
        if not self._inspection_active:
            return
        if (time.time() - self._inspection_start) < self._finalize_sec:
            return
        n = len(self._captured_images)
        if n >= 1:
            self.get_logger().warn(
                f'검사 타임아웃({self._finalize_sec}s) — 확보 {n}/{len(self._angles)}장으로 판정 강행')
            self._run_inspection()   # _surface_analysis 는 누락 각도를 건너뛴다
        else:
            # 한 장도 못 잡음 — 판정 불가. 활성 해제하고 경고(오케스트레이터는 타임아웃 처리).
            self.get_logger().error('검사 타임아웃 — 캡처 이미지 0장, 판정 불가')
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

    def _dataset_capture_trigger_callback(self, msg: Bool):
        """데이터셋 촬영 트리거 수신 (ML 정상품 데이터셋 수집용 별도 병렬 모드)."""
        if msg.data:
            if self._inspection_active or self._ref_capture_active:
                self.get_logger().warn('검사/기준 캡처 진행 중 — 데이터셋 캡처 무시')
                return
            self._dataset_capture_active = True
            self._captured_images.clear()
            self.get_logger().info(
                f'데이터셋 촬영 모드 활성화 | '
                f'저장 경로: {self._ds_dir} | '
                f'턴테이블 {self._angles}° 순서로 회전시키세요')
        else:
            self._dataset_capture_active = False

    def _on_dataset_settle_elapsed(self):
        """데이터셋 촬영 모드: 안정화 지연 경과 후 캡처 수행 (일회성)."""
        self._ds_settle_timer.cancel()

        if not self._dataset_capture_active:
            return
        for angle in self._angles:
            if angle not in self._captured_images:
                self._capture_dataset_angle(angle)
                break

    def _capture_dataset_angle(self, angle: int):
        """현재 프레임을 컬러 원본 그대로 데이터셋 디렉토리에 저장.

        기준 이미지(_reference_images)와 무관한 별도 경로로,
        grayscale 전처리 없이 원본을 저장한다.
        """
        if self._latest_frame is None:
            self.get_logger().warn(f'{angle}° 데이터셋 캡처 실패: 카메라 프레임 없음')
            return

        frame = self._latest_frame.copy()
        self._captured_images[angle] = frame

        angle_dir = os.path.join(self._ds_dir, str(angle))
        os.makedirs(angle_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(angle_dir, f'{timestamp}.png')
        cv2.imwrite(path, frame)
        self.get_logger().info(f'데이터셋 이미지 저장: {path}')

        if len(self._captured_images) == len(self._angles):
            self._dataset_capture_active = False
            self._captured_images.clear()
            self.get_logger().info(
                f'데이터셋 {len(self._angles)}장 저장 완료 — 경로: {self._ds_dir}')

    # ─────────────────────────────────────────────
    # 메인 검사 로직
    # ─────────────────────────────────────────────
    def _run_inspection(self):
        """4방향 이미지로 표면 특징 분석 검사 실행."""
        with self._inspection_lock:
            # 먼저 들어간 호출이 finally에서 _inspection_active를 내리므로,
            # 락을 기다리다 뒤늦게 들어온 두번째 호출은 여기서 걸러진다.
            if not self._inspection_active:
                return
            try:
                self._run_inspection_inner()
            except Exception as e:   # noqa: BLE001 — 분석 예외로 노드가 고착되지 않도록 (#16)
                self.get_logger().error(f'검사 중 예외 발생 — 검사 상태 해제: {e}')
            finally:
                self._inspection_active = False
                self._captured_images.clear()

    def _run_inspection_inner(self):
        start_time = time.time()
        self.get_logger().info('=' * 50)
        self.get_logger().info('양불 판정 시작 (표면 특징 분석)')

        surface_results = self._surface_analysis()
        final_pass      = surface_results['passed']
        fail_reason     = surface_results['fail_detail'] if not final_pass else ''

        elapsed = time.time() - start_time

        result = InspectionResult()
        result.header.stamp        = self.get_clock().now().to_msg()
        result.header.frame_id     = 'inspection_chamber'
        result.passed              = final_pass
        result.fail_reason         = fail_reason
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

        # ── 섀도우 모드 로그: ML 은 참고용, passed 에는 반영되지 않음 ──
        ml_passed = surface_results['ml_passed']
        worst = surface_results['anomaly_score_worst']
        if ml_passed is None:
            ml_str, worst_str, agree = 'N/A', 'N/A', 'N/A'
        else:
            ml_str = 'PASS' if ml_passed else 'FAIL'
            worst_str = f'{worst:.2f}'
            agree = '일치' if ml_passed == final_pass else '불일치'
        rule_str = 'PASS' if final_pass else 'FAIL'
        self.get_logger().info(
            f'[섀도우] 룰={rule_str} | ML={ml_str} (worst={worst_str}) | {agree}')
        self.get_logger().info('=' * 50)

        if self._pub_debug:
            self._publish_debug_image(final_pass, surface_results)
        if self._save_images:
            self._save_inspection_log(final_pass, surface_results)
        # 상태 해제는 _run_inspection 의 try/finally 가 일괄 처리 (#16)

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

            # ── 면적비: 정렬(크롭·확대) 전 전체 프레임끼리 비교 ──
            # 기준 이미지는 전체 프레임으로 저장되므로, 캡처도 정렬 전 면적을
            # 써야 조건이 일치한다. 정렬된 ROI 면적으로 비교하면 크롭 배율만큼
            # 비율이 부풀어 항상 FAIL 이 난다.
            # 배치 각도가 틀어지면 캡처 뷰가 다른 각도의 기준과 대응하므로,
            # 4개 각도 기준 전부와 비교해 1.0 에 가장 가까운 면적비를 채택한다.
            cap_area = cache.largest_external_area()
            a_ratio  = float('nan')
            for ref in self._reference_images.values():
                if ref is None:
                    continue
                ref_resized = cv2.resize(ref, (cache.gray.shape[1], cache.gray.shape[0]))
                ref_area = BinaryCache(ref_resized, self._bin_thresh).largest_external_area()
                r = cap_area / ref_area if ref_area > 0 else 0.0
                if math.isnan(a_ratio) or abs(r - 1.0) < abs(a_ratio - 1.0):
                    a_ratio = r
            if math.isnan(a_ratio):
                # 기준 이미지 전무 시 면적비 검출 축이 통째로 빠진 채 검사가 진행됨을 알린다
                self.get_logger().warning(
                    '기준 이미지 없음 — 면적비 검사 스킵됨 (기준 캡처 필요)',
                    throttle_duration_sec=30.0)

            # ── 소프트웨어 정렬 (정렬된 이미지로 표면 분석) ──
            if self._align_enabled:
                aligned = cache.get_aligned_roi(
                    max_dim=self._align_max_dim,
                    padding_pct=self._align_padding,
                    min_area=self._align_min_area)
                if aligned is not None:
                    cache = BinaryCache(aligned, self._bin_thresh)

            solidity = cache.solidity()

            h_count, h_area_ratio = cache.holes(self._min_hole_px)

            lap     = cv2.Laplacian(gray, cv2.CV_64F)
            tex_var = float(lap.var())

            # ── ML 이상탐지 (섀도우 모드 — passed 판정에는 반영하지 않음) ──
            a_score = None
            detector = self._anomaly_detectors.get(angle)
            if detector is not None:
                try:
                    ml_input = preprocess_for_ml(captured, self._bin_thresh)
                    a_score = detector.score(ml_input)
                except Exception as exc:  # noqa: BLE001 — ML 실패가 검사 전체를 막지 않음
                    self.get_logger().warn(f'{angle}° ML 점수 계산 실패: {exc}')

            angle_features[angle] = {
                'solidity':        solidity,
                'area_ratio':      a_ratio,
                'hole_count':      h_count,
                'hole_area_ratio': h_area_ratio,
                'texture_variance': tex_var,
                'anomaly_score':   a_score,
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

        # ── ML 이상탐지 집계 (섀도우 모드 — passed 에는 절대 반영하지 않음) ──
        ml_scores = {a: f['anomaly_score'] for a, f in angle_features.items()
                     if f['anomaly_score'] is not None}
        anomaly_score_worst = max(ml_scores.values()) if ml_scores else None
        ml_passed = None
        ml_over: List[str] = []
        if ml_scores:
            ml_passed = True
            for angle, score in ml_scores.items():
                threshold = self._anomaly_thresholds.get(angle)
                if threshold is not None and score > threshold:
                    ml_passed = False
                    ml_over.append(f'{angle}°={score:.2f}(임계{threshold:.2f})')
        ml_detail = '; '.join(ml_over)

        return {
            'passed':              all_pass,
            'solidity':            min(all_sol)     if all_sol     else 1.0,
            'area_ratio':          worst_area,
            'hole_count':          max(all_holes)   if all_holes   else 0,
            'hole_area_ratio':     max(all_hole_ar) if all_hole_ar else 0.0,
            'texture_variance':    max(all_tex)     if all_tex     else 0.0,
            'fail_detail':         '; '.join(fail_details) if fail_details else '',
            'anomaly_score_worst': anomaly_score_worst,
            'ml_passed':           ml_passed,
            'ml_detail':           ml_detail,
        }

    # ─────────────────────────────────────────────
    # 디버그 / 로깅
    # ─────────────────────────────────────────────
    def _publish_debug_image(self, passed: bool, surface: Dict):
        """원본 이미지를 타일링하고 표면 특징 분석 결과를 오버레이한다."""
        TILE_W, TILE_H = 320, 240
        n = len(self._angles)
        # 4장(기본 0/90/180/270°)은 2x2 격자로 — HMI 판정 결과 셀 비율에 맞춤
        cols = 2 if n == 4 else (n if n <= 4 else 4)

        tiles = []
        for angle in self._angles:
            img = self._captured_images.get(angle)
            tile = cv2.resize(
                img if img is not None else np.zeros((TILE_H, TILE_W, 3), np.uint8),
                (TILE_W, TILE_H))
            cv2.putText(tile, f'{angle}deg', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            tiles.append(tile)

        while len(tiles) % cols:
            tiles.append(np.zeros((TILE_H, TILE_W, 3), np.uint8))

        rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
        debug_img = np.vstack(rows)

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

            ml_passed = surface['ml_passed']
            worst = surface['anomaly_score_worst']
            ml_str = 'N/A' if ml_passed is None else ('PASS' if ml_passed else 'FAIL')
            worst_str = 'N/A' if worst is None else f'{worst:.4f}'
            f.write(f'ML판정: {ml_str}\n')
            f.write(f'ML점수(worst): {worst_str}\n')
            f.write(f'ML상세: {surface["ml_detail"]}\n')

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
