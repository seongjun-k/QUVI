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
import re
import time
import math
import json
import threading
import subprocess
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, String

from quvi_msgs.msg import GraspGoal, InspectionResult
from quvi_robot_control.utils import decode_compressed, BinaryCache, encode_bgr
from quvi_robot_control import topics
from quvi_inspect.ml_preprocess import preprocess_for_ml

# 품종 id는 디렉토리명으로 직접 쓰인다 — 경로 이탈 방지(트리거 문자열이 신뢰 경계).
PRODUCT_ID_RE = re.compile(r'^[A-Za-z0-9_-]{1,64}$')


class InspectNode(Node):
    """양불 판정 노드 — 표면 특징 분석 (CAD 비교는 제거됨)."""

    def __init__(self, **kwargs):
        # RobotControlNode 와 동일 패턴 — parameter_overrides 등을 그대로 전달해
        # 노드/런치 없이도 오프라인 테스트에서 파라미터를 주입할 수 있게 한다.
        super().__init__('inspect_node', **kwargs)

        # ─── 파라미터 선언 + 로드 (1단계) ───
        self._load_params()
        # 품종별 params.json 오버라이드가 없을 때 되돌아갈 전역 기본값 (SSoT: yaml/기본 파라미터).
        self._global_f_area_min = self._f_area_min
        self._global_f_area_max = self._f_area_max

        # ─── 다품종 검사 자산 ───
        # 자산은 각도로만 키잉하던 기존 구조를 품종(product_id) 하위로 감싼다.
        # 기동 시에는 UNSELECTED — 마지막 품종을 자동 로드하지 않는다(오판정 방지, 필수3).
        self._current_product: Optional[str] = None
        self._products_cache: List[dict] = []
        self._reference_images: Dict[int, np.ndarray] = {}
        self._anomaly_detectors: Dict[int, object] = {}
        self._anomaly_thresholds: Dict[int, float] = {}
        # ACT _act_reload_lock(robot_control_node.py) 과 동일한 논블로킹 락 패턴 —
        # 원자적 스왑 도중 중복 전환 요청을 거부한다.
        self._product_reload_lock = threading.Lock()
        self._product_loading = False

        # ─── ROS 통신 ───
        self._img_sub = self.create_subscription(
            CompressedImage, self._camera_topic,
            self._image_callback, 10)

        self._turntable_done_sub = self.create_subscription(
            Bool, topics.TOPIC_MOTOR_TURNTABLE_DONE,
            self._turntable_done_callback, 10)

        self._trigger_sub = self.create_subscription(
            Bool, topics.TOPIC_INSPECTION_TRIGGER,
            self._trigger_callback, 10)

        # turntable_done 누락(0도->0도 무이동 등) 시 검사 모드 캡처가 밀리는 문제를
        # 방지하기 위해 오케스트레이터가 각도별로 명시 발행하는 캡처 명령.
        self._capture_now_sub = self.create_subscription(
            Bool, topics.TOPIC_INSPECTION_CAPTURE_NOW,
            self._capture_now_callback, 10)

        self._ref_capture_sub = self.create_subscription(
            Bool, '/inspection/capture_reference',
            self._ref_capture_trigger_callback, 10)

        self._dataset_capture_sub = self.create_subscription(
            Bool, '/inspection/capture_dataset',
            self._dataset_capture_trigger_callback, 10)

        self._grasp_cmd_sub = self.create_subscription(
            GraspGoal, topics.TOPIC_ROBOT_GRASP_CMD,
            self._grasp_cmd_callback, 10)

        # 검사 조명 ON/OFF에 맞춰 검사캠 노출 전환 (orchestrator·HMI 어느 쪽이 켜도 잡힘)
        self._led_exposure_sub = self.create_subscription(
            Bool, topics.TOPIC_MOTOR_LED, self._led_exposure_cb, 10)

        # 품종 선택/생성 (HMI→inspect_node)
        self._product_select_sub = self.create_subscription(
            String, topics.TOPIC_INSPECTION_PRODUCT_SELECT,
            self._on_product_select, 10)
        self._product_create_sub = self.create_subscription(
            String, topics.TOPIC_INSPECTION_PRODUCT_CREATE,
            self._on_product_create, 10)

        self._result_pub = self.create_publisher(
            InspectionResult, topics.TOPIC_INSPECTION_RESULT, 10)

        if self._pub_debug:
            self._debug_pub = self.create_publisher(
                Image, self._debug_topic, 5)

        # 품종 목록/현재상태 (latched — 늦게 붙는 HMI 구독자도 즉시 최신 상태를 받는다.
        # ACT 모델 목록·현재상태(robot_control_node.py)와 동일 패턴).
        _latched = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._products_pub = self.create_publisher(
            String, topics.TOPIC_INSPECTION_PRODUCTS, _latched)
        self._product_current_pub = self.create_publisher(
            String, topics.TOPIC_INSPECTION_PRODUCT_CURRENT, _latched)

        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._captured_images: Dict[int, np.ndarray] = {}
        self._inspection_active = False
        # MultiThreadedExecutor에서 _capture_angle과 _watchdog_cb가 동시에
        # _run_inspection에 진입할 수 있어 이중 판정을 막는 락.
        self._inspection_lock = threading.Lock()
        self._ref_capture_active = False
        self._dataset_capture_active = False
        self._current_object_index = 0

        # ─── 캡처 안정화 타이머 ───
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

        # ─── 검사 워치독 ───
        # turntable_done 누락(예: 0°에서 0°로 '이동' 시 done 미발행)으로 캡처가
        # 4장을 못 채우면 판정이 영영 실행되지 않아 오케스트레이터가 타임아웃/ERROR
        # 로 빠진다. 검사 활성 후 일정 시간 내 미완료면 확보된 캡처로 마무리한다.
        self._inspection_start = 0.0
        self._inspection_watchdog = self.create_timer(1.0, self._watchdog_cb)

        # ─── 품종 목록 스캔 + 최초 발행 (UNSELECTED로 시작, 필수3) ───
        self._publish_products()
        self._publish_current_product()

        self.get_logger().info(
            f'INSPECT_NODE 초기화 완료 | '
            f'촬영 각도: {self._angles} | 판정 타임아웃: {self._finalize_sec}s | '
            f'품종 디렉토리: {self._products_dir} (미선택으로 시작)')

    # ─── 파라미터 (선언 + 로드 통합) ───
    def _load_params(self):
        """모든 파라미터를 선언하고 로컬 멤버 변수로 로드합니다."""
        params = [
            ('camera_topic',            '/camera2/image_raw/compressed',    '_camera_topic'),
            ('reference_image_dir',     '/workspace/data/reference_images',  '_ref_dir'),
            # 다품종 검사 자산 루트 — 품종별 하위 디렉토리(reference_images/models/anomaly_dataset/params.json)를 관리.
            # 위 reference_image_dir(_ref_dir)은 레거시 전역 경로로 남겨두되(자동 마이그레이션 금지),
            # 다품종 도입 후에는 사용하지 않는다.
            ('inspection_products_dir', '/workspace/data/inspection_products', '_products_dir'),
            # ─── 표면 특징 분석 임계값 ───
            ('solidity_min',            0.85,                               '_sol_min'),
            ('solidity_max',            1.00,                               '_sol_max'),
            ('feature_area_ratio_min',  0.80,                               '_f_area_min'),
            ('feature_area_ratio_max',  1.50,                               '_f_area_max'),
            ('hole_count_max',          0,                                  '_hole_max'),
            ('hole_area_ratio_max',     0.05,                               '_hole_area_max'),
            ('texture_variance_max',    500.0,                              '_tex_var_max'),
            ('min_hole_area_px',        50,                                 '_min_hole_px'),
            # ─── 턴테이블 / 전처리 ───
            ('turntable_angles',        [0, 90, 180, 270],                  '_angles'),
            # 데이터셋 촬영(1.5s 노출 안정)과 실검사 노출 조건 정합 — train/infer skew 방지
            ('capture_settle_sec',      1.5,                                '_capture_settle'),
            # 한 바퀴 캡처 ≈ 20s(각도당 ~5s) + LED 안정화 5s + 여유 — 12s는 270° 캡처 전에 판정을 강행했다
            ('inspection_finalize_sec', 45.0,                               '_finalize_sec'),
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
            # ─── 검사 LED 연동 노출 (흰 출력물 링조명 정반사 과노출 방지, 2026-09-05 실측) ───
            ('inspection_cam_device',   '/dev/fixed_cam',                   '_insp_cam_dev'),
            ('led_on_exposure',         3,                                  '_led_on_exp'),
            ('led_on_brightness',       -30,                                '_led_on_bright'),
            ('led_on_gain',             0,                                  '_led_on_gain'),
            ('led_off_exposure',        400,                                '_led_off_exp'),
            ('led_off_brightness',      0,                                  '_led_off_bright'),
            ('led_off_gain',            30,                                 '_led_off_gain'),
        ]

        for name, default, attr_name in params:
            self.declare_parameter(name, default)
            setattr(self, attr_name, self.get_parameter(name).value)

    # ─── 다품종 자산 경로 헬퍼 ───
    def _product_dir(self, pid: str) -> str:
        return os.path.join(self._products_dir, pid)

    def _product_ref_dir(self, pid: str) -> str:
        return os.path.join(self._product_dir(pid), 'reference_images')

    def _product_model_dir(self, pid: str) -> str:
        return os.path.join(self._product_dir(pid), 'models')

    def _product_ds_dir(self, pid: str) -> str:
        return os.path.join(self._product_dir(pid), 'anomaly_dataset', 'raw')

    def _product_params_path(self, pid: str) -> str:
        return os.path.join(self._product_dir(pid), 'params.json')

    # ─── 품종 스캔/발행 ───
    def _product_status(self, pid: str) -> dict:
        """품종 하나의 자산 무결성을 점검한다.

        ready = 4각도 기준이미지 전부 존재 AND (anomaly_enabled면 4뱅크+thresholds 전부).
        부분 자산으로 로드하고 해당 축만 스킵하는 관용은 다품종에서 없앤다(필수1).
        """
        missing: List[str] = []
        ref_dir = self._product_ref_dir(pid)
        for angle in self._angles:
            if not os.path.isfile(os.path.join(ref_dir, f'ref_{angle}.png')):
                missing.append(f'ref_{angle}')
        if self._anomaly_enabled:
            model_dir = self._product_model_dir(pid)
            if not os.path.isfile(os.path.join(model_dir, 'thresholds.json')):
                missing.append('thresholds.json')
            for angle in self._angles:
                if not os.path.isfile(os.path.join(model_dir, f'bank_{angle}.pt')):
                    missing.append(f'bank_{angle}')
        return {'id': pid, 'ready': len(missing) == 0, 'missing': missing}

    def _scan_products(self) -> List[dict]:
        """products_dir 하위 품종 디렉토리를 스캔해 무결성 상태 목록을 만든다."""
        os.makedirs(self._products_dir, exist_ok=True)
        products = []
        for entry in sorted(os.listdir(self._products_dir)):
            if os.path.isdir(os.path.join(self._products_dir, entry)):
                products.append(self._product_status(entry))
        return products

    def _publish_products(self):
        """품종 목록을 재스캔해 latched 토픽으로 발행 (ACT 모델 목록과 동일 패턴)."""
        try:
            products = self._scan_products()
        except OSError as exc:
            self.get_logger().error(f'품종 스캔 실패: {exc}')
            products = []
        self._products_cache = products
        self._products_pub.publish(String(data=json.dumps(products, ensure_ascii=False)))

    def _publish_current_product(self):
        """현재 선택된 품종 id를 latched 토픽으로 발행 (미선택이면 빈 문자열)."""
        self._product_current_pub.publish(String(data=self._current_product or ''))

    # ─── 품종 생성 ───
    def _on_product_create(self, msg: String):
        """products_dir/<pid>/ 하위 디렉토리 골격을 생성한다. 생성 직후는 ready=false 가 정상."""
        pid = msg.data.strip()
        if not PRODUCT_ID_RE.match(pid):
            self.get_logger().warn(f'품종 생성 거부 — 유효하지 않은 id: {pid!r}')
            return
        try:
            os.makedirs(self._product_ref_dir(pid), exist_ok=True)
            os.makedirs(self._product_model_dir(pid), exist_ok=True)
            for angle in self._angles:
                os.makedirs(os.path.join(self._product_ds_dir(pid), str(angle)), exist_ok=True)
        except OSError as exc:
            self.get_logger().error(f'품종 생성 실패({pid}): {exc}')
            return
        self.get_logger().info(f'품종 생성: {pid}')
        self._publish_products()

    # ─── 품종 선택 (원자적 스왑, 필수1·필수2) ───
    def _on_product_select(self, msg: String):
        pid = msg.data.strip()
        if not pid:
            return
        if self._inspection_active:
            self.get_logger().warn(f'검사 진행 중 — 품종 전환 거부: {pid} (현재 품종 유지)')
            return
        threading.Thread(target=self._reload_product, args=(pid,), daemon=True).start()

    def _reload_product(self, pid: str):
        """대상 품종 자산을 임시 dict에 전부 로드 후 원자적으로 스왑한다.

        ACT `_load_act_policy`(robot_control_node.py) 의 논블로킹 락 + 백그라운드
        재로드 패턴을 그대로 따른다. 로딩 도중 검사가 시작되면 스왑을 취소해
        구/신 자산 혼합을 막는다.
        """
        if not self._product_reload_lock.acquire(blocking=False):
            self.get_logger().warn('품종 전환 이미 진행 중 — 요청 무시')
            return
        try:
            self._product_loading = True
            info = self._product_status(pid)
            if not os.path.isdir(self._product_dir(pid)):
                self.get_logger().error(f'품종 전환 거부 — 존재하지 않는 품종: {pid}')
                return
            # 불완전 품종도 '캡처 대상'으로 선택 허용한다 — 자산을 쌓아 완성하려면 먼저 선택돼야
            # 하기 때문(선택 거부하면 교착). 판정 실행은 _run_inspection_inner 의 메모리 완전성
            # 게이트가 별도로 막으므로, 불완전 상태로 PASS 가 나갈 일은 없다.
            if not info['ready']:
                self.get_logger().info(
                    f'품종 선택(자산 미비): {pid} (누락: {info["missing"]}) — 캡처 대상 지정, 검사는 불가')

            # ── 임시 dict에 자산 로드 (있는 것만) — 부분 로드는 판정 게이트가 걸러낸다 ──
            new_refs: Dict[int, np.ndarray] = {}
            ref_dir = self._product_ref_dir(pid)
            for angle in self._angles:
                path = os.path.join(ref_dir, f'ref_{angle}.png')
                if not os.path.isfile(path):
                    continue
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    new_refs[angle] = img

            new_f_area_min, new_f_area_max = self._global_f_area_min, self._global_f_area_max
            params_path = self._product_params_path(pid)
            if os.path.isfile(params_path):
                try:
                    with open(params_path, encoding='utf-8') as f:
                        p = json.load(f)
                    new_f_area_min = float(p.get('feature_area_ratio_min', new_f_area_min))
                    new_f_area_max = float(p.get('feature_area_ratio_max', new_f_area_max))
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    self.get_logger().warn(f'{pid} params.json 파싱 실패({exc}) — 전역 임계값 사용')

            new_detectors: Dict[int, object] = {}
            new_thresholds: Dict[int, float] = {}
            model_dir = self._product_model_dir(pid)
            thresholds_path = os.path.join(model_dir, 'thresholds.json')
            # 뱅크·임계값이 4각도 전부 있을 때만 ML 로드. 부분이면 이 품종은 ML off 로 선택되고,
            # 판정 게이트가 anomaly_enabled 인데 detectors 미완이면 검사를 막는다(ML 우회 방지).
            banks_present = (self._anomaly_enabled and os.path.isfile(thresholds_path) and all(
                os.path.isfile(os.path.join(model_dir, f'bank_{a}.pt')) for a in self._angles))
            if banks_present:
                try:
                    # torch는 이 모듈 내부에서만 import되므로 비활성 시 로드 비용이 없다.
                    from quvi_inspect.anomaly_detector import PatchCoreDetector

                    with open(thresholds_path, encoding='utf-8') as f:
                        thresholds = json.load(f)
                    # 백본은 품종 무관 — 기존 전역 anomaly_model_dir 에서 로드(공용 유지).
                    backbone_path = os.path.join(self._anomaly_model_dir, 'wide_resnet50.pth')
                    shared_backbone = None
                    for angle in self._angles:
                        bank_path = os.path.join(model_dir, f'bank_{angle}.pt')
                        detector = PatchCoreDetector.load(
                            bank_path, device=self._anomaly_device,
                            backbone_weights_path=backbone_path, backbone=shared_backbone)
                        if shared_backbone is None:
                            shared_backbone = detector.backbone
                        new_detectors[angle] = detector
                        new_thresholds[angle] = float(thresholds[str(angle)]['threshold'])
                except Exception as exc:  # noqa: BLE001 — ML 로드 실패는 ML만 끄고 선택은 진행
                    self.get_logger().error(f'{pid} ML 자산 로드 실패({exc}) — 이 품종 ML off 로 선택')
                    new_detectors = {}
                    new_thresholds = {}

            # ── 원자적 스왑 — 로딩 도중 검사가 시작됐으면 취소(구/신 자산 혼합 금지) ──
            if self._inspection_active:
                self.get_logger().warn(f'품종 전환 취소 — 로딩 중 검사가 시작됨: {pid}')
                return
            self._reference_images = new_refs
            self._anomaly_detectors = new_detectors
            self._anomaly_thresholds = new_thresholds
            self._f_area_min = new_f_area_min
            self._f_area_max = new_f_area_max
            self._current_product = pid
            self._publish_current_product()
            self.get_logger().info(
                f'품종 전환 완료: {pid} | 면적비 임계=[{new_f_area_min}, {new_f_area_max}] | '
                f'ML={"ON" if new_detectors else "OFF"}')
        finally:
            self._product_loading = False
            self._product_reload_lock.release()

    # ─── 콜백 ───
    def _image_callback(self, msg: CompressedImage):
        frame = decode_compressed(msg)
        if frame is not None:
            # 검사캠이 거꾸로 장착되어 상하 반전 + 좌우 반전 → 동시 -1
            flipped = cv2.flip(frame, -1)
            with self._frame_lock:
                self._latest_frame = flipped

    def _grasp_cmd_callback(self, msg: GraspGoal):
        self._current_object_index = msg.object_index
        self.get_logger().info(f'Object index 동기화: {self._current_object_index}')

    # ─── 검사 LED 연동 노출 ───
    def _led_exposure_cb(self, msg: Bool):
        """검사 LED ON/OFF에 맞춰 검사캠 노출을 전환한다.
        흰 출력물이 링조명 정반사로 과노출되어 표면 결이 사라지는 것을 막는다 —
        LED ON 시 저노출 고정, OFF 시 일반 노출로 원복. usb_cam은 기동 시 한 번만
        컨트롤을 걸므로 스트리밍 중 외부 v4l2 설정이 그대로 유지된다."""
        if msg.data:
            self._set_cam_exposure(self._led_on_exp, self._led_on_bright, self._led_on_gain)
        else:
            self._set_cam_exposure(self._led_off_exp, self._led_off_bright, self._led_off_gain)

    def _set_cam_exposure(self, exposure: int, brightness: int, gain: int):
        """v4l2-ctl로 검사캠 노출/밝기/게인을 즉시 설정한다. auto_exposure=1(Manual)을
        먼저 걸어야 exposure_time_absolute가 적용된다(콤마 인자는 좌→우 순차 적용)."""
        try:
            subprocess.run(
                ['v4l2-ctl', '-d', self._insp_cam_dev, '--set-ctrl',
                 f'auto_exposure=1,exposure_time_absolute={int(exposure)},'
                 f'gain={int(gain)},brightness={int(brightness)}'],
                check=True, timeout=2, capture_output=True)
            self.get_logger().info(
                f'검사캠 노출 전환: exposure={exposure}, brightness={brightness}, gain={gain}')
        except (subprocess.SubprocessError, OSError) as exc:
            self.get_logger().warn(f'검사캠 노출 설정 실패({exc}) — 무시하고 진행')

    def _turntable_done_callback(self, msg: Bool):
        """턴테이블 이동 완료 시 안정화 지연 후 캡처를 예약한다.

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
        """오케스트레이터의 명시 캡처 명령 수신 (검사 모드 전용)."""
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
        with self._frame_lock:
            frame = self._latest_frame.copy() if self._latest_frame is not None else None
        if frame is not None:
            self._captured_images[angle] = frame
            self.get_logger().info(f'캡처 완료: {angle}°')
            if len(self._captured_images) == len(self._angles):
                self._run_inspection()
        else:
            self.get_logger().warn(f'{angle}° 캡처 실패: 카메라 프레임 없음')

    def _trigger_callback(self, msg: Bool):
        """검사 트리거 수신."""
        if msg.data:
            if self._ref_capture_active:
                # 기준 캡처 중 동시 진입 방지 — _captured_images 공유로 인한 교차 오염 방지
                self.get_logger().warn('기준 이미지 캡처 진행 중 — 검사 트리거 무시')
                return
            self._inspection_active = True
            self._captured_images.clear()
            self._inspection_start = time.time()   # 워치독 기준
            self.get_logger().info('검사 모드 활성화 — 턴테이블 회전 대기 중')
        else:
            self._inspection_active = False

    def _watchdog_cb(self):
        """검사 완료 워치독. turntable_done 누락 등으로 캡처가 부족해도
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
        """기준 이미지 캡처 트리거 수신 (정상품을 챔버에 올려둔 상태에서 발행).

        현재 품종이 선택돼 있어야 한다 — 자산이 어디 저장될지 모호한 상태로
        촬영을 시작하지 않는다.
        """
        if msg.data:
            if self._inspection_active:
                self.get_logger().warn('검사 진행 중 — 기준 캡처 무시')
                return
            if self._current_product is None:
                self.get_logger().warn('품종 미선택 — 기준 캡처 거부')
                return
            self._ref_capture_active = True
            self._captured_images.clear()
            self.get_logger().info(
                f'기준 이미지 캡처 모드 활성화 | 품종: {self._current_product} | '
                f'저장 경로: {self._product_ref_dir(self._current_product)} | '
                f'턴테이블 {self._angles}° 순서로 회전시키세요')
        else:
            self._ref_capture_active = False

    def _capture_reference_angle(self, angle: int):
        """현재 프레임을 기준 이미지로 캡처 후 현재 품종 하위에 파일 저장."""
        if self._current_product is None:
            self.get_logger().warn('품종 미선택 — 기준 캡처 중단')
            self._ref_capture_active = False
            return
        with self._frame_lock:
            frame = self._latest_frame.copy() if self._latest_frame is not None else None
        if frame is None:
            self.get_logger().warn(f'{angle}° 기준 캡처 실패: 카메라 프레임 없음')
            return

        gray = self._preprocess(frame)
        self._captured_images[angle] = gray

        ref_dir = self._product_ref_dir(self._current_product)
        os.makedirs(ref_dir, exist_ok=True)
        path = os.path.join(ref_dir, f'ref_{angle}.png')
        cv2.imwrite(path, gray)
        self.get_logger().info(f'기준 이미지 저장: {path}')

        if len(self._captured_images) == len(self._angles):
            self._reference_images = dict(self._captured_images)
            self._captured_images.clear()
            self._ref_capture_active = False
            self.get_logger().info(
                f'기준 이미지 {len(self._angles)}장 캡처 완료 — 즉시 적용됨')
            # ready 상태가 바뀌었을 수 있으니 목록 재발행 (HMI 반영).
            self._publish_products()

    def _dataset_capture_trigger_callback(self, msg: Bool):
        """데이터셋 촬영 트리거 수신 (ML 정상품 데이터셋 수집용 별도 병렬 모드).

        현재 품종이 선택돼 있어야 한다 — 자산이 어디 저장될지 모호한 상태로
        촬영을 시작하지 않는다.
        """
        if msg.data:
            if self._inspection_active or self._ref_capture_active:
                self.get_logger().warn('검사/기준 캡처 진행 중 — 데이터셋 캡처 무시')
                return
            if self._current_product is None:
                self.get_logger().warn('품종 미선택 — 데이터셋 캡처 거부')
                return
            self._dataset_capture_active = True
            self._captured_images.clear()
            self.get_logger().info(
                f'데이터셋 촬영 모드 활성화 | 품종: {self._current_product} | '
                f'저장 경로: {self._product_ds_dir(self._current_product)} | '
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
        """현재 프레임을 컬러 원본 그대로 현재 품종 데이터셋 디렉토리에 저장.

        기준 이미지(_reference_images)와 무관한 별도 경로로,
        grayscale 전처리 없이 원본을 저장한다.
        """
        if self._current_product is None:
            self.get_logger().warn('품종 미선택 — 데이터셋 캡처 중단')
            self._dataset_capture_active = False
            return
        with self._frame_lock:
            frame = self._latest_frame.copy() if self._latest_frame is not None else None
        if frame is None:
            self.get_logger().warn(f'{angle}° 데이터셋 캡처 실패: 카메라 프레임 없음')
            return

        frame = self._latest_frame.copy()
        self._captured_images[angle] = frame

        ds_dir = self._product_ds_dir(self._current_product)
        angle_dir = os.path.join(ds_dir, str(angle))
        os.makedirs(angle_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(angle_dir, f'{timestamp}.png')
        cv2.imwrite(path, frame)
        self.get_logger().info(f'데이터셋 이미지 저장: {path}')

        if len(self._captured_images) == len(self._angles):
            self._dataset_capture_active = False
            self._captured_images.clear()
            self.get_logger().info(
                f'데이터셋 {len(self._angles)}장 저장 완료 — 경로: {ds_dir}')

    # ─── 메인 검사 로직 ───
    def _run_inspection(self):
        """4방향 이미지로 표면 특징 분석 검사 실행."""
        with self._inspection_lock:
            # 먼저 들어간 호출이 finally에서 _inspection_active를 내리므로,
            # 락을 기다리다 뒤늦게 들어온 두번째 호출은 여기서 걸러진다.
            if not self._inspection_active:
                return
            try:
                self._run_inspection_inner()
            except Exception as e:   # noqa: BLE001 — 분석 예외로 노드가 고착되지 않도록
                self.get_logger().error(f'검사 중 예외 발생 — 검사 상태 해제: {e}')
            finally:
                self._inspection_active = False
                self._captured_images.clear()

    def _publish_unselected_result(self, start_time: float, reason: str = '품종 미선택'):
        """품종 미선택·자산 미비 상태에서 검사가 트리거되면 판정을 실행하지 않고 즉시 FAIL 발행한다."""
        fail_reason = f'{reason} — 검사 불가'
        result = InspectionResult()
        result.header.stamp        = self.get_clock().now().to_msg()
        result.header.frame_id     = 'inspection_chamber'
        result.passed               = False
        result.fail_reason          = fail_reason
        result.solidity             = 0.0
        result.area_ratio           = 0.0
        result.hole_count           = 0
        result.hole_area_ratio      = 0.0
        result.texture_variance     = 0.0
        result.anomaly_score_worst  = -1.0
        result.ml_passed            = -1
        result.object_index         = self._current_object_index
        result.inspection_time_sec  = time.time() - start_time
        self._result_pub.publish(result)
        self.get_logger().error(f'판정 불가: {fail_reason}')
        self.get_logger().info('=' * 50)

    def _run_inspection_inner(self):
        start_time = time.time()
        self.get_logger().info('=' * 50)
        self.get_logger().info('양불 판정 시작 (표면 특징 분석)')

        # ── 필수3: 실제 '메모리에 로드된' 자산의 완전성으로 판정 실행을 게이트한다(디스크 stat 아님).
        # 뱅크는 오프라인 스크립트가 디스크에 직접 쓰므로, stat 기준이면 '뱅크 완성됐지만 아직
        # 재선택 안 함 → detectors 비어있음' 상태에서 ml_passed=None 으로 룰단독 PASS(ML 우회)가
        # 나갈 수 있다. 판정에 실제로 쓰이는 메모리 자산으로 게이트해야 그 구멍이 닫힌다.
        refs_full = len(self._reference_images) == len(self._angles)
        ml_ready = (not self._anomaly_enabled) or (len(self._anomaly_detectors) == len(self._angles))
        if self._current_product is None or not refs_full or not ml_ready:
            reason = '품종 미선택' if self._current_product is None else '자산 미비(기준이미지/뱅크 부족)'
            self._publish_unselected_result(start_time, reason)
            return

        surface_results = self._surface_analysis()
        rule_pass = surface_results['passed']
        ml_passed = surface_results['ml_passed']
        # ML 미로드(None)면 룰 단독 폴백 — ML 이 명시적으로 FAIL(False) 일 때만 최종 판정에 반영
        final_pass = rule_pass and (ml_passed is not False)
        if not rule_pass and ml_passed is False:
            fail_reason = surface_results['fail_detail'] + '; ML 이상탐지: ' + surface_results['ml_detail']
        elif not rule_pass:
            fail_reason = surface_results['fail_detail']
        elif ml_passed is False:
            fail_reason = f"ML 이상탐지: {surface_results['ml_detail']}"
        else:
            fail_reason = ''

        elapsed = time.time() - start_time

        worst = surface_results['anomaly_score_worst']
        anomaly_score_worst = worst if worst is not None else -1.0
        # msg 의 ml_passed 는 int8 이라 None 표현 불가 — -1=미사용, 0=FAIL, 1=PASS 로 매핑
        if ml_passed is None:
            ml_passed_msg = -1
        elif ml_passed is False:
            ml_passed_msg = 0
        else:
            ml_passed_msg = 1

        result = InspectionResult()
        result.header.stamp        = self.get_clock().now().to_msg()
        result.header.frame_id     = 'inspection_chamber'
        result.passed               = final_pass
        result.fail_reason          = fail_reason
        result.solidity             = surface_results['solidity']
        result.area_ratio           = surface_results['area_ratio']
        result.hole_count           = surface_results['hole_count']
        result.hole_area_ratio      = surface_results['hole_area_ratio']
        result.texture_variance     = surface_results['texture_variance']
        result.anomaly_score_worst  = anomaly_score_worst
        result.ml_passed            = ml_passed_msg
        result.object_index         = self._current_object_index
        result.inspection_time_sec  = elapsed
        self._result_pub.publish(result)

        status = 'PASS ✓' if final_pass else f'FAIL ✗ ({fail_reason})'
        self.get_logger().info(f'판정: {status} | 품종: {self._current_product} | 소요: {elapsed:.2f}s')
        self.get_logger().info(
            f'  Solidity: {surface_results["solidity"]:.3f} | '
            f'구멍: {surface_results["hole_count"]}개 | '
            f'텍스처: {surface_results["texture_variance"]:.1f}')

        # ── 하이브리드 판정 로그: 룰 PASS 라도 ML 이 FAIL 이면 최종 판정에 반영됨 ──
        if ml_passed is None:
            ml_str, worst_str, agree = 'N/A', 'N/A', 'N/A'
        else:
            ml_str = 'PASS' if ml_passed else 'FAIL'
            worst_str = f'{worst:.2f}'
            agree = '일치' if ml_passed == rule_pass else '불일치'
        rule_str = 'PASS' if rule_pass else 'FAIL'
        self.get_logger().info(
            f'[하이브리드] 룰={rule_str} | ML={ml_str} (worst={worst_str}) | {agree}')
        self.get_logger().info('=' * 50)

        if self._pub_debug:
            self._publish_debug_image(final_pass, surface_results)
        if self._save_images:
            self._save_inspection_log(final_pass, surface_results)
        # 상태 해제는 _run_inspection 의 try/finally 가 일괄 처리

    # ─── 이미지 전처리 ───
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """이미지 전처리: 그레이스케일 + 가우시안 블러."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        return cv2.GaussianBlur(gray, (self._blur_k, self._blur_k), 0)

    # ─── 표면 특징 분석 ───
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
            # 턴테이블 편심으로 물체-카메라 거리가 위상마다 변해 raw 면적이
            # 거리 제곱으로 흔들리므로(정상품 0.78~1.71 실측),
            # 면적/폭² 끼리 비교해 거리 배율을 상쇄한다. 미출력(높이 손실)은
            # 폭 대비 면적이 줄어 여전히 비율 하락으로 검출된다.
            cap_area = cache.largest_external_area()
            cap_w    = cache.largest_external_width()
            cap_norm = cap_area / (cap_w * cap_w) if cap_w > 0 else 0.0
            a_ratio  = float('nan')
            for ref in self._reference_images.values():
                if ref is None:
                    continue
                ref_resized = cv2.resize(ref, (cache.gray.shape[1], cache.gray.shape[0]))
                ref_cache = BinaryCache(ref_resized, self._bin_thresh)
                ref_area  = ref_cache.largest_external_area()
                ref_w     = ref_cache.largest_external_width()
                ref_norm  = ref_area / (ref_w * ref_w) if ref_w > 0 else 0.0
                r = cap_norm / ref_norm if ref_norm > 0 else 0.0
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

            # ── ML 이상탐지 점수 계산 (최종 판정 반영은 _run_inspection_inner 의 하이브리드 로직에서) ──
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

        # worst_area: 범위 위반 값 우선, 없으면 1.0 에서 가장 멀리 벗어난 값 (NaN 제외)
        # 합격 범위가 비대칭이라 |1-a| 최대값이 범위 기준 최악과 다를 수 있다 —
        # 위반 값을 두고 편차 큰 통과 값을 표시하면 표는 OK 인데 사유는 면적비인 모순이 생긴다
        valid_areas = [a for a in all_area if not math.isnan(a)]
        violating_areas = [a for a in valid_areas
                           if not (self._f_area_min <= a <= self._f_area_max)]
        pick = violating_areas or valid_areas
        worst_area = max(pick, key=lambda a: abs(1.0 - a)) if pick else float('nan')

        # ── ML 이상탐지 집계 (여기서는 판정만 계산 — 최종 반영은 _run_inspection_inner 에서) ──
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

    # ─── 디버그 / 로깅 ───
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
            f.write(f'품종: {self._current_product}\n')
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

        # result.json — 기계 판독용(shadow_report 등). result.txt 는 사람이 읽는 용도로 유지.
        # json.dump 의 기본 NaN 출력은 비표준이라 파서가 깨진다 — None 으로 정규화.
        def _n(v):
            return None if isinstance(v, float) and math.isnan(v) else v

        result_json = {
            'passed':              bool(passed),
            'product_id':          self._current_product,
            'solidity':            _n(surface['solidity']),
            'area_ratio':          _n(surface['area_ratio']),
            'hole_count':          surface['hole_count'],
            'hole_area_ratio':     _n(surface['hole_area_ratio']),
            'texture_variance':    _n(surface['texture_variance']),
            'ml_passed':           surface['ml_passed'],
            'anomaly_score_worst': _n(surface['anomaly_score_worst']),
            'ml_detail':           surface['ml_detail'],
        }
        with open(os.path.join(log_subdir, 'result.json'), 'w', encoding='utf-8') as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2)

        self.get_logger().info(f'검사 로그 저장: {log_subdir}')


def main(args=None):
    rclpy.init(args=args)
    node = InspectNode()
    # 캡처 지연 타이머와 이미지/트리거 콜백이 서로를 막지 않도록 멀티스레드 실행.
    from rclpy.executors import MultiThreadedExecutor, ExternalShutdownException
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
