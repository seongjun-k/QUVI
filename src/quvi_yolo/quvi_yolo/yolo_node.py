"""
QUVI YOLO_NODE
──────────────
3D 프린터 베드(Zone 1) 위의 출력물을 YOLO로 다중 감지하고,
좌표 목록을 /detection/objects 토픽으로 발행한다.

기능:
  - 핸드캠(compressed) 이미지 구독
  - YOLOv8n 추론 → Bounding Box + 신뢰도
  - 중심 좌표 계산 → 왼쪽→오른쪽 정렬
  - 근접 충돌 체크 (거리 < proximity_warning_px)
  - 디버그 이미지 발행 (탐지 결과 시각화)
"""

import math
from typing import List, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool

from quvi_msgs.msg import DetectedObject, ObjectArray


class YoloNode(Node):
    """YOLO 기반 3D 프린터 출력물 탐지 노드."""

    def __init__(self):
        super().__init__('yolo_node')

        # ─── 파라미터 선언 ───
        self.declare_parameter('camera_topic', '/camera1/image_raw/compressed')
        self.declare_parameter('use_compressed', True)
        self.declare_parameter('model_path', '')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('max_detections', 20)
        self.declare_parameter('target_classes', ['print_object'])
        self.declare_parameter('min_bbox_area', 500)
        self.declare_parameter('proximity_warning_px', 60)
        self.declare_parameter('sort_direction', 'left_to_right')
        self.declare_parameter('input_width', 640)
        self.declare_parameter('input_height', 480)
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('debug_image_topic', '/yolo/debug_image')

        # ─── 파라미터 로드 ───
        self._camera_topic = self.get_parameter('camera_topic').value
        self._use_compressed = self.get_parameter('use_compressed').value
        self._model_path = self.get_parameter('model_path').value
        self._conf_thresh = self.get_parameter('confidence_threshold').value
        self._iou_thresh = self.get_parameter('iou_threshold').value
        self._max_det = self.get_parameter('max_detections').value
        self._target_classes = self.get_parameter('target_classes').value
        self._min_bbox_area = self.get_parameter('min_bbox_area').value
        self._proximity_px = self.get_parameter('proximity_warning_px').value
        self._sort_dir = self.get_parameter('sort_direction').value
        self._input_w = self.get_parameter('input_width').value
        self._input_h = self.get_parameter('input_height').value
        self._pub_debug = self.get_parameter('publish_debug_image').value
        self._debug_topic = self.get_parameter('debug_image_topic').value

        # ─── YOLO 모델 로드 ───
        self._model = self._load_model()

        # ─── ROS 통신 ───
        self._bridge = CvBridge()

        # Subscriber: 카메라 이미지
        if self._use_compressed:
            self._img_sub = self.create_subscription(
                CompressedImage, self._camera_topic,
                self._image_callback, 10)
        else:
            self._img_sub = self.create_subscription(
                Image, self._camera_topic,
                self._image_callback_raw, 10)

        # Subscriber: 탐지 트리거 (MAIN에서 요청 시에만 탐지 수행)
        self._trigger_sub = self.create_subscription(
            Bool, '/detection/trigger',
            self._trigger_callback, 10)

        # Publisher: 탐지 결과
        self._obj_pub = self.create_publisher(ObjectArray, '/detection/objects', 10)

        # Publisher: 디버그 이미지
        if self._pub_debug:
            self._debug_pub = self.create_publisher(Image, self._debug_topic, 5)

        # Publisher: 근접 경고
        self._proximity_pub = self.create_publisher(Bool, '/detection/proximity_warning', 10)

        # 내부 상태
        self._latest_frame: np.ndarray | None = None
        self._detection_enabled = False

        self.get_logger().info(
            f'YOLO_NODE 초기화 완료 | 카메라: {self._camera_topic} | '
            f'신뢰도: {self._conf_thresh} | 최대 탐지: {self._max_det}')

    # ─────────────────────────────────────────────
    # 모델 로드
    # ─────────────────────────────────────────────
    def _load_model(self):
        """Ultralytics YOLO 모델 로드."""
        try:
            from ultralytics import YOLO
        except ImportError:
            self.get_logger().error('ultralytics 패키지가 설치되지 않음. pip install ultralytics')
            raise

        model_path = self._model_path if self._model_path else 'yolov8n.pt'
        self.get_logger().info(f'YOLO 모델 로드: {model_path}')
        model = YOLO(model_path)
        return model

    # ─────────────────────────────────────────────
    # 이미지 콜백
    # ─────────────────────────────────────────────
    def _image_callback(self, msg: CompressedImage):
        """Compressed 이미지 수신 → numpy 변환."""
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is not None:
            self._latest_frame = frame
            if self._detection_enabled:
                self._run_detection(frame, msg.header)

    def _image_callback_raw(self, msg: Image):
        """Raw 이미지 수신 → numpy 변환."""
        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if frame is not None:
            self._latest_frame = frame
            if self._detection_enabled:
                self._run_detection(frame, msg.header)

    def _trigger_callback(self, msg: Bool):
        """MAIN에서 탐지 트리거 수신."""
        if msg.data:
            self._detection_enabled = True
            self.get_logger().info('탐지 활성화됨')
            # 최신 프레임이 있으면 즉시 탐지
            if self._latest_frame is not None:
                from std_msgs.msg import Header
                header = Header()
                header.stamp = self.get_clock().now().to_msg()
                header.frame_id = 'camera_handcam'
                self._run_detection(self._latest_frame, header)
        else:
            self._detection_enabled = False
            self.get_logger().info('탐지 비활성화됨')

    # ─────────────────────────────────────────────
    # YOLO 추론
    # ─────────────────────────────────────────────
    def _run_detection(self, frame: np.ndarray, header):
        """YOLO 추론 실행 → 결과 발행."""
        # 리사이즈
        resized = cv2.resize(frame, (self._input_w, self._input_h))
        scale_x = frame.shape[1] / self._input_w
        scale_y = frame.shape[0] / self._input_h

        # YOLO 추론
        results = self._model(
            resized,
            conf=self._conf_thresh,
            iou=self._iou_thresh,
            max_det=self._max_det,
            verbose=False,
        )

        # 결과 파싱
        detections: List[DetectedObject] = []
        debug_frame = frame.copy() if self._pub_debug else None

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # 바운딩 박스 (xywh 형식)
                    x_c, y_c, w, h = box.xywh[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = result.names.get(cls_id, f'class_{cls_id}')

                    # 원본 해상도로 스케일링
                    x_c *= scale_x
                    y_c *= scale_y
                    w *= scale_x
                    h *= scale_y

                    # 최소 면적 필터
                    if w * h < self._min_bbox_area:
                        continue

                    # 커스텀 클래스 필터 (학습 전에는 모든 클래스 허용)
                    if self._target_classes and 'print_object' not in self._target_classes:
                        if cls_name not in self._target_classes:
                            continue

                    obj = DetectedObject()
                    obj.x = float(x_c)
                    obj.y = float(y_c)
                    obj.width = float(w)
                    obj.height = float(h)
                    obj.confidence = conf
                    obj.class_name = cls_name
                    detections.append(obj)

                    # 디버그 시각화
                    if debug_frame is not None:
                        x1 = int(x_c - w / 2)
                        y1 = int(y_c - h / 2)
                        x2 = int(x_c + w / 2)
                        y2 = int(y_c + h / 2)
                        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f'{cls_name} {conf:.2f}'
                        cv2.putText(debug_frame, label, (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # ─── 정렬: 왼쪽 → 오른쪽 ───
        if self._sort_dir == 'left_to_right':
            detections.sort(key=lambda d: d.x)
        elif self._sort_dir == 'right_to_left':
            detections.sort(key=lambda d: -d.x)

        # ─── 근접 충돌 체크 ───
        proximity_warning = self._check_proximity(detections)
        prox_msg = Bool()
        prox_msg.data = proximity_warning
        self._proximity_pub.publish(prox_msg)

        if proximity_warning:
            self.get_logger().warn(
                f'근접 충돌 경고! 출력물 간 거리 < {self._proximity_px}px')

        # ─── ObjectArray 발행 ───
        obj_array = ObjectArray()
        obj_array.header = header
        obj_array.objects = detections
        obj_array.total_count = len(detections)
        self._obj_pub.publish(obj_array)

        self.get_logger().info(
            f'탐지 완료: {len(detections)}개 출력물 | 근접경고: {proximity_warning}')

        # ─── 디버그 이미지 발행 ───
        if self._pub_debug and debug_frame is not None:
            # 탐지 정보 오버레이
            info_text = f'Objects: {len(detections)} | Proximity: {proximity_warning}'
            cv2.putText(debug_frame, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            debug_msg = self._bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
            self._debug_pub.publish(debug_msg)

        # 1회 탐지 후 비활성화
        self._detection_enabled = False

    # ─────────────────────────────────────────────
    # 근접 충돌 체크
    # ─────────────────────────────────────────────
    def _check_proximity(self, detections: List[DetectedObject]) -> bool:
        """출력물 간 거리 < proximity_warning_px인 쌍이 있는지 확인."""
        n = len(detections)
        for i in range(n):
            for j in range(i + 1, n):
                dist = math.sqrt(
                    (detections[i].x - detections[j].x) ** 2 +
                    (detections[i].y - detections[j].y) ** 2
                )
                if dist < self._proximity_px:
                    return True
        return False


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
