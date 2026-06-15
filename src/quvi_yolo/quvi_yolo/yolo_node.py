"""
QUVI YOLO_NODE
──────────────
3D 프린터 베드(Zone 1) 위의 출력물을 YOLO로 다중 감지하고,
좌표 목록을 /detection/objects 토픽으로 발행한다.

기능:
  - 핸드캠(compressed) 이미지 구독
  - YOLOv8n 추론 → Bounding Box + 신뢰도
  - 중심 좌표 계산 → 왼쪽→오른쪽 정렬
  - 근접 충돌 체크 (거리² 비교, sqrt 제거)
  - 디버그 이미지 발행 (탐지 결과 시각화)
"""

from typing import List

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool

from quvi_msgs.msg import DetectedObject, ObjectArray
from quvi_robot_control.utils import decode_compressed, decode_raw, declare_and_get, encode_bgr


class YoloNode(Node):
    """YOLO 기반 3D 프린터 출력물 탐지 노드."""

    def __init__(self):
        super().__init__('yolo_node')

        self._load_params()
        self._model = self._load_model()

        if self._use_compressed:
            self._img_sub = self.create_subscription(
                CompressedImage, self._camera_topic,
                self._image_callback, 10)
        else:
            self._img_sub = self.create_subscription(
                Image, self._camera_topic,
                self._image_callback_raw, 10)

        self._trigger_sub = self.create_subscription(
            Bool, '/detection/trigger',
            self._trigger_callback, 10)

        self._obj_pub       = self.create_publisher(ObjectArray, '/detection/objects', 10)
        self._proximity_pub = self.create_publisher(Bool, '/detection/proximity_warning', 10)

        if self._pub_debug:
            self._debug_pub = self.create_publisher(Image, self._debug_topic, 5)

        self._latest_frame: np.ndarray | None = None
        self._detection_enabled = False

        self.get_logger().info(
            f'YOLO_NODE 초기화 완료 | 카메라: {self._camera_topic} | '
            f'신뢰도: {self._conf_thresh} | 최대 탐지: {self._max_det}')

    # ─────────────────────────────────────────────
    # 파라미터 (선언 + 로드 통합)
    # ─────────────────────────────────────────────
    def _load_params(self):
        g = lambda name, default: declare_and_get(self, name, default)

        self._camera_topic    = g('camera_topic',           '/camera1/image_raw/compressed')
        self._use_compressed  = g('use_compressed',         True)
        self._model_path      = g('model_path',             '')
        self._conf_thresh     = g('confidence_threshold',   0.5)
        self._iou_thresh      = g('iou_threshold',          0.45)
        self._max_det         = g('max_detections',         20)
        self._target_classes  = g('target_classes',         ['print_object'])
        self._min_bbox_area   = g('min_bbox_area',          500)
        self._proximity_px    = g('proximity_warning_px',   60)
        self._sort_dir        = g('sort_direction',         'left_to_right')
        self._input_w         = g('input_width',            640)
        self._input_h         = g('input_height',           480)
        self._pub_debug       = g('publish_debug_image',    True)
        self._debug_topic     = g('debug_image_topic',      '/yolo/debug_image')

    # ─────────────────────────────────────────────
    # 모델 로드
    # ─────────────────────────────────────────────
    def _load_model(self):
        try:
            from ultralytics import YOLO
        except ImportError:
            self.get_logger().error('ultralytics 패키지가 설치되지 않음. pip install ultralytics')
            raise
        model_path = self._model_path if self._model_path else 'yolov8n.pt'
        self.get_logger().info(f'YOLO 모델 로드: {model_path}')
        return YOLO(model_path)

    # ─────────────────────────────────────────────
    # 이미지 콜백
    # ─────────────────────────────────────────────
    def _image_callback(self, msg: CompressedImage):
        frame = decode_compressed(msg)
        if frame is not None:
            self._latest_frame = frame
            if self._detection_enabled:
                self._run_detection(frame, msg.header)

    def _image_callback_raw(self, msg: Image):
        frame = decode_raw(msg)
        if frame is not None:
            self._latest_frame = frame
            if self._detection_enabled:
                self._run_detection(frame, msg.header)

    def _trigger_callback(self, msg: Bool):
        if msg.data:
            self._detection_enabled = True
            self.get_logger().info('탐지 활성화됨')
            if self._latest_frame is not None:
                from std_msgs.msg import Header
                header = Header()
                header.stamp    = self.get_clock().now().to_msg()
                header.frame_id = 'camera_handcam'
                self._run_detection(self._latest_frame, header)
        else:
            self._detection_enabled = False
            self.get_logger().info('탐지 비활성화됨')

    # ─────────────────────────────────────────────
    # YOLO 추론
    # ─────────────────────────────────────────────
    def _run_detection(self, frame: np.ndarray, header):
        resized  = cv2.resize(frame, (self._input_w, self._input_h))
        scale_x  = frame.shape[1] / self._input_w
        scale_y  = frame.shape[0] / self._input_h

        results  = self._model(
            resized,
            conf=self._conf_thresh,
            iou=self._iou_thresh,
            max_det=self._max_det,
            verbose=False,
        )

        detections: List[DetectedObject] = []
        debug_frame = frame.copy() if self._pub_debug else None

        if results and len(results) > 0:
            result = results[0]
            boxes  = result.boxes

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x_c, y_c, w, h = box.xywh[0].cpu().numpy()
                    conf    = float(box.conf[0].cpu().numpy())
                    cls_id  = int(box.cls[0].cpu().numpy())
                    cls_name = result.names.get(cls_id, f'class_{cls_id}')

                    x_c *= scale_x;  y_c *= scale_y
                    w   *= scale_x;  h   *= scale_y

                    if w * h < self._min_bbox_area:
                        continue
                    if self._target_classes and cls_name not in self._target_classes:
                        continue

                    obj = DetectedObject()
                    obj.x          = float(x_c)
                    obj.y          = float(y_c)
                    obj.width      = float(w)
                    obj.height     = float(h)
                    obj.confidence = conf
                    obj.class_name = cls_name
                    detections.append(obj)

                    if debug_frame is not None:
                        x1, y1 = int(x_c - w / 2), int(y_c - h / 2)
                        x2, y2 = int(x_c + w / 2), int(y_c + h / 2)
                        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(debug_frame, f'{cls_name} {conf:.2f}',
                                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if self._sort_dir == 'left_to_right':
            detections.sort(key=lambda d: d.x)
        elif self._sort_dir == 'right_to_left':
            detections.sort(key=lambda d: -d.x)

        proximity_warning = self._check_proximity(detections)
        prox_msg      = Bool()
        prox_msg.data = proximity_warning
        self._proximity_pub.publish(prox_msg)

        if proximity_warning:
            self.get_logger().warn(
                f'근접 충돌 경고! 출력물 간 거리 < {self._proximity_px}px')

        obj_array             = ObjectArray()
        obj_array.header      = header
        obj_array.objects     = detections
        obj_array.total_count = len(detections)
        self._obj_pub.publish(obj_array)

        self.get_logger().info(
            f'탐지 완료: {len(detections)}개 출력물 | 근접경고: {proximity_warning}')

        if self._pub_debug and debug_frame is not None:
            cv2.putText(debug_frame,
                        f'Objects: {len(detections)} | Proximity: {proximity_warning}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            self._debug_pub.publish(encode_bgr(debug_frame))

        self._detection_enabled = False

    # ─────────────────────────────────────────────
    # 근접 충돌 체크
    # ─────────────────────────────────────────────
    def _check_proximity(self, detections: List[DetectedObject]) -> bool:
        """출력물 간 거리 < proximity_warning_px 인 쌍 검사.
        math.sqrt 대신 거리² 비교로 sqrt 연산을 제거한다.
        """
        threshold_sq = self._proximity_px ** 2
        n = len(detections)
        for i in range(n):
            for j in range(i + 1, n):
                dx = detections[i].x - detections[j].x
                dy = detections[i].y - detections[j].y
                if dx * dx + dy * dy < threshold_sq:
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
