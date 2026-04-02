"""
QUVI STL Renderer
─────────────────
STL 파일을 4방향(0°/90°/180°/270°)으로 2D 렌더링하여
검사용 기준 이미지를 생성하는 유틸리티.

사용법:
  ros2 run quvi_inspect stl_renderer --ros-args \
    -p stl_path:=/workspace/data/reference_stl/box_40x40x20.stl \
    -p output_dir:=/workspace/data/reference_images \
    -p image_size:=640
"""

import os
import sys

import cv2
import numpy as np


def render_stl_to_images(stl_path: str, output_dir: str,
                         angles: list = None, image_size: int = 640):
    """STL 파일을 4방향으로 렌더링하여 기준 이미지 생성.

    trimesh + pyrender를 사용한 오프스크린 렌더링.
    GPU 없는 환경에서도 동작 (osmesa/egl 폴백).
    """
    if angles is None:
        angles = [0, 90, 180, 270]

    os.makedirs(output_dir, exist_ok=True)

    try:
        import trimesh
    except ImportError:
        print('[ERROR] trimesh가 설치되지 않음: pip install trimesh')
        return False

    # STL 로드
    mesh = trimesh.load(stl_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    print(f'STL 로드 완료: {stl_path}')
    print(f'  정점 수: {len(mesh.vertices)}')
    print(f'  면 수: {len(mesh.faces)}')
    print(f'  바운딩 박스: {mesh.bounds}')

    # 메시 중심을 원점으로 이동
    mesh.vertices -= mesh.centroid

    for angle in angles:
        # 실루엣 기반 렌더링 (pyrender 없이도 동작)
        rendered = _render_silhouette(mesh, angle, image_size)

        # 저장
        out_path = os.path.join(output_dir, f'ref_{angle}.png')
        cv2.imwrite(out_path, rendered)
        print(f'  기준 이미지 저장: {out_path}')

    print(f'렌더링 완료: {len(angles)}개 이미지 → {output_dir}')
    return True


def _render_silhouette(mesh, angle_deg: float, image_size: int) -> np.ndarray:
    """메시를 특정 각도에서 바라본 2D 실루엣(바이너리 이미지)으로 렌더링.

    pyrender 없이 trimesh의 프로젝션만으로 구현.
    """
    import trimesh

    # Y축 기준 회전
    angle_rad = np.radians(angle_deg)
    rotation = trimesh.transformations.rotation_matrix(angle_rad, [0, 1, 0])
    rotated = mesh.copy()
    rotated.apply_transform(rotation)

    # 정면(XY 평면)으로 프로젝션
    vertices_2d = rotated.vertices[:, :2]  # X, Y만 사용

    # 이미지 좌표로 변환
    min_xy = vertices_2d.min(axis=0)
    max_xy = vertices_2d.max(axis=0)
    span = max_xy - min_xy
    max_span = max(span) * 1.2  # 마진 20%

    # 정규화
    center = (min_xy + max_xy) / 2
    normalized = (vertices_2d - center) / max_span + 0.5  # 0~1

    # 이미지 좌표
    img_coords = (normalized * image_size).astype(np.int32)
    img_coords[:, 1] = image_size - img_coords[:, 1]  # Y축 반전

    # 실루엣 이미지 생성
    canvas = np.zeros((image_size, image_size), dtype=np.uint8)

    # 삼각형 면 채우기
    for face in rotated.faces:
        pts = img_coords[face]
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(canvas, [pts], 255)

    return canvas


def main(args=None):
    """ROS 2 노드로 실행 또는 스탠드얼론."""
    try:
        import rclpy
        from rclpy.node import Node

        rclpy.init(args=args)
        node = Node('stl_renderer')

        node.declare_parameter('stl_path', '')
        node.declare_parameter('output_dir', '/workspace/data/reference_images')
        node.declare_parameter('image_size', 640)

        stl_path = node.get_parameter('stl_path').value
        output_dir = node.get_parameter('output_dir').value
        image_size = node.get_parameter('image_size').value

        if not stl_path:
            node.get_logger().error('stl_path 파라미터가 지정되지 않음')
            rclpy.shutdown()
            return

        success = render_stl_to_images(stl_path, output_dir, image_size=image_size)

        if success:
            node.get_logger().info('STL 렌더링 완료')
        else:
            node.get_logger().error('STL 렌더링 실패')

        node.destroy_node()
        rclpy.shutdown()

    except ImportError:
        # ROS 2 없이 스탠드얼론 실행
        if len(sys.argv) < 3:
            print('Usage: python3 stl_renderer.py <stl_path> <output_dir> [image_size]')
            sys.exit(1)

        stl_path = sys.argv[1]
        output_dir = sys.argv[2]
        image_size = int(sys.argv[3]) if len(sys.argv) > 3 else 640

        render_stl_to_images(stl_path, output_dir, image_size=image_size)


if __name__ == '__main__':
    main()
