# 🔧 QUVI 3순위 품질/유지보수 개선 계획서

**작성일**: 2026-06-20
**대상 파일**:
- `src/quvi_robot_control/quvi_robot_control/robot_control_node.py`
- `src/quvi_bringup/launch/vision_pipeline.launch.py`
- `src/quvi_yolo/quvi_yolo/yolo_node.py`
- `src/quvi_inspect/quvi_inspect/inspect_node.py`
- `src/quvi_inspect/quvi_inspect/stl_renderer.py`
- `src/quvi_hmi/quvi_hmi/hmi_node.py`

---

## 개요

코드 리뷰에서 식별된 3순위(품질/유지보수) 이슈 5건에 대한 상세 해결 계획입니다. 이 이슈들은 기능적 결함은 아니지만 코드 품질, 가독성, 설정 유연성, 확장성을 개선하기 위한 항목입니다.

| # | 이슈 | 영향도 | 예상 공수 |
|---|------|--------|----------|
| 1 | ACTPolicy import 중복 (`_load_act_policy`) | **낮음** — dead code | 5분 |
| 2 | `yolo_config`/`inspect_config` YAML 실제 로드 미구현 | **중간** — 설정 변경이 코드 수정 필요 | 1일 |
| 3 | `_load_params()`의 lambda 패턴 가독성 | **낮음** — 코드 스타일 | 10분 |
| 4 | `stl_renderer.py` 미사용 import 정리 | **낮음** — dead code | 5분 |
| 5 | HMI 텔레옵 상태 관리 개선 | **낮음** — race condition 가능성 | 0.5일 |

---

## 이슈 1: `_load_act_policy()` 내 ACTPolicy import 중복

### 현재 상태

`robot_control_node.py` L290-318에서 `ACTPolicy`가 두 번 import 됩니다:

```python
# L290-297: 첫 번째 import (try-except)
def _load_act_policy(self):
    """LeRobot ACTPolicy 로드."""
    try:
        import torch
        from lerobot.policies.act.modeling_act import ACTPolicy
    except ImportError as e:
        self.get_logger().error(f'LeRobot/torch 미설치: {e}')
        return

    # ... 경로 확인 ...

    # L305-306: 두 번째 import (다시)
    try:
        import torch                                       # ← 중복
        from lerobot.policies.act.modeling_act import ACTPolicy  # ← 중복
        if not resolved_path.exists():
            ...
```

첫 번째 try-except에서 import 성공 후 return하지 않았으므로, 두 번째 try에서 다시 import할 필요가 없습니다.

### 해결 방안

두 번째 import를 제거하고 첫 번째 import의 결과를 그대로 사용합니다:

```python
def _load_act_policy(self):
    """LeRobot ACTPolicy 로드."""
    try:
        import torch
        from lerobot.policies.act.modeling_act import ACTPolicy
    except ImportError as e:
        self.get_logger().error(f'LeRobot/torch 미설치: {e}')
        return

    resolved_path = Path(self._act_model_path)
    if not resolved_path.is_absolute():
        resolved_path = Path('/workspace') / resolved_path
    resolved_path = resolved_path.resolve()

    self.get_logger().info(f'ACT 모델 로드 중: {resolved_path}')
    try:
        # import torch, ACTPolicy — 이미 상단에서 성공했으므로 재import 불필요
        if not resolved_path.exists():
            raise FileNotFoundError(
                f'로컬 모델 디렉토리가 존재하지 않습니다: {resolved_path}')
        self._act_policy = ACTPolicy.from_pretrained(str(resolved_path))
        self._act_policy.eval()
        device = self._act_device
        self._act_policy = self._act_policy.to(device)
        self._act_device_obj = device
        self._act_ready = True
        self.get_logger().info(f'ACT 모델 로드 완료 (device={device})')
    except Exception as e:
        self.get_logger().error(f'ACT 모델 로드 실패: {e}')
        self._act_ready = False
```

### 변경 위치 요약

| 파일 | 위치 | 변경 내용 |
|------|------|----------|
| `robot_control_node.py` | L305-306 | `import torch` 및 `from lerobot...import ACTPolicy` 제거 |

---

## 이슈 2: YAML Config 파일 실제 로드 구현

### 현재 상태

`vision_pipeline.launch.py`에는 `yolo_config`와 `inspect_config` launch argument가 선언되어 있지만(L72-78), **실제 YAML 파일을 로드하는 로직이 없습니다**. 대신 모든 파라미터가 launch 파일 내에 하드코딩되어 있습니다:

```python
# vision_pipeline.launch.py L141-158
yolo_node = Node(
    package='quvi_yolo',
    executable='yolo_node',
    name='yolo_node',
    parameters=[{
        'camera_topic': LaunchConfiguration('handcam_topic'),
        'use_compressed': True,
        'model_path': LaunchConfiguration('yolo_model_path'),
        'confidence_threshold': 0.5,       # ← 하드코딩
        'iou_threshold': 0.45,             # ← 하드코딩
        'max_detections': 20,              # ← 하드코딩
        # ... 10개 이상의 하드코딩 파라미터
    }],
)
```

세 개의 YAML 파일이 이미 작성되어 있습니다:
- `src/quvi_yolo/config/yolo_params.yaml`
- `src/quvi_inspect/config/inspect_params.yaml`
- `src/quvi_hmi/config/hmi_params.yaml`

하지만 launch 파일에서는 전혀 참조되지 않습니다.

### 해결 방안

ROS 2의 표준 방식을 따라 YAML 파일을 로드합니다. `get_package_share_directory()`로 YAML 경로를 찾고, launch argument로 재정의 가능하게 합니다.

#### 2-A) YAML 파일 로드 함수 추가

```python
# vision_pipeline.launch.py 상단
from ament_index_python.packages import get_package_share_directory
import os

def _load_yaml_or_default(package_name: str, default_name: str, override: str = ''):
    """YAML 파라미터 파일 경로를 결정한다.

    Args:
        package_name: ROS 2 패키지 이름 ('quvi_yolo' 등).
        default_name: 기본 YAML 파일명 ('yolo_params.yaml' 등).
        override: launch argument로 전달된 사용자 지정 경로.

    Returns:
        파라미터 파일의 절대 경로. override가 있으면 그대로,
        없으면 {package}/config/{default_name} 경로.
    """
    if override:
        return override
    pkg_dir = get_package_share_directory(package_name)
    return os.path.join(pkg_dir, 'config', default_name)
```

#### 2-B) 노드에서 YAML 로드 적용

```python
yolo_node = Node(
    package='quvi_yolo',
    executable='yolo_node',
    name='yolo_node',
    parameters=[
        _load_yaml_or_default(
            'quvi_yolo', 'yolo_params.yaml',
            LaunchConfiguration('yolo_config')),
        # YAML에 없는 런타임 동적 값만 오버라이드
        {
            'camera_topic': LaunchConfiguration('handcam_topic'),
            'model_path': LaunchConfiguration('yolo_model_path'),
        },
    ],
    output='screen',
)

inspect_node = Node(
    package='quvi_inspect',
    executable='inspect_node',
    name='inspect_node',
    parameters=[
        _load_yaml_or_default(
            'quvi_inspect', 'inspect_params.yaml',
            LaunchConfiguration('inspect_config')),
        # YAML에 없는 런타임 동적 값만 오버라이드
        {
            'camera_topic': LaunchConfiguration('inspect_topic'),
            'reference_image_dir': LaunchConfiguration('reference_image_dir'),
            'inspection_log_dir': LaunchConfiguration('inspection_log_dir'),
        },
    ],
    output='screen',
)
```

#### 2-C) YAML 파일과 launch 파라미터의 우선순위

ROS 2 파라미터는 리스트의 **뒤쪽 항목이 앞쪽을 덮어씁니다**. 위 코드에서는:
1. YAML 파일에서 기본값 로드
2. launch argument로 전달된 값이 YAML 값을 오버라이드

이렇게 하면 YAML 파일을 수정하면 기본 동작이 바뀌고, launch argument로 필요 시 재정의할 수 있습니다.

#### 2-D) `_load_yaml_or_default` 헬퍼 함수 상세

`LaunchConfiguration`은 런타임에 평가되므로, `LaunchConfiguration('yolo_config')`가 빈 문자열(`''`)일 때만 기본 YAML을 사용하도록 합니다:

```python
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition

# 주의: LaunchConfiguration은 launch 파일의 generate_launch_description() 내에서만
# 유효하므로, 헬퍼 함수 내에서 직접 사용할 수 없습니다.
# 대신 아래와 같이 인라인으로 처리합니다.
```

실제 구현은 `LaunchConfiguration`의 특성상 `PythonExpression`을 사용해야 합니다:

```python
from launch_ros.parameter_descriptions import ParameterFile

# 방법 1: ParameterFile 사용 (권장 — ROS 2 표준)
yolo_default_config = os.path.join(
    get_package_share_directory('quvi_yolo'), 'config', 'yolo_params.yaml')

yolo_node = Node(
    package='quvi_yolo',
    executable='yolo_node',
    name='yolo_node',
    parameters=[
        # 사용자 지정 YAML이 있으면 사용, 없으면 기본값
        ParameterFile(LaunchConfiguration('yolo_config'),
                      allow_substs=True),
        # 런타임 오버라이드
        {
            'camera_topic': LaunchConfiguration('handcam_topic'),
            'model_path': LaunchConfiguration('yolo_model_path'),
        },
    ],
    output='screen',
)
```

> **주의**: `ParameterFile`은 지정된 파일이 존재하지 않으면 오류가 발생할 수 있습니다. `yolo_config`의 기본값이 빈 문자열이면 이 방식은 적합하지 않습니다. 대신 아래 접근법을 사용합니다.

#### 2-E) 권장 구현: PythonExpression 으로 조건부 경로 선택

```python
from launch.substitutions import PythonExpression, LaunchConfiguration, IfCondition, UnlessCondition

yolo_default_yaml = os.path.join(
    get_package_share_directory('quvi_yolo'), 'config', 'yolo_params.yaml')
inspect_default_yaml = os.path.join(
    get_package_share_directory('quvi_inspect'), 'config', 'inspect_params.yaml')

# YOLO 노드 — 2개 버전 (사용자 YAML 사용 vs 기본 YAML 사용)
yolo_node_custom = Node(
    package='quvi_yolo',
    executable='yolo_node',
    name='yolo_node',
    parameters=[
        LaunchConfiguration('yolo_config'),  # 사용자 지정 YAML
        {'camera_topic': LaunchConfiguration('handcam_topic'),
         'model_path': LaunchConfiguration('yolo_model_path')},
    ],
    output='screen',
    condition=IfCondition(
        PythonExpression(['"', LaunchConfiguration('yolo_config'), '" != ""'])),
)

yolo_node_default = Node(
    package='quvi_yolo',
    executable='yolo_node',
    name='yolo_node',
    parameters=[
        yolo_default_yaml,  # 기본 YAML
        {'camera_topic': LaunchConfiguration('handcam_topic'),
         'model_path': LaunchConfiguration('yolo_model_path')},
    ],
    output='screen',
    condition=UnlessCondition(
        PythonExpression(['"', LaunchConfiguration('yolo_config'), '" != ""'])),
)

# LaunchDescription에 두 노드 모두 추가 (IfCondition/UnlessCondition이 하나만 활성화)
```

이 방식은 복잡하므로, **더 간단한 접근법**을 권장합니다:

#### 2-F) 최종 권장: launch argument 기본값을 YAML 경로로 설정

```python
yolo_config_arg = DeclareLaunchArgument(
    'yolo_config',
    default_value=os.path.join(
        get_package_share_directory('quvi_yolo'), 'config', 'yolo_params.yaml'),
    description='YOLO 파라미터 YAML 경로')

inspect_config_arg = DeclareLaunchArgument(
    'inspect_config',
    default_value=os.path.join(
        get_package_share_directory('quvi_inspect'), 'config', 'inspect_params.yaml'),
    description='검사 파라미터 YAML 경로')

# 노드 정의 — 항상 LaunchConfiguration 사용
yolo_node = Node(
    package='quvi_yolo',
    executable='yolo_node',
    name='yolo_node',
    parameters=[
        LaunchConfiguration('yolo_config'),  # ← 기본값이 YAML 경로
        {
            'camera_topic': LaunchConfiguration('handcam_topic'),
            'model_path': LaunchConfiguration('yolo_model_path'),
        },
    ],
    output='screen',
)
```

**장점**:
- `ros2 launch` 시 기본적으로 YAML 파일이 로드됨
- `yolo_config:=/custom/path.yaml` 로 재정의 가능
- 빈 문자열 전달 시 ROS 2가 무시하므로 기존 하드코딩 fallback 동작 유지
- IfCondition/UnlessCondition 없이 코드가 간결함
- `full_system.launch.py`는 `vision_pipeline`에 `yolo_config`/`inspect_config`를 전달하지 않으므로 기본 YAML이 사용됨

### 변경 위치 요약

| 파일 | 위치 | 변경 내용 |
|------|------|----------|
| `vision_pipeline.launch.py` | L72-78 | `yolo_config`/`inspect_config` 기본값을 YAML 경로로 변경 |
| `vision_pipeline.launch.py` | L141-193 | YOLO/Inspect 노드의 하드코딩 파라미터를 LaunchConfiguration으로 대체 |
| `vision_pipeline.launch.py` | L23 | `get_package_share_directory` import 추가 |

### 하드코딩 제거 대상 파라미터 목록

**YOLO 노드**에서 YAML로 이관 (이미 yolo_params.yaml에 존재):
- `confidence_threshold: 0.5` → YAML
- `iou_threshold: 0.45` → YAML
- `max_detections: 20` → YAML
- `min_bbox_area: 500` → YAML
- `proximity_warning_px: 60` → YAML
- `sort_direction: 'left_to_right'` → YAML
- `input_width: 640` → YAML
- `input_height: 480` → YAML
- `publish_debug_image: True` → YAML

**Inspect 노드**에서 YAML로 이관 (이미 inspect_params.yaml에 존재):
- `ssim_threshold: 0.85` → YAML
- `area_ratio_min: 0.90` → YAML
- `area_ratio_max: 1.10` → YAML
- `pixel_diff_threshold: 0.10` → YAML
- `solidity_min: 0.85` → YAML
- `solidity_max: 1.00` → YAML
- `feature_area_ratio_min: 0.90` → YAML
- `feature_area_ratio_max: 1.10` → YAML
- `hole_count_max: 2` → YAML
- `hole_area_ratio_max: 0.05` → YAML
- `texture_variance_max: 500.0` → YAML
- `min_hole_area_px: 50` → YAML
- `turntable_angles: [0, 90, 180, 270]` → YAML
- `save_inspection_images: True` → YAML
- `publish_debug_image: True` → YAML
- `alignment_enabled: True` → YAML
- `align_max_dimension: 200` → YAML
- `align_padding_pct: 0.15` → YAML
- `align_min_bbox_area: 500` → YAML

**launch 파일에 남겨야 하는 동적 파라미터** (기기/환경 의존):
- `camera_topic`, `handcam_topic`, `inspect_topic` — 카메라 설정 의존
- `use_compressed` — 카메라 설정 의존
- `model_path` — launch argument로 전달
- `reference_image_dir`, `inspection_log_dir` — 파일시스템 경로

---

## 이슈 3: `_load_params()` lambda 패턴 가독성 개선

### 현재 상태

`yolo_node.py` L67과 `inspect_node.py` L95에서 lambda를 사용한 파라미터 로드 패턴:

```python
# yolo_node.py L67
def _load_params(self):
    g = lambda name, default: declare_and_get(self, name, default)

    self._camera_topic    = g('camera_topic',           '/camera1/image_raw/compressed')
    self._use_compressed  = g('use_compressed',         True)
    self._model_path      = g('model_path',             '')
    # ... 14줄
```

`g`라는 한 글자 이름과 lambda 사용은 숙련된 개발자에게는 간결하지만, 새로 합류하는 팀원에게는 불필요한 혼란을 줄 수 있습니다. 또한 파라미터가 추가/제거될 때 모든 행을 수정해야 합니다.

### 해결 방안

#### 3-A) 권장: 딕셔너리 기반 일괄 로드

```python
def _load_params(self):
    """모든 파라미터를 선언+로드 한다."""
    # (name, default) 쌍으로 정의
    param_defaults = {
        'camera_topic':           '/camera1/image_raw/compressed',
        'use_compressed':         True,
        'model_path':             '',
        'confidence_threshold':   0.5,
        'iou_threshold':          0.45,
        'max_detections':         20,
        'target_classes':         ['print_object'],
        'min_bbox_area':          500,
        'proximity_warning_px':   60,
        'sort_direction':         'left_to_right',
        'input_width':            640,
        'input_height':           480,
        'publish_debug_image':    True,
        'debug_image_topic':      '/yolo/debug_image',
    }

    for name, default in param_defaults.items():
        self.declare_parameter(name, default)
        setattr(self, '_' + name, self.get_parameter(name).value)

    # 별도 처리가 필요한 파라미터는 여기서
    # self._target_classes = self.get_parameter('target_classes').value  # 이미 위에서 처리됨
```

**장점**:
- 파라미터 추가/제거가 한 곳에서 이뤄짐
- `setattr`를 사용하므로 속성 접근 방식은 동일하게 유지 (`self._camera_topic`)
- `declare_and_get` 헬퍼를 제거할 수 있음 (`utils.py`에서 함수 하나 줄임)
- YAML config 로드(이슈 2)와 조합 시, `param_defaults`를 YAML에서 로드한 값으로 업데이트하는 방식으로 확장 가능

#### 3-B) ROS 2 네이티브 방식 활용 (더 나은 접근)

```python
def _load_params(self):
    """모든 파라미터를 선언 후 일괄 로드한다."""
    defaults = [
        ('camera_topic',           '/camera1/image_raw/compressed'),
        ('use_compressed',         True),
        ('model_path',             ''),
        ('confidence_threshold',   0.5),
        ('iou_threshold',          0.45),
        ('max_detections',         20),
        ('target_classes',         ['print_object']),
        ('min_bbox_area',          500),
        ('proximity_warning_px',   60),
        ('sort_direction',         'left_to_right'),
        ('input_width',            640),
        ('input_height',           480),
        ('publish_debug_image',    True),
        ('debug_image_topic',      '/yolo/debug_image'),
    ]

    for name, default in defaults:
        self.declare_parameter(name, default)

    # 한 번에 모든 파라미터 로드
    params = {name: self.get_parameter(name).value for name, _ in defaults}

    # 속성 할당
    for name, value in params.items():
        setattr(self, '_' + name, value)
```

### 변경 위치 요약

| 파일 | 위치 | 변경 내용 |
|------|------|----------|
| `yolo_node.py` | L66-82 `_load_params()` | lambda 제거, 딕셔너리 기반으로 변경 |
| `inspect_node.py` | L93-128 `_load_params()` | lambda 제거, 딕셔너리 기반으로 변경 |
| `utils.py` | L50-53 `declare_and_get()` | 사용처가 없으면 제거 (선택적) |

> **참고**: `declare_and_get`은 `yolo_node.py`와 `inspect_node.py`에서만 사용됩니다. 두 곳 모두 변경하면 `utils.py`에서 제거할 수 있습니다. 하지만 `robot_control_node.py`는 별도의 `_declare_params` + `_load_params` 패턴을 사용하므로 영향 없습니다.

---

## 이슈 4: `stl_renderer.py` 미사용 import 정리

### 현재 상태

```python
# stl_renderer.py L107-112
def main(args=None):
    """ROS 2 노드로 실행 또는 스탠드얼론."""
    try:
        import rclpy
        from rclpy.node import Node      # ← L111: Node 클래스 import

        rclpy.init(args=args)
        node = Node('stl_renderer')       # ← L113: Node 직접 생성

        node.declare_parameter('stl_path', '')
        # ... (Node 클래스의 메서드 사용)
```

`from rclpy.node import Node`로 import한 `Node`는 L113에서 `Node('stl_renderer')`로 직접 사용됩니다. 이 import는 **실제로 사용 중**이므로 제거 대상이 아닙니다.

> **정정**: 이전 리뷰에서 "미사용 import"로 지적했으나, 실제로는 `Node('stl_renderer')` 생성자 호출에 사용되고 있습니다. 이 이슈는 **무효(invalid)** 처리합니다.

### 해결 방안

이슈 취소. `stl_renderer.py`는 정상입니다.

### 변경 위치 요약

| 파일 | 변경 내용 |
|------|----------|
| 없음 | - |

---

## 이슈 5: HMI 텔레옵 상태 관리 개선

### 현재 상태

`hmi_node.py`의 `api_teleop()` (L383-401)에서 HMI가 직접 `_system_status`를 수정합니다:

```python
@app.route('/api/teleop/<action>', methods=['POST'])
def api_teleop(action):
    if action == 'on':
        msg = Bool()
        msg.data = True
        hmi_node._teleop_pub.publish(msg)
        with hmi_node._lock:
            hmi_node._system_status['teleop_active'] = True
            hmi_node._system_status['current_state'] = 'TELEOPING'  # ← 직접 덮어씀
        return jsonify({'ok': True, 'teleop': 'on'})
    elif action == 'off':
        msg = Bool()
        msg.data = False
        hmi_node._teleop_pub.publish(msg)
        with hmi_node._lock:
            hmi_node._system_status['teleop_active'] = False
            hmi_node._system_status['current_state'] = 'IDLE'       # ← 직접 덮어씀
        return jsonify({'ok': True, 'teleop': 'off'})
```

또한 `_status_cb` (L138-152)에서도 보정 로직이 있습니다:

```python
def _status_cb(self, msg: SystemStatus):
    with self._lock:
        if self._system_status.get('teleop_active', False):
            self._system_status['current_state'] = 'TELEOPING'  # ← 보정
        else:
            self._system_status['current_state'] = msg.current_state
        # ...
```

**문제점**:
1. 상태 변경이 HMI API → 직접 lock → 상태 변경 순서로 이뤄지고, 동시에 `_status_cb`가 오케스트레이터로부터 상태를 수신하면서 race condition이 발생할 수 있음
2. 텔레옵이 `off`로 바뀌었을 때, 오케스트레이터의 실제 상태가 `ERROR`이면 HMI는 `IDLE`로 표시 → 사용자가 시스템 상태를 오인할 수 있음
3. `_status_cb`의 보정 로직이 HMI의 직접 수정에 의존하는 순환 구조

### 해결 방안

#### 5-A) `teleop_active`를 별도 플래그로 분리하고, 상태 덮어쓰기 제거

```python
@app.route('/api/teleop/<action>', methods=['POST'])
def api_teleop(action):
    if action == 'on':
        msg = Bool()
        msg.data = True
        hmi_node._teleop_pub.publish(msg)
        with hmi_node._lock:
            hmi_node._system_status['teleop_active'] = True
            # current_state 는 _status_cb 가 오케스트레이터로부터 수신한 값을
            # 그대로 사용하므로 여기서 덮어쓰지 않음
        return jsonify({'ok': True, 'teleop': 'on'})
    elif action == 'off':
        msg = Bool()
        msg.data = False
        hmi_node._teleop_pub.publish(msg)
        with hmi_node._lock:
            hmi_node._system_status['teleop_active'] = False
            # current_state 도 덮어쓰지 않음
        return jsonify({'ok': True, 'teleop': 'off'})
```

#### 5-B) `_status_cb` 보정 로직 개선

오케스트레이터가 보내는 `current_state`를 신뢰하고, `teleop_active`는 HMI의 로컬 플래그로만 관리:

```python
def _status_cb(self, msg: SystemStatus):
    with self._lock:
        # teleop_active 가 True 이면, 오케스트레이터 상태와 무관하게
        # HMI 표시용으로 TELEOPING 상태를 보여줄 수 있음
        if self._system_status.get('teleop_active', False):
            # 단, ERROR 상태면 텔레옵 중이라도 ERROR를 우선 표시
            if msg.current_state == 'ERROR':
                self._system_status['current_state'] = 'ERROR'
            else:
                self._system_status['current_state'] = 'TELEOPING'
        else:
            self._system_status['current_state'] = msg.current_state

        self._system_status['total_objects'] = msg.total_objects
        # ... 나머지 필드
```

#### 5-C) `robot_control_node` 텔레옵 상태를 오케스트레이터에 전파

현재 오케스트레이터는 텔레옵 상태를 모릅니다. `/robot/status` 토픽에 텔레옵 활성화 상태를 포함시키면 HMI가 추측 없이 실제 상태를 반영할 수 있습니다. (선택적 — 3순위 범위 초과이므로 참고로만)

```python
# robot_control_node.py _publish_status 의 상위 개념으로
# 주기적으로 RobotState + teleop_active 를 포함한 상태 메시지 발행
```

### 변경 위치 요약

| 파일 | 위치 | 변경 내용 |
|------|------|----------|
| `hmi_node.py` | L371-373 (`api_teleop` on) | `current_state` 덮어쓰기 제거 |
| `hmi_node.py` | L380-382 (`api_teleop` off) | `current_state` 덮어쓰기 제거 |
| `hmi_node.py` | L139-143 (`_status_cb`) | ERROR 상태 우선 표시 로직 추가 |

---

## 이슈 6 (추가): `docker/` 디렉토리 산출물 정리

### 현재 상태

`.gitignore`에 Docker 관련 항목이 이미 정의되어 있지만(L50-58), 일부 파일이 이미 git에 추적되고 있습니다:

```bash
docker/build/       # 빌드 캐시 산출물
docker/install/     # colcon install 산출물
docker/yolov8n.pt   # 대용량 모델 파일 (6.5MB)
```

### 해결 방안

#### 6-A) git 추적에서 제거

```bash
# 이미 추적 중인 파일을 git에서 제거 (파일은 유지)
git rm --cached docker/build/.built_by docker/build/COLCON_IGNORE
git rm --cached -r docker/install/
git rm --cached docker/yolov8n.pt
```

#### 6-B) `.gitignore` 보강

```gitignore
# Docker build artifacts/logs
docker/build/
docker/install/
docker/log/
docker/*.pt          # 모델 파일은 data/models/ 에서 관리
docker/CACHED
docker/ERROR
docker/\[internal\]
docker/reading
docker/resolve
docker/transferring
```

> **참고**: `yolov8n.pt`는 기본 COCO 모델로, `YOLO('yolov8n.pt')` 호출 시 ultralytics가 자동 다운로드하므로 저장소에 포함할 필요가 없습니다. 커스텀 학습 모델(`best.pt`)은 `data/models/`에 별도 관리합니다.

### 변경 위치 요약

| 파일 | 위치 | 변경 내용 |
|------|------|----------|
| `.gitignore` | L50-58 | `docker/build/`, `docker/install/`, `docker/*.pt` 추가 |
| git index | - | `git rm --cached` 로 기존 추적 파일 제거 |

---

## 테스트 계획

### 이슈 1 (중복 import) 테스트

```
1. ACT 모델 없이 로봇 제어 노드 실행 (use_act:=false)
2. 확인사항:
   - "LeRobot/torch 미설치" 로그가 한 번만 출력됨 (중복 없음)
   - 노드가 정상 기동됨 (ACT 미로드 상태로)

3. ACT 모델 있는 환경에서 실행
   - ACT 모델 로드 완료 로그 정상 출력
   - ACT 파지 기능 정상 동작
```

### 이슈 2 (YAML 로드) 테스트

```
[기본 YAML 사용]
1. ros2 launch quvi_bringup vision_pipeline.launch.py
2. 확인사항:
   - YOLO 노드가 yolo_params.yaml 값을 사용하는지 로그로 확인
   - Inspect 노드가 inspect_params.yaml 값을 사용하는지 확인
   - 기존과 동일한 동작 (YAML 값 = 기존 하드코딩 값)

[사용자 YAML 사용]
3. cp yolo_params.yaml ~/custom_yolo.yaml && 값 일부 수정
4. ros2 launch ... yolo_config:=~/custom_yolo.yaml
5. 확인사항:
   - 수정된 값이 반영됨 (예: confidence_threshold=0.7)

[YAML 파일 없음]
6. yolo_config:=/nonexistent.yaml
7. 확인사항:
   - ROS 2가 파일 없음 에러 출력 후 기본 파라미터로 폴백
```

### 이슈 3 (lambda 제거) 테스트

```
[회귀 테스트]
1. 전체 시스템 런치
2. 모든 파라미터가 정상 로드되는지 확인
3. get_parameter() 호출이 동일한 값을 반환하는지 확인
4. pytest tests/ 실행 → 기존 테스트 통과 확인
```

### 이슈 5 (HMI 텔레옵) 테스트

```
[텔레옵 ON/OFF]
1. HMI에서 텔레옵 ON 클릭
2. 확인사항:
   - teleop_active=True, current_state='IDLE' (오케스트레이터에서 받은 값)
   - HMI 화면에는 'TELEOPING'으로 표시 (프론트엔드 로직)
3. 텔레옵 OFF
4. 확인사항:
   - teleop_active=False, current_state는 오케스트레이터 값 유지

[텔레옵 중 ESTOP]
5. 텔레옵 ON 상태에서 ESTOP
6. 확인사항:
   - current_state='ERROR' (오케스트레이터 상태)
   - ERROR가 우선 표시됨 (텔레옵 중이어도)
```

### 이슈 6 (Docker 정리) 테스트

```
1. git rm --cached 실행 후 git status 확인
2. docker/build/, docker/install/, docker/yolov8n.pt 가 untracked 상태인지 확인
3. docker compose build → docker compose up 정상 동작 확인
```

---

## 구현 순서

1. **1단계** (5분): 이슈 1 — ACTPolicy 중복 import 제거
2. **2단계** (10분): 이슈 3 — `_load_params()` lambda → 딕셔너리 방식으로 변경
3. **3단계** (10분): 이슈 6 — Docker 산출물 정리 (.gitignore + git rm --cached)
4. **4단계** (1시간): 이슈 2 — YAML config 로드 구현
5. **5단계** (30분): 이슈 5 — HMI 텔레옵 상태 관리 개선
6. **6단계** (20분): 통합 테스트

---

## 관련 파일 요약

| 파일 | 변경 규모 | 위험도 |
|------|----------|--------|
| `robot_control_node.py` | **-2줄** (중복 import 제거) | 없음 |
| `yolo_node.py` | **~20줄** (lambda → 딕셔너리) | 낮음 |
| `inspect_node.py` | **~20줄** (lambda → 딕셔너리) | 낮음 |
| `vision_pipeline.launch.py` | **~30줄** (YAML 로드 + 하드코딩 제거) | 중간 — 런치 파라미터 변경 |
| `hmi_node.py` | **~10줄** (teleop 상태 관리) | 낮음 |
| `utils.py` | **-15줄** (`declare_and_get` 제거, 선택적) | 낮음 |
| `.gitignore` | **+3줄** (Docker 경로 추가) | 없음 |