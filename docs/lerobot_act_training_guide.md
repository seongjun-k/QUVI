# LeRobot ACT 모방학습 작업 가이드북 (QUVI 프로젝트)

이 가이드는 로봇팔 파지 제어를 위한 **LeRobot ACT(Action Chunking with Transformers) 모방학습 데이터 녹화, 저장, 시각화 분석, 학습 및 최종 적용**까지의 전 과정을 상세히 설명합니다.

---

## 1. 환경 준비 및 터미널 기동

이 프로젝트는 Docker 컨테이너 및 호스트 환경 모두를 지원합니다. 아래 절차에 따라 가상환경 및 하드웨어 포트를 설정합니다.

### 1-1. 터미널 열기 및 작업 디렉토리 이동
새 터미널을 열고 프로젝트 폴더로 이동합니다.
```bash
cd ~/QUVI
```

### 1-2. 다이나믹셀 및 카메라 하드웨어 포트 권한 부여
다이나믹셀 U2D2 및 카메라가 정상 인식되도록 권한을 설정합니다.
```bash
sudo chmod a+rw /dev/ttyUSB*
sudo chmod a+rw /dev/video*
```

### 1-3. 가상환경 활성화 (필요한 경우)
Conda 환경을 사용하는 경우 해당 환경을 활성화합니다.
```bash
conda activate lerobot
```

---

## 2. 데이터 녹화 및 수집 (Record)

리더-팔로워 원격 조종을 통해 파지 동작을 직접 수행하며 데이터를 녹화합니다. 성공적인 학습을 위해 최소 **30~50회 이상의 성공 에피소드**가 필요합니다.

### 2-1. 녹화 스크립트 실행
아래 명령어를 입력하여 카메라 피드와 관절 각도의 기록을 시작합니다.
```bash
python3 lerobot/src/lerobot/record.py \
  --robot.path lerobot/configs/robot/omx.yaml \
  --fps 30 \
  --repo-id <HuggingFace유저ID>/quvi_act_grasp \
  --warmup-time-s 2 \
  --episode-time-s 15 \
  --num-episodes 50
```

### 주요 옵션 설명
* `--robot.path`: 로봇 하드웨어 및 포트 정의 파일 (`omx.yaml` 사용)
* `--fps 30`: 제어 주기 및 이미지 캡처 주기 (30Hz 고정)
* `--repo-id`: 데이터가 저장될 Hugging Face 리포지토리명 (로컬에도 동시에 캐싱 저장됨)
* `--episode-time-s`: 1개 에피소드(파지 ➔ 검사장 배치 완료까지) 녹화 시간 (초 단위)
* `--num-episodes`: 총 수집할 에피소드 횟수

> [!TIP]
> **고품질 데이터 수집 팁**
> 1. **시작 자세 일관성**: 에피소드 시작 시 로봇팔은 항상 거의 동일한 초기 자세(Home)에서 출발해야 합니다.
> 2. **부드러운 조작**: 급작스러운 모터 흔들림은 학습에 악영향을 줍니다. 최대한 부드럽게 조작해 주세요.
> 3. **다양성 확보**: 타겟 객체의 위치, 각도, 작업 영역 조명을 미세하게 변경해가며 수집해야 강인한(Robust) 모델이 학습됩니다.

---

## 3. 데이터 분석 및 시각화 (Visualization & Analysis)

수집된 데이터셋이 제대로 저장되었는지 확인하고 분석합니다.

### 3-1. 로컬 캐시 확인
수집된 이미지 및 모터 상태 데이터는 로컬 사용자 폴더 아래에 캐싱됩니다.
* 경로: `~/.cache/huggingface/lerobot/datasets/<HuggingFace유저ID>/quvi_act_grasp`

### 3-2. 웹 시각화 (추천)
Hugging Face에 데이터셋을 푸시한 경우, 브라우저에서 인터랙티브하게 분석할 수 있습니다.
1. 웹 브라우저를 열고 `https://huggingface.co/datasets/<HuggingFace유저ID>/quvi_act_grasp` 페이지로 이동합니다.
2. 내장된 **LeRobot Dataset Viewer**를 통해 카메라 1(handcam) 비디오 프레임과 각 모터의 궤적(Trajectory Plot)을 한눈에 검증할 수 있습니다.

### 3-3. 로컬 재생 테스트
실제 수집된 데이터셋의 동작을 로봇팔로 복제(Replay) 구동하여 정상 녹화 여부를 시각적으로 확인할 수 있습니다.
```bash
python3 lerobot/src/lerobot/replay.py \
  --robot.path lerobot/configs/robot/omx.yaml \
  --repo-id <HuggingFace유저ID>/quvi_act_grasp \
  --episode 0
```

---

## 4. ACT 모델 학습 (Train)

수집된 비주오모터 데이터셋을 기반으로 트랜스포머 파지 인공지능 모델을 학습합니다.

```bash
python3 lerobot/src/lerobot/scripts/train.py \
  dataset_repo_id=<HuggingFace유저ID>/quvi_act_grasp \
  policy.type=act \
  output_dir=outputs/train/quvi_act_grasp \
  device=cuda \
  env.type=real \
  wandb.enable=true
```

### 주요 옵션 설명
* `dataset_repo_id`: 학습에 활용할 Hugging Face 데이터셋 ID
* `policy.type=act`: 학습할 알고리즘 방식 (ACT 정책 선택)
* `output_dir`: 학습 가중치와 로그가 저장될 출력 디렉토리
* `device=cuda`: GPU 연산 코어 활성화 (CPU 대비 최소 10~20배 빠름)
* `env.type=real`: 실제 로봇 환경 설정
* `wandb.enable=true`: 학습 과정을 실시간 웹 모니터링하기 위한 Weights & Biases 대시보드 연동

> [!NOTE]
> **학습 완료 체크포인트**:
> 학습이 끝나면 지정한 `outputs/train/quvi_act_grasp/checkpoints/last/pretrained_model` 폴더 안에 PyTorch 모델 가중치(`model.safetensors`) 및 메타 설정 파일들이 자동 생성됩니다.

---

## 5. 학습 완료 모델 실기 적용 (Deployment)

학습한 가중치 파일을 QUVI 로봇 제어 시스템에 연결하여 실제 동작을 수행합니다.

### 5-1. 모델 파일 연결
수정 완료된 로봇 제어 노드 파라미터에 학습 완료된 ACT 가중치 디렉토리 경로를 지정합니다.
* **대상 파일**: [robot_control_node.py](file:///home/ksj/QUVI/src/quvi_robot_control/quvi_robot_control/robot_control_node.py)
* **설정 매핑**: `act_model_path` 파라미터 기본값을 새로 학습된 경로로 변경합니다.
  ```python
  self.declare_parameter('act_model_path', 'outputs/train/quvi_act_grasp/checkpoints/last/pretrained_model')
  ```

### 5-2. 자율 이송 시스템 기동
전체 런치 스크립트를 실행하고 HMI Dashboard에서 **START**를 누르면, 로봇이 베드로 이동하여 새로 학습된 ACT 모델 기반으로 단일 출력물을 안전하게 집어 검사장으로 이송합니다.
```bash
ros2 launch quvi_bringup full_system.launch.py
```

---

## 6. 안전 경고 (Safety Warning)

* **12번 모터(shoulder_lift) 안전 가드**:
  * 캘리브레이션 맵 파일(`quvi_follower.json`)에 `"range_min": 1024` 제한이 인가되어, 하드웨어 캘리브레이션 단계부터 모터 12번의 90도 아래 하강을 물리적으로 원천 차단합니다.
  * 로봇 제어 노드([robot_control_node.py](file:///home/ksj/QUVI/src/quvi_robot_control/quvi_robot_control/robot_control_node.py))에서도 `_clip_shoulder_lift` 안전 가드가 활성화되어 실시간 조종(Teleop) 및 ACT 추론 중에도 90도 미만으로 하강할 위험이 없습니다.
