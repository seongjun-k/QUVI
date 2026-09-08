# VLA 추론 사이드카

quvi-dev(lerobot 0.3.4, numpy<2)에서는 lerobot 0.6.1로 학습·export된 SmolVLA
2캠 체크포인트를 로드할 수 없다. 이 디렉토리는 격리 venv(lerobot 0.6.1)에서
상주 실행되는 별도 추론 프로세스이며, Unix domain socket으로 관측→액션 청크를
주고받는다. 로봇 클라이언트 재배선은 별도 작업이며 여기 포함되지 않는다.

## 구성 (1회)

```bash
bash scripts/vla_sidecar/setup_venv.sh
# 기본 경로: /home/ksj/QUVI/data/vla_sidecar/venv (data/는 .gitignore, 대용량 OK)
```

torch(cu128) 다운로드가 수 GB라 오래 걸린다. 완료 후 출력되는 버전이
lerobot 0.6.1 / numpy 2.2.6 / transformers 5.5.4 / torch 2.11.0+cu128인지
확인한다 (lerobot_server 컨테이너와 동일 핀이어야 함).

## 서버 실행

```bash
/home/ksj/QUVI/data/vla_sidecar/venv/bin/python \
  scripts/vla_sidecar/server.py \
  --model-path /home/ksj/QUVI/data/models/smolvla_120ep/checkpoints/006000/pretrained_model \
  --socket /dev/shm/quvi_vla.sock
```

## 오프라인 수치 검증 (하드웨어 없음)

```bash
/home/ksj/QUVI/data/vla_sidecar/venv/bin/python \
  scripts/vla_sidecar/test_equivalence.py \
  --model-path /home/ksj/QUVI/data/models/smolvla_120ep/checkpoints/006000/pretrained_model
```

확인 항목:
1. 모델 로드 성공 (서버 서브프로세스 기동)
2. 결정성 - 동일 입력 + 동일 seed 2회 호출 결과 일치
3. 소켓(IPC) 경로 vs 인프로세스 직접호출 경로 수치 일치
4. 왕복 latency 평균/최대(ms) - 30Hz 예산(33ms) 대비 출력

## IPC 프로토콜

`protocol.py` 참조. 4바이트 빅엔디언 길이 프리픽스 + payload(pickle). ndarray 는
dtype+shape+bytes 로 변환해 실어 numpy 2.x↔1.26 pickle 불일치를 피한다. 소켓:
`/dev/shm/quvi_vla.sock` (quvi-dev와 lerobot_server 컨테이너가 host network +
`/dev/shm` 공유).

요청: `{camera1: <HWC uint8 ndarray>, camera3: <HWC uint8 ndarray>, state: [6 floats], task: str, seed?: int}`
(`seed`는 검증 전용 - 지정 시 SmolVLA의 확률적 디노이징 노이즈를 고정한다.
정상 운용에서는 보내지 않는다.)

응답: `{success: bool, action_chunk: (T,6) ndarray, chunk_size: int, action_dim: int, message: str}`
