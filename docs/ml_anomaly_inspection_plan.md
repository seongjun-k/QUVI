# 양불판정 ML(이상탐지) 전환 계획서 — v2 (검증·수정판)

> v1: 2026-07-03 작성 · v2: 2026-07-03 — v1의 가정을 실제 환경에서 전수 검증 후 수정
> 결정 사항: **이상탐지(Anomaly Detection) / 정상품(PASS) 위주 데이터 수집**

---

## 0. v1 → v2 변경 요약 (검증에서 드러난 것)

| # | v1의 가정/결함 | 실측 결과 | v2 반영 |
|---|---|---|---|
| 1 | "PyTorch 이미 설치·구동 중" | **호스트엔 없음.** `quvi:latest` 컨테이너에 torch 2.12.1+cu130, torchvision 0.27.1. RTX 4060 Laptop 8GB + `runtime: nvidia` 확인 | GPU 추론 전제로 설계 (CPU 폴백 포함). 학습·추론 모두 quvi-dev 컨테이너에서 |
| 2 | 백본 가중치 리스크 "확인 필요" | 컨테이너 캐시 **비어 있음** — 최초 1회 다운로드 필요 | 가중치를 `/workspace/data/models/`(= 호스트 `./data/models/`, 볼륨 영속)에 저장·재사용 |
| 3 | **(v1 누락) 라벨 오염** | 수집될 "정상" 라벨은 **룰 판정기가 붙임** — 교체하려는 바로 그 판정기 | 사람 검수 단계 필수 (Phase 0) |
| 4 | "ROC로 임계값 캘리브레이션" | 검증셋 6장으로 ROC는 무의미 | 정상점수 분포 기반 임계값(§4). FAIL 5장은 sanity check로만 |
| 5 | **(v1 누락) 전처리 일관성** | 백본은 3채널 입력. 현재 파이프라인은 grayscale 정렬 | 학습/추론이 **동일 전처리 함수 공유** (§3). 원본은 1920×1080 컬러로 저장돼 있어 재처리 가능 |
| 6 | anomalib vs 자체구현 "Phase 1에서 결정" | anomalib은 lightning 등 무거운 의존성 유입 (이미지 이미 11.5GB) | **결정: 자체구현 PatchCore** (~150줄, 신규 의존성 0 — torch/torchvision만) |
| 7 | 컷오버 기준 모호 | — | 정량 게이트 정의 (§4 Phase 3) |

**확정(2026-07-03 사용자 확인)**: 검사 대상은 **단일 품종**으로 진행. 각도별 메모리뱅크 1세트 체계 확정. (향후 품종 변경 시 뱅크 재구축 필요)

## 1. 현재 상태 (룰 기반 클래식 CV)

`src/quvi_inspect/quvi_inspect/inspect_node.py :: _surface_analysis()` — ML 아님.
4방향(0/90/180/270°) 이미지에서 임계값 규칙으로 판정:

| 특징 | 계산 | 잡는 불량 | 임계값(기본) |
|---|---|---|---|
| Solidity | 컨투어면적/볼록껍질 | 워핑 | 0.85–1.00 |
| Area ratio | 기준이미지 대비 면적비 | 미출력 | 0.90–1.10 |
| Hole count / area | 이진화 후 구멍 | 레이어분리 | ≤2개 / ≤0.05 |
| Texture variance | Laplacian 분산 | 스트링잉 | ≤500 |

- 판정 = 4각도 worst-case, 모든 임계값 통과해야 PASS (AND).
- 반환 dict 중 **`passed`만 액추에이션에 영향** (orchestrator가 사용) → 통합 지점 깨끗함.
- `_save_inspection_log()`가 검사마다 `..._PASS/FAIL` 폴더에 4각도 **원본 컬러 PNG(1920×1080)** 저장 → 수집 파이프라인 존재, 볼륨만 부족 (현재 PASS 1 / FAIL 5).

## 2. 기법: 자체구현 PatchCore (경량)

- 사전학습 CNN 백본(torchvision **WideResNet-50**, ImageNet weights)의 중간층(layer2+layer3) 패치 특징 → 정상품 특징 **메모리뱅크**(coreset 서브샘플링 ~10%) 구성.
- 추론: 입력 패치별 뱅크 kNN 거리의 최댓값 = 이상점수. **학습 루프 없음**(특징추출뿐) — 소량 정상 데이터에 최적, MVTec 산업검사 표준.
- **각도별 뱅크 4개**: `data/models/bank_{0,90,180,270}.pt`. 각도마다 뷰가 달라 분리가 정확도·단순성 모두 우위.
- 신규 의존성 없음. RTX 4060 8GB에서 4장 추론 <1초 (검사 finalize 12초 예산 대비 여유). CUDA 불가 시 CPU 폴백(수 초, 여전히 예산 내).

## 3. 전처리 통일 (학습·추론 공통, 핵심 설계)

train/infer skew가 이상탐지 최대 오탐 원인이므로 **단일 함수로 공유**:

```
preprocess_for_ml(bgr_image, bin_thresh) -> 256×256 RGB tensor
  1. 기존 BinaryCache 로직으로 객체 bbox 검출 (+15% 패딩)  ← 룰 파이프라인과 동일 정렬
  2. 컬러 원본에서 해당 ROI crop
  3. 256×256 resize + ImageNet 정규화
```

- 위치: `quvi_inspect/ml_preprocess.py` — 학습 스크립트와 노드가 **같은 모듈 import**.
- 저장된 로그가 원본 컬러라서 기존 6장도 동일 함수로 재처리 가능.

## 4. 단계별 실행 (Surgical / 롤백 보장)

### Phase 0 — 데이터 수집 + 검수 (선행 게이트, 성패 좌우)
- [ ] `scripts/build_anomaly_dataset.py`: `inspection_logs`의 `*_PASS` 폴더를 각도별로 모아 `data/anomaly_dataset/{angle}/`로 정리 + 썸네일 시트 생성
- [ ] 실제 장비로 정상품 반복 검사 촬영 — **목표 각도당 ≥50장, 최소 20장**. 조명·거치 조건은 실검사와 동일하게, 자연스러운 놓임 편차 포함(정상 변동성 학습)
- [ ] **사람 검수(필수)**: 룰 판정이 붙인 PASS 라벨을 그대로 믿지 않음 — 썸네일 시트에서 사람이 불량 혼입 제거. *오염된 정상셋 = 불량을 정상으로 학습*
- 게이트: 검수 통과 정상 이미지 각도당 20장 미만이면 중단하고 재평가

### Phase 1 — 오프라인 프로토타입 (노드 무수정)
- [ ] `quvi_inspect/ml_preprocess.py` (§3) + `quvi_inspect/anomaly_detector.py` (PatchCore 코어: 특징추출→coreset→kNN 점수)
- [ ] `scripts/train_anomaly_bank.py`: 정상셋 → 각도별 뱅크 4개 저장. 백본 가중치는 최초 1회 다운로드 후 `data/models/wide_resnet50.pth`로 영속화(오프라인 재현성)
- [ ] **임계값 산정**: 정상셋을 8:2 분할, held-out 정상 20%의 점수 분포에서 `threshold = max(held-out) × 1.15` (마진은 실측 조정). 기존 FAIL 5장은 "임계값 초과하는가" sanity check로만 사용 — 통계적 검증 아님
- [ ] 산출물: 각도별 정상/불량 점수 분포 리포트 (분리도 눈으로 확인)

### Phase 2 — 노드 통합 (판정은 아직 룰)
- [ ] `_load_params`에 추가: `anomaly_enabled(False)`, `anomaly_model_dir`, `anomaly_threshold`, `anomaly_device('cuda')`
- [ ] init에서 뱅크 로드 (파일 없거나 로드 실패 → 경고 로그 + 자동 비활성, **노드는 절대 죽지 않음**)
- [ ] `_surface_analysis()`에서 각도별 이상점수 계산, worst(최대) 점수를 결과 dict에 추가

### Phase 3 — 섀도우 모드 (액추에이션 불변)
- [ ] `passed`는 룰 유지. ML 점수·ML 판정·룰 판정을 `result.txt`와 노드 로그에 병행 기록
- [ ] `scripts/shadow_report.py`: 누적 로그에서 룰 vs ML 일치율·불일치 케이스 리포트
- [ ] **컷오버 정량 게이트**: ① 실검사 ≥30회 누적 ② 알려진 불량품 투입 시험에서 ML 미검출(false-accept) 0건 ③ 불일치 케이스 전건 사람 판독 완료
- HMI 노출은 이 단계에선 로그만 (InspectionResult.msg 변경은 Phase 4 선택 항목 — msg 변경은 의존 패키지 리빌드 필요해 후순위)

### Phase 4 — 컷오버
- [ ] `use_ml_judgment` 파라미터(기본 False)로 `passed` 소스 전환 — **파라미터 하나로 즉시 롤백**
- [ ] 초기 운영은 보수적 옵션 `ML AND 룰` 지원 (둘 다 PASS여야 PASS)
- [ ] (선택) `InspectionResult.msg`에 `float32[] anomaly_scores` 추가 + HMI 대시보드 표시

## 5. 리스크 관리

| 리스크 | 대응 |
|---|---|
| 정상셋 라벨 오염 (룰 판정기가 라벨러) | Phase 0 사람 검수 필수화 |
| 촬영 조건(조명/거치) drift → 오탐 | 실검사와 동일 조건 수집 + 자연 편차 포함, 섀도우 모드에서 오탐률 실측 |
| 백본 가중치 다운로드 불가 환경 | 최초 1회 후 `data/models/` 볼륨 영속화 |
| CUDA 런타임 문제 | device 파라미터 + CPU 폴백, 로드 실패 시 자동 비활성 |
| 데이터 볼륨 미달(각도당 <20) | Phase 0 게이트에서 중단·재평가 (진행 강행 금지) |
| 다품종 전환 | 현 계획은 obj0 단일 제품 가정 — 바뀌면 제품별 뱅크 체계로 계획 수정 |
| 신규 불량 유형(뱅크가 못 본 것) | 이상탐지 특성상 "정상에서 벗어남"은 유형 불문 검출 — 단 fail_reason 세분류는 룰 특징 병기로 보완 |

## 6. 다음 액션

**Phase 0-a**: `scripts/build_anomaly_dataset.py` 작성 (기존 로그 정리 + 썸네일 검수 시트) → 이후 실장비 정상품 촬영은 사용자 작업.
Phase 1의 코드(전처리·PatchCore·학습 스크립트)는 데이터 수집과 병렬로 진행 가능 — 기존 6장으로 파이프라인 동작 검증까지는 가능하나, **임계값 확정은 반드시 Phase 0 완료 후**.
