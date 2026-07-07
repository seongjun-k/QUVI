"""
QUVI 자체구현 PatchCore (경량 이상탐지)
──────────────────────────────────
사전학습 WideResNet-50(ImageNet)의 중간층(layer2+layer3) 패치 특징으로
정상품 메모리뱅크를 구성하고, 추론 시 패치별 최근접 거리의 최댓값을
이상점수로 사용하는 자체구현 PatchCore.

신규 의존성 없음 — torch/torchvision만 사용 (anomalib/lightning/faiss/
sklearn 등 금지). 학습 루프가 없는 특징추출 기반 기법이라 소량 정상 데이터
(각도당 20~50장)에서도 동작한다.

참고: docs/ml_anomaly_inspection_plan.md §2, §4 Phase 1
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

# ImageNet 정규화 상수 (백본 사전학습 시 사용된 값과 동일해야 함)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# 뱅크 대상 kNN 거리 계산 시 한 번에 비교할 뱅크 청크 크기 (OOM 방지)
_DIST_CHUNK_SIZE = 4096


# ─────────────────────────────────────────────
# 디바이스 헬퍼
# ─────────────────────────────────────────────
def _resolve_device(device: str) -> torch.device:
    """요청 디바이스가 사용 불가하면 cpu 로 자동 폴백."""
    if device.startswith('cuda') and not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device(device)


# ─────────────────────────────────────────────
# PatchCoreDetector
# ─────────────────────────────────────────────
class PatchCoreDetector:
    """WideResNet-50 중간층 패치 특징 기반 자체구현 PatchCore.

    Attributes:
        device: 실제 사용 중인 torch.device (cuda 불가 시 cpu 로 폴백됨).
        bank: fit() 이후 채워지는 메모리뱅크 텐서 (N, C). fit 전에는 None.
        coreset_ratio: 마지막 fit() 에 사용된 coreset 비율.
        out_size: 학습에 사용된 입력 이미지 한 변 크기 (참고용 메타).
        meta: save() 시 함께 저장되는 임의의 메타 정보 딕셔너리
              (호출자가 자유롭게 채워 넣을 수 있음 — 예: 학습 이미지 수, 각도 등).
    """

    def __init__(
        self,
        device: str = 'cuda',
        backbone_weights_path: Optional[str] = None,
        backbone: Optional[nn.Module] = None,
    ) -> None:
        """백본을 로드한다.

        Args:
            device: 'cuda' 또는 'cpu'. cuda 요청이지만 사용 불가하면 자동으로
                cpu 로 폴백한다.
            backbone_weights_path: WideResNet-50 state_dict 저장 경로.
                파일이 있으면 그대로 로드(오프라인 재현). 없으면 torchvision
                기본 ImageNet 가중치를 다운로드한 뒤 이 경로에 state_dict 를
                저장해 이후 실행에서 재사용(영속화)한다. None 이면 매번 다운로드.
            backbone: 이미 로드된 백본을 재사용하고 싶을 때 전달(각도별 뱅크
                4개를 로드하며 백본을 4번 새로 만드는 GPU 메모리 낭비 방지용).
                전달되면 backbone_weights_path 는 무시된다.
        """
        self.device = _resolve_device(device)
        self.backbone_weights_path = backbone_weights_path
        if backbone is not None:
            self.backbone = backbone.to(self.device)
        else:
            self.backbone = self._load_backbone(backbone_weights_path).to(self.device)
        self.backbone.eval()
        self.backbone.requires_grad_(False)

        # layer2/layer3 출력을 가로채기 위한 forward hook
        self._hook_outputs: Dict[str, torch.Tensor] = {}
        self.backbone.layer2.register_forward_hook(self._make_hook('layer2'))
        self.backbone.layer3.register_forward_hook(self._make_hook('layer3'))

        self._smooth_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.bank: Optional[torch.Tensor] = None
        self.coreset_ratio: Optional[float] = None
        self.out_size: Optional[int] = None
        self.meta: Dict[str, Any] = {}

    # ── 백본 로드 / 영속화 ───────────────────────
    def _load_backbone(self, weights_path: Optional[str]) -> nn.Module:
        if weights_path and os.path.isfile(weights_path):
            net = wide_resnet50_2(weights=None)
            state = torch.load(weights_path, map_location='cpu')
            net.load_state_dict(state)
            return net

        net = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        if weights_path:
            os.makedirs(os.path.dirname(weights_path) or '.', exist_ok=True)
            torch.save(net.state_dict(), weights_path)
        return net

    def _make_hook(self, name: str):
        def _hook(_module: nn.Module, _inp: Any, output: torch.Tensor) -> None:
            self._hook_outputs[name] = output
        return _hook

    # ── 전처리: RGB uint8 리스트 → 정규화 텐서 ──────
    def _to_tensor(self, images: List[np.ndarray]) -> torch.Tensor:
        """256×256 RGB uint8 이미지 리스트 → ImageNet 정규화 (B,3,H,W) 텐서."""
        arr = np.stack(images).astype(np.float32) / 255.0   # (B,H,W,3)
        arr = arr.transpose(0, 3, 1, 2)                       # (B,3,H,W)
        t = torch.from_numpy(arr).to(self.device)
        mean = torch.tensor(_IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        return (t - mean) / std

    # ── 패치 임베딩 추출 ─────────────────────────
    def _extract_patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """layer2+layer3 결합 패치 임베딩을 반환한다.

        layer3 을 layer2 의 공간 크기로 업샘플 후 채널 방향 concat, 이어서
        AvgPool2d(3, stride=1, padding=1) 로 로컬 스무딩(표준 PatchCore 기법).

        Returns:
            (B, C2+C3, H2, W2) 텐서.
        """
        with torch.no_grad():
            self.backbone(x)
        f2 = self._hook_outputs['layer2']
        f3 = self._hook_outputs['layer3']
        f3_up = F.interpolate(
            f3, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        embedding = torch.cat([f2, f3_up], dim=1)
        return self._smooth_pool(embedding)

    def _flatten_patches(self, embedding: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B*H*W, C) 패치별 벡터로 펼친다."""
        b, c, h, w = embedding.shape
        return embedding.permute(0, 2, 3, 1).reshape(b * h * w, c)

    # ── 학습(=특징 수집 + coreset) ────────────────
    def fit(
        self,
        images: List[np.ndarray],
        coreset_ratio: float = 0.1,
        batch_size: int = 8,
        seed: Optional[int] = None,
    ) -> None:
        """정상품 이미지들로 메모리뱅크를 구성한다.

        Args:
            images: 256×256 RGB uint8 이미지 리스트 (ml_preprocess.preprocess_for_ml 출력).
            coreset_ratio: greedy k-center 로 서브샘플링할 패치 비율 (0~1).
            batch_size: 특징 추출 시 배치 크기.
            seed: coreset 시작점 선택 재현성을 위한 시드. None 이면 비고정.
        """
        if not images:
            raise ValueError('fit()에 빈 이미지 리스트가 전달됨 — 최소 1장 필요')

        self.out_size = images[0].shape[0]

        all_patches: List[torch.Tensor] = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            x = self._to_tensor(batch)
            emb = self._extract_patch_embeddings(x)
            all_patches.append(self._flatten_patches(emb).cpu())

        embeddings = torch.cat(all_patches, dim=0).to(self.device)  # (N, C)

        self.coreset_ratio = coreset_ratio
        self.bank = self._greedy_coreset(embeddings, coreset_ratio, seed=seed).cpu()

    def _greedy_coreset(
        self,
        embeddings: torch.Tensor,
        ratio: float,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Greedy k-center coreset 서브샘플링 (GPU 텐서 반복 argmax).

        수천 패치 규모에서는 반복 cdist 만으로 충분히 빠르다 (표준 PatchCore
        coreset 방식).
        """
        n = embeddings.shape[0]
        n_select = max(1, int(n * ratio))

        if seed is not None:
            gen = torch.Generator(device='cpu').manual_seed(seed)
            start_idx = int(torch.randint(0, n, (1,), generator=gen).item())
        else:
            start_idx = int(torch.randint(0, n, (1,)).item())

        selected = [start_idx]
        min_dists = torch.cdist(
            embeddings, embeddings[start_idx:start_idx + 1]).squeeze(1)  # (n,)

        for _ in range(n_select - 1):
            next_idx = int(torch.argmax(min_dists).item())
            selected.append(next_idx)
            new_dists = torch.cdist(
                embeddings, embeddings[next_idx:next_idx + 1]).squeeze(1)
            min_dists = torch.minimum(min_dists, new_dists)

        return embeddings[selected]

    # ── 추론(이상점수) ───────────────────────────
    def score(self, image: np.ndarray) -> float:
        """단일 이미지의 이상점수(패치별 뱅크 최근접 거리의 최댓값)를 반환한다.

        Args:
            image: 256×256 RGB uint8 이미지 (ml_preprocess.preprocess_for_ml 출력).

        Returns:
            이상점수 (클수록 정상에서 벗어남). 뱅크가 없으면 예외.
        """
        if self.bank is None:
            raise RuntimeError('bank 가 비어 있음 — fit() 또는 load() 를 먼저 호출하세요')

        x = self._to_tensor([image])
        emb = self._extract_patch_embeddings(x)
        patches = self._flatten_patches(emb)  # (P, C)

        min_dists = self._min_dist_to_bank(patches)
        return float(min_dists.max().item())

    def _min_dist_to_bank(
        self, query: torch.Tensor, chunk_size: int = _DIST_CHUNK_SIZE,
    ) -> torch.Tensor:
        """query 패치별 뱅크 최근접 L2 거리. 뱅크를 청크 처리해 OOM을 방지한다."""
        bank = self.bank.to(query.device)
        min_dists = torch.full(
            (query.shape[0],), float('inf'), device=query.device)
        for i in range(0, bank.shape[0], chunk_size):
            chunk = bank[i:i + chunk_size]
            d = torch.cdist(query, chunk)          # (P, chunk)
            chunk_min, _ = d.min(dim=1)
            min_dists = torch.minimum(min_dists, chunk_min)
        return min_dists

    # ── 저장 / 로드 ──────────────────────────────
    def save(self, path: str) -> None:
        """메모리뱅크 + 메타 정보를 저장한다 (백본 가중치는 별도 파일)."""
        if self.bank is None:
            raise RuntimeError('bank 가 비어 있음 — fit() 을 먼저 호출하세요')
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'bank': self.bank,
            'coreset_ratio': self.coreset_ratio,
            'out_size': self.out_size,
            'meta': self.meta,
        }, path)

    @classmethod
    def load(
        cls,
        path: str,
        device: str = 'cuda',
        backbone_weights_path: Optional[str] = None,
        backbone: Optional[nn.Module] = None,
    ) -> 'PatchCoreDetector':
        """저장된 뱅크를 로드해 즉시 score() 가능한 인스턴스를 만든다.

        Args:
            path: save() 로 저장된 뱅크 파일 경로.
            device: 추론에 사용할 디바이스.
            backbone_weights_path: 백본 가중치 경로 (없으면 재다운로드).
            backbone: 이미 로드된 백본을 재사용(각도별 4회 로드 시 백본 공유용).
        """
        instance = cls(
            device=device,
            backbone_weights_path=backbone_weights_path,
            backbone=backbone)
        data = torch.load(path, map_location='cpu')
        instance.bank = data['bank'].to(instance.device)
        instance.coreset_ratio = data.get('coreset_ratio')
        instance.out_size = data.get('out_size')
        instance.meta = data.get('meta', {})
        return instance
