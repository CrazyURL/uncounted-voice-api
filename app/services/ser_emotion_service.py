# -*- coding: utf-8 -*-
"""세부감정(SER) 오디오 추론 — 학습된 wav2vec2 모델로 발화 오디오 → 7분류 감정.

세부감정은 텍스트가 아니라 **오디오**가 정답(실측: 텍스트 0.34 < 오디오 0.51).
발화 오디오 클립(16kHz)을 받아 7분류(기쁨/놀라움/사랑스러움/화남/두려움/슬픔/없음) 예측.
auto_label_service(텍스트 감정)와 별개 — 오디오 전용. 모델 없으면 graceful(None).

학습/데이터 교훈(2026-06-07): 데이터 질>양. 실내(전부 '보통' 강도)는 모델 열화 →
v2(실외+샘플, 강함 보유) 0.51 채택. 모델: models/ser_emotion_v2/current.
"""
from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.environ.get("SER_MODEL_DIR", "models/ser_emotion_v2"))
CURRENT = MODEL_DIR / "current"
SR = 16000
MAX_SEC = 4.0          # 학습과 동일 하드 트런케이션
MIN_SEC = 0.4
BATCH = 16
BASE_FEAT = "kresnik/wav2vec2-large-xlsr-korean"


def _resolve() -> Optional[Path]:
    if CURRENT.is_symlink() and CURRENT.exists():
        r = CURRENT.resolve()
        return r if r.is_dir() else None
    return CURRENT if CURRENT.is_dir() else None


class SEREmotionService:
    def __init__(self) -> None:
        self._feat = None
        self._encoder = None
        self._head = None
        self._labels: list[str] = []
        self._version = ""
        self._device = "cpu"
        self._loaded = False

    def is_available(self) -> bool:
        if not self._loaded:
            self._try_load()
        return self._encoder is not None

    def _try_load(self) -> None:
        self._loaded = True
        mp = _resolve()
        if mp is None:
            logger.info("SEREmotion: 모델 없음 (graceful degradation)")
            return
        enc_dir, head_path, lm_path = mp / "encoder", mp / "head.pt", mp / "label_map.json"
        if not enc_dir.is_dir() or not head_path.exists():
            logger.warning("SEREmotion: 모델 구조 불완전 (%s)", mp)
            return
        try:
            import torch
            import torch.nn as nn
            from transformers import AutoFeatureExtractor, Wav2Vec2Model

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._feat = AutoFeatureExtractor.from_pretrained(BASE_FEAT)
            self._encoder = Wav2Vec2Model.from_pretrained(str(enc_dir)).eval().to(self._device)
            if lm_path.exists():
                lm = json.loads(lm_path.read_text(encoding="utf-8"))
                self._labels = next((v for v in lm.values() if isinstance(v, list) and len(v) >= 5), [])
            sd = torch.load(str(head_path), map_location="cpu")
            if isinstance(sd, dict) and "head" in sd and isinstance(sd["head"], dict):
                sd = sd["head"]
            self._head = nn.Linear(self._encoder.config.hidden_size, len(self._labels))
            self._head.load_state_dict(sd)
            self._head.eval().to(self._device)
            self._version = mp.name
            logger.info("SEREmotion: 로드 완료 — %s (감정 %d종, device=%s)",
                        self._version, len(self._labels), self._device)
        except Exception as exc:
            logger.error("SEREmotion: 로드 실패 (%s) — graceful degradation", exc)
            self._encoder = self._head = self._feat = None

    def predict(self, clips: list) -> list[tuple[Optional[str], float]]:
        """발화 오디오 클립 리스트(각 16kHz 1D float32 numpy) → [(감정, conf)].

        너무 짧은 클립(<0.4s)은 (None, 0.0). 모델 없으면 전부 (None, 0.0).
        """
        if not clips or not self.is_available():
            return [(None, 0.0) for _ in clips]
        import numpy as np
        import torch

        out: list[tuple[Optional[str], float]] = []
        for i in range(0, len(clips), BATCH):
            batch = clips[i : i + BATCH]
            prepped, idx_map = [], []
            for j, a in enumerate(batch):
                a = np.asarray(a, dtype=np.float32).ravel()
                if a.size < int(MIN_SEC * SR):
                    continue
                a = a[: int(MAX_SEC * SR)]
                prepped.append(a); idx_map.append(j)
            results = [(None, 0.0)] * len(batch)
            if prepped:
                try:
                    enc = self._feat(prepped, sampling_rate=SR, return_tensors="pt", padding=True)
                    iv = enc["input_values"].to(self._device)
                    with torch.no_grad():
                        h = self._encoder(iv).last_hidden_state.mean(dim=1)
                        probs = torch.softmax(self._head(h), dim=-1)
                        conf, pred = probs.max(dim=-1)
                    for k, j in enumerate(idx_map):
                        results[j] = (self._labels[pred[k].item()], round(conf[k].item(), 4))
                except Exception as exc:
                    logger.error("SEREmotion.predict 배치 오류: %s", exc)
            out.extend(results)
        return out


ser_emotion_service = SEREmotionService()
