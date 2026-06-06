# -*- coding: utf-8 -*-
"""주제(topic) 세그먼트 분류기 — 학습된 KR-ELECTRA 모델 독립 로드.

세그먼트(발화 묶음) 텍스트 → 19분류 주제 예측. 세그먼트 단위가 정답 단위이며
(발화단위 0.49 → 세그먼트단위 0.79, 누수차단 실측), 본 분류기는 topic_segmentation_service
의 경계탐지로 묶인 세그먼트 텍스트에 적용된다.

auto_label_service(감정 인코더)와 별개로 둔다 — topic 전용으로 fine-tune된 인코더라
헤드만 공유 못 함(인코더 특징 불일치). speech_age 독립모델 패턴과 동일.
모델 없으면 graceful degradation(None 반환) → 호출측이 키워드 fallback.
"""
from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.environ.get("TOPIC_MODEL_DIR", "models/topic_grouped"))
CURRENT_LINK = MODEL_DIR / "current"
CURRENT_TXT = MODEL_DIR / "current.txt"
MAX_LEN = 256
BATCH = 16


def _resolve_model_path() -> Optional[Path]:
    if CURRENT_LINK.is_symlink() and CURRENT_LINK.exists():
        r = CURRENT_LINK.resolve()
        if r.is_dir():
            return r
    if CURRENT_LINK.is_dir():
        return CURRENT_LINK
    if CURRENT_TXT.exists():
        c = Path(CURRENT_TXT.read_text().strip())
        if c.is_dir():
            return c
    return None


class TopicClassifierService:
    def __init__(self) -> None:
        self._tokenizer = None
        self._encoder = None
        self._head = None
        self._labels: list[str] = []
        self._version: str = ""
        self._load_attempted = False

    def is_available(self) -> bool:
        if not self._load_attempted:
            self._try_load()
        return self._encoder is not None

    def _try_load(self) -> None:
        self._load_attempted = True
        mp = _resolve_model_path()
        if mp is None:
            logger.info("TopicClassifier: 모델 없음 (graceful degradation)")
            return
        enc_dir, tok_dir = mp / "encoder", mp / "tokenizer"
        head_path, lm_path = mp / "head.pt", mp / "label_map.json"
        if not enc_dir.is_dir() or not tok_dir.is_dir() or not head_path.exists():
            logger.warning("TopicClassifier: 모델 구조 불완전 (%s)", mp)
            return
        try:
            import torch
            import torch.nn as nn
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
            self._encoder = AutoModel.from_pretrained(str(enc_dir)).eval()
            if lm_path.exists():
                lm = json.loads(lm_path.read_text(encoding="utf-8"))
                self._labels = next(
                    (v for v in lm.values() if isinstance(v, list) and len(v) >= 5), []
                )
            sd = torch.load(str(head_path), map_location="cpu")
            if isinstance(sd, dict) and "head" in sd and isinstance(sd["head"], dict):
                sd = sd["head"]
            self._head = nn.Linear(self._encoder.config.hidden_size, len(self._labels))
            self._head.load_state_dict(sd)
            self._head.eval()
            self._version = mp.name
            logger.info("TopicClassifier: 로드 완료 — %s (주제 %d종)", self._version, len(self._labels))
        except Exception as exc:
            logger.error("TopicClassifier: 로드 실패 (%s) — graceful degradation", exc)
            self._encoder = self._tokenizer = self._head = None

    def classify(self, text: str) -> tuple[Optional[str], float]:
        """세그먼트 텍스트 1개 → (주제, confidence). 모델 없거나 빈 텍스트면 (None, 0.0)."""
        if not text or not text.strip() or not self.is_available():
            return None, 0.0
        res = self.classify_batch([text])
        return res[0] if res else (None, 0.0)

    def classify_batch(self, texts: list[str]) -> list[tuple[Optional[str], float]]:
        if not self.is_available():
            return [(None, 0.0) for _ in texts]
        import torch

        out: list[tuple[Optional[str], float]] = []
        for i in range(0, len(texts), BATCH):
            chunk = texts[i : i + BATCH]
            try:
                e = self._tokenizer(
                    chunk, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt"
                )
                with torch.no_grad():
                    h = self._encoder(
                        input_ids=e["input_ids"],
                        attention_mask=e["attention_mask"],
                        token_type_ids=e.get("token_type_ids"),
                    ).last_hidden_state[:, 0]
                    probs = torch.softmax(self._head(h), dim=-1)
                    conf, idx = probs.max(dim=-1)
                for j in range(len(chunk)):
                    lab = self._labels[idx[j].item()] if self._labels else None
                    out.append((lab, round(conf[j].item(), 4)))
            except Exception as exc:
                logger.error("TopicClassifier.classify_batch 오류: %s", exc)
                out.extend((None, 0.0) for _ in chunk)
        return out


topic_classifier_service = TopicClassifierService()
