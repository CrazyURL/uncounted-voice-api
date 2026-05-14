"""
AutoLabelService — KcELECTRA 기반 감정/대화행위 자동 예측 (CPU 추론)

모델 없을 때: is_available() == False, predict() → None 필드로 graceful degradation.
모델 있을 때: models/emotion/current/ 디렉토리에서 lazy load.

저장 포맷 (train_emotion_model.py 와 일치):
  {version}/encoder/          AutoModel.from_pretrained 로드
  {version}/tokenizer/        AutoTokenizer.from_pretrained 로드
  {version}/heads.pt          {"emotion_head": state_dict, "dialog_act_head": state_dict}
  {version}/label_map.json    {"emotion_labels": [...], "dialog_act_labels": [...]}
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

FALLBACK_EMOTION_LABELS = ["긍정", "중립", "부정"]
FALLBACK_DIALOG_ACT_LABELS = [
    "진술", "질문", "요청", "감사", "인사", "사과",
    "동의", "반대", "확인", "부정", "응답", "제안",
    "명령", "감탄", "기타",
]

MODEL_BASE_DIR = Path(os.environ.get("EMOTION_MODEL_DIR", "models/emotion"))
CURRENT_LINK = MODEL_BASE_DIR / "current"
CURRENT_TXT = MODEL_BASE_DIR / "current.txt"  # Windows / symlink 미지원 환경 fallback

BATCH_SIZE = 32
MAX_LEN = 256


@dataclass
class LabelResult:
    emotion: Optional[str]                # 긍정 | 중립 | 부정
    emotion_confidence: float             # 0.0–1.0
    dialog_act: Optional[str]             # 15종
    dialog_act_confidence: float
    model_version: str                    # v{YYYYMMDD_HHMMSS}


def _resolve_current_model_path() -> Optional[Path]:
    if CURRENT_LINK.is_symlink() and CURRENT_LINK.exists():
        resolved = CURRENT_LINK.resolve()
        if resolved.is_dir():
            return resolved
    if CURRENT_TXT.exists():
        candidate = Path(CURRENT_TXT.read_text().strip())
        if candidate.is_dir():
            return candidate
    return None


class AutoLabelService:
    def __init__(self) -> None:
        self._tokenizer = None
        self._encoder = None
        self._emotion_head = None
        self._dialog_act_head = None
        self._emotion_labels: list[str] = FALLBACK_EMOTION_LABELS
        self._dialog_act_labels: list[str] = FALLBACK_DIALOG_ACT_LABELS
        self._model_version: str = ""
        self._load_attempted = False

    def is_available(self) -> bool:
        if not self._load_attempted:
            self._try_load()
        return self._encoder is not None

    def predict(self, texts: list[str]) -> list[LabelResult]:
        if not self.is_available():
            return [_null_result() for _ in texts]

        import torch

        results: list[LabelResult] = []

        for batch_start in range(0, len(texts), BATCH_SIZE):
            batch = texts[batch_start : batch_start + BATCH_SIZE]
            try:
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LEN,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    out = self._encoder(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                        token_type_ids=encoded.get("token_type_ids"),
                    )
                    cls = out.last_hidden_state[:, 0]

                    emotion_probs = torch.softmax(self._emotion_head(cls), dim=-1)
                    dialog_probs = torch.softmax(self._dialog_act_head(cls), dim=-1)

                    e_conf, e_idx = emotion_probs.max(dim=-1)
                    d_conf, d_idx = dialog_probs.max(dim=-1)

                for j in range(len(batch)):
                    results.append(LabelResult(
                        emotion=self._emotion_labels[e_idx[j].item()],
                        emotion_confidence=round(e_conf[j].item(), 4),
                        dialog_act=self._dialog_act_labels[d_idx[j].item()],
                        dialog_act_confidence=round(d_conf[j].item(), 4),
                        model_version=self._model_version,
                    ))

            except Exception as exc:
                logger.error("AutoLabelService.predict 배치 오류: %s", exc)
                results.extend(_null_result(self._model_version) for _ in batch)

        return results

    # ------------------------------------------------------------------
    def _try_load(self) -> None:
        self._load_attempted = True
        model_path = _resolve_current_model_path()
        if model_path is None:
            logger.info("AutoLabelService: 사용 가능한 모델 없음 (graceful degradation)")
            return

        encoder_dir = model_path / "encoder"
        tokenizer_dir = model_path / "tokenizer"
        heads_path = model_path / "heads.pt"
        label_map_path = model_path / "label_map.json"

        if not encoder_dir.is_dir() or not tokenizer_dir.is_dir():
            logger.warning("AutoLabelService: 모델 디렉토리 구조 불완전 (%s)", model_path)
            return

        try:
            import torch
            import torch.nn as nn
            from transformers import AutoModel, AutoTokenizer

            logger.info("AutoLabelService: 모델 로딩 — %s", model_path)
            self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
            self._encoder = AutoModel.from_pretrained(str(encoder_dir))
            self._encoder.eval()

            if label_map_path.exists():
                lm = json.loads(label_map_path.read_text(encoding="utf-8"))
                self._emotion_labels = lm.get("emotion_labels", FALLBACK_EMOTION_LABELS)
                self._dialog_act_labels = lm.get("dialog_act_labels", FALLBACK_DIALOG_ACT_LABELS)

            hidden = self._encoder.config.hidden_size
            self._emotion_head = nn.Linear(hidden, len(self._emotion_labels))
            self._dialog_act_head = nn.Linear(hidden, len(self._dialog_act_labels))

            if heads_path.exists():
                heads = torch.load(str(heads_path), map_location="cpu")
                self._emotion_head.load_state_dict(heads["emotion_head"])
                self._dialog_act_head.load_state_dict(heads["dialog_act_head"])
                logger.info("AutoLabelService: heads.pt 로드 완료")
            else:
                logger.warning("AutoLabelService: heads.pt 없음 — 랜덤 가중치로 초기화")

            self._emotion_head.eval()
            self._dialog_act_head.eval()
            self._model_version = model_path.name
            logger.info("AutoLabelService: 로드 완료 — %s", self._model_version)

        except Exception as exc:
            logger.error("AutoLabelService: 로드 실패 (%s) — graceful degradation", exc)
            self._encoder = None
            self._tokenizer = None
            self._emotion_head = None
            self._dialog_act_head = None


def _null_result(version: str = "") -> LabelResult:
    return LabelResult(
        emotion=None,
        emotion_confidence=0.0,
        dialog_act=None,
        dialog_act_confidence=0.0,
        model_version=version,
    )


# 모듈 레벨 싱글톤 — main.py lifespan에서 초기화, stt_processor에서 import
auto_label_service = AutoLabelService()
