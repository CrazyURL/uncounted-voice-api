"""
AutoLabelService — KcELECTRA 기반 감정/대화행위 자동 예측 (CPU 추론)

모델 없을 때: is_available() == False, predict() → None 필드로 graceful degradation.
모델 있을 때: models/emotion/current/ 디렉토리에서 lazy load.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

EMOTION_LABELS = ["기쁨", "놀람", "슬픔", "분노", "불안", "당황", "중립"]

# 세부감정 → 상위 카테고리 (모델 미학습, 추론 후 계산)
EMOTION_TO_CATEGORY: dict[str, str] = {
    "기쁨": "긍정", "놀람": "긍정",
    "슬픔": "부정", "분노": "부정", "불안": "부정", "당황": "부정",
    "중립": "중립",
}

DIALOG_ACT_LABELS = [
    "진술", "질문", "요청", "감사", "인사", "사과",
    "동의", "반대", "확인", "부정", "응답", "제안",
    "명령", "감탄", "기타",
]

MODEL_BASE_DIR = Path(os.environ.get("EMOTION_MODEL_DIR", "models/emotion"))
CURRENT_LINK = MODEL_BASE_DIR / "current"
CURRENT_TXT = MODEL_BASE_DIR / "current.txt"  # Windows fallback


@dataclass
class LabelResult:
    emotion: Optional[str]            # 세부감정 7종: 기쁨|놀람|슬픔|분노|불안|당황|중립
    emotion_category: Optional[str]   # 상위 카테고리 3종: 긍정|중립|부정 (파생값)
    emotion_confidence: float
    dialog_act: Optional[str]
    dialog_act_confidence: float
    model_version: str


def _resolve_current_model_path() -> Optional[Path]:
    """current symlink 또는 current.txt 로 현재 모델 경로를 반환한다."""
    if CURRENT_LINK.exists():
        resolved = CURRENT_LINK.resolve()
        if resolved.is_dir():
            return resolved
    if CURRENT_TXT.exists():
        version = CURRENT_TXT.read_text().strip()
        candidate = MODEL_BASE_DIR / version
        if candidate.is_dir():
            return candidate
    return None


class AutoLabelService:
    def __init__(self) -> None:
        self._tokenizer = None
        self._model = None
        self._emotion_head = None
        self._dialog_act_head = None
        self._model_version: str = ""
        self._load_attempted = False

    def is_available(self) -> bool:
        if not self._load_attempted:
            self._try_load()
        return self._model is not None

    def predict(self, texts: list[str]) -> list[LabelResult]:
        """texts 각 항목에 대해 감정 + 대화행위를 예측한다."""
        if not self.is_available():
            return [
                LabelResult(
                    emotion=None,
                    emotion_category=None,
                    emotion_confidence=0.0,
                    dialog_act=None,
                    dialog_act_confidence=0.0,
                    model_version="",
                )
                for _ in texts
            ]

        import torch

        results: list[LabelResult] = []
        batch_size = 32

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            try:
                encoded = self._tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    outputs = self._model(**encoded)
                    cls_hidden = outputs.last_hidden_state[:, 0, :]  # [B, H]

                    emotion_logits = self._emotion_head(cls_hidden)  # [B, 3]
                    dialog_logits = self._dialog_act_head(cls_hidden)  # [B, 15]

                    emotion_probs = torch.softmax(emotion_logits, dim=-1)
                    dialog_probs = torch.softmax(dialog_logits, dim=-1)

                    emotion_conf, emotion_idx = emotion_probs.max(dim=-1)
                    dialog_conf, dialog_idx = dialog_probs.max(dim=-1)

                for j in range(len(batch_texts)):
                    emotion = EMOTION_LABELS[emotion_idx[j].item()]
                    results.append(
                        LabelResult(
                            emotion=emotion,
                            emotion_category=EMOTION_TO_CATEGORY.get(emotion),
                            emotion_confidence=round(emotion_conf[j].item(), 4),
                            dialog_act=DIALOG_ACT_LABELS[dialog_idx[j].item()],
                            dialog_act_confidence=round(dialog_conf[j].item(), 4),
                            model_version=self._model_version,
                        )
                    )
            except Exception as exc:
                logger.error("AutoLabelService.predict 배치 오류: %s", exc)
                for _ in batch_texts:
                    results.append(
                        LabelResult(
                            emotion=None,
                            emotion_category=None,
                            emotion_confidence=0.0,
                            dialog_act=None,
                            dialog_act_confidence=0.0,
                            model_version=self._model_version,
                        )
                    )

        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _try_load(self) -> None:
        self._load_attempted = True
        model_path = _resolve_current_model_path()
        if model_path is None:
            logger.info("AutoLabelService: 사용 가능한 모델 없음 (graceful degradation)")
            return

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            logger.info("AutoLabelService: 모델 로딩 중 — %s", model_path)
            self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            self._model = AutoModel.from_pretrained(str(model_path))
            self._model.eval()

            hidden_size = self._model.config.hidden_size

            # 저장된 head 가중치 로드 (없으면 랜덤 초기화 — 첫 학습 전)
            emotion_head_path = model_path / "emotion_head.pt"
            dialog_head_path = model_path / "dialog_act_head.pt"

            import torch.nn as nn

            self._emotion_head = nn.Linear(hidden_size, len(EMOTION_LABELS))
            self._dialog_act_head = nn.Linear(hidden_size, len(DIALOG_ACT_LABELS))

            if emotion_head_path.exists():
                self._emotion_head.load_state_dict(
                    torch.load(str(emotion_head_path), map_location="cpu")
                )
            if dialog_head_path.exists():
                self._dialog_act_head.load_state_dict(
                    torch.load(str(dialog_head_path), map_location="cpu")
                )

            self._emotion_head.eval()
            self._dialog_act_head.eval()
            self._model_version = model_path.name

            logger.info("AutoLabelService: 모델 로드 완료 — v%s", self._model_version)

        except Exception as exc:
            logger.error("AutoLabelService: 모델 로드 실패 (%s) — graceful degradation", exc)
            self._tokenizer = None
            self._model = None
            self._emotion_head = None
            self._dialog_act_head = None
