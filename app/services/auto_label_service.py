"""
AutoLabelService — KcELECTRA 기반 감정/대화행위/말투연령 자동 예측 (CPU 추론)

모델 없을 때: is_available() == False, predict() → None 필드로 graceful degradation.
모델 있을 때: models/emotion/current/ 디렉토리에서 lazy load.

말투연령 모델 (predict_speech_age): models/speech_age/current/ 에서 별도 lazy load.
  — KcELECTRA backbone 공유, 말투연령 헤드는 CPU 전용 (GPU 적재 금지).

저장 포맷 (train_emotion_model.py / train_speech_age_model.py 와 일치):
  {version}/encoder/          AutoModel.from_pretrained 로드
  {version}/tokenizer/        AutoTokenizer.from_pretrained 로드
  감정: heads.pt              {"emotion_head": state_dict, "dialog_act_head": state_dict,
                              "emotion_category_head": state_dict (선택)}
       label_map.json         {"emotion_labels": [...], "dialog_act_labels": [...],
                              "emotion_category_labels": [...] (선택)}
  말투: heads.pt              {"speech_age_head": state_dict}
       label_map.json         {"speech_age_labels": [...]}
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
# 세부감정 6대분류 (헤드 미학습 시 라벨만 fallback; heads.pt에 emotion_category_head
# 없으면 추론은 None 산출)
FALLBACK_EMOTION_CATEGORY_LABELS = ["분노", "슬픔", "불안", "상처", "당황", "기쁨"]
# 주제 20분류 (heads.pt에 topic_head 없으면 추론 None 산출). label_map.json의
# topic_labels가 정본 — 아래는 헤드 미학습 시 슬롯 유지용 fallback일 뿐.
FALLBACK_TOPIC_LABELS = [
    "가족", "건강", "게임", "계절/날씨", "교육", "교통", "군대", "미용",
    "반려동물", "방송/연예", "사회이슈", "상거래 전반", "스포츠/레저", "식음료",
    "여행", "연애/결혼", "영화/만화", "주거와 생활", "타 국가 이슈", "회사/아르바이트",
]
# 방언 권역 (heads.pt에 dialect_head 없으면 추론 None 산출)
FALLBACK_DIALECT_LABELS = [
    "수도권", "강원", "충청", "전라", "경북", "경남", "제주",
]
SPEECH_AGE_LABELS = ["20대", "30대", "40대", "50대+"]

MODEL_BASE_DIR = Path(os.environ.get("EMOTION_MODEL_DIR", "models/emotion"))
CURRENT_LINK = MODEL_BASE_DIR / "current"
CURRENT_TXT = MODEL_BASE_DIR / "current.txt"  # Windows / symlink 미지원 환경 fallback

SPEECH_AGE_MODEL_BASE_DIR = Path(os.environ.get("SPEECH_AGE_MODEL_DIR", "models/speech_age"))
SPEECH_AGE_CURRENT_LINK = SPEECH_AGE_MODEL_BASE_DIR / "current"
SPEECH_AGE_CURRENT_TXT = SPEECH_AGE_MODEL_BASE_DIR / "current.txt"

BATCH_SIZE = 32
MAX_LEN = 256


@dataclass
class LabelResult:
    emotion: Optional[str]                # 긍정 | 중립 | 부정
    emotion_confidence: float             # 0.0–1.0
    dialog_act: Optional[str]             # 15종
    dialog_act_confidence: float
    model_version: str                    # v{YYYYMMDD_HHMMSS}
    emotion_category: Optional[str] = None        # 세부감정 6대분류 (헤드 없으면 None)
    emotion_category_confidence: float = 0.0      # 0.0–1.0
    topic_category: Optional[str] = None          # 주제 20분류 (헤드 없으면 None)
    topic_category_confidence: float = 0.0        # 0.0–1.0
    dialect: Optional[str] = None                 # 방언 권역 (헤드 없으면 None)
    dialect_confidence: float = 0.0               # 0.0–1.0


@dataclass
class SpeechAgeResult:
    speech_age: Optional[str]            # 20대 | 30대 | 40대 | 50대+  (None = 신호 없음)
    speech_age_confidence: float         # 0.0–1.0
    model_version: str                   # v{YYYYMMDD_HHMMSS}


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


def _resolve_speech_age_model_path() -> Optional[Path]:
    if SPEECH_AGE_CURRENT_LINK.is_symlink() and SPEECH_AGE_CURRENT_LINK.exists():
        resolved = SPEECH_AGE_CURRENT_LINK.resolve()
        if resolved.is_dir():
            return resolved
    if SPEECH_AGE_CURRENT_TXT.exists():
        candidate = Path(SPEECH_AGE_CURRENT_TXT.read_text().strip())
        if candidate.is_dir():
            return candidate
    return None


class AutoLabelService:
    def __init__(self) -> None:
        self._tokenizer = None
        self._encoder = None
        self._emotion_head = None
        self._dialog_act_head = None
        self._emotion_category_head = None
        self._topic_head = None
        self._dialect_head = None
        self._emotion_labels: list[str] = FALLBACK_EMOTION_LABELS
        self._dialog_act_labels: list[str] = FALLBACK_DIALOG_ACT_LABELS
        self._emotion_category_labels: list[str] = FALLBACK_EMOTION_CATEGORY_LABELS
        self._topic_labels: list[str] = FALLBACK_TOPIC_LABELS
        self._dialect_labels: list[str] = FALLBACK_DIALECT_LABELS
        self._model_version: str = ""
        self._load_attempted = False

        # 말투연령 모델 (별도 디렉토리, CPU 전용)
        self._sa_tokenizer = None
        self._sa_encoder = None
        self._sa_head = None
        self._sa_labels: list[str] = SPEECH_AGE_LABELS
        self._sa_model_version: str = ""
        self._sa_load_attempted = False

    def is_available(self) -> bool:
        if not self._load_attempted:
            self._try_load()
        return self._encoder is not None

    def is_speech_age_available(self) -> bool:
        if not self._sa_load_attempted:
            self._try_load_speech_age()
        return self._sa_encoder is not None

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
                    e_conf, e_idx = emotion_probs.max(dim=-1)

                    if self._dialog_act_head is not None:
                        dialog_probs = torch.softmax(self._dialog_act_head(cls), dim=-1)
                        d_conf, d_idx = dialog_probs.max(dim=-1)
                    else:
                        d_conf, d_idx = None, None

                    if self._emotion_category_head is not None:
                        ec_probs = torch.softmax(self._emotion_category_head(cls), dim=-1)
                        ec_conf, ec_idx = ec_probs.max(dim=-1)
                    else:
                        ec_conf, ec_idx = None, None

                    if self._topic_head is not None:
                        tp_probs = torch.softmax(self._topic_head(cls), dim=-1)
                        tp_conf, tp_idx = tp_probs.max(dim=-1)
                    else:
                        tp_conf, tp_idx = None, None

                    if self._dialect_head is not None:
                        dl_probs = torch.softmax(self._dialect_head(cls), dim=-1)
                        dl_conf, dl_idx = dl_probs.max(dim=-1)
                    else:
                        dl_conf, dl_idx = None, None

                for j in range(len(batch)):
                    results.append(LabelResult(
                        emotion=self._emotion_labels[e_idx[j].item()],
                        emotion_confidence=round(e_conf[j].item(), 4),
                        dialog_act=(
                            self._dialog_act_labels[d_idx[j].item()]
                            if d_idx is not None else None
                        ),
                        dialog_act_confidence=(
                            round(d_conf[j].item(), 4) if d_conf is not None else 0.0
                        ),
                        model_version=self._model_version,
                        emotion_category=(
                            self._emotion_category_labels[ec_idx[j].item()]
                            if ec_idx is not None else None
                        ),
                        emotion_category_confidence=(
                            round(ec_conf[j].item(), 4) if ec_conf is not None else 0.0
                        ),
                        topic_category=(
                            self._topic_labels[tp_idx[j].item()]
                            if tp_idx is not None else None
                        ),
                        topic_category_confidence=(
                            round(tp_conf[j].item(), 4) if tp_conf is not None else 0.0
                        ),
                        dialect=(
                            self._dialect_labels[dl_idx[j].item()]
                            if dl_idx is not None else None
                        ),
                        dialect_confidence=(
                            round(dl_conf[j].item(), 4) if dl_conf is not None else 0.0
                        ),
                    ))

            except Exception as exc:
                logger.error("AutoLabelService.predict 배치 오류: %s", exc)
                results.extend(_null_result(self._model_version) for _ in batch)

        return results

    def encode(self, texts: list[str]):
        """[CLS] 임베딩을 numpy 배열 (N, hidden) 로 반환한다.

        모델이 없으면 None 반환.
        """
        if not self.is_available():
            return None
        import numpy as np
        import torch

        all_vecs: list[np.ndarray] = []
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
                    cls = out.last_hidden_state[:, 0].cpu().numpy()
                all_vecs.append(cls)
            except Exception as exc:
                logger.error("AutoLabelService.encode 배치 오류: %s", exc)
                hidden = self._encoder.config.hidden_size
                all_vecs.append(np.zeros((len(batch), hidden), dtype=np.float32))

        return np.concatenate(all_vecs, axis=0) if all_vecs else None

    def predict_speech_age(self, texts: list[str]) -> list[SpeechAgeResult]:
        """말투연령 4분류 예측 (CPU 전용).

        모델 없거나 텍스트 비어있으면 speech_age=None 로 graceful degradation.
        """
        if not texts:
            return []
        if not self.is_speech_age_available():
            return [_null_speech_age_result() for _ in texts]

        import torch

        results: list[SpeechAgeResult] = []

        for batch_start in range(0, len(texts), BATCH_SIZE):
            batch = texts[batch_start : batch_start + BATCH_SIZE]
            try:
                encoded = self._sa_tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LEN,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    out = self._sa_encoder(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                        token_type_ids=encoded.get("token_type_ids"),
                    )
                    cls = out.last_hidden_state[:, 0]
                    probs = torch.softmax(self._sa_head(cls), dim=-1)
                    conf, idx = probs.max(dim=-1)

                for j in range(len(batch)):
                    results.append(SpeechAgeResult(
                        speech_age=self._sa_labels[idx[j].item()],
                        speech_age_confidence=round(conf[j].item(), 4),
                        model_version=self._sa_model_version,
                    ))

            except Exception as exc:
                logger.error("AutoLabelService.predict_speech_age 배치 오류: %s", exc)
                results.extend(_null_speech_age_result(self._sa_model_version) for _ in batch)

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
                self._emotion_category_labels = lm.get(
                    "emotion_category_labels", FALLBACK_EMOTION_CATEGORY_LABELS
                )
                self._topic_labels = lm.get("topic_labels", FALLBACK_TOPIC_LABELS)
                self._dialect_labels = lm.get("dialect_labels", FALLBACK_DIALECT_LABELS)

            hidden = self._encoder.config.hidden_size
            self._emotion_head = nn.Linear(hidden, len(self._emotion_labels))
            self._dialog_act_head = nn.Linear(hidden, len(self._dialog_act_labels))
            self._emotion_category_head = nn.Linear(hidden, len(self._emotion_category_labels))
            self._topic_head = nn.Linear(hidden, len(self._topic_labels))
            self._dialect_head = nn.Linear(hidden, len(self._dialect_labels))

            if heads_path.exists():
                heads = torch.load(str(heads_path), map_location="cpu")
                self._emotion_head.load_state_dict(heads["emotion_head"])
                if "dialog_act_head" in heads:
                    self._dialog_act_head.load_state_dict(heads["dialog_act_head"])
                    logger.info("AutoLabelService: heads.pt 로드 완료 (emotion + dialog_act)")
                else:
                    self._dialog_act_head = None
                    logger.info("AutoLabelService: emotion-only 모델 — dialog_act_head 없음")
                # 세부감정 헤드 — heads.pt에 있을 때만 로드, 없으면 None (추론 시 None 산출)
                if "emotion_category_head" in heads:
                    self._emotion_category_head.load_state_dict(heads["emotion_category_head"])
                    logger.info(
                        "AutoLabelService: emotion_category_head 로드 완료 (세부감정 %d종)",
                        len(self._emotion_category_labels),
                    )
                else:
                    self._emotion_category_head = None
                    logger.info(
                        "AutoLabelService: emotion_category_head 없음 — 세부감정 None 산출"
                    )
                # 주제 20분류 헤드 — heads.pt에 있을 때만 로드, 없으면 None
                if "topic_head" in heads:
                    self._topic_head.load_state_dict(heads["topic_head"])
                    logger.info(
                        "AutoLabelService: topic_head 로드 완료 (주제 %d종)",
                        len(self._topic_labels),
                    )
                else:
                    self._topic_head = None
                    logger.info("AutoLabelService: topic_head 없음 — 주제 None 산출")
                # 방언 권역 헤드 — heads.pt에 있을 때만 로드, 없으면 None
                if "dialect_head" in heads:
                    self._dialect_head.load_state_dict(heads["dialect_head"])
                    logger.info(
                        "AutoLabelService: dialect_head 로드 완료 (방언 %d권역)",
                        len(self._dialect_labels),
                    )
                else:
                    self._dialect_head = None
                    logger.info("AutoLabelService: dialect_head 없음 — 방언 None 산출")
            else:
                logger.warning("AutoLabelService: heads.pt 없음 — 랜덤 가중치로 초기화")

            self._emotion_head.eval()
            if self._dialog_act_head is not None:
                self._dialog_act_head.eval()
            if self._emotion_category_head is not None:
                self._emotion_category_head.eval()
            if self._topic_head is not None:
                self._topic_head.eval()
            if self._dialect_head is not None:
                self._dialect_head.eval()
            self._model_version = model_path.name
            logger.info("AutoLabelService: 로드 완료 — %s", self._model_version)

        except Exception as exc:
            logger.error("AutoLabelService: 로드 실패 (%s) — graceful degradation", exc)
            self._encoder = None
            self._tokenizer = None
            self._emotion_head = None
            self._dialog_act_head = None
            self._emotion_category_head = None
            self._topic_head = None
            self._dialect_head = None


    def _try_load_speech_age(self) -> None:
        self._sa_load_attempted = True
        model_path = _resolve_speech_age_model_path()
        if model_path is None:
            logger.info("AutoLabelService: 말투연령 모델 없음 (graceful degradation)")
            return

        encoder_dir = model_path / "encoder"
        tokenizer_dir = model_path / "tokenizer"
        heads_path = model_path / "heads.pt"
        label_map_path = model_path / "label_map.json"

        if not encoder_dir.is_dir() or not tokenizer_dir.is_dir():
            logger.warning("AutoLabelService: 말투연령 모델 디렉토리 구조 불완전 (%s)", model_path)
            return

        try:
            import torch
            import torch.nn as nn
            from transformers import AutoModel, AutoTokenizer

            logger.info("AutoLabelService: 말투연령 모델 로딩 — %s", model_path)
            self._sa_tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
            self._sa_encoder = AutoModel.from_pretrained(str(encoder_dir))
            self._sa_encoder.eval()

            if label_map_path.exists():
                lm = json.loads(label_map_path.read_text(encoding="utf-8"))
                self._sa_labels = lm.get("speech_age_labels", SPEECH_AGE_LABELS)

            hidden = self._sa_encoder.config.hidden_size
            self._sa_head = nn.Linear(hidden, len(self._sa_labels))

            if heads_path.exists():
                heads = torch.load(str(heads_path), map_location="cpu")
                self._sa_head.load_state_dict(heads["speech_age_head"])
                logger.info("AutoLabelService: 말투연령 heads.pt 로드 완료")
            else:
                logger.warning("AutoLabelService: 말투연령 heads.pt 없음 — 랜덤 가중치")

            self._sa_head.eval()
            self._sa_model_version = model_path.name
            logger.info("AutoLabelService: 말투연령 로드 완료 — %s", self._sa_model_version)

        except Exception as exc:
            logger.error("AutoLabelService: 말투연령 로드 실패 (%s) — graceful degradation", exc)
            self._sa_encoder = None
            self._sa_tokenizer = None
            self._sa_head = None


def _null_result(version: str = "") -> LabelResult:
    return LabelResult(
        emotion=None,
        emotion_confidence=0.0,
        dialog_act=None,
        dialog_act_confidence=0.0,
        model_version=version,
        emotion_category=None,
        emotion_category_confidence=0.0,
        topic_category=None,
        topic_category_confidence=0.0,
        dialect=None,
        dialect_confidence=0.0,
    )


def _null_speech_age_result(version: str = "") -> SpeechAgeResult:
    return SpeechAgeResult(
        speech_age=None,
        speech_age_confidence=0.0,
        model_version=version,
    )


# 모듈 레벨 싱글톤 — main.py lifespan에서 초기화, stt_processor에서 import
auto_label_service = AutoLabelService()
