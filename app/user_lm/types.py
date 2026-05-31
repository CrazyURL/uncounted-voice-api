"""R2 user LM types — frozen dataclasses, pure data only."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ConfusionPair:
    """사용자 confirm 한 1:1 단어/구문 정정 쌍.

    user_id 격리 보장 — rescoring helper 가 user_id 불일치 시 거부.
    """
    from_word: str
    to_word: str
    user_id: str
    confirm_count: int = 0
    reject_count: int = 0
    contexts: tuple[str, ...] = ()
    last_used_at: Optional[float] = None
    weight: float = 1.0


@dataclass(frozen=True)
class PhraseMemoryEntry:
    """사용자별 phrase memory entry (도메인 어휘 boost)."""
    surface: str
    weight: float
    user_id: str
    last_used_at: Optional[float] = None


@dataclass(frozen=True)
class PiiInterval:
    """voice-api detect_pii_spans 출력 구간."""
    start: float
    end: float
    type: str = "ALL"


@dataclass(frozen=True)
class WordCandidate:
    """WhisperX word-level 출력."""
    word: str
    start: float
    end: float
    score: float = 1.0


@dataclass(frozen=True)
class ConfusionScore:
    """confusion pair scoring 결과."""
    score_delta: float
    confidence_delta: float
    reason: str
    applied: bool
    needs_review: bool


@dataclass(frozen=True)
class RescoringInput:
    """rescore_transcript 의 단일 진입 구조체."""
    user_id: str
    text: str
    words: tuple[WordCandidate, ...] = ()
    confusion_pairs: tuple[ConfusionPair, ...] = ()
    phrase_memory: tuple[PhraseMemoryEntry, ...] = ()
    pii_intervals: tuple[PiiInterval, ...] = ()
    whisper_score: float = 0.8
    global_lm_score: float = 0.5
    alpha: float = 0.3
    beta: float = 0.2
    lm_version: Optional[str] = None


@dataclass(frozen=True)
class RescoringOutput:
    """rescore_transcript 출력 + provenance."""
    corrected_text: str
    applied_rules: tuple[dict, ...]
    final_score: float
    margin: float
    confidence_delta: float
    safe_to_auto_apply: bool
    needs_review: bool
    needs_review_reason: Optional[str] = None
    source: str = "user_lm"
    lm_version: Optional[str] = None
