"""STAGE 15: 화자 자동 식별 + 속성 분석 서비스.

STT + 화자분리 완료 후 각 SPEAKER 레이블에 대해:
  - 발화자 식별: WeSpeaker 임베딩 cosine similarity vs. reference_embedding
  - 성별 감지: librosa F0 중앙값 기반
  - 목소리 연령: jitter·shimmer·F0 std → 10세 단위
  - 호칭어 관계 룰엔진: pre-mask 텍스트 기반
  - 말투 연령: KcELECTRA speech-age 헤드 (모델 없으면 skip)

반환: dict[speaker_label, SpeakerAnalysisResult]
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

_COSINE_THRESHOLD = float(os.environ.get("SPEAKER_EMBED_COSINE_THRESHOLD", "0.75"))

# ── 호칭어 → 관계 매핑 ─────────────────────────────────────────────────────
_SALUTATION_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"엄마|어머니|어머님|모친"), "부모"),
    (re.compile(r"아빠|아버지|아버님|부친"), "부모"),
    (re.compile(r"자기야|여보|당신|오빠야|언니야"), "배우자"),
    (re.compile(r"형|누나|언니|오빠|남동생|여동생"), "형제자매"),
    (re.compile(r"사장님|대표님|회장님"), "직장상사"),
    (re.compile(r"부장님|과장님|팀장님|차장님|실장님|본부장님"), "직장상사"),
    (re.compile(r"선생님|교수님|교수|선생"), "교사"),
    (re.compile(r"친구야|친구|야|어이"), "친구"),
]

# ── 성별 F0 기준 (Hz) ────────────────────────────────────────────────────────
_MALE_F0_MAX = 180.0
_FEMALE_F0_MIN = 165.0


@dataclass
class SpeakerAnalysisResult:
    speaker_label: str
    speaker_role: str | None = None          # 'self' | 'other'
    speaker_role_source: str | None = None   # 'profile_match' | 'heuristic'
    speaker_gender: str | None = None        # 'male' | 'female'
    speaker_voice_age_range: str | None = None  # '20대'|'30대'|'40대'|'50대+'
    speaker_speech_age_range: str | None = None
    speaker_speech_age_model_version: str | None = None
    speaker_relation: str | None = None


# ---------------------------------------------------------------------------
# 오디오 청크 추출 헬퍼
# ---------------------------------------------------------------------------

def _extract_speaker_audio(
    audio: np.ndarray,
    sample_rate: int,
    segments: list[dict],
    speaker_label: str,
) -> np.ndarray:
    """segments에서 speaker_label에 해당하는 오디오 구간을 이어붙여 반환한다."""
    chunks: list[np.ndarray] = []
    for seg in segments:
        if seg.get("speaker") != speaker_label:
            continue
        start_sample = int(seg["start"] * sample_rate)
        end_sample = int(seg["end"] * sample_rate)
        chunk = audio[start_sample:end_sample]
        if len(chunk) > 0:
            chunks.append(chunk)
    if not chunks:
        return np.array([], dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# 성별 / 목소리 연령 감지
# ---------------------------------------------------------------------------

def _detect_gender_and_voice_age(
    audio_chunk: np.ndarray,
    sample_rate: int,
) -> tuple[str | None, str | None]:
    """librosa F0 분석 → (gender, voice_age_range)."""
    if len(audio_chunk) < sample_rate * 0.5:
        return None, None
    try:
        import librosa  # type: ignore

        f0, _, _ = librosa.pyin(
            audio_chunk.astype(np.float32),
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sample_rate,
        )
        valid_f0 = f0[~np.isnan(f0)]
        if len(valid_f0) < 10:
            return None, None

        f0_median = float(np.median(valid_f0))
        f0_std = float(np.std(valid_f0))

        # 성별 판정
        if f0_median < _MALE_F0_MAX and f0_median < _FEMALE_F0_MIN:
            gender = "male"
        elif f0_median >= _FEMALE_F0_MIN:
            gender = "female"
        else:
            gender = None  # 겹치는 구간 → 미상

        # 목소리 연령 (jitter 근사: F0 표준편차 / 중앙값)
        jitter_ratio = f0_std / f0_median if f0_median > 0 else 0.0

        if gender == "male":
            if f0_median > 130 and jitter_ratio < 0.08:
                voice_age = "20대"
            elif f0_median > 110:
                voice_age = "30대"
            elif f0_median > 95:
                voice_age = "40대"
            else:
                voice_age = "50대+"
        elif gender == "female":
            if f0_median > 220 and jitter_ratio < 0.08:
                voice_age = "20대"
            elif f0_median > 195:
                voice_age = "30대"
            elif f0_median > 175:
                voice_age = "40대"
            else:
                voice_age = "50대+"
        else:
            voice_age = None

        return gender, voice_age

    except Exception as exc:
        logger.warning("F0 분석 실패: %s", exc)
        return None, None


# ---------------------------------------------------------------------------
# 호칭어 관계 감지
# ---------------------------------------------------------------------------

def _detect_relation(texts: list[str]) -> str | None:
    """pre-mask 텍스트에서 호칭어를 탐지해 관계 레이블을 반환한다."""
    combined = " ".join(texts)
    for pattern, relation in _SALUTATION_RULES:
        if pattern.search(combined):
            return relation
    return None


# ---------------------------------------------------------------------------
# 말투 연령 (KcELECTRA 4th head) — optional
# ---------------------------------------------------------------------------

def _detect_speech_age(
    texts: list[str],
) -> tuple[str | None, str | None]:
    """KcELECTRA speech-age 헤드로 말투 연령을 예측한다.

    모델이 없으면 (None, None)을 반환한다.
    """
    try:
        from app.services.auto_label_service import auto_label_service
        if not auto_label_service.is_available():
            return None, None
        results = auto_label_service.predict_speech_age(texts)
        if not results:
            return None, None
        # 다수결
        from collections import Counter
        votes = Counter(r.speech_age for r in results if r.speech_age)
        if not votes:
            return None, None
        speech_age = votes.most_common(1)[0][0]
        model_ver = results[0].model_version if results else None
        return speech_age, model_ver
    except (ImportError, AttributeError):
        return None, None


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

def analyze_speakers(
    audio: np.ndarray,
    sample_rate: int,
    segments: list[dict],
    pre_mask_texts_by_speaker: dict[str, list[str]],
    reference_embedding: list[float] | None,
    embedding_model,  # SpeakerEmbeddingModel | None
) -> dict[str, SpeakerAnalysisResult]:
    """각 speaker_label에 대해 SpeakerAnalysisResult를 계산한다.

    Args:
        audio: 전체 오디오 배열 (float32, 1D)
        sample_rate: 오디오 샘플레이트
        segments: WhisperX 출력 segments (speaker 키 포함)
        pre_mask_texts_by_speaker: PII 마스킹 전 발화자별 텍스트 목록
        reference_embedding: 사용자 voice_profile 임베딩 (L2 정규화된 float list)
        embedding_model: SpeakerEmbeddingModel 인스턴스 (None이면 임베딩 skip)

    Returns:
        {speaker_label: SpeakerAnalysisResult}
    """
    speaker_labels = sorted({
        seg.get("speaker") for seg in segments if seg.get("speaker")
    })
    if not speaker_labels:
        return {}

    ref_emb = np.array(reference_embedding, dtype=np.float32) if reference_embedding else None

    # 화자별 누적 발화 시간 (fallback heuristic용)
    speaker_duration: dict[str, float] = {}
    for seg in segments:
        lbl = seg.get("speaker")
        if lbl:
            dur = float(seg.get("end", 0)) - float(seg.get("start", 0))
            speaker_duration[lbl] = speaker_duration.get(lbl, 0.0) + dur

    # embedding cosine similarity per speaker
    speaker_similarity: dict[str, float] = {}
    if ref_emb is not None and embedding_model is not None:
        for lbl in speaker_labels:
            chunk = _extract_speaker_audio(audio, sample_rate, segments, lbl)
            if len(chunk) < sample_rate * 0.5:
                logger.debug("[speaker_analysis] %s 오디오 너무 짧음 — 임베딩 skip", lbl)
                continue
            emb = embedding_model.extract_embedding(chunk, sample_rate)
            from app.services.speaker_embedding import EmbeddingUnavailable
            if isinstance(emb, EmbeddingUnavailable):
                logger.debug("[speaker_analysis] %s 임베딩 실패: %s", lbl, emb.reason)
                continue
            sim = _cosine_similarity(ref_emb, emb)
            speaker_similarity[lbl] = sim
            logger.debug("[speaker_analysis] %s cosine_sim=%.3f", lbl, sim)

    # 발화자(self) 결정
    self_label: str | None = None
    role_source: str = "heuristic"

    if speaker_similarity:
        best_lbl = max(speaker_similarity, key=speaker_similarity.__getitem__)
        if speaker_similarity[best_lbl] >= _COSINE_THRESHOLD:
            self_label = best_lbl
            role_source = "profile_match"
            logger.info(
                "[speaker_analysis] profile_match: %s (sim=%.3f)",
                self_label, speaker_similarity[best_lbl],
            )

    if self_label is None and speaker_duration:
        self_label = max(speaker_duration, key=speaker_duration.__getitem__)
        role_source = "heuristic"
        logger.info("[speaker_analysis] heuristic self: %s (dur=%.1fs)", self_label, speaker_duration[self_label])

    results: dict[str, SpeakerAnalysisResult] = {}
    for lbl in speaker_labels:
        role = "self" if lbl == self_label else "other"
        chunk = _extract_speaker_audio(audio, sample_rate, segments, lbl)
        gender, voice_age = _detect_gender_and_voice_age(chunk, sample_rate)

        pre_texts = pre_mask_texts_by_speaker.get(lbl, [])
        relation = _detect_relation(pre_texts) if role == "other" else None
        speech_age, speech_age_ver = _detect_speech_age(pre_texts) if pre_texts else (None, None)

        results[lbl] = SpeakerAnalysisResult(
            speaker_label=lbl,
            speaker_role=role,
            speaker_role_source=role_source if lbl == self_label else "heuristic",
            speaker_gender=gender,
            speaker_voice_age_range=voice_age,
            speaker_speech_age_range=speech_age,
            speaker_speech_age_model_version=speech_age_ver,
            speaker_relation=relation,
        )

    return results
