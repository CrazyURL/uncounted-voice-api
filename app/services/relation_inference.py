"""LLM 기반 화자 관계(relation) 추론 — Ollama qwen2.5.

기존 `speaker_analysis_service._detect_relation` 의 호칭 정규식은 IT/업무 통화의
"여보세요"(전화 인사)를 "배우자"로 오탐한다. 본 모듈은 양쪽 화자 대화 전체를 LLM 에
주어 맥락 기반으로 관계를 추정한다.

안전 설계:
  - env gate `RELATION_INFER_OLLAMA_ENABLED` (기본 false). 꺼져 있으면 호출 자체를
    하지 않아 기존 동작과 100% 동일(무회귀).
  - Ollama 미응답 / 타임아웃 / 파싱 실패 / 저신뢰(conf < threshold) → None 반환 →
    호출자가 기존 호칭 정규식으로 fallback.
  - 외부 의존성(requests 등) 없이 표준 라이브러리 urllib 만 사용.

PII: transcript 를 Ollama(로컬 localhost:11434)로만 보내고, 본 모듈은 어떤
경로로도 raw 텍스트를 로깅하지 않는다(relation 레이블·confidence 만 로깅).
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

# session_speakers.speaker_relation / peers.relationship 와 정합되는 관계 레이블.
# 기존 _SALUTATION_RULES 라벨(부모/배우자/형제자매/직장상사/교사/친구) 상위집합 +
# 업무 통화에서 필요한 직장동료/거래처/고객.
RELATION_LABELS = (
    "부모", "배우자", "형제자매", "자녀", "친구",
    "직장상사", "직장동료", "거래처", "교사", "고객", "기타",
)

_PROMPT = """다음은 한국어 통화 녹취록이다. 두 화자 사이의 관계를 추정하라.

판단 기준:
- 호칭, 말투(존댓말/반말), 대화 주제, 업무/사적 맥락을 종합한다.
- "여보세요"는 전화 인사일 뿐 배우자 신호가 아니다. 실제 호칭/맥락만 본다.
- 업무·기술·고객지원 대화면 직장동료/거래처/고객/직장상사를 우선 고려한다.

가능한 관계: {labels}

녹취록:
{transcript}

JSON 한 줄로만 답하라. 형식: {{"relation": "<관계>", "confidence": <0.0~1.0>}}"""


def is_enabled() -> bool:
    return os.environ.get("RELATION_INFER_OLLAMA_ENABLED", "false").strip().lower() == "true"


def _ollama_url() -> str:
    return os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")


def _model() -> str:
    return os.environ.get("RELATION_INFER_MODEL", "qwen2.5:7b-instruct-q4_K_M")


def _threshold() -> float:
    try:
        return float(os.environ.get("RELATION_INFER_MIN_CONFIDENCE", "0.70"))
    except ValueError:
        return 0.70


def _max_chars() -> int:
    try:
        return int(os.environ.get("RELATION_INFER_MAX_CHARS", "8000"))
    except ValueError:
        return 8000


def build_dialogue(texts_by_speaker: dict[str, list[str]]) -> str:
    """화자별 텍스트 dict → "SPEAKER_xx: ..." 줄 결합 (시간순 보장 못해도 맥락 충분).

    관계 추정에는 양쪽 화자 발화가 모두 필요하므로 self/other 구분 없이 전부 포함.
    """
    lines: list[str] = []
    for spk in sorted(texts_by_speaker):
        for t in texts_by_speaker[spk]:
            t = (t or "").strip()
            if t:
                lines.append(f"{spk}: {t}")
    return "\n".join(lines)


def infer(transcript: str, *, timeout: float = 120.0) -> tuple[str, float] | None:
    """관계 추론. 반환: (relation, confidence) 또는 None(게이트 OFF/실패/저신뢰).

    호출자는 None 이면 기존 호칭 정규식으로 fallback 한다.
    """
    if not is_enabled():
        return None
    transcript = (transcript or "").strip()
    if not transcript:
        return None

    prompt = _PROMPT.format(labels=", ".join(RELATION_LABELS), transcript=transcript[: _max_chars()])
    payload = json.dumps({
        "model": _model(),
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0},
    }).encode("utf-8")
    req = urllib.request.Request(
        _ollama_url(), data=payload, headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.warning("[relation_infer] Ollama 호출 실패 — 정규식 fallback: %s", type(exc).__name__)
        return None
    except json.JSONDecodeError:
        logger.warning("[relation_infer] Ollama 응답 JSON 파싱 실패 — fallback")
        return None

    raw = body.get("response", "{}")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("[relation_infer] 모델 출력 JSON 파싱 실패 — fallback")
        return None

    relation = parsed.get("relation")
    if relation not in RELATION_LABELS:
        logger.info("[relation_infer] 비정상 라벨(%r) — fallback", relation)
        return None
    try:
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
    except (TypeError, ValueError):
        return None

    if confidence < _threshold():
        logger.info("[relation_infer] 저신뢰 conf=%.2f < %.2f — fallback", confidence, _threshold())
        return None

    logger.info("[relation_infer] relation=%s conf=%.2f", relation, confidence)
    return relation, confidence
