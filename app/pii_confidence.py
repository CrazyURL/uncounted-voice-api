"""PII confidence tier 합성 (PII-1A 부트스트랩 휴리스틱).

detect_pii_spans 가 반환한 span(type/offset/matched_text) 을 받아
confidence / high_precision_pattern / confidence_tier 를 합성한다.

⚠️ 이것은 ML 확률이 아니라 **정규식 pattern class 기반 부트스트랩 휴리스틱**이다.
high_precision 패턴(주민/전화/이메일/카드/계좌/여권/IP)은 상수 confidence 를 배정하고,
이름(ambiguous)은 사람 판단 큐로 보낸다. PII-5 에서 실제 모델 confidence 로 대체한다.

안전 계약: score_candidates 출력에는 matched_text(원문 span) 가 **절대 포함되지 않는다**.
offset(char_start/char_end)·type·confidence·tier·high_precision 만 노출한다.
"""

from __future__ import annotations

# 명확한 정규식 패턴으로 잡히는 고정밀 유형 → 큐에 보내지 않는 자동 확정 후보.
HIGH_PRECISION_TYPES: frozenset[str] = frozenset({
    "주민등록번호",
    "운전면허번호",
    "여권번호",
    "카드번호",
    "이메일",
    "전화번호",
    "계좌번호",
    "IP주소",
})

# 유형 자체가 애매해 사람 판단이 필요한 유형(오탐 가능성 높음).
AMBIGUOUS_TYPES: frozenset[str] = frozenset({"이름"})

# 부트스트랩 confidence 상수 (ML 아님).
_CONFIDENCE_HIGH_PRECISION = 0.95
_CONFIDENCE_AMBIGUOUS = 0.70
_CONFIDENCE_WEAK = 0.40

# tier 임계값.
_AUTO_CONFIRMED_MIN = 0.90
_NEEDS_HUMAN_MIN = 0.50

# 출력 키 화이트리스트 — matched_text 누출 방지의 단일 지점.
_OUTPUT_KEYS = (
    "type",
    "char_start",
    "char_end",
    "confidence",
    "high_precision_pattern",
    "confidence_tier",
)


def classify_tier(
    confidence: float,
    high_precision_pattern: bool,
    type_ambiguous: bool = False,
) -> str:
    """confidence/패턴/애매성으로 confidence_tier 를 결정한다 (total function).

    - auto_confirmed: confidence ≥ 0.90 AND high_precision_pattern
    - auto_rejected:  confidence < 0.50 AND NOT high_precision_pattern (애매유형 제외)
    - needs_human_decision: 그 외 전부 (애매유형은 항상 여기)
    """
    if confidence >= _AUTO_CONFIRMED_MIN and high_precision_pattern:
        return "auto_confirmed"
    if type_ambiguous:
        return "needs_human_decision"
    if confidence < _NEEDS_HUMAN_MIN and not high_precision_pattern:
        return "auto_rejected"
    return "needs_human_decision"


def _score_one(pii_type: str) -> dict:
    high_precision = pii_type in HIGH_PRECISION_TYPES
    ambiguous = pii_type in AMBIGUOUS_TYPES
    if high_precision:
        confidence = _CONFIDENCE_HIGH_PRECISION
    elif ambiguous:
        confidence = _CONFIDENCE_AMBIGUOUS
    else:
        confidence = _CONFIDENCE_WEAK
    return {
        "confidence": confidence,
        "high_precision_pattern": high_precision,
        "confidence_tier": classify_tier(confidence, high_precision, ambiguous),
    }


def score_candidates(spans: list[dict]) -> list[dict]:
    """detect_pii_spans 결과 → tier 합성 후보 리스트.

    입력 span 의 matched_text 는 **버린다**. char_start/char_end 오프셋만 보존한다.
    """
    out: list[dict] = []
    for span in spans:
        scored = _score_one(span["type"])
        candidate = {
            "type": span["type"],
            "char_start": span["char_start"],
            "char_end": span["char_end"],
            **scored,
        }
        # 화이트리스트 강제 — matched_text 등 비허용 키 제거.
        out.append({k: candidate[k] for k in _OUTPUT_KEYS})
    return out
