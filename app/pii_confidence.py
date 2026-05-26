"""PII confidence tier 합성 (PII-1A 부트스트랩 휴리스틱).

detect_pii_spans 가 반환한 span(type/offset/matched_text) 을 받아
confidence / high_precision_pattern / confidence_tier 를 합성한다.

⚠️ 이것은 ML 확률이 아니라 **정규식 pattern class 기반 부트스트랩 휴리스틱**이다.
PII-5 에서 실제 모델 confidence 로 대체한다.

── tier 정책 (PR-S2) ────────────────────────────────────────────────
패턴이 "구체적"인 것과 "오탐이 낮은" 것은 다르다. 구조적 패턴이라도 컨텍스트
오탐 위험이 큰 유형은 사람 검수 큐로 보낸다.

- AUTO_CONFIRM (rigid): 주민/운전면허/여권/카드/이메일 — 구분자·구조가 강해 오탐 낮음.
- REVIEW_REQUIRED: IP주소/계좌번호/전화번호 — 구조적이나 컨텍스트 오탐 위험.
    · IP주소: octet 0~255 통과해도 버전/빌드번호 in-range 오탐 가능.
    · 계좌번호: \\d{11,14} 연속 — 주문/송장 등 일반 긴 숫자열과 구분 불가(가장 loose).
    · 전화번호: 기본 검수. 단 detect 단계에서 구분자(하이픈/공백/점) 형식이면
      high_precision_pattern=True hint 를 받아 auto_confirm 된다(붙여쓰기 raw 숫자열은 검수).
      음성 전사형(한글 숫자어)은 강한 discriminator hint 로 auto_confirm 유지.
- AMBIGUOUS: 이름 — 항상 사람 판단.

주의: 이 tier 는 **후보 검수 큐(detect-batch/admin)** 에만 영향한다. 납품 마스킹
(mask_pii)은 tier 를 보지 않으므로 본 정책 변경에 영향받지 않는다.

안전 계약: score_candidates 출력에는 matched_text(원문 span) 가 **절대 포함되지 않는다**.
offset(char_start/char_end)·type·confidence·tier·high_precision 만 노출한다.
"""

from __future__ import annotations

# rigid: 구분자·구조가 강해 컨텍스트 오탐이 낮음 → 사람 검수 없이 자동 확정 가능.
AUTO_CONFIRM_TYPES: frozenset[str] = frozenset({
    "주민등록번호",
    "운전면허번호",
    "여권번호",
    "카드번호",
    "이메일",
})

# 구조적이나 컨텍스트 오탐 위험이 커 사람 검수 큐로 보내는 유형.
# 전화번호는 기본 검수이되, 구분자 형식 span 은 high_precision_pattern hint 로 auto_confirm.
REVIEW_REQUIRED_TYPES: frozenset[str] = frozenset({
    "IP주소",
    "계좌번호",
    "전화번호",
})

# 하위호환 별칭 (외부 import 대비). 자동 확정 대상 유형 집합.
HIGH_PRECISION_TYPES: frozenset[str] = AUTO_CONFIRM_TYPES

# 유형 자체가 애매해 사람 판단이 필요한 유형(오탐 가능성 높음).
AMBIGUOUS_TYPES: frozenset[str] = frozenset({"이름"})

# 부트스트랩 confidence 상수 (ML 아님).
_CONFIDENCE_HIGH_PRECISION = 0.95
_CONFIDENCE_AMBIGUOUS = 0.70
# 검수 필요 유형 기본 confidence — auto_rejected(<0.50) 를 피해 needs_human 으로 보낸다.
_CONFIDENCE_REVIEW = 0.70
_CONFIDENCE_WEAK = 0.40

# 이름 graded confidence: detect_pii_spans 가 부착한 name_context 범주 → confidence.
# 목적은 마스킹(탐지)은 유지하되 **검수 큐 우선순위만** 차등화하는 것이다.
#   - honorific(0.85)/mid(0.70): ambiguous 유지 → needs_human_decision (큐 유지, confidence 만 차등)
#   - weak_trailing(0.40):       ambiguous 해제 → <0.50 → auto_rejected (사람 큐에서 이탈)
# 어떤 경우에도 span 은 detect_pii_spans 가 그대로 emit 하므로 mask_pii recall 은 불변이다.
_NAME_CONTEXT_CONFIDENCE: dict[str, float] = {
    "honorific": 0.85,
    "mid": 0.70,
    "weak_trailing": 0.40,
}

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


def _score_one(span: dict) -> dict:
    """span 의 type 기본값을 쓰되, span 이 per-span hint 를 주면 그것을 우선한다.

    hint(있을 때만): confidence / high_precision_pattern / type_ambiguous.
    음성 전사형 등 같은 type 안에서도 정밀도가 다른 후보를 보수적으로 분류하기 위함.
    hint 가 없는 기존 패턴은 종전과 동일하게 type 기반으로 합성된다(하위호환).
    """
    pii_type = span["type"]
    # 이름 graded confidence 문맥 범주(있으면). 탐지/마스킹과 무관, 검수 tier 강등용.
    name_context = span.get("name_context")
    high_precision = span.get("high_precision_pattern")
    if high_precision is None:
        high_precision = pii_type in AUTO_CONFIRM_TYPES
    ambiguous = span.get("type_ambiguous")
    if ambiguous is None:
        if name_context is not None:
            # weak_trailing 은 약한 후보 → ambiguous 해제해 confidence 기반 auto_rejected 를 허용.
            # honorific/mid 는 ambiguous 유지 → needs_human_decision.
            ambiguous = name_context != "weak_trailing"
        else:
            ambiguous = pii_type in AMBIGUOUS_TYPES
    confidence = span.get("confidence")
    if confidence is None:
        if name_context is not None:
            confidence = _NAME_CONTEXT_CONFIDENCE.get(name_context, _CONFIDENCE_AMBIGUOUS)
        elif high_precision:
            confidence = _CONFIDENCE_HIGH_PRECISION
        elif ambiguous:
            confidence = _CONFIDENCE_AMBIGUOUS
        elif pii_type in REVIEW_REQUIRED_TYPES:
            # 구조적이나 검수 필요 → needs_human (auto_rejected 방지).
            confidence = _CONFIDENCE_REVIEW
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
        scored = _score_one(span)
        candidate = {
            "type": span["type"],
            "char_start": span["char_start"],
            "char_end": span["char_end"],
            **scored,
        }
        # 화이트리스트 강제 — matched_text 등 비허용 키 제거.
        out.append({k: candidate[k] for k in _OUTPUT_KEYS})
    return out
