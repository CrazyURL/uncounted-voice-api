"""PR-B: 한국어 PII 사각지대 보강 — credential / foreign_id / payment / numeric_sensitive / korean_name 후보.

목적:
    기존 `app/pii_masker.py` 의 `detect_pii_spans` 는 한국 PII (주민번호 [1-4]
    한국인만 / 카드 / 계좌 / 전화 / IP / 이메일 / 운전면허 / 여권) 만 잡고,
    한국어 password 컨텍스트 / 외국인등록증 / 결제 컨텍스트 / 한국어 이름 후보 등
    **사각지대**가 있었다(세션 9fa79d3cbf1fb2c9 export QA 실측, 메모리
    `[[project-pii-detector-blind-spot]]`).

본 모듈은 그 사각지대를 보강한다. **기존 detect_pii_spans 시그니처/형식은
유지** — 본 모듈의 결과는 같은 dict 형식(type/char_start/char_end/matched_text)을
반환하므로 호출자(mask_pii / audio_pii_masker / pii_confidence / routers/pii.py /
worker.py build_pii_intervals) 가 변경 없이 동작한다.

설계 정본: PR-A safety preflight (`scripts/analysis/session_9fa79d3c_export_qa_diagnosis`)
PR-A 와 직교 = export-side 안전망(PR-A) + worker-side detector 보강(본 PR-B).

⚠️ candidate 성격 (보고서 명시):
    - credential_like / korean_name_like_candidate 등 일부 카테고리는
      confirmed PII 가 아닌 **후보(candidate)** 마킹이다.
    - training label / confirmed PII 출처로 자동 투입 금지.
    - 관리자 검수 후 confirmed 트랙은 별도 PR.

D4b 정책 정합:
    - 본 모듈은 detect 단계에서 후보 span 만 반환.
    - `maskType` 결정은 호출자 (`worker.py:build_pii_intervals`) 가
      `PII_INTERVAL_MASK_TYPE = "text_only"` (D4b 정책) 그대로 사용.
    - audio masking (1kHz beep) 정책 변경 0.
"""

from __future__ import annotations

import os
import re
from typing import Iterable, Mapping, Optional


# ── feature flag (worker_config 컨벤션 동형) ──────────────────────────────

_TRUE_STRS = frozenset({"true", "1", "yes", "on"})
_FALSE_STRS = frozenset({"false", "0", "no", "off"})


def _parse_bool_env(raw: Optional[str], default: bool, env_name: str) -> bool:
    if raw is None or raw == "":
        return default
    v = raw.strip().lower()
    if v in _TRUE_STRS:
        return True
    if v in _FALSE_STRS:
        return False
    raise ValueError(
        f"invalid bool env {env_name}={raw!r} "
        f"(허용: {sorted(_TRUE_STRS)} / {sorted(_FALSE_STRS)})"
    )


def resolve_pii_detector_extended_enabled(
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """``WORKER_PII_DETECTOR_EXTENDED`` env 해석.

    의미론:
      - 미설정 / true 류 → ``True`` (기본, 본 PR-B 신규 룰 활성)
      - false 류 → ``False`` (기존 PII_PATTERNS 만, 신규 룰 우회)
      - invalid → ``ValueError`` import 시점 fail-loud (WORKER_CONCURRENCY 컨벤션)

    Rollback: env 1줄 → 다음 신규 처리분부터 적용.
    """
    e = env if env is not None else os.environ
    return _parse_bool_env(
        e.get("WORKER_PII_DETECTOR_EXTENDED"),
        default=True,
        env_name="WORKER_PII_DETECTOR_EXTENDED",
    )


# ── 카테고리 라벨 (PR-A 카테고리명과 정합) ────────────────────────────────

LABEL_CREDENTIAL_LIKE = "credential_like"
LABEL_FOREIGN_ID_LIKE = "foreign_id_like"
LABEL_PAYMENT_LIKE = "payment_like"
LABEL_NUMERIC_SENSITIVE_LIKE = "numeric_sensitive_like"
LABEL_KOREAN_NAME_CANDIDATE = "korean_name_like_candidate"


# ── 키워드 / 정규식 ───────────────────────────────────────────────────────

# credential_like — 비밀번호 컨텍스트 (같은 줄 / 가까운 단어 안에 키워드 + 영문대소+숫자 토큰)
_CREDENTIAL_KEYWORDS = (
    "비밀번호", "패스워드", "암호", "로그인 정보", "계정 정보", "계정번호",
    "password", "passwd", "login", "credentials",
)

# 영문대소+숫자 혼합 4자 이상 (Korean speech 의 "비밀번호 ABC123" 류).
_CREDENTIAL_TOKEN_RE = re.compile(r"(?=[A-Za-z])(?=.*\d)[A-Za-z0-9]{4,}")

# foreign_id_like — 6자리 + 구분자 필수 + 5-8 시작 7자리 (외국인등록증/주민 외국인).
#   - 기존 PII_PATTERNS 의 주민등록번호는 두번째 자리 `[1-4]` (한국인) 만 잡음.
#   - 본 룰은 `[5-8]` (외국인) 보강. `[1-4]` 는 기존 룰에서 잡히므로 중복 제거.
_FOREIGN_ID_RE = re.compile(r"\b(\d{6})[-_\s]([5-8]\d{6})\b")

# payment_like — 결제 컨텍스트 키워드 + 인접 6+ 자리 숫자.
#   - 카드/계좌 정규식 자체는 기존 PII_PATTERNS 에 있어 중복 X.
#   - 본 룰은 **컨텍스트 키워드 + 숫자** 조합 (e.g. "송금 1234567890") 보강.
_PAYMENT_KEYWORDS = (
    "전자결제", "이체", "송금", "입금", "출금", "출국정산",
    "CVC", "CVV", "승인번호",
)
_PAYMENT_DIGIT_RE = re.compile(r"\b\d{6,16}\b")

# numeric_sensitive_like — 6+ 자리 Arabic 숫자 (소수점 부분 제외).
#   - 핵심 식별번호 / 전화 잔여 / 시리얼 번호 류 후보.
#   - 기존 룰(카드 16자리, 계좌 11-14자리, 주민 13자리, 전화) 과 겹치는 영역은
#     중복 마킹되더라도 안전 (호출자 `mask_pii` 가 non-overlapping 처리).
_NUMERIC_SENSITIVE_RE = re.compile(r"(?<![\d.])\d{6,}(?![\d.])")

# korean_name_like_candidate — 성씨 + 한글 정확히 2자 + 호칭 26종 (PR-A 와 정합).
#   - 기존 pii_masker._SURNAME_PATTERN 은 별도 컨텍스트 검사로 더 엄격
#     (enable_name_masking 옵션, NAME_EXCLUDE_PREFIX 다수).
#   - 본 룰은 export QA 사각지대 단순 candidate 마킹 — confirmed name 아님,
#     호출자가 후속 단계에서 검수 필요.
_KOREAN_SURNAMES_LIST = (
    "김", "이", "박", "최", "정", "강", "조", "윤", "장", "임",
    "한", "오", "서", "신", "권", "황", "안", "송", "류", "전",
    "홍", "고", "문", "양", "손", "배", "백", "허", "유", "남",
    "심", "노", "하", "곽", "성", "차", "주", "우", "구", "민",
)
_KOREAN_TITLES_LIST = (
    "씨", "님", "사장", "대표", "대리", "과장", "차장", "부장",
    "팀장", "실장", "이사", "상무", "전무", "부사장", "회장",
    "교수", "박사", "선생", "의원", "의사", "간호사", "변호사",
    "소장", "관장", "회계사",
)
_KOREAN_NAME_RE = re.compile(
    f"({'|'.join(re.escape(s) for s in _KOREAN_SURNAMES_LIST)})"
    "([가-힣]{2})"
    rf"\s*(?:{'|'.join(re.escape(t) for t in _KOREAN_TITLES_LIST)})"
)


# ── 본 함수: detect_extended_spans ────────────────────────────────────────

def detect_extended_spans(text: str, enable_name_masking: bool = False) -> list[dict]:
    """PR-B 신규 5 카테고리 감지.

    반환 형식 = 기존 `detect_pii_spans` 와 동일 dict 리스트:
        {"type": <라벨>, "char_start": int, "char_end": int, "matched_text": str}

    원문(matched_text)은 기존 detect_pii_spans 도 포함 — 본 모듈만 다른 정책으로
    가지 않는다. 호출자가 후속 단계에서 원문 노출 책임을 짐 (worker.py
    `build_pii_intervals` 는 matched_text 를 pii_intervals JSON 에 미포함,
    startSec/endSec/maskType/piiType 만 emit — 기존 정책 그대로).
    """
    if not isinstance(text, str) or text == "":
        return []

    spans: list[dict] = []

    # 1. credential_like — 키워드 + 영숫자 토큰 (같은 텍스트 안)
    if any(kw.lower() in text.lower() for kw in _CREDENTIAL_KEYWORDS):
        for m in _CREDENTIAL_TOKEN_RE.finditer(text):
            spans.append({
                "type": LABEL_CREDENTIAL_LIKE,
                "char_start": m.start(),
                "char_end": m.end(),
                "matched_text": m.group(0),
            })

    # 2. foreign_id_like — 정규식 단독 ([5-8] 시작 7자리, 외국인등록증)
    for m in _FOREIGN_ID_RE.finditer(text):
        spans.append({
            "type": LABEL_FOREIGN_ID_LIKE,
            "char_start": m.start(),
            "char_end": m.end(),
            "matched_text": m.group(0),
        })

    # 3. payment_like — 결제 키워드 + 6+ 자리 숫자
    if any(kw in text for kw in _PAYMENT_KEYWORDS):
        for m in _PAYMENT_DIGIT_RE.finditer(text):
            spans.append({
                "type": LABEL_PAYMENT_LIKE,
                "char_start": m.start(),
                "char_end": m.end(),
                "matched_text": m.group(0),
            })

    # 4. numeric_sensitive_like — 6+ 자리 Arabic 숫자
    for m in _NUMERIC_SENSITIVE_RE.finditer(text):
        spans.append({
            "type": LABEL_NUMERIC_SENSITIVE_LIKE,
            "char_start": m.start(),
            "char_end": m.end(),
            "matched_text": m.group(0),
        })

    # 5. korean_name_like_candidate — 성씨+2자+호칭 후보
    #    기존 pii_masker._SURNAME_PATTERN 의 premium grade 정책(enable_name_masking=False
    #    이면 이름 보존)을 동일하게 따른다 — premium 양측 개인 통화에서 이름은 정상
    #    화제(개인 흐름 보존). standard/excluded 는 호출자가 enable_name_masking=True
    #    로 전달해 와 자동 활성.
    if enable_name_masking:
        for m in _KOREAN_NAME_RE.finditer(text):
            spans.append({
                "type": LABEL_KOREAN_NAME_CANDIDATE,
                "char_start": m.start(),
                "char_end": m.end(),
                "matched_text": m.group(0),
            })

    return spans


# ── 분류 helper (호출자 메타 라벨링 용) ───────────────────────────────────

_CANDIDATE_TYPES = frozenset({
    LABEL_CREDENTIAL_LIKE,
    LABEL_KOREAN_NAME_CANDIDATE,
})

_CONFIDENCE_HIGH = frozenset({
    LABEL_CREDENTIAL_LIKE,
    LABEL_FOREIGN_ID_LIKE,
    LABEL_PAYMENT_LIKE,
})

_CONFIDENCE_MEDIUM = frozenset({
    LABEL_NUMERIC_SENSITIVE_LIKE,
    LABEL_KOREAN_NAME_CANDIDATE,
})


def is_candidate_type(label: str) -> bool:
    """본 라벨이 **후보** 성격인지 — confirmed PII 자동 학습 금지 마커."""
    return label in _CANDIDATE_TYPES


def suggested_confidence(label: str) -> str:
    """제안 confidence tier — pii_confidence 와 정합 별 합성용 hint.

    호출자(worker.py build_pii_intervals 또는 pii_confidence)가 본 값을
    저장 vs 무시 결정. 본 함수는 단순 hint.
    """
    if label in _CONFIDENCE_HIGH:
        return "high"
    if label in _CONFIDENCE_MEDIUM:
        return "medium"
    return "low"


def category_labels() -> Iterable[str]:
    """본 모듈이 정의하는 모든 라벨 목록 (테스트/문서용)."""
    return (
        LABEL_CREDENTIAL_LIKE,
        LABEL_FOREIGN_ID_LIKE,
        LABEL_PAYMENT_LIKE,
        LABEL_NUMERIC_SENSITIVE_LIKE,
        LABEL_KOREAN_NAME_CANDIDATE,
    )
