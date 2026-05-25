"""PR-S1: 구조 PII 정규식 정밀화 + overlap/substring dedup 검증.

검증 범위 (DB/원문 무관, 합성·shape-only):
  1. IP octet 0~255 검증 (코퍼스 일반 위생용)
  2. detect_pii_spans / mask_pii 가 같은 dedup 규칙 공유
  3. 전화⊂계좌 substring 이중태깅 제거
  4. IP 시프트/겹침 중복 정리
  5. 기존 auto_confirmed 7건 shape 기준 dry-run (offset-only, 원문 미포함)

핵심 레버: IP octet 검증 '단독' 이 아니라 overlap/substring dedup 이다.
기존 7건 IP 5건은 모두 in_range → octet 검증만으로는 0건 제거. dedup 으로 IP 5→4,
전화/계좌 2→1, 전체 7→5.

안전: 실제 통화 원문/matched_text/snippet 을 일절 사용하지 않는다.
dry-run 은 dev PC 에서 추출한 '형태(shape)' 정보(offset/길이/type)만 사용한다.

tier 정책 불변: 본 테스트는 auto_confirmed/needs_human 강등을 검증/변경하지 않는다
(그 정책 변경은 PR-S2 로 분리).
"""

from __future__ import annotations

import pytest

from app.pii_masker import (
    PII_PATTERNS,
    _resolve_overlapping_spans,
    detect_pii_spans,
    mask_pii,
)

_IP_RE = next(p for p, _, label in PII_PATTERNS if label == "IP주소")


def _types(spans: list[dict]) -> list[str]:
    return [s["type"] for s in spans]


# ── 1. IP octet 0~255 검증 (코퍼스 위생용) ──────────────────────────
class TestIpOctetValidation:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("서버 192.168.0.1 에 접속", "192.168.0.1"),
            ("게이트웨이 10.0.0.255 확인", "10.0.0.255"),
            ("주소 255.255.255.0 입니다", "255.255.255.0"),
            ("로컬 127.0.0.1 루프백", "127.0.0.1"),
        ],
    )
    def test_valid_ip_detected(self, text, expected):
        spans = [s for s in detect_pii_spans(text) if s["type"] == "IP주소"]
        assert len(spans) == 1
        assert spans[0]["matched_text"] == expected

    @pytest.mark.parametrize(
        "text",
        [
            "값 300.1.2.3 임",          # octet > 255
            "코드 256.256.256.256",      # 전 octet > 255
            "수치 10.45.49.999 기록",     # 마지막 octet > 255 (버전형)
            "버전 1.2.3.4.5 릴리스",       # 5분절 버전형
            "주소 192.168.0.1000 잘못",    # 마지막 octet 4자리 (긴 숫자열)
            "ip 999.999.999.999 오류",
        ],
    )
    def test_invalid_or_version_not_detected(self, text):
        spans = [s for s in detect_pii_spans(text) if s["type"] == "IP주소"]
        assert spans == [], f"오탐 IP span: {[s['matched_text'] for s in spans]}"

    def test_octet_alone_keeps_in_range_versionlike(self):
        """in-range 4-octet 문자열(예: 10.0.0.1)은 octet 검증만으로는 안 빠진다.

        기존 auto_confirmed IP 5건이 모두 in_range 였던 사실의 근거.
        octet 검증은 코퍼스 위생용이며, 실제 7건의 레버는 dedup 이다.
        """
        for s in ["10.0.0.1", "10.45.49.20", "1.2.3.4", "8.8.8.8", "172.16.0.1"]:
            assert _IP_RE.search(s) is not None, f"in-range IP 가 탈락: {s}"


# ── IP 미탐 기준 명확화 (PR-S1 regex 층의 탐지 경계 계약) ────────────────
class TestIpVersionAmbiguityCriteria:
    """regex 단독으로 IP 와 버전번호를 구분할 수 없는 경계를 '계약'으로 고정한다.

    핵심 사실: 'in-range 4-octet 점표기 문자열'은 IP 와 버전번호가 **동일 문자열**이다.
    예) "버전 10.45.49.12" 의 10.45.49.12 는 모든 octet 0~255 인 유효 IPv4 shape 이므로
    regex 가 버전번호임을 알 방법이 없다 → PR-S1(regex/마스킹 층)은 이를 IP 로 마스킹한다.

    ── PR-S1 IP 미탐(non-detection) 기준 (형태만으로 확정 가능한 경우로 한정) ──
      [미탐 = IP 아님]  octet > 255  /  5분절 이상  /  앞뒤 숫자·점 인접(긴 숫자열)
      [탐지·마스킹됨]   모든 octet 0~255 인 정확히 4분절  (버전번호여도 마스킹됨 — 알려진 한계)

    in-range 버전번호의 오마스킹 회피는 regex 층이 아니라 **후보 큐/사람 검수(tier) 트랙**의
    책임이다(design 문서 §2 근거: "octet 검증 통과해도 in-range 오탐"). 본 클래스는 그
    경계를 회귀 테스트로 못박아, "regex 가 버전번호까지 걸러줄 것"이라는 잘못된 기대를 막는다.
    """

    @pytest.mark.parametrize(
        "text,is_masked",
        [
            # in-range 4-octet → 버전번호여도 IP 로 마스킹됨 (regex 한계, tier 트랙이 처리).
            ("버전 10.45.49.12 릴리스", True),
            ("릴리스 1.2.3.4 노트", True),
            # 아래는 형태만으로 IP 아님이 확정 → 미탐 (오마스킹 방지).
            ("빌드 10.45.49.999 배포", False),   # octet > 255
            ("버전 1.2.3.4.5 릴리스", False),     # 5분절
            ("ver 192.168.0.1000 패치", False),  # 마지막 octet 4자리(긴 숫자열)
        ],
    )
    def test_inrange_version_masked_out_of_range_not(self, text, is_masked):
        result = mask_pii(text)
        has_ip_mask = "***.***.***.***" in result["masked_text"]
        assert has_ip_mask is is_masked, (text, result["masked_text"])


# ── 2 & 3. 전화/계좌 이중태깅 제거 ──────────────────────────────────
class TestPhoneAccountDedup:
    def test_normal_phone_detected(self):
        spans = detect_pii_spans("번호 010-1234-5678 입니다")
        assert _types(spans) == ["전화번호"]

    def test_normal_account_detected(self):
        # 14자리 연속 숫자(010 미시작) → 계좌번호 단일.
        spans = detect_pii_spans("계좌 35219008123456 으로 입금")
        accts = [s for s in spans if s["type"] == "계좌번호"]
        assert len(accts) == 1
        assert "전화번호" not in _types(spans)

    def test_11digit_010_kept_as_phone(self):
        """정책 2: 정확히 11자리 010 단독은 전화+계좌 동시 매칭돼도 전화로 유지."""
        spans = detect_pii_spans("연락처 01012345678 로")
        assert _types(spans) == ["전화번호"], spans

    def test_phone_substring_inside_longer_account_dropped(self):
        """정책 1: 14자리 010 숫자열 → 앞 11자리 전화 substring 탈락, 계좌 1건."""
        spans = detect_pii_spans("입금 01012345678901 계좌")
        assert len(spans) == 1, spans
        assert spans[0]["type"] == "계좌번호"
        assert spans[0]["char_end"] - spans[0]["char_start"] == 14


# ── 4. IP 시프트/겹침 중복 정리 (shape-level) ───────────────────────
class TestOverlapResolver:
    def test_shift_overlap_collapses_to_one(self):
        """정책 3: 1글자 시프트된 IP 후보 2건 → 1건."""
        spans = [
            {"type": "IP주소", "char_start": 5, "char_end": 19},
            {"type": "IP주소", "char_start": 6, "char_end": 20},
        ]
        kept = _resolve_overlapping_spans(spans)
        assert len(kept) == 1

    def test_non_overlapping_preserved(self):
        spans = [
            {"type": "이메일", "char_start": 0, "char_end": 10},
            {"type": "전화번호", "char_start": 12, "char_end": 25},
        ]
        kept = _resolve_overlapping_spans(spans)
        assert len(kept) == 2

    def test_input_order_preserved(self):
        spans = [
            {"type": "전화번호", "char_start": 12, "char_end": 25},
            {"type": "이메일", "char_start": 0, "char_end": 10},
        ]
        kept = _resolve_overlapping_spans(spans)
        assert _types(kept) == ["전화번호", "이메일"]  # 입력 순서 보존

    def test_hints_preserved(self):
        """per-span hint(confidence 등) 키가 dedup 후에도 보존된다."""
        spans = [
            {"type": "전화번호", "char_start": 5, "char_end": 19,
             "confidence": 0.95, "high_precision_pattern": True},
            {"type": "전화번호", "char_start": 5, "char_end": 16,
             "confidence": 0.95, "high_precision_pattern": True},
        ]
        kept = _resolve_overlapping_spans(spans)
        assert len(kept) == 1
        assert kept[0]["confidence"] == 0.95
        assert kept[0]["high_precision_pattern"] is True


# ── mask_pii 동일 dedup 정책 (오마스킹 방지가 본 트랙 최우선) ─────────
class TestMaskPiiSharesPolicy:
    def test_mask_does_not_over_mask_version(self):
        """버전/긴 숫자열을 [IP] 로 오마스킹하지 않는다 (납품 transcript 보호)."""
        for text in ["빌드 1.2.3.4.5 배포", "오탐 300.1.2.3 값", "ver 192.168.0.1000"]:
            result = mask_pii(text)
            assert "***.***.***.***" not in result["masked_text"], text
            ip_items = [d for d in result["pii_detected"] if d["type"] == "IP주소"]
            assert ip_items == [], text

    def test_mask_still_masks_valid_ip(self):
        result = mask_pii("서버 192.168.0.1 접속")
        assert "***.***.***.***" in result["masked_text"]

    def test_mask_11digit_phone_single(self):
        """11자리 010: 전화 1건만 마스킹 (전화+계좌 중복 없음)."""
        result = mask_pii("연락처 01012345678 임")
        assert result["total_masked"] == 1
        assert [d["type"] for d in result["pii_detected"]] == ["전화번호"]

    def test_mask_14digit_account_single(self):
        result = mask_pii("입금 01012345678901 계좌")
        assert result["total_masked"] == 1
        assert [d["type"] for d in result["pii_detected"]] == ["계좌번호"]


# ── 5. 기존 auto_confirmed 7건 shape 기준 dry-run (offset-only) ──────
# dev PC 추출 shape (원문/matched_text 없음):
#   IP: 5건 모두 octets=4, in_range=true, dirty=false
#       그중 session e61debb9 의 2건이 offset 5-19, 6-20 (1글자 시프트 중복)
#   전화/계좌: session 18b98cca, 전화 5-16(11자리, phoneRe&acctRe), 계좌 5-19(14자리)
# dedup 은 utterance(발화) 단위로 적용된다. 7건은 세션/발화별로 나뉘므로
# 발화 그룹 단위로 모델링한다 (실제 detect_pii_spans 호출 단위와 일치).
_SHAPE_IP_SHIFT_PAIR = [
    {"type": "IP주소", "char_start": 5, "char_end": 19},   # e61debb9
    {"type": "IP주소", "char_start": 6, "char_end": 20},   # e61debb9 (시프트 중복)
]
_SHAPE_PHONE_ACCT = [
    {"type": "전화번호", "char_start": 5, "char_end": 16},  # 18b98cca 전화(11자리)
    {"type": "계좌번호", "char_start": 5, "char_end": 19},  # 18b98cca 계좌(14자리, 전화 포함)
]
# 비중첩 standalone IP 3건 — 각자 다른 발화. (offset 은 임의, 같은 발화 아님)
_SHAPE_IP_STANDALONE_GROUPS = [
    [{"type": "IP주소", "char_start": 3, "char_end": 16}],
    [{"type": "IP주소", "char_start": 0, "char_end": 13}],
    [{"type": "IP주소", "char_start": 8, "char_end": 21}],
]
# 7건 발화 그룹 전체.
_SHAPE_SEVEN_GROUPS = (
    [_SHAPE_IP_SHIFT_PAIR]
    + _SHAPE_IP_STANDALONE_GROUPS
    + [_SHAPE_PHONE_ACCT]
)


class TestExistingSevenDryRun:
    def test_ip_shift_pair_2_to_1(self):
        kept = _resolve_overlapping_spans(_SHAPE_IP_SHIFT_PAIR)
        assert len(kept) == 1  # 2 → 1

    def test_phone_account_2_to_1(self):
        kept = _resolve_overlapping_spans(_SHAPE_PHONE_ACCT)
        assert len(kept) == 1  # 2 → 1 (substring 전화 탈락, 계좌 1건)
        assert kept[0]["type"] == "계좌번호"

    def test_octet_validation_removes_zero_of_five(self):
        """IP 5건이 모두 in_range → octet 검증만으로 제거되는 건 0건."""
        in_range_samples = ["10.0.0.1", "10.45.49.20", "192.168.1.1",
                            "172.16.254.1", "203.0.113.5"]
        survived = [s for s in in_range_samples if _IP_RE.search(s)]
        assert len(survived) == 5  # octet 검증으로 0건 제거

    def test_seven_candidate_total_after_dedup(self):
        """7건 발화별 shape dry-run: IP 5→4, 전화/계좌 2→1 ⇒ 전체 7→5."""
        before = sum(len(g) for g in _SHAPE_SEVEN_GROUPS)
        assert before == 7

        kept_groups = [_resolve_overlapping_spans(g) for g in _SHAPE_SEVEN_GROUPS]
        kept_total = sum(len(g) for g in kept_groups)

        ip_kept = sum(
            len([s for s in g if s["type"] == "IP주소"]) for g in kept_groups
        )
        pa_kept = sum(
            len([s for s in g if s["type"] in ("전화번호", "계좌번호")])
            for g in kept_groups
        )
        assert ip_kept == 4   # 5 → 4 (시프트 1쌍 제거)
        assert pa_kept == 1   # 2 → 1 (전화 substring 탈락)
        assert kept_total == 5  # 7 → 5
