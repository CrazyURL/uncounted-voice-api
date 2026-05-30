"""PR-B2 — Extended PII detector → find_pii_word_ranges → time_ranges → build_pii_intervals 배선.

배경:
  PR-B (#18) 가 detect_pii_spans 안에 5종 extended 룰(credential_like / foreign_id_like /
  payment_like / numeric_sensitive_like / korean_name_like_candidate) 을 통합했으나,
  운영 canary (9fa79d3c, 6e682a45) 의 pii_intervals 는 여전히 0. 단절점 =
  app/stt_processor.py 의 join 단계 (mask_segments → pii_summary 와
  find_pii_word_ranges → type_to_ranges 의 join) 가 mask_pii pattern_order 에
  미정의 type 을 silently drop.

본 테스트는 다음을 검증:
  1. find_pii_word_ranges 가 PR-B 룰을 word timestamp 에 매핑해 time_range emit.
  2. find_pii_word_ranges 가 같은 (start, end, type) 튜플을 중복 emit 하지 않음.
  3. stt_processor join 시뮬레이션이 extended type 의 pii_summary 항목을
     신규 추가 (PR-B2 단절점 해소).
  4. build_pii_intervals 가 extended type 을 piiType 으로 그대로 emit.
  5. D4b text_only 정책 보존 (maskType 변경 0).
  6. enable_name_masking=False 시 korean_name_like_candidate range 미생성.
  7. 9fa79d3c 동등 패턴: direct detector + find_pii_word_ranges 양쪽 hit, count/type 만 검증
     (원문/민감 substring 출력 금지).
  8. 기존 PII (전화/주민/카드/IP) range 회귀 0.
  9. mask_pii / mask_segments contract 변경 0 (외부 API schema 호환).

원문 노출 금지: 본 테스트 자체에 실 transcript 미포함, 정규식 의도 표현만.
"""

import os

# worker import 용 더미 env (build_pii_intervals 만 단위 호출).
os.environ.setdefault("SUPABASE_URL", "http://localhost")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "dummy")
os.environ.setdefault("S3_ENDPOINT_URL", "http://localhost")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "dummy")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "dummy")

from app import worker  # noqa: E402
from app.pii_extended import (  # noqa: E402
    LABEL_CREDENTIAL_LIKE,
    LABEL_FOREIGN_ID_LIKE,
    LABEL_KOREAN_NAME_CANDIDATE,
    LABEL_NUMERIC_SENSITIVE_LIKE,
    LABEL_PAYMENT_LIKE,
    is_candidate_type,
)
from app.pii_masker import detect_pii_spans, mask_segments  # noqa: E402
from app.services.audio_pii_masker import find_pii_word_ranges  # noqa: E402


# ── helper: stt_processor.py L713-743 join 단계의 동등 재현 ─────────────────
# 본 함수는 stt_processor 의 본 분기를 단위 테스트 가능 표면으로 추출한 시뮬레이션.
# 운영 코드와 동일 알고리즘 = PR-B2 단절점 해소 분기 포함.

def _join_pii_summary_with_audio_ranges(pii_summary, pii_audio_ranges, pad_round=2):
    if not pii_audio_ranges:
        return list(pii_summary)
    type_to_ranges: dict[str, list[dict]] = {}
    for r_start, r_end, p_type in pii_audio_ranges:
        type_to_ranges.setdefault(p_type, []).append(
            {"start": round(r_start, pad_round), "end": round(r_end, pad_round)}
        )
    existing_types = {item["type"] for item in pii_summary}
    out = [
        {**item, "time_ranges": type_to_ranges[item["type"]]}
        if item["type"] in type_to_ranges
        else {**item}
        for item in pii_summary
    ]
    # PR-B2: pii_summary 에 미존재하는 type 도 신규 항목.
    for p_type, ranges in type_to_ranges.items():
        if p_type not in existing_types:
            out.append({"type": p_type, "count": len(ranges), "time_ranges": ranges})
    return out


# ── helper: 가짜 word-level segment fixture ─────────────────────────────────
# WhisperX align 결과와 동등 형식 — text 와 words(start/end/word) 만 사용.

def _make_segment(text: str, words: list[tuple[str, float, float]]) -> dict:
    return {
        "text": text,
        "words": [{"word": w, "start": s, "end": e} for w, s, e in words],
    }


# ── 1. find_pii_word_ranges 가 extended span 의 time_range 를 emit ─────────

class TestFindPiiWordRangesExtended:
    def test_credential_span_to_time_range(self):
        # "비밀번호 Abc1234" — keyword + 영숫자 토큰
        seg = _make_segment(
            "비밀번호 Abc1234",
            [("비밀번호", 0.0, 0.5), ("Abc1234", 0.6, 1.0)],
        )
        ranges = find_pii_word_ranges([seg], enable_name_masking=False)
        types = [t for _, _, t in ranges]
        assert LABEL_CREDENTIAL_LIKE in types

    def test_foreign_id_span_to_time_range(self):
        # 외국인등록증: 6자리 + 하이픈 + [5-8] 시작 7자리.
        seg = _make_segment(
            "외국인등록증 850515-5876543",
            [("외국인등록증", 0.0, 0.8), ("850515-5876543", 0.9, 1.8)],
        )
        ranges = find_pii_word_ranges([seg], enable_name_masking=False)
        types = [t for _, _, t in ranges]
        assert LABEL_FOREIGN_ID_LIKE in types

    def test_payment_span_to_time_range(self):
        # "이체" 키워드 + 6+ 자리 숫자.
        seg = _make_segment(
            "이체 1234567",
            [("이체", 0.0, 0.3), ("1234567", 0.4, 1.0)],
        )
        ranges = find_pii_word_ranges([seg], enable_name_masking=False)
        types = [t for _, _, t in ranges]
        assert LABEL_PAYMENT_LIKE in types

    def test_numeric_sensitive_span_to_time_range(self):
        # 6+ 자리 Arabic 숫자 (키워드 무관).
        seg = _make_segment(
            "코드 987654",
            [("코드", 0.0, 0.3), ("987654", 0.4, 1.0)],
        )
        ranges = find_pii_word_ranges([seg], enable_name_masking=False)
        types = [t for _, _, t in ranges]
        assert LABEL_NUMERIC_SENSITIVE_LIKE in types

    def test_korean_name_candidate_with_name_masking_true(self):
        # enable_name_masking=True 일 때만 candidate range 생성.
        # 성씨 + 한글 2자 + 호칭 (씨).
        seg = _make_segment(
            "김민수 씨께",
            [("김민수", 0.0, 0.5), ("씨께", 0.6, 0.9)],
        )
        ranges = find_pii_word_ranges([seg], enable_name_masking=True)
        types = [t for _, _, t in ranges]
        assert LABEL_KOREAN_NAME_CANDIDATE in types

    def test_korean_name_candidate_with_name_masking_false_no_range(self):
        # enable_name_masking=False → detect_extended_spans 가 candidate emit 안함
        # → time_range 도 안 생김.
        seg = _make_segment(
            "김민수 씨께",
            [("김민수", 0.0, 0.5), ("씨께", 0.6, 0.9)],
        )
        ranges = find_pii_word_ranges([seg], enable_name_masking=False)
        types = [t for _, _, t in ranges]
        assert LABEL_KOREAN_NAME_CANDIDATE not in types


# ── 2. find_pii_word_ranges dedup ───────────────────────────────────────────

class TestFindPiiWordRangesDedup:
    def test_no_duplicate_tuples_for_same_word(self):
        # 6+ 자리 숫자 + payment 키워드 동시 매칭 → 동일 word 매핑 시
        # (start, end, type) 동일 튜플 중복 가능. dedup 으로 1건만.
        seg = _make_segment(
            "이체 9876543",  # payment_like + numeric_sensitive_like 둘 다 7자리 매칭
            [("이체", 0.0, 0.3), ("9876543", 0.4, 1.2)],
        )
        ranges = find_pii_word_ranges([seg], enable_name_masking=False)
        # 같은 (start, end, type) 가 두 번 들어가지 않는다.
        keys = [(round(s, 4), round(e, 4), t) for s, e, t in ranges]
        assert len(keys) == len(set(keys))

    def test_different_types_overlap_kept(self):
        # 같은 word 영역이 다른 type 으로 매핑되는 건 dedup 대상 아님
        # (build_pii_intervals 가 piiType 별 별도 행 emit).
        seg = _make_segment(
            "이체 9876543",
            [("이체", 0.0, 0.3), ("9876543", 0.4, 1.2)],
        )
        ranges = find_pii_word_ranges([seg], enable_name_masking=False)
        types = {t for _, _, t in ranges}
        # payment_like 와 numeric_sensitive_like 모두 emit.
        assert LABEL_PAYMENT_LIKE in types
        assert LABEL_NUMERIC_SENSITIVE_LIKE in types


# ── 3. stt_processor join — PR-B2 단절점 해소 ───────────────────────────────

class TestStttProcessorJoinExtended:
    def test_extended_type_added_to_pii_summary_when_missing(self):
        # mask_segments 의 pii_summary 에 extended type 항목이 없는 상태로 시작.
        segments = [_make_segment(
            "비밀번호 Abc1234",
            [("비밀번호", 0.0, 0.5), ("Abc1234", 0.6, 1.0)],
        )]
        # mask_segments 는 PII_PATTERNS + "이름" 만 emit → extended type 미존재.
        pii_summary_before = mask_segments(
            [dict(s) for s in segments],  # 새 dict (in-place 보호)
            enable_name_masking=False,
        )
        existing_types = {item["type"] for item in pii_summary_before}
        assert LABEL_CREDENTIAL_LIKE not in existing_types  # 단절점 재현

        pii_audio_ranges = find_pii_word_ranges(segments, enable_name_masking=False)
        joined = _join_pii_summary_with_audio_ranges(pii_summary_before, pii_audio_ranges)

        # PR-B2 분기로 신규 추가.
        joined_types = {item["type"] for item in joined}
        assert LABEL_CREDENTIAL_LIKE in joined_types
        cred_item = next(item for item in joined if item["type"] == LABEL_CREDENTIAL_LIKE)
        assert cred_item["count"] >= 1
        assert "time_ranges" in cred_item
        assert len(cred_item["time_ranges"]) >= 1

    def test_existing_pii_summary_items_get_time_ranges(self):
        # 기존 type (전화번호) 은 join 으로 time_ranges 부착 (기존 동작 회귀 0).
        segments = [_make_segment(
            "전화 010-1234-5678",
            [("전화", 0.0, 0.3), ("010-1234-5678", 0.4, 1.5)],
        )]
        pii_summary_before = mask_segments(
            [dict(s) for s in segments],
            enable_name_masking=False,
        )
        pii_audio_ranges = find_pii_word_ranges(segments, enable_name_masking=False)
        joined = _join_pii_summary_with_audio_ranges(pii_summary_before, pii_audio_ranges)

        phone = next(item for item in joined if item["type"] == "전화번호")
        assert "time_ranges" in phone
        assert len(phone["time_ranges"]) >= 1

    def test_extended_count_equals_time_range_count(self):
        # PR-B2 신규 항목의 count = time_ranges 길이.
        segments = [_make_segment(
            "비밀번호 Abc1234 Def5678",
            [("비밀번호", 0.0, 0.5), ("Abc1234", 0.6, 1.0), ("Def5678", 1.1, 1.5)],
        )]
        pii_audio_ranges = find_pii_word_ranges(segments, enable_name_masking=False)
        joined = _join_pii_summary_with_audio_ranges([], pii_audio_ranges)
        cred = next(item for item in joined if item["type"] == LABEL_CREDENTIAL_LIKE)
        assert cred["count"] == len(cred["time_ranges"])

    def test_pii_summary_without_audio_ranges_unchanged(self):
        # pii_audio_ranges 가 빈 리스트면 pii_summary 변경 0.
        segments = [_make_segment(
            "전화 010-1234-5678",
            [("전화", 0.0, 0.3), ("010-1234-5678", 0.4, 1.5)],
        )]
        pii_summary_before = mask_segments(
            [dict(s) for s in segments],
            enable_name_masking=False,
        )
        joined = _join_pii_summary_with_audio_ranges(pii_summary_before, [])
        assert joined == pii_summary_before


# ── 4. build_pii_intervals 통합 — extended piiType 흐름 ─────────────────────

class TestBuildPiiIntervalsWithExtended:
    def test_credential_like_emitted_as_pii_interval(self):
        # joined pii_summary 를 worker.build_pii_intervals 입력으로.
        pii_summary = [
            {
                "type": LABEL_CREDENTIAL_LIKE,
                "count": 1,
                "time_ranges": [{"start": 0.6, "end": 1.0}],
            },
        ]
        intervals = worker.build_pii_intervals(
            pii_summary, utt_start=0.0, utt_end=2.0,
        )
        assert len(intervals) == 1
        item = intervals[0]
        assert item["piiType"] == LABEL_CREDENTIAL_LIKE
        # D4b text_only 정책 보존.
        assert item["maskType"] == "text_only"
        # 기존 contract: 4키만, 원문 미포함.
        assert set(item.keys()) == {"startSec", "endSec", "maskType", "piiType"}

    def test_korean_name_candidate_emitted_as_pii_interval(self):
        pii_summary = [
            {
                "type": LABEL_KOREAN_NAME_CANDIDATE,
                "count": 1,
                "time_ranges": [{"start": 0.0, "end": 0.9}],
            },
        ]
        intervals = worker.build_pii_intervals(
            pii_summary, utt_start=0.0, utt_end=2.0,
        )
        assert len(intervals) == 1
        # piiType 자체에 `_candidate` suffix → 호출자가 is_candidate_type 으로 판별.
        assert intervals[0]["piiType"] == LABEL_KOREAN_NAME_CANDIDATE
        assert is_candidate_type(intervals[0]["piiType"]) is True

    def test_non_candidate_extended_types_marked_confirmed(self):
        # foreign_id / payment / numeric_sensitive 는 candidate 아님.
        for label in (LABEL_FOREIGN_ID_LIKE, LABEL_PAYMENT_LIKE, LABEL_NUMERIC_SENSITIVE_LIKE):
            assert is_candidate_type(label) is False

    def test_d4b_mask_type_unchanged_for_extended(self):
        # 모든 piiType 에 대해 maskType="text_only" 유지 (D4b 정책 보존).
        pii_summary = [
            {"type": LABEL_FOREIGN_ID_LIKE, "time_ranges": [{"start": 0.0, "end": 1.0}]},
            {"type": LABEL_PAYMENT_LIKE, "time_ranges": [{"start": 1.0, "end": 2.0}]},
            {"type": LABEL_NUMERIC_SENSITIVE_LIKE, "time_ranges": [{"start": 2.0, "end": 3.0}]},
            {"type": LABEL_CREDENTIAL_LIKE, "time_ranges": [{"start": 3.0, "end": 4.0}]},
            {"type": LABEL_KOREAN_NAME_CANDIDATE, "time_ranges": [{"start": 4.0, "end": 5.0}]},
        ]
        intervals = worker.build_pii_intervals(
            pii_summary, utt_start=0.0, utt_end=10.0,
        )
        assert len(intervals) == 5
        for item in intervals:
            assert item["maskType"] == "text_only"


# ── 5. 9fa79d3c 동등 fixture 회귀 ─────────────────────────────────────────
# 원문/민감 substring 노출 금지. piiType/count 만 검증.

class TestSession9fa79d3cRegressionFixture:
    def test_direct_detector_and_wiring_both_hit(self):
        # 5 카테고리 동등 표현 — credential / foreign_id / payment / numeric / korean_name.
        # 각 카테고리 1건씩 매칭되는 동등 패턴.
        seg = _make_segment(
            "비밀번호 Abc1234 외국인등록증 850515-5876543 이체 1234567 코드 987654 김민수 씨께",
            [
                ("비밀번호", 0.0, 0.5),
                ("Abc1234", 0.6, 1.0),
                ("외국인등록증", 1.1, 1.7),
                ("850515-5876543", 1.8, 2.6),
                ("이체", 2.7, 3.0),
                ("1234567", 3.1, 3.7),
                ("코드", 3.8, 4.1),
                ("987654", 4.2, 4.8),
                ("김민수", 4.9, 5.3),
                ("씨께", 5.4, 5.7),
            ],
        )

        # 1) direct detector hit > 0.
        spans = detect_pii_spans(seg["text"], enable_name_masking=True)
        types = {s["type"] for s in spans}
        assert len(types) > 0
        # 5 카테고리 모두 표지 — piiType set 으로만 검증 (원문 미포함).
        assert LABEL_CREDENTIAL_LIKE in types
        assert LABEL_FOREIGN_ID_LIKE in types
        assert LABEL_PAYMENT_LIKE in types
        assert LABEL_NUMERIC_SENSITIVE_LIKE in types
        assert LABEL_KOREAN_NAME_CANDIDATE in types

        # 2) find_pii_word_ranges hit > 0 — 각 카테고리 time_range 한 건 이상.
        ranges = find_pii_word_ranges([seg], enable_name_masking=True)
        range_types = {t for _, _, t in ranges}
        assert LABEL_CREDENTIAL_LIKE in range_types
        assert LABEL_FOREIGN_ID_LIKE in range_types
        assert LABEL_PAYMENT_LIKE in range_types
        assert LABEL_NUMERIC_SENSITIVE_LIKE in range_types
        assert LABEL_KOREAN_NAME_CANDIDATE in range_types

        # 3) join → build_pii_intervals → piiType emit (count 만 검증).
        pii_summary_before = mask_segments(
            [dict(seg)], enable_name_masking=True,
        )
        joined = _join_pii_summary_with_audio_ranges(pii_summary_before, ranges)
        intervals = worker.build_pii_intervals(joined, utt_start=0.0, utt_end=10.0)
        pii_types_in_intervals = {item["piiType"] for item in intervals}
        assert LABEL_CREDENTIAL_LIKE in pii_types_in_intervals
        assert LABEL_FOREIGN_ID_LIKE in pii_types_in_intervals
        assert LABEL_PAYMENT_LIKE in pii_types_in_intervals
        assert LABEL_NUMERIC_SENSITIVE_LIKE in pii_types_in_intervals
        assert LABEL_KOREAN_NAME_CANDIDATE in pii_types_in_intervals


# ── 6. 기존 PII 회귀 — 주민/전화/카드/IP range emit 정상 ────────────────────

class TestExistingPiiRegression:
    def test_phone_number_range_emit(self):
        seg = _make_segment(
            "전화 010-1234-5678",
            [("전화", 0.0, 0.3), ("010-1234-5678", 0.4, 1.5)],
        )
        ranges = find_pii_word_ranges([seg], enable_name_masking=False)
        types = [t for _, _, t in ranges]
        assert "전화번호" in types

    def test_resident_number_range_emit(self):
        seg = _make_segment(
            "주민 900101-1234567",
            [("주민", 0.0, 0.3), ("900101-1234567", 0.4, 1.5)],
        )
        ranges = find_pii_word_ranges([seg], enable_name_masking=False)
        types = [t for _, _, t in ranges]
        assert "주민등록번호" in types

    def test_card_number_range_emit(self):
        seg = _make_segment(
            "카드 1234-5678-9012-3456",
            [("카드", 0.0, 0.3), ("1234-5678-9012-3456", 0.4, 1.8)],
        )
        ranges = find_pii_word_ranges([seg], enable_name_masking=False)
        types = [t for _, _, t in ranges]
        assert "카드번호" in types

    def test_ip_address_range_emit(self):
        seg = _make_segment(
            "IP 192.168.1.1",
            [("IP", 0.0, 0.2), ("192.168.1.1", 0.3, 1.0)],
        )
        ranges = find_pii_word_ranges([seg], enable_name_masking=False)
        types = [t for _, _, t in ranges]
        assert "IP주소" in types

    def test_empty_segments_returns_empty(self):
        assert find_pii_word_ranges([], enable_name_masking=False) == []

    def test_segment_without_words_skipped(self):
        seg = {"text": "전화 010-1234-5678", "words": []}
        assert find_pii_word_ranges([seg], enable_name_masking=False) == []


# ── 7. mask_pii / mask_segments contract 변경 0 (외부 API schema 호환) ─────

class TestContractPreservation:
    def test_mask_segments_output_unchanged_for_legacy_pii(self):
        # mask_segments 가 emit 하는 pii_summary 형식 (PII_PATTERNS + "이름") 그대로.
        segments = [_make_segment(
            "전화 010-1234-5678",
            [("전화", 0.0, 0.3), ("010-1234-5678", 0.4, 1.5)],
        )]
        out = mask_segments(segments, enable_name_masking=False)
        # 외부 schema 호환: type, count 키만.
        for item in out:
            assert "type" in item
            assert "count" in item

    def test_mask_segments_pii_summary_does_not_emit_extended_types(self):
        # 단절점 재현 — mask_segments 만으로는 extended type 미emit (이게 PR-B2 단절점).
        # PR-B2 의 join 단계가 채워준다.
        segments = [_make_segment(
            "비밀번호 Abc1234",
            [("비밀번호", 0.0, 0.5), ("Abc1234", 0.6, 1.0)],
        )]
        out = mask_segments(segments, enable_name_masking=False)
        types = {item["type"] for item in out}
        assert LABEL_CREDENTIAL_LIKE not in types  # 단절점 정확 재현
