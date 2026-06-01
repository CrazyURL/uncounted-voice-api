"""D4b-2 — worker pii_intervals_only 전송 + maskType provenance + 정책 P 회귀 테스트.

검증:
  - submit params 에 pii_intervals_only="true" 만 추가, mask_audio_pii 부재
  - build_pii_intervals: 발화 겹침 필터 + maskType="text_only"(저장 WAV 원본 provenance), 원문 미포함
  - curated_sequence_orders: pii_reviewed_at/pii_masked_at 흔적 행만 보존 대상
  - strip_pii_if_curated: 정책 P — curated/precheck 실패 시 pii_intervals 제거(보존)

DB/S3/HTTP 불요 — 순수 함수 + params 빌더 단위 테스트. worker.py import 용 더미 env 만 설정.
"""

import os

# worker.py 는 import 시 env 를 요구한다(create_client 는 lazy 라 더미면 충분).
os.environ.setdefault("SUPABASE_URL", "http://localhost")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "dummy")
os.environ.setdefault("S3_ENDPOINT_URL", "http://localhost")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "dummy")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "dummy")

from app import worker  # noqa: E402


# ── submit params ──────────────────────────────────────────────────────────

def test_submit_params_add_only_pii_intervals_only():
    params = worker.build_submit_params()
    assert params["pii_intervals_only"] == "true"
    # mask_audio_pii 는 전달하지 않는다(D5 비프 분리, voice-api 기본 false 유지).
    assert "mask_audio_pii" not in params
    assert "mask_audio_names" not in params
    # 기존 키 불변
    for k in ("language", "diarize", "split_by_utterance", "split_by_speaker", "mask_pii", "denoise"):
        assert k in params


def test_submit_params_enables_name_masking():
    # 한국어 이름 마스킹(텍스트). voice-api enable_name_masking 기본값 false 이므로
    # worker 가 명시 전달해야 성씨+이름 패턴이 감지된다(누락 시 transcript 평문 노출).
    params = worker.build_submit_params()
    assert params["enable_name_masking"] == "true"


def test_submit_params_name_masking_is_text_only_not_audio_beep():
    # 텍스트 마스킹과 오디오 비프(D5)는 분리. 이름 텍스트 마스킹을 켜도 오디오 변형 게이트
    # (mask_audio_pii / mask_audio_names)는 켜지 않는다(저장 WAV 원본 provenance 유지).
    params = worker.build_submit_params()
    assert params["enable_name_masking"] == "true"
    assert "mask_audio_names" not in params
    assert "mask_audio_pii" not in params


def test_submit_params_regression_missing_name_masking_fails():
    # 회귀 가드: enable_name_masking 키가 빠지거나 "true" 가 아니면 실패.
    # 세션 93c28f57 재처리에서 이름 PII 0 건이 된 배선 결함 재발 방지.
    params = worker.build_submit_params()
    assert "enable_name_masking" in params, "enable_name_masking 누락 — 이름 PII 미감지 회귀"
    assert params["enable_name_masking"] == "true"


# ── build_pii_intervals: maskType + overlap ─────────────────────────────────

def test_build_pii_intervals_masktype_text_only_and_no_original():
    pii_summary = [{"type": "전화번호", "count": 1, "time_ranges": [{"start": 1.35, "end": 3.0}]}]
    out = worker.build_pii_intervals(pii_summary, utt_start=0.0, utt_end=5.0)
    assert len(out) == 1
    item = out[0]
    assert item["maskType"] == "text_only"   # 저장 WAV 는 원본(비프 안 됨) → provenance text_only
    assert item["maskType"] != "audio"
    assert item["piiType"] == "전화번호"
    assert item["startSec"] == 1.35 and item["endSec"] == 3.0
    # 원문 텍스트 등 추가 키 없음(startSec/endSec/maskType/piiType 4키만)
    assert set(item.keys()) == {"startSec", "endSec", "maskType", "piiType"}


def test_build_pii_intervals_overlap_filter_excludes_nonoverlapping():
    pii_summary = [{"type": "전화번호", "time_ranges": [
        {"start": 10.0, "end": 11.0},   # 발화 [0,5) 밖 → 제외
        {"start": 2.0, "end": 4.0},     # 겹침 → 포함
    ]}]
    out = worker.build_pii_intervals(pii_summary, utt_start=0.0, utt_end=5.0)
    assert len(out) == 1
    assert out[0]["startSec"] == 2.0


def test_build_pii_intervals_empty_summary():
    assert worker.build_pii_intervals([], 0.0, 5.0) == []


# ── curated_sequence_orders: 정책 P 대상 식별 ────────────────────────────────

def test_curated_sequence_orders_detects_reviewed_and_masked():
    rows = [
        {"sequence_order": 1, "pii_reviewed_at": None, "pii_masked_at": None},   # 자동 → 덮어쓰기 허용
        {"sequence_order": 2, "pii_reviewed_at": "2026-05-27T00:00:00Z", "pii_masked_at": None},  # 검수됨
        {"sequence_order": 3, "pii_reviewed_at": None, "pii_masked_at": "2026-05-27T00:00:00Z"},  # 마스킹됨
    ]
    assert worker.curated_sequence_orders(rows) == {2, 3}


def test_curated_sequence_orders_empty_and_none():
    assert worker.curated_sequence_orders([]) == set()
    assert worker.curated_sequence_orders(None) == set()


# ── strip_pii_if_curated: 정책 P 적용 ────────────────────────────────────────

def _row(seq):
    return {"sequence_order": seq, "pii_intervals": [{"startSec": 1.0, "endSec": 2.0,
            "maskType": "text_only", "piiType": "전화번호"}], "transcript_text": "x"}


def test_strip_keeps_pii_for_non_curated_row():
    row = worker.strip_pii_if_curated(_row(1), seq=1, curated_seqs={2, 3}, precheck_ok=True)
    assert "pii_intervals" in row          # 자동 행 → worker 가 갱신
    assert "transcript_text" in row        # 다른 컬럼은 그대로


def test_strip_removes_pii_for_curated_row():
    row = worker.strip_pii_if_curated(_row(2), seq=2, curated_seqs={2, 3}, precheck_ok=True)
    assert "pii_intervals" not in row      # 검수/마스킹된 행 → 기존 pii_intervals 보존(미덮어씀)
    assert "transcript_text" in row        # STT/품질/라벨 컬럼은 정상 갱신


def test_strip_precheck_failure_preserves_all():
    # 사전조회 실패 시 보수적으로 전 행 pii_intervals 보존(덮어쓰기 회피).
    row = worker.strip_pii_if_curated(_row(1), seq=1, curated_seqs=set(), precheck_ok=False)
    assert "pii_intervals" not in row
