"""NaN/Inf sanitize 회귀 테스트.

검증 대상:
  - app.sanitize_json.sanitize_json_safe 의 scalar / dict / list / nested / WhisperX word-shape 처리.
  - **결정적 oracle**: ``json.dumps(sanitize_json_safe(row), allow_nan=False)`` 가 raise 하지 않는다.
    이것이 본 helper 가 worker → supabase REST → PostgreSQL JSONB 경로의 NaN reject 를 차단함을 의미.

GPU / WhisperX deps 불요. worker.py 의 supabase env 의존을 피하기 위해 helper 만 직접 import.
"""

from __future__ import annotations

import json
import math

import pytest

from app.sanitize_json import sanitize_json_safe


# ─────────────────────────────────────────────────────────────────────────────
# 1. Scalar — NaN / +Inf / -Inf → None
# ─────────────────────────────────────────────────────────────────────────────

def test_nan_scalar_becomes_none():
    assert sanitize_json_safe(float("nan")) is None


def test_positive_infinity_becomes_none():
    assert sanitize_json_safe(float("inf")) is None


def test_negative_infinity_becomes_none():
    assert sanitize_json_safe(float("-inf")) is None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Scalar — 정상값 보존
# ─────────────────────────────────────────────────────────────────────────────

def test_normal_float_preserved():
    assert sanitize_json_safe(0.0) == 0.0
    assert sanitize_json_safe(-1.5) == -1.5
    assert sanitize_json_safe(3.14159) == 3.14159


def test_int_preserved():
    assert sanitize_json_safe(0) == 0
    assert sanitize_json_safe(42) == 42
    assert sanitize_json_safe(-7) == -7


def test_bool_preserved_not_coerced():
    # bool 은 int 의 subclass 이지만 본 helper 는 그대로 통과.
    assert sanitize_json_safe(True) is True
    assert sanitize_json_safe(False) is False


def test_str_preserved():
    assert sanitize_json_safe("") == ""
    assert sanitize_json_safe("hello") == "hello"
    # "NaN" / "inf" 같은 문자열은 절대 None 으로 바꾸지 않는다(자료형이 다름).
    assert sanitize_json_safe("NaN") == "NaN"
    assert sanitize_json_safe("inf") == "inf"


def test_none_preserved():
    assert sanitize_json_safe(None) is None


# ─────────────────────────────────────────────────────────────────────────────
# 3. Dict 재귀
# ─────────────────────────────────────────────────────────────────────────────

def test_dict_with_nan_value_sanitized():
    row = {"snr_db": float("nan"), "speech_ratio": 0.85}
    out = sanitize_json_safe(row)
    assert out == {"snr_db": None, "speech_ratio": 0.85}


def test_dict_with_inf_value_sanitized():
    row = {"score": float("inf"), "label": "A"}
    assert sanitize_json_safe(row) == {"score": None, "label": "A"}


def test_dict_keys_preserved_even_when_value_nan():
    # 키는 절대 잃지 않는다 — JSONB 컬럼 shape 보존.
    row = {"a": float("nan"), "b": float("inf"), "c": float("-inf"), "d": 1.0}
    out = sanitize_json_safe(row)
    assert set(out.keys()) == {"a", "b", "c", "d"}
    assert out["a"] is None and out["b"] is None and out["c"] is None
    assert out["d"] == 1.0


def test_empty_dict_preserved():
    assert sanitize_json_safe({}) == {}


# ─────────────────────────────────────────────────────────────────────────────
# 4. List / tuple 재귀
# ─────────────────────────────────────────────────────────────────────────────

def test_list_with_nan_items_sanitized():
    assert sanitize_json_safe([1.0, float("nan"), 2.0]) == [1.0, None, 2.0]


def test_tuple_normalized_to_list():
    # tuple 은 JSON serialization 시 어차피 list 가 되므로 정규화.
    out = sanitize_json_safe((1.0, float("nan"), 3.0))
    assert isinstance(out, list)
    assert out == [1.0, None, 3.0]


def test_empty_list_preserved():
    assert sanitize_json_safe([]) == []


# ─────────────────────────────────────────────────────────────────────────────
# 5. 깊은 중첩
# ─────────────────────────────────────────────────────────────────────────────

def test_deeply_nested_structure_sanitized():
    row = {
        "session_id": "s1",
        "metrics": {
            "snr_db": float("nan"),
            "clipping": {
                "ratio": float("inf"),
                "ok": True,
            },
        },
        "words": [
            {"w": "안", "start": 0.1, "end": float("nan"), "score": 0.9},
            {"w": "녕", "start": float("-inf"), "end": 0.4, "score": float("nan")},
        ],
    }
    out = sanitize_json_safe(row)
    assert out["session_id"] == "s1"
    assert out["metrics"]["snr_db"] is None
    assert out["metrics"]["clipping"]["ratio"] is None
    assert out["metrics"]["clipping"]["ok"] is True
    assert out["words"][0] == {"w": "안", "start": 0.1, "end": None, "score": 0.9}
    assert out["words"][1] == {"w": "녕", "start": None, "end": 0.4, "score": None}


# ─────────────────────────────────────────────────────────────────────────────
# 6. WhisperX word-shape (실측 시나리오)
# ─────────────────────────────────────────────────────────────────────────────

def test_whisperx_word_shape_partial_nan():
    # alignment 실패한 단어가 섞인 경우 — start/score 만 NaN, 나머지 정상.
    words = [
        {"word": "여보세요", "start": 0.0, "end": 0.5, "score": 0.95},
        {"word": "?", "start": float("nan"), "end": float("nan"), "score": float("nan")},
        {"word": "안녕", "start": 0.7, "end": 1.0, "score": 0.88},
    ]
    out = sanitize_json_safe(words)
    assert out[0] == {"word": "여보세요", "start": 0.0, "end": 0.5, "score": 0.95}
    assert out[1] == {"word": "?", "start": None, "end": None, "score": None}
    assert out[2] == {"word": "안녕", "start": 0.7, "end": 1.0, "score": 0.88}


# ─────────────────────────────────────────────────────────────────────────────
# 7. 결정적 oracle — json.dumps(..., allow_nan=False) raise 없음
#    (PostgreSQL JSONB 가 거부하는 NaN/Infinity 리터럴이 페이로드에 없음을 보증)
# ─────────────────────────────────────────────────────────────────────────────

def test_json_dumps_allow_nan_false_raises_on_raw_nan():
    """guard test — sanitize 없이는 raise (helper 가 의미 있는 경로임을 lock)."""
    with pytest.raises(ValueError):
        json.dumps({"x": float("nan")}, allow_nan=False)


def test_sanitized_row_passes_json_dumps_allow_nan_false():
    # 실제 worker.py 가 supabase 로 보내는 row 와 비슷한 shape — NaN 가 곳곳에 섞여 있어도
    # sanitize 후엔 strict JSON 으로 직렬화 가능해야 한다.
    row = {
        "id": "utt_s1_001",
        "session_id": "s1",
        "sequence_order": 1,
        "speaker_id": "SPEAKER_00",
        "start_sec": 0.0,
        "end_sec": 1.0,
        "duration_sec": 1.0,
        "storage_path": "utterances/s1/utt_s1_001.wav",
        "file_size_bytes": 32000,
        "upload_status": "uploaded",
        "transcript_text": "안녕하세요",
        "transcript_words": [
            {"word": "안녕", "start": 0.0, "end": 0.5, "score": 0.9},
            {"word": "하세요", "start": float("nan"), "end": float("nan"), "score": float("nan")},
        ],
        "quality_score": float("nan"),
        "quality_grade": "B",
        "snr_db": float("nan"),
        "speech_ratio": 1.0,
        "pii_intervals": [],
        "emotion": None,
        "emotion_confidence": float("inf"),
        "dialog_act": None,
        "dialog_act_confidence": None,
        "speech_rate_wpm": float("-inf"),
        "silence_before_sec": 0.0,
        "filler_word_count": 0,
        "honorific_level": None,
        "question_type": None,
        "language_mix_flag": False,
    }
    sanitized = sanitize_json_safe(row)
    # 이 한 줄이 핵심 oracle — supabase REST 가 본질적으로 수행하는 strict-JSON 직렬화와 동등.
    serialized = json.dumps(sanitized, allow_nan=False, ensure_ascii=False)
    # 직렬화 결과에 NaN/Infinity 리터럴 없음(이중 확인).
    assert "NaN" not in serialized
    assert "Infinity" not in serialized
    # 키 갯수 보존 — shape 가 깨지지 않음.
    assert len(sanitized) == len(row)


# ─────────────────────────────────────────────────────────────────────────────
# 8. 원본 mutate 금지
# ─────────────────────────────────────────────────────────────────────────────

def test_input_not_mutated():
    original = {"snr_db": float("nan"), "words": [{"score": float("nan")}]}
    original_snr = original["snr_db"]
    _ = sanitize_json_safe(original)
    # 원본의 NaN 은 여전히 NaN(helper 가 새 dict 를 반환했으므로).
    assert math.isnan(original["snr_db"]) and math.isnan(original_snr)
    assert math.isnan(original["words"][0]["score"])
