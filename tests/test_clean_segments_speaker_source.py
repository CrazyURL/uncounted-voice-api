"""PR-β — _clean_segments / all_words 의 raw_direct 메타 보존 회귀 테스트.

검증:
  - _clean_segments 가 word.speaker_source 를 보존 (1차 소실 차단)
  - segment.source_distribution / overlap_ranges / parent_segment_text 보존
  - 기존 word/start/end/speaker 회귀 없음
  - legacy(메타 없는) word 는 기존 schema 그대로 (speaker_source 키 미추가)
  - speaker=None 은 문자열 "None" 으로 저장되지 않음 (PR #22 가드 유지)
  - utterance_segmenter.segment 통과 후에도 word 메타가 남는지 (2차 경로 통합)

본 PR 은 non-chunked 경로 한정. chunked(chunk_utterance_emitter)는 PR-β2 별도.
"""

from app.stt_processor import _clean_segments, _clean_word
from app.services.utterance_segmenter import segment as segment_utterances


# ── _clean_word: 메타 보존 + legacy 호환 ──────────────────────────────────

def test_clean_word_preserves_speaker_source():
    w = {"word": "여보세요", "start": 0.05, "end": 0.55,
         "speaker": "SPEAKER_00", "speaker_source": "exact"}
    out = _clean_word(w)
    assert out["speaker_source"] == "exact"
    assert out["word"] == "여보세요"
    assert out["speaker"] == "SPEAKER_00"
    assert out["start"] == 0.05 and out["end"] == 0.55


def test_clean_word_legacy_has_no_speaker_source_key():
    # legacy whisperx word — speaker_source 부재 → 추가되지 않아야 기존 schema 호환
    w = {"word": "hello", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_01"}
    out = _clean_word(w)
    assert "speaker_source" not in out
    assert set(out.keys()) == {"word", "start", "end", "speaker"}


def test_clean_word_none_speaker_not_stringified():
    # PR #22 가드: speaker=None 은 문자열 "None" 으로 바뀌지 않는다.
    w = {"word": "x", "start": 1.0, "end": 1.5, "speaker": None,
         "speaker_source": "overlap_SPEAKER_00_SPEAKER_01"}
    out = _clean_word(w)
    assert out["speaker"] is None
    assert out["speaker"] != "None"
    assert out["speaker_source"] == "overlap_SPEAKER_00_SPEAKER_01"


# ── _clean_segments: word + segment 메타 보존 ────────────────────────────

def _raw_seg(**kw):
    base = {"start": 0.0, "end": 2.0, "text": "t", "speaker": "SPEAKER_00",
            "words": [{"word": "t", "start": 0.0, "end": 2.0,
                       "speaker": "SPEAKER_00", "speaker_source": "exact"}]}
    base.update(kw)
    return base


def test_clean_segments_preserves_word_speaker_source():
    out = _clean_segments([_raw_seg()])
    assert out[0]["words"][0]["speaker_source"] == "exact"


def test_clean_segments_preserves_source_distribution():
    seg = _raw_seg(source_distribution={"exact": 3, "overlap": 1})
    out = _clean_segments([seg])
    assert out[0]["source_distribution"] == {"exact": 3, "overlap": 1}


def test_clean_segments_preserves_overlap_ranges():
    ov = [{"start": 1.0, "end": 1.2, "speakers": ["SPEAKER_00", "SPEAKER_01"]}]
    out = _clean_segments([_raw_seg(overlap_ranges=ov)])
    assert out[0]["overlap_ranges"] == ov


def test_clean_segments_preserves_parent_segment_text():
    out = _clean_segments([_raw_seg(parent_segment_text="원본 문장 전체")])
    assert out[0]["parent_segment_text"] == "원본 문장 전체"


def test_clean_segments_legacy_no_meta_keys():
    # legacy whisperx segment — 메타 키 부재 → 추가되지 않아야 한다 (기존 동작 무변경)
    legacy = {"start": 0.0, "end": 1.0, "text": "hi", "speaker": "SPEAKER_00",
              "words": [{"word": "hi", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]}
    out = _clean_segments([legacy])
    assert "source_distribution" not in out[0]
    assert "overlap_ranges" not in out[0]
    assert "parent_segment_text" not in out[0]
    assert "speaker_source" not in out[0]["words"][0]


def test_clean_segments_existing_fields_regression():
    # 기존 word/start/end/speaker/segment 필드 회귀 없음
    out = _clean_segments([_raw_seg()])
    s = out[0]
    assert s["start"] == 0.0 and s["end"] == 2.0
    assert s["text"] == "t" and s["speaker"] == "SPEAKER_00"
    w = s["words"][0]
    assert w["word"] == "t" and w["speaker"] == "SPEAKER_00"


def test_clean_segments_drops_words_without_timestamps():
    # 기존 동작: start/end None 인 word 는 제외 (회귀 없음)
    seg = {"start": 0.0, "end": 1.0, "text": "x", "speaker": "SPEAKER_00",
           "words": [{"word": "ok", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
                     {"word": "drop", "start": None, "end": None, "speaker": "SPEAKER_00"}]}
    out = _clean_segments([seg])
    assert len(out[0]["words"]) == 1
    assert out[0]["words"][0]["word"] == "ok"


# ── 2차 경로 통합: segment_utterances 통과 후에도 메타 보존 ────────────────

def test_speaker_source_survives_segmentation():
    # _clean_segments 출력의 word 를 segment_utterances 에 흘렸을 때 메타가 남는지.
    # (stt_processor all_words 재구성도 동일하게 speaker_source 를 passthrough)
    words = [
        {"word": "여보세요", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00", "speaker_source": "exact"},
        {"word": "네", "start": 0.6, "end": 1.0, "speaker": "SPEAKER_00", "speaker_source": "tolerance_120ms"},
    ]
    out = segment_utterances(words, 5.0)
    assert len(out) >= 1
    all_w = [w for u in out for w in u.words]
    assert any(w.get("speaker_source") == "exact" for w in all_w)
    assert any(w.get("speaker_source") == "tolerance_120ms" for w in all_w)


def test_segmentation_still_drops_none_speaker_words():
    # PR #22 회귀 가드: None speaker word 는 발화 입력에서 제외(미emit) 유지.
    words = [
        {"word": "x", "start": 0.0, "end": 0.4, "speaker": None, "speaker_source": "ambiguous"},
        {"word": "안녕", "start": 0.5, "end": 1.0, "speaker": "SPEAKER_00", "speaker_source": "exact"},
    ]
    out = segment_utterances(words, 5.0)
    # None word 는 제외되고, 남은 발화에 speaker_id="None" 문자열이 없어야 함
    for u in out:
        assert u.speaker_id != "None"
        assert str(u.speaker_id).strip().lower() != "none"
