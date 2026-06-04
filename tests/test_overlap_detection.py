"""Unit tests for app.services.overlap_detection (pure, no I/O)."""
from types import SimpleNamespace

from app.services.overlap_detection import (
    DEFAULT_CUTOFF_SEC,
    extract_overlap_regions,
    overlap_regions_from_diarization,
    utterance_overlap_features,
)


# ── overlap_regions_from_diarization (0-GPU, 메인 df 재사용) ──────────────

def test_diar_two_speakers_overlap():
    # 다른 화자 두 세그먼트가 1.0s 겹침 → 1 region
    segs = [(0.0, 3.0, "A"), (2.0, 5.0, "B")]
    assert overlap_regions_from_diarization(segs, cutoff_sec=0.2) == [(2.0, 3.0)]


def test_diar_same_speaker_no_overlap():
    # 같은 화자 겹침은 cross-talk 아님 → 무시
    segs = [(0.0, 3.0, "A"), (2.0, 5.0, "A")]
    assert overlap_regions_from_diarization(segs, cutoff_sec=0.2) == []


def test_diar_below_cutoff_dropped():
    segs = [(0.0, 2.1, "A"), (2.0, 4.0, "B")]  # 0.1s 겹침 < 0.2
    assert overlap_regions_from_diarization(segs, cutoff_sec=0.2) == []


def test_diar_no_overlap():
    segs = [(0.0, 2.0, "A"), (2.0, 4.0, "B")]  # 인접, 겹침 0
    assert overlap_regions_from_diarization(segs, cutoff_sec=0.2) == []


def test_diar_merges_adjacent_regions():
    # 두 겹침 구간이 인접/연속 → 병합
    segs = [(0.0, 3.0, "A"), (2.0, 6.0, "B"), (2.5, 7.0, "A")]
    out = overlap_regions_from_diarization(segs, cutoff_sec=0.2)
    assert len(out) == 1
    assert out[0][0] == 2.0 and out[0][1] >= 6.0


def test_diar_empty():
    assert overlap_regions_from_diarization([], cutoff_sec=0.2) == []


def _seg(start, end):
    return SimpleNamespace(start=start, end=end)


class _FakeAnnotation:
    def __init__(self, segs):
        self._segs = segs

    def get_overlap(self):
        return self._segs


class _FakeDiarizeOutput:
    def __init__(self, segs):
        self.speaker_diarization = _FakeAnnotation(segs)


# ── extract_overlap_regions ──────────────────────────────────────────────

def test_extract_drops_subcutoff_blips():
    out = _FakeDiarizeOutput([_seg(1.0, 1.05), _seg(5.0, 5.4), _seg(10.0, 10.1)])
    regions = extract_overlap_regions(out, cutoff_sec=0.2)
    assert regions == [(5.0, 5.4)]  # only the >=0.2s region survives


def test_extract_accepts_raw_annotation():
    ann = _FakeAnnotation([_seg(2.0, 2.5)])
    assert extract_overlap_regions(ann, cutoff_sec=0.2) == [(2.0, 2.5)]


def test_extract_returns_empty_when_no_get_overlap():
    # whisperx exclusive dataframe has no get_overlap -> "unknown", not crash
    assert extract_overlap_regions(object()) == []


def test_extract_sorted():
    out = _FakeDiarizeOutput([_seg(9.0, 9.5), _seg(1.0, 1.6)])
    assert extract_overlap_regions(out) == [(1.0, 1.6), (9.0, 9.5)]


def test_default_cutoff_is_point_two():
    assert DEFAULT_CUTOFF_SEC == 0.2


# ── utterance_overlap_features ───────────────────────────────────────────

def test_no_overlap():
    f = utterance_overlap_features(0.0, 10.0, [])
    assert f["is_overlapping"] is False
    assert f["overlap_count"] == 0
    assert f["overlap_total_sec"] == 0.0
    assert f["overlap_ratio"] == 0.0
    assert f["overlap_intervals"] == []


def test_single_region_inside():
    f = utterance_overlap_features(0.0, 10.0, [(2.0, 3.0)])
    assert f["is_overlapping"] is True
    assert f["overlap_count"] == 1
    assert f["overlap_total_sec"] == 1.0
    assert f["overlap_ratio"] == 0.1  # 1.0 / 10.0
    assert f["overlap_intervals"] == [{"start_sec": 2.0, "end_sec": 3.0}]


def test_region_clipped_to_utterance_bounds():
    # region straddles the right edge; only the inside part is attributed
    f = utterance_overlap_features(0.0, 5.0, [(4.5, 6.0)])
    assert f["overlap_count"] == 1
    assert f["overlap_intervals"] == [{"start_sec": 4.5, "end_sec": 5.0}]
    assert f["overlap_total_sec"] == 0.5


def test_multiple_regions_count_and_total():
    f = utterance_overlap_features(0.0, 20.0, [(1.0, 2.0), (5.0, 5.5), (10.0, 11.5)])
    assert f["overlap_count"] == 3
    assert f["overlap_total_sec"] == 3.0  # 1.0 + 0.5 + 1.5
    assert f["overlap_ratio"] == 0.15


def test_region_outside_utterance_ignored():
    f = utterance_overlap_features(10.0, 20.0, [(0.0, 5.0)])
    assert f["is_overlapping"] is False
    assert f["overlap_intervals"] == []


def test_float_sliver_ignored():
    # a region touching the edge by < 1ms must not create a phantom interval
    f = utterance_overlap_features(0.0, 5.0, [(4.9995, 6.0)])
    assert f["is_overlapping"] is False


def test_zero_duration_utterance_safe():
    f = utterance_overlap_features(3.0, 3.0, [(2.0, 4.0)])
    # clipped region is [3,3] -> zero length -> not counted; ratio guarded
    assert f["overlap_ratio"] == 0.0
    assert f["is_overlapping"] is False


def test_none_bounds_safe():
    f = utterance_overlap_features(None, None, [(1.0, 2.0)])
    assert f["is_overlapping"] is False
    assert f["overlap_ratio"] == 0.0
