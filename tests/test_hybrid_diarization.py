"""하이브리드 도입부 발화분리 (NeMo 오버라이트) 단위 테스트.

검증:
  - 게이트 OFF(기본): NeMo 미호출, result 무변경 (무회귀)
  - 코사인 ID매핑 주력: pyannote↔NeMo 임베딩 코사인으로 1:1
  - overlap 백업: 임베딩 없을 때 시간 overlap 매핑
  - NeMo 미응답/실패: result 무변경 (fallback)
  - 하드 오버라이트: 도입부 window 내 word.speaker 만 교체, window 밖 보존
  - GT1 ABAB 복원 (실데이터 패턴)

requests/NeMo 미호출 — monkeypatch 로 가짜 응답 주입(네트워크 0).
"""
import os

from app.services import hybrid_diarization as hd


# ── 게이트 ──────────────────────────────────────────────────────────────────

def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VOICE_HYBRID_DIAR_ENABLED", raising=False)
    assert hd.is_enabled() is False
    result = {"segments": [{"start": 0, "end": 1, "speaker": "SPEAKER_00",
                            "words": [{"word": "x", "start": 0.1, "end": 0.5, "speaker": "SPEAKER_00"}]}]}
    out = hd.apply_hybrid_intro(result, "/tmp/x.wav")
    assert out is result   # 게이트 OFF → 입력 그대로(무변경)


# ── 코사인 ID 매핑 ──────────────────────────────────────────────────────────

def test_map_by_cosine_clear_one_to_one():
    py = {"SPEAKER_00": [1.0, 0.0], "SPEAKER_01": [0.0, 1.0]}
    nemo = {"speaker_0": [0.0, 1.0], "speaker_1": [1.0, 0.0]}
    m = hd._map_by_cosine(py, nemo)
    assert m == {"speaker_0": "SPEAKER_01", "speaker_1": "SPEAKER_00"}


def test_map_by_cosine_ambiguous_returns_none():
    # 두 NeMo 가 같은 pyannote 와 가장 가까움 → 1:1 깨짐 → None(overlap 위임)
    py = {"SPEAKER_00": [1.0, 0.0], "SPEAKER_01": [0.99, 0.01]}
    nemo = {"speaker_0": [1.0, 0.0], "speaker_1": [1.0, 0.0]}
    assert hd._map_by_cosine(py, nemo) is None


def test_map_by_cosine_empty_returns_none():
    assert hd._map_by_cosine({}, {"speaker_0": [1.0]}) is None
    assert hd._map_by_cosine({"SPEAKER_00": [1.0]}, {}) is None


def test_map_by_cosine_requires_mutual_best_match():
    # 비대칭(오배정 위험): NeMo 두 화자가 모두 SPEAKER_00 쪽으로 기울고 SPEAKER_01 은
    # 어느 NeMo 와도 mutual best 가 아님 → None (정답지 없이 오배정 차단).
    py = {"SPEAKER_00": [1.0, 0.0], "SPEAKER_01": [0.9, 0.1]}
    nemo = {"speaker_0": [1.0, 0.0], "speaker_1": [0.95, 0.05]}
    assert hd._map_by_cosine(py, nemo) is None


def test_map_by_cosine_mutual_match_succeeds():
    # 양방향 일치(직교 임베딩): 정상 1:1
    py = {"SPEAKER_00": [1.0, 0.0], "SPEAKER_01": [0.0, 1.0]}
    nemo = {"speaker_0": [0.0, 1.0], "speaker_1": [1.0, 0.0]}
    m = hd._map_by_cosine(py, nemo)
    assert m == {"speaker_0": "SPEAKER_01", "speaker_1": "SPEAKER_00"}


def test_completeness_guard_assigns_new_label_for_unmapped():
    # pyannote 도입부 단일화자 + 임베딩 없음 → overlap 백업이 1명만 매핑 → 완전성 가드가
    # 누락 NeMo 화자에 신규 라벨 발급(모든 NeMo 화자가 서로 다른 라벨).
    import os
    os.environ["VOICE_HYBRID_DIAR_ENABLED"] = "true"
    try:
        result = {"segments": [{
            "start": 0.0, "end": 3.7, "speaker": "SPEAKER_00",
            "words": [
                {"word": "a", "start": 0.4, "end": 0.6, "speaker": "SPEAKER_00"},
                {"word": "b", "start": 1.1, "end": 1.3, "speaker": "SPEAKER_00"},
            ],
        }]}
        import unittest.mock as mock
        nemo_payload = {
            "status": "success",
            "turns": [
                {"start": 0.0, "end": 0.9, "nemo_spk": "speaker_0"},
                {"start": 1.0, "end": 1.5, "nemo_spk": "speaker_1"},
            ],
            "embeddings": {},  # 임베딩 없음 → overlap 백업
        }
        with mock.patch.object(hd, "_call_nemo", return_value=nemo_payload):
            out = hd.apply_hybrid_intro(result, "/tmp/x.wav", pyannote_embeddings={})
        spk = [w["speaker"] for w in out["segments"][0]["words"]]
        # 두 word 가 서로 다른 화자로 갈려야(둘 다 SPEAKER_00 로 남으면 실패)
        assert spk[0] != spk[1]
    finally:
        os.environ.pop("VOICE_HYBRID_DIAR_ENABLED", None)


# ── F0 어쿠스틱 앵커 매핑 (매핑 메인) ───────────────────────────────────────

def test_map_by_f0_rank_low_to_low():
    # 저음↔저음, 고음↔고음 (성대 물리 순위). speaker_1 저음 → 저음 pyannote SP01.
    py = {"SPEAKER_01": 118.9, "SPEAKER_00": 133.9}
    nemo = {"speaker_1": 106.9, "speaker_0": 130.1}
    m = hd._map_by_f0(py, nemo)
    assert m == {"speaker_1": "SPEAKER_01", "speaker_0": "SPEAKER_00"}


def test_map_by_f0_ambiguous_when_pitch_too_close():
    # 두 화자 F0 차이 < 8Hz → 앵커 신뢰 불가 → None (코사인/overlap 위임)
    py = {"SPEAKER_00": 120.0, "SPEAKER_01": 124.0}   # 4Hz 차
    nemo = {"speaker_0": 121.0, "speaker_1": 126.0}
    assert hd._map_by_f0(py, nemo) is None


def test_map_by_f0_requires_two_speakers():
    assert hd._map_by_f0({"SPEAKER_00": 120.0}, {"speaker_0": 120.0}) is None


def test_f0_overrides_cosine_when_they_disagree(monkeypatch):
    # 매핑 메인 = F0. 코사인이 거꾸로(speaker_1→SP00) 가리켜도 F0(저음↔저음)가 이긴다.
    monkeypatch.setenv("VOICE_HYBRID_DIAR_ENABLED", "true")
    _mock_nemo(monkeypatch, {
        "status": "success",
        "turns": [
            {"start": 0.0, "end": 0.88, "nemo_spk": "speaker_1"},   # 본인(저음)
            {"start": 0.88, "end": 1.12, "nemo_spk": "speaker_0"},  # 상대(고음)
            {"start": 1.12, "end": 2.17, "nemo_spk": "speaker_1"},
            {"start": 2.62, "end": 3.85, "nemo_spk": "speaker_0"},
        ],
        # 코사인은 거꾸로 유도: speaker_1↔SP00, speaker_0↔SP01 (틀린 방향)
        "embeddings": {"speaker_1": [0.0, 1.0], "speaker_0": [1.0, 0.0]},
    })
    py_embs = {"SPEAKER_00": [0.0, 1.0], "SPEAKER_01": [1.0, 0.0]}  # 코사인→speaker_1=SP00(오답)
    # F0: 저음 speaker_1 ↔ 저음 SP01, 고음 speaker_0 ↔ 고음 SP00 (정답)
    py_f0 = {"SPEAKER_01": 118.9, "SPEAKER_00": 133.9}
    nemo_f0 = {"speaker_1": 106.9, "speaker_0": 130.1}
    out = hd.apply_hybrid_intro(
        _result_gt1(), "/tmp/x.wav",
        pyannote_embeddings=py_embs, pyannote_f0=py_f0, nemo_f0=nemo_f0, window_sec=30.0,
    )
    spk = [w["speaker"] for w in out["segments"][0]["words"]]
    # F0 앵커 채택 → GT1 정답: 0.4=본인SP01 1.1=상대SP00 1.9=본인SP01 3.3=상대SP00
    assert spk[0] == "SPEAKER_01" and spk[1] == "SPEAKER_00"
    assert spk[2] == "SPEAKER_01" and spk[3] == "SPEAKER_00"


def test_f0_alone_maps_without_embeddings(monkeypatch):
    # 임베딩 전무여도 F0 만으로 매핑 (앵커가 메인이므로)
    monkeypatch.setenv("VOICE_HYBRID_DIAR_ENABLED", "true")
    _mock_nemo(monkeypatch, {
        "status": "success",
        "turns": [
            {"start": 0.0, "end": 0.88, "nemo_spk": "speaker_1"},
            {"start": 0.88, "end": 1.12, "nemo_spk": "speaker_0"},
            {"start": 1.12, "end": 2.17, "nemo_spk": "speaker_1"},
            {"start": 2.62, "end": 3.85, "nemo_spk": "speaker_0"},
        ],
        "embeddings": {},  # 임베딩 없음
    })
    out = hd.apply_hybrid_intro(
        _result_gt1(), "/tmp/x.wav",
        pyannote_embeddings={},
        pyannote_f0={"SPEAKER_01": 118.9, "SPEAKER_00": 133.9},
        nemo_f0={"speaker_1": 106.9, "speaker_0": 130.1},
        window_sec=30.0,
    )
    spk = [w["speaker"] for w in out["segments"][0]["words"]]
    assert spk[0] == "SPEAKER_01" and spk[1] == "SPEAKER_00"
    assert spk[2] == "SPEAKER_01" and spk[3] == "SPEAKER_00"


# ── 중첩(cross-talk) 구간 보존 ──────────────────────────────────────────────

def test_overlap_region_words_not_overwritten(monkeypatch):
    # 중첩 구간 word 는 단일 화자로 오버라이트하지 않고 원본 유지(양쪽 화자 보존·timeline 정합).
    monkeypatch.setenv("VOICE_HYBRID_DIAR_ENABLED", "true")
    _mock_nemo(monkeypatch, {
        "status": "success",
        "turns": [
            {"start": 0.0, "end": 0.88, "nemo_spk": "speaker_1"},
            {"start": 0.88, "end": 1.12, "nemo_spk": "speaker_0"},
            {"start": 1.12, "end": 2.17, "nemo_spk": "speaker_1"},
            {"start": 2.62, "end": 3.85, "nemo_spk": "speaker_0"},
        ],
        "embeddings": {},
    })
    py_f0 = {"SPEAKER_01": 118.9, "SPEAKER_00": 133.9}
    nemo_f0 = {"speaker_1": 106.9, "speaker_0": 130.1}
    # 1.9s word("네예") 를 중첩 구간으로 지정 → NeMo 가 speaker_1(SP01) 로 보더라도 원본 유지.
    out = hd.apply_hybrid_intro(
        _result_gt1(), "/tmp/x.wav",
        pyannote_embeddings={}, pyannote_f0=py_f0, nemo_f0=nemo_f0,
        overlap_regions=[(1.85, 2.20)], window_sec=30.0,
    )
    words = out["segments"][0]["words"]
    # 0.4(상대 아님-본인 SP01) 1.1(상대 SP00) 은 오버라이트, 1.9(중첩) 는 원본 SPEAKER_00 유지
    by_t = {round(w["start"], 1): w for w in words}
    assert by_t[1.9]["speaker"] == "SPEAKER_00"            # 중첩 → 원본 보존
    assert by_t[1.9].get("speaker_source") != "hybrid_nemo_intro"
    # 중첩 밖 word 는 정상 F0 매핑 적용(보존이 전체를 막지 않음)
    assert by_t[0.4]["speaker"] == "SPEAKER_01"
    assert by_t[1.1]["speaker"] == "SPEAKER_00"


def test_no_overlap_regions_is_noop(monkeypatch):
    # overlap_regions=None(기본) 이면 기존 동작과 동일(전부 오버라이트 후보).
    monkeypatch.setenv("VOICE_HYBRID_DIAR_ENABLED", "true")
    _mock_nemo(monkeypatch, {
        "status": "success",
        "turns": [
            {"start": 0.0, "end": 0.88, "nemo_spk": "speaker_1"},
            {"start": 0.88, "end": 1.12, "nemo_spk": "speaker_0"},
            {"start": 1.12, "end": 2.17, "nemo_spk": "speaker_1"},
            {"start": 2.62, "end": 3.85, "nemo_spk": "speaker_0"},
        ],
        "embeddings": {},
    })
    out = hd.apply_hybrid_intro(
        _result_gt1(), "/tmp/x.wav",
        pyannote_embeddings={},
        pyannote_f0={"SPEAKER_01": 118.9, "SPEAKER_00": 133.9},
        nemo_f0={"speaker_1": 106.9, "speaker_0": 130.1},
        window_sec=30.0,
    )
    spk = [w["speaker"] for w in out["segments"][0]["words"]]
    assert spk[0] == "SPEAKER_01" and spk[1] == "SPEAKER_00"
    assert spk[2] == "SPEAKER_01" and spk[3] == "SPEAKER_00"


# ── overlap 백업 매핑 ───────────────────────────────────────────────────────

def test_map_by_overlap_basic():
    pyannote_turns = [(0.0, 2.0, "SPEAKER_00"), (2.0, 4.0, "SPEAKER_01")]
    nemo_turns = [{"start": 0.0, "end": 1.9, "nemo_spk": "speaker_0"},
                  {"start": 2.1, "end": 3.9, "nemo_spk": "speaker_1"}]
    m = hd._map_by_overlap(pyannote_turns, nemo_turns, 30.0)
    assert m == {"speaker_0": "SPEAKER_00", "speaker_1": "SPEAKER_01"}


# ── NeMo 호출 mock ──────────────────────────────────────────────────────────

def _mock_nemo(monkeypatch, payload):
    monkeypatch.setattr(hd, "_call_nemo", lambda audio_path, win: payload)


def _result_gt1():
    # GT1: 본인0.4 / 상대1.1 / 본인1.9 / 상대3.3 — pyannote 가 전부 SPEAKER_00 으로 뭉침
    return {"segments": [{
        "start": 0.0, "end": 3.7, "speaker": "SPEAKER_00",
        "words": [
            {"word": "여보세요?", "start": 0.4, "end": 0.6, "speaker": "SPEAKER_00"},
            {"word": "여보세요", "start": 1.1, "end": 1.3, "speaker": "SPEAKER_00"},
            {"word": "네예", "start": 1.9, "end": 2.1, "speaker": "SPEAKER_00"},
            {"word": "바쁘세요?", "start": 3.3, "end": 3.6, "speaker": "SPEAKER_00"},
            {"word": "끝", "start": 40.0, "end": 40.5, "speaker": "SPEAKER_00"},  # window 밖
        ],
    }]}


def test_hybrid_overwrites_intro_gt1_abab(monkeypatch):
    monkeypatch.setenv("VOICE_HYBRID_DIAR_ENABLED", "true")
    # NeMo: 본인(speaker_1) 상대(speaker_0) 교대 + 임베딩
    _mock_nemo(monkeypatch, {
        "status": "success",
        "turns": [
            {"start": 0.0, "end": 0.88, "nemo_spk": "speaker_1"},
            {"start": 0.88, "end": 1.12, "nemo_spk": "speaker_0"},
            {"start": 1.12, "end": 2.17, "nemo_spk": "speaker_1"},
            {"start": 2.62, "end": 3.85, "nemo_spk": "speaker_0"},
        ],
        "embeddings": {"speaker_1": [1.0, 0.0], "speaker_0": [0.0, 1.0]},
    })
    # pyannote 임베딩: SPEAKER_00=본인쪽? 여기선 코사인으로 speaker_1→SP01, speaker_0→SP00 유도
    py_embs = {"SPEAKER_00": [0.0, 1.0], "SPEAKER_01": [1.0, 0.0]}
    out = hd.apply_hybrid_intro(_result_gt1(), "/tmp/x.wav", pyannote_embeddings=py_embs, window_sec=30.0)
    words = out["segments"][0]["words"]
    spk = [w["speaker"] for w in words]
    # GT1 ABAB: 0.4=본인 1.1=상대 1.9=본인 3.3=상대 → SP01 SP00 SP01 SP00
    assert spk[0] == "SPEAKER_01" and spk[1] == "SPEAKER_00"
    assert spk[2] == "SPEAKER_01" and spk[3] == "SPEAKER_00"
    assert spk[0] != spk[1] and spk[0] == spk[2] and spk[1] == spk[3]  # ABAB
    # window 밖(40s) word 는 원본 보존
    assert spk[4] == "SPEAKER_00"


def test_hybrid_nemo_failure_returns_unchanged(monkeypatch):
    monkeypatch.setenv("VOICE_HYBRID_DIAR_ENABLED", "true")
    _mock_nemo(monkeypatch, None)   # NeMo 미응답
    r = _result_gt1()
    out = hd.apply_hybrid_intro(r, "/tmp/x.wav", pyannote_embeddings={})
    assert out is r   # 무변경


def test_hybrid_overlap_backup_when_no_embeddings(monkeypatch):
    monkeypatch.setenv("VOICE_HYBRID_DIAR_ENABLED", "true")
    _mock_nemo(monkeypatch, {
        "status": "success",
        "turns": [{"start": 0.0, "end": 2.0, "nemo_spk": "speaker_0"},
                  {"start": 2.0, "end": 3.7, "nemo_spk": "speaker_1"}],
        "embeddings": {},   # 임베딩 없음 → overlap 백업
    })
    # pyannote 도입부 2화자 turn (overlap 매핑 가능)
    result = {"segments": [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00",
         "words": [{"word": "a", "start": 0.5, "end": 1.0, "speaker": "SPEAKER_00"}]},
        {"start": 2.0, "end": 3.7, "speaker": "SPEAKER_01",
         "words": [{"word": "b", "start": 2.5, "end": 3.0, "speaker": "SPEAKER_01"}]},
    ]}
    out = hd.apply_hybrid_intro(result, "/tmp/x.wav", pyannote_embeddings={})
    # overlap 매핑: speaker_0→SP00, speaker_1→SP01 → word 유지(이미 일치)
    assert out["segments"][0]["words"][0]["speaker"] == "SPEAKER_00"
    assert out["segments"][1]["words"][0]["speaker"] == "SPEAKER_01"
