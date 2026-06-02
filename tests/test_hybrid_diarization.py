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
