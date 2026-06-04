# -*- coding: utf-8 -*-
"""NeMo 전체재분리 (단기 통화 ≤임계) 단위 테스트 — NeMo/F0 mock."""
import numpy as np
import pytest
from app.services import nemo_full_diarization as nf
from app.services import hybrid_diarization as hd
from app.services import anchor_diarization as ad


def _result():
    return {"segments": [{"words": [
        {"word": "여보세요", "start": 0.0, "end": 0.8, "speaker": "SPEAKER_00"},
        {"word": "여보세요", "start": 0.9, "end": 1.6, "speaker": "SPEAKER_00"},
        {"word": "네", "start": 2.0, "end": 2.3, "speaker": "SPEAKER_00"},
    ]}]}


def test_disabled_returns_input(monkeypatch):
    monkeypatch.setenv("VOICE_NEMO_FULL_DIAR_ENABLED", "false")
    r = _result()
    assert nf.apply_nemo_full_diarization(r, "/x.wav", np.zeros(16000), 16000, 100.0) is r


def test_nemo_fail_fallback(monkeypatch):
    monkeypatch.setenv("VOICE_NEMO_FULL_DIAR_ENABLED", "true")
    monkeypatch.setattr(hd, "_call_nemo", lambda p, w: None)
    r = _result()
    assert nf.apply_nemo_full_diarization(r, "/x.wav", np.zeros(16000), 16000, 100.0) is r


def test_not_two_speakers_fallback(monkeypatch):
    monkeypatch.setenv("VOICE_NEMO_FULL_DIAR_ENABLED", "true")
    monkeypatch.setattr(hd, "_call_nemo", lambda p, w: {
        "status": "success", "turns": [{"nemo_spk": "speaker_0", "start": 0.0, "end": 3.0}]})
    r = _result()
    assert nf.apply_nemo_full_diarization(r, "/x.wav", np.zeros(16000), 16000, 100.0) is r


def test_two_speakers_assigns_by_turn_and_f0(monkeypatch):
    monkeypatch.setenv("VOICE_NEMO_FULL_DIAR_ENABLED", "true")
    # NeMo: 여보세요#1=spk0(0~0.85), 여보세요#2=spk1(0.85~1.7), 네=spk0(1.9~2.4)
    turns = [
        {"nemo_spk": "speaker_0", "start": 0.0, "end": 0.85},
        {"nemo_spk": "speaker_1", "start": 0.85, "end": 1.7},
        {"nemo_spk": "speaker_0", "start": 1.9, "end": 2.4},
    ]
    monkeypatch.setattr(hd, "_call_nemo", lambda p, w: {"status": "success", "turns": turns})
    # F0: speaker_0 저음(100Hz)=SP01, speaker_1 고음(180Hz)=SP00
    monkeypatch.setattr(hd, "_f0_medians_from_segments",
                        lambda a, sr, by: {"speaker_0": 100.0, "speaker_1": 180.0})
    out = nf.apply_nemo_full_diarization(_result(), "/x.wav", np.zeros(48000), 16000, 100.0)
    spk = [w["speaker"] for seg in out["segments"] for w in seg["words"]]
    # 여보세요#1=spk0→SP01, #2=spk1→SP00, 네=spk0→SP01
    assert spk == [ad._LABEL_LOW, ad._LABEL_HIGH, ad._LABEL_LOW]
    srcs = [w.get("speaker_source") for seg in out["segments"] for w in seg["words"]]
    assert "nemo_full" in srcs


def test_f0_too_close_fallback(monkeypatch):
    monkeypatch.setenv("VOICE_NEMO_FULL_DIAR_ENABLED", "true")
    turns = [{"nemo_spk": "speaker_0", "start": 0.0, "end": 1.0},
             {"nemo_spk": "speaker_1", "start": 1.0, "end": 2.0}]
    monkeypatch.setattr(hd, "_call_nemo", lambda p, w: {"status": "success", "turns": turns})
    monkeypatch.setattr(hd, "_f0_medians_from_segments",
                        lambda a, sr, by: {"speaker_0": 150.0, "speaker_1": 153.0})  # 차 3Hz < 8
    r = _result()
    assert nf.apply_nemo_full_diarization(r, "/x.wav", np.zeros(16000), 16000, 100.0) is r
