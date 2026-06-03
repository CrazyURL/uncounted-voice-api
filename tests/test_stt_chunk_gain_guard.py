# -*- coding: utf-8 -*-
"""chunked STT gain-guard 분기 핀. _transcribe_chunk 가:
- raw 와 전처리 audio 길이 동일 → STT(transcribe+align)에 raw 사용(게인왜곡 truncation 방지)
- 길이 불일치(무음압축) → 전처리 audio 폴백(타임라인 정합; ⚠️ 이 분기는 truncation 미해결 = known-degraded,
  완전 해결은 preprocess gain/silence 분리 리팩터 백로그)
"""
import numpy as np
import pytest
from app import stt_processor


@pytest.fixture
def _mock_gpu(monkeypatch):
    monkeypatch.setattr(stt_processor.job_store, "update_gpu_acquired", lambda *a, **k: None)
    monkeypatch.setattr(stt_processor.job_store, "update_gpu_released", lambda *a, **k: None)
    monkeypatch.setattr(stt_processor.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(stt_processor.whisperx, "align", lambda *a, **k: {"segments": []})


def _capture_stt_audio(monkeypatch):
    captured = {}

    def fake_transcribe(a, tid):
        captured["audio"] = a
        return {"segments": []}

    monkeypatch.setattr(stt_processor, "_transcribe_with_oom_guard", fake_transcribe)
    return captured


def test_chunk_uses_raw_when_length_preserved(monkeypatch, _mock_gpu):
    captured = _capture_stt_audio(monkeypatch)
    audio = np.zeros(16000, dtype=np.float32)   # 전처리본(게인 적용)
    raw = np.ones(16000, dtype=np.float32)      # raw, 동일 길이
    stt_processor._transcribe_chunk(audio, "t", enable_diarize=False, raw_audio=raw)
    assert captured["audio"] is raw  # 길이 동일 → STT 는 raw


def test_chunk_falls_back_when_length_differs(monkeypatch, _mock_gpu):
    captured = _capture_stt_audio(monkeypatch)
    audio = np.zeros(16000, dtype=np.float32)   # 전처리본(무음압축으로 짧아짐)
    raw = np.ones(16500, dtype=np.float32)      # raw, 길이 다름
    stt_processor._transcribe_chunk(audio, "t", enable_diarize=False, raw_audio=raw)
    assert captured["audio"] is audio  # 길이 불일치 → 전처리 audio 폴백(known-degraded)


def test_chunk_falls_back_when_raw_none(monkeypatch, _mock_gpu):
    captured = _capture_stt_audio(monkeypatch)
    audio = np.zeros(16000, dtype=np.float32)
    stt_processor._transcribe_chunk(audio, "t", enable_diarize=False, raw_audio=None)
    assert captured["audio"] is audio  # raw 없음 → 전처리 audio
