"""D4b — pii_intervals_only 디커플링 회귀 테스트.

검증 대상(D4a 설계 §4 불변식):
  - range 산출 게이트 = (mask_audio_pii OR pii_intervals_only)
  - 오디오 변형(1kHz beep) 게이트 = mask_audio_pii 단독
  - (F,T) interval-only 경로의 저장 WAV 는 (F,F) 비-PII 경로와 바이트 동일
  - (T,F) 기존 mask_audio_pii beep 동작 불변
  - (T,T) lenient: mask_audio_pii 가 오디오를 지배(beep) + interval 도 emit

플래그 표기: (mask_audio_pii, pii_intervals_only).
GPU/모델 불요 — whisperx deps 를 mock(audio=np.zeros)하여 dev PC 에서 실행 가능.
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from pathlib import Path

from app.stt_processor import transcribe, _transcribe_chunked
from app.services.audio_pii_masker import mask_audio_ranges


# ---------------------------------------------------------------------------
# 일반 모드 (transcribe) — 기존 test_stt_processor_audio_pii.py 와 동일한 mock 하네스
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_whisperx_deps():
    with patch("whisperx.load_audio") as mock_load, \
         patch("subprocess.run") as mock_run, \
         patch("app.stt_processor._model") as mock_model, \
         patch("app.stt_processor._align_model"), \
         patch("whisperx.align") as mock_align, \
         patch("app.stt_processor.job_store"), \
         patch("app.stt_processor.preprocess") as mock_preprocess:

        # 무음(전부 0) 5초 오디오 — beep 가 적용되면 그 구간만 비-0 으로 바뀌어 바이트가 달라진다.
        mock_load.return_value = np.zeros(16000 * 5, dtype=np.float32)
        mock_run.return_value = MagicMock(stdout="5.0")
        mock_preprocess.side_effect = lambda x, sr: x

        seg = {
            "text": "제 번호는 010-1234-5678입니다.",
            "start": 1.0,
            "end": 3.0,
        }
        mock_model.transcribe.return_value = {"segments": [seg]}
        mock_align.return_value = {
            "segments": [{
                **seg,
                "words": [
                    {"word": "제", "start": 1.0, "end": 1.2},
                    {"word": "번호는", "start": 1.2, "end": 1.5},
                    {"word": "010-1234-5678입니다.", "start": 1.5, "end": 3.0},
                ],
            }]
        }
        yield


def _run_transcribe(tmp_path, *, mask_audio_pii, pii_intervals_only):
    """주어진 플래그 조합으로 transcribe 를 실행하고 (result, mask_call_count) 반환."""
    dummy = tmp_path / "in.wav"
    dummy.write_text("dummy")
    with patch.object(Path, "exists", return_value=True), \
         patch.object(Path, "stat") as mock_stat, \
         patch("os.unlink"), \
         patch("app.stt_processor.mask_audio_ranges", wraps=mask_audio_ranges) as spy_mask:
        mock_stat.return_value.st_size = 1000
        result = transcribe(
            str(dummy),
            task_id="t_pii_only",
            mask_pii=True,
            mask_audio_pii=mask_audio_pii,
            pii_intervals_only=pii_intervals_only,
        )
    return result, spy_mask.call_count


def _preprocessed_wav(result):
    return result["_audio_files"]["_preprocessed_audio.wav"]


def _phone_time_ranges(result):
    for item in result["pii_summary"]:
        if item["type"] == "전화번호":
            return item.get("time_ranges")
    return None


# T1 — bit-identical WAV (핵심 게이트): (F,F) vs (F,T) 저장 WAV 바이트 동일
def test_T1_intervals_only_wav_bit_identical_to_no_pii(mock_whisperx_deps, tmp_path):
    res_ff, mask_ff = _run_transcribe(tmp_path, mask_audio_pii=False, pii_intervals_only=False)
    res_ft, mask_ft = _run_transcribe(tmp_path, mask_audio_pii=False, pii_intervals_only=True)

    # 두 경로 모두 audio 변형(beep) 호출이 일어나지 않아야 한다.
    assert mask_ff == 0
    assert mask_ft == 0
    # 저장되는 발화/전처리 WAV 가 바이트 단위로 동일해야 한다 = 원본성 보장.
    assert _preprocessed_wav(res_ft) == _preprocessed_wav(res_ff)


# T1b — (T,F) beep 경로는 WAV 가 달라져야 한다(마스킹이 실제로 오디오를 바꿈)
def test_T1b_mask_audio_pii_changes_wav(mock_whisperx_deps, tmp_path):
    res_ff, _ = _run_transcribe(tmp_path, mask_audio_pii=False, pii_intervals_only=False)
    res_tf, mask_tf = _run_transcribe(tmp_path, mask_audio_pii=True, pii_intervals_only=False)

    assert mask_tf >= 1  # beep 호출됨
    assert _preprocessed_wav(res_tf) != _preprocessed_wav(res_ff)


# T2 — interval 산출 확인: (F,T) 에서 time_ranges 가 emit 된다
def test_T2_intervals_only_emits_time_ranges(mock_whisperx_deps, tmp_path):
    res_ft, _ = _run_transcribe(tmp_path, mask_audio_pii=False, pii_intervals_only=True)
    ranges = _phone_time_ranges(res_ft)
    assert ranges is not None and len(ranges) > 0
    # pad 0.15 적용 → 1.5 - 0.15 = 1.35
    assert ranges[0]["start"] == pytest.approx(1.35)

    # 대조군: 두 플래그 모두 off 면 time_ranges 가 없어야 한다.
    res_ff, _ = _run_transcribe(tmp_path, mask_audio_pii=False, pii_intervals_only=False)
    assert _phone_time_ranges(res_ff) is None


# T3 — beep 미발생 확인: (F,T) 는 mask_audio_ranges 를 호출하지 않는다
def test_T3_intervals_only_does_not_beep(mock_whisperx_deps, tmp_path):
    _, mask_ft = _run_transcribe(tmp_path, mask_audio_pii=False, pii_intervals_only=True)
    assert mask_ft == 0


# T4 / (T,T) — lenient lock: 둘 다 켜면 beep 적용(오디오 지배) + interval 도 emit
def test_T4_both_flags_beep_dominates_and_emits_intervals(mock_whisperx_deps, tmp_path):
    res_tt, mask_tt = _run_transcribe(tmp_path, mask_audio_pii=True, pii_intervals_only=True)
    res_tf, _ = _run_transcribe(tmp_path, mask_audio_pii=True, pii_intervals_only=False)
    res_ff, _ = _run_transcribe(tmp_path, mask_audio_pii=False, pii_intervals_only=False)

    assert mask_tt >= 1  # beep 적용됨(오디오 지배)
    # (T,T) 오디오는 (T,F) 와 동일(beep 동일) — pii_intervals_only 가 beep 를 약화시키지 않음
    assert _preprocessed_wav(res_tt) == _preprocessed_wav(res_tf)
    # 그리고 (F,F) 와는 달라야 함(실제 마스킹됨)
    assert _preprocessed_wav(res_tt) != _preprocessed_wav(res_ff)
    # interval 도 함께 emit
    assert _phone_time_ranges(res_tt) is not None


# ---------------------------------------------------------------------------
# 청크 모드 (_transcribe_chunked) — advisor 가 지적한 별도 게이트 경로
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_chunk_deps():
    with patch("app.stt_processor._detect_silence_points") as mock_silence, \
         patch("app.stt_processor._extract_chunk"), \
         patch("whisperx.load_audio") as mock_load, \
         patch("app.stt_processor.preprocess") as mock_preprocess, \
         patch("app.stt_processor._transcribe_chunk") as mock_trans_chunk, \
         patch("app.stt_processor.emit_chunk_utterances") as mock_emit, \
         patch("app.stt_processor._diarize_model", MagicMock()):

        mock_silence.return_value = [5.0]
        mock_load.return_value = np.zeros(16000 * 5, dtype=np.float32)
        mock_preprocess.side_effect = lambda x, sr: x
        mock_trans_chunk.return_value = [{
            "text": "전화번호는 010-1111-2222입니다.",
            "start": 1.0,
            "end": 4.0,
            "speaker": "SPEAKER_00",
            "words": [
                {"word": "전화번호는", "start": 1.0, "end": 1.5, "speaker": "SPEAKER_00"},
                {"word": "010-1111-2222입니다.", "start": 1.5, "end": 4.0, "speaker": "SPEAKER_00"},
            ],
        }]
        mock_emit.return_value = ([], {}, 0)
        yield {"emit": mock_emit}


def _run_chunked(*, mask_audio_pii, pii_intervals_only):
    with patch("app.stt_processor.mask_audio_ranges", wraps=mask_audio_ranges) as spy_mask, \
         patch("app.config.CHUNK_DURATION_SEC", 5), \
         patch("app.config.CHUNK_MARGIN_SEC", 2):
        segments, utterances, audio_files, pii_ranges = _transcribe_chunked(
            Path("dummy.wav"),
            task_id="chunk_pii_only",
            total_duration=10.0,
            enable_diarize=True,
            split_by_utterance=True,
            mask_audio_pii=mask_audio_pii,
            pii_intervals_only=pii_intervals_only,
        )
    return pii_ranges, spy_mask.call_count


# 청크 T1 — (F,T) 는 range 만 산출하고 beep 는 호출하지 않는다(저장 WAV 원본 유지)
def test_chunk_intervals_only_emits_ranges_without_beep(mock_chunk_deps):
    pii_ranges, mask_calls = _run_chunked(mask_audio_pii=False, pii_intervals_only=True)
    assert len(pii_ranges) == 2          # 청크당 1개 = range 산출됨
    assert mask_calls == 0               # beep 미발생 = chunk_audio 미변형 → emit WAV 원본 동일


# 청크 (F,F) — 아무 플래그도 없으면 range 도 beep 도 없다
def test_chunk_no_flags_no_ranges_no_beep(mock_chunk_deps):
    pii_ranges, mask_calls = _run_chunked(mask_audio_pii=False, pii_intervals_only=False)
    assert len(pii_ranges) == 0
    assert mask_calls == 0


# 청크 (T,F) — 기존 mask_audio_pii beep 동작 불변(청크당 beep)
def test_chunk_mask_audio_pii_beeps_per_chunk(mock_chunk_deps):
    pii_ranges, mask_calls = _run_chunked(mask_audio_pii=True, pii_intervals_only=False)
    assert len(pii_ranges) == 2
    assert mask_calls == 2


# 청크 bit-identical — emit 에 전달되는 chunk_audio 가 (F,F) 와 (F,T) 에서 바이트 동일
def test_chunk_emit_audio_bit_identical_ff_vs_ft(mock_chunk_deps):
    emit = mock_chunk_deps["emit"]

    def captured_chunk_audios():
        return [call.args[0].copy() for call in emit.call_args_list]

    emit.reset_mock()
    _run_chunked(mask_audio_pii=False, pii_intervals_only=False)
    ff_audios = captured_chunk_audios()

    emit.reset_mock()
    _run_chunked(mask_audio_pii=False, pii_intervals_only=True)
    ft_audios = captured_chunk_audios()

    assert len(ff_audios) == len(ft_audios) == 2
    for a_ff, a_ft in zip(ff_audios, ft_audios):
        assert np.array_equal(a_ff, a_ft)  # interval-only 가 chunk_audio 를 변형하지 않음
