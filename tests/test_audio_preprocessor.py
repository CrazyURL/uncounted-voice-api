"""Tests for audio preprocessing pipeline."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.services.audio_preprocessor import (
    compress_silence,
    denoise,
    local_normalize_gain,
    normalize_gain,
    preprocess,
    remove_duplicates,
)

SR = 16000


def _sine(freq: float, duration: float, sr: int = SR) -> np.ndarray:
    """Generate a sine wave at the given frequency and duration."""
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    return np.sin(2 * np.pi * freq * t)


def _silence(duration: float, sr: int = SR) -> np.ndarray:
    """Generate silence of the given duration."""
    return np.zeros(int(sr * duration), dtype=np.float32)


# ---------------------------------------------------------------------------
# TestDenoise
# ---------------------------------------------------------------------------

class TestDenoise:
    """Denoise는 DeepFilterNet 상주 subprocess 워커 기반이다.

    워커가 로드되지 않았거나 죽어있으면 입력을 그대로 반환한다.
    파일 IPC 경로(input.raw → request → output.raw)는 통합 테스트로 분리.
    """

    def test_returns_input_when_worker_not_ready(self):
        audio = _sine(440, 1.0)
        with patch("app.services.audio_preprocessor._ensure_worker", return_value=False):
            result = denoise(audio, SR)
        np.testing.assert_array_equal(result, audio)

    def test_returns_ndarray_same_dtype(self):
        audio = _sine(440, 1.0).astype(np.float32)
        with patch("app.services.audio_preprocessor._ensure_worker", return_value=False):
            result = denoise(audio, SR)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_does_not_mutate_input(self):
        audio = _sine(440, 1.0)
        original = audio.copy()
        with patch("app.services.audio_preprocessor._ensure_worker", return_value=False):
            denoise(audio, SR)
        np.testing.assert_array_equal(audio, original)

    def test_skips_when_worker_none(self):
        audio = _sine(440, 1.0)
        with patch("app.services.audio_preprocessor._df_process", new=None):
            result = denoise(audio, SR)
        np.testing.assert_array_equal(result, audio)

    def test_output_is_1d(self):
        audio = _sine(440, 1.0)
        with patch("app.services.audio_preprocessor._ensure_worker", return_value=False):
            result = denoise(audio, SR)
        assert result.ndim == 1
        assert result.shape == audio.shape

    def test_handles_silence_input(self):
        audio = _silence(1.0)
        with patch("app.services.audio_preprocessor._ensure_worker", return_value=False):
            result = denoise(audio, SR)
        assert len(result) == len(audio)


# ---------------------------------------------------------------------------
# TestRemoveDuplicates
# ---------------------------------------------------------------------------

class TestRemoveDuplicates:
    def test_removes_repeated_segment(self):
        # Arrange — same 3-second segment repeated twice
        segment = _sine(440, 3.0)
        audio = np.concatenate([segment, segment])

        # Act
        result = remove_duplicates(audio, SR)

        # Assert — should be shorter (one copy removed)
        assert len(result) < len(audio)
        assert len(result) >= len(segment)

    def test_keeps_unique_segments(self):
        # Arrange — two different frequencies
        seg_a = _sine(440, 3.0)
        seg_b = _sine(880, 3.0)
        audio = np.concatenate([seg_a, seg_b])

        # Act
        result = remove_duplicates(audio, SR)

        # Assert — no removal, length preserved
        assert len(result) == len(audio)

    def test_skips_silent_regions(self):
        # Arrange — two silence blocks (should NOT be flagged as duplicates)
        audio = np.concatenate([_silence(3.0), _sine(440, 1.0), _silence(3.0)])

        # Act
        result = remove_duplicates(audio, SR)

        # Assert — no removal
        assert len(result) == len(audio)

    def test_does_not_mutate_input(self):
        # Arrange
        segment = _sine(440, 3.0)
        audio = np.concatenate([segment, segment])
        original = audio.copy()

        # Act
        remove_duplicates(audio, SR)

        # Assert
        np.testing.assert_array_equal(audio, original)

    def test_handles_short_audio(self):
        # Arrange — shorter than one window
        audio = _sine(440, 1.0)

        # Act
        result = remove_duplicates(audio, SR)

        # Assert — returned as-is
        np.testing.assert_array_equal(result, audio)


# ---------------------------------------------------------------------------
# TestCompressSilence
# ---------------------------------------------------------------------------

class TestCompressSilence:
    def test_compresses_long_silence(self):
        # Arrange — speech + 3s silence + speech
        speech = _sine(440, 1.0) * 0.5
        silence = _silence(3.0)
        audio = np.concatenate([speech, silence, speech])

        # Act
        result = compress_silence(audio, SR)

        # Assert — 3s silence compressed to ~0.5s
        expected_max = len(speech) * 2 + int(0.5 * SR) + SR  # some tolerance
        assert len(result) < len(audio)
        assert len(result) < expected_max

    def test_preserves_short_silence(self):
        # Arrange — speech + 0.3s silence + speech (below threshold)
        speech = _sine(440, 1.0) * 0.5
        silence = _silence(0.3)
        audio = np.concatenate([speech, silence, speech])

        # Act
        result = compress_silence(audio, SR)

        # Assert — no compression, length preserved
        assert len(result) == len(audio)

    def test_does_not_mutate_input(self):
        # Arrange
        speech = _sine(440, 1.0) * 0.5
        silence = _silence(3.0)
        audio = np.concatenate([speech, silence, speech])
        original = audio.copy()

        # Act
        compress_silence(audio, SR)

        # Assert
        np.testing.assert_array_equal(audio, original)

    def test_output_shorter_with_gaps(self):
        # Arrange — multiple long silence gaps
        speech = _sine(440, 0.5) * 0.5
        silence = _silence(2.0)
        audio = np.concatenate([speech, silence, speech, silence, speech])

        # Act
        result = compress_silence(audio, SR)

        # Assert — both gaps compressed
        assert len(result) < len(audio)
        # Original: 0.5*3 + 2.0*2 = 5.5s → compressed: 0.5*3 + 0.5*2 = 2.5s
        expected_approx = int(2.5 * SR)
        assert abs(len(result) - expected_approx) < SR  # within 1s tolerance


# ---------------------------------------------------------------------------
# TestPreprocess
# ---------------------------------------------------------------------------

class TestPreprocess:
    """Preprocess 파이프라인은 config flag 기반으로 각 단계가 독립 제어된다."""

    @staticmethod
    def _enable_all(monkeypatch):
        from app import config as cfg
        monkeypatch.setattr(cfg, "PREPROCESS_GAIN_ENABLED", True)
        monkeypatch.setattr(cfg, "PREPROCESS_DENOISE_ENABLED", True)
        monkeypatch.setattr(cfg, "PREPROCESS_DEDUP_ENABLED", True)
        monkeypatch.setattr(cfg, "PREPROCESS_SILENCE_ENABLED", True)

    @staticmethod
    def _disable_all(monkeypatch):
        from app import config as cfg
        monkeypatch.setattr(cfg, "PREPROCESS_GAIN_ENABLED", False)
        monkeypatch.setattr(cfg, "PREPROCESS_DENOISE_ENABLED", False)
        monkeypatch.setattr(cfg, "PREPROCESS_DEDUP_ENABLED", False)
        monkeypatch.setattr(cfg, "PREPROCESS_SILENCE_ENABLED", False)

    @patch("app.services.audio_preprocessor.denoise")
    def test_chains_all_steps_when_all_enabled(self, mock_denoise, monkeypatch):
        # Arrange — all flags on, denoise mocked to pass through
        self._enable_all(monkeypatch)
        speech = _sine(440, 1.0) * 0.5
        silence = _silence(3.0)
        audio = np.concatenate([speech, silence, speech])
        mock_denoise.side_effect = lambda a, sr: a.copy()

        # Act
        result = preprocess(audio, SR)

        # Assert — denoise called, silence compressed
        mock_denoise.assert_called_once()
        assert len(result) < len(audio)

    @patch("app.services.audio_preprocessor.denoise")
    def test_returns_numpy_float32(self, mock_denoise, monkeypatch):
        # Arrange
        self._enable_all(monkeypatch)
        audio = _sine(440, 2.0).astype(np.float32)
        mock_denoise.side_effect = lambda a, sr: a.copy()

        # Act
        result = preprocess(audio, SR)

        # Assert
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    @patch("app.services.audio_preprocessor.denoise")
    def test_all_flags_off_returns_unchanged_length(self, mock_denoise, monkeypatch):
        # Arrange — all flags off
        self._disable_all(monkeypatch)
        audio = _sine(440, 2.0).astype(np.float32) * 0.5
        silence = _silence(3.0)
        full = np.concatenate([audio, silence, audio])
        mock_denoise.side_effect = lambda a, sr: a.copy()

        # Act
        result = preprocess(full, SR)

        # Assert — denoise never called, length preserved
        mock_denoise.assert_not_called()
        assert len(result) == len(full)

    @patch("app.services.audio_preprocessor.denoise")
    def test_only_gain_enabled_round_1(self, mock_denoise, monkeypatch):
        # Arrange — Round 1 default: only gain
        from app import config as cfg
        monkeypatch.setattr(cfg, "PREPROCESS_GAIN_ENABLED", True)
        monkeypatch.setattr(cfg, "PREPROCESS_DENOISE_ENABLED", False)
        monkeypatch.setattr(cfg, "PREPROCESS_DEDUP_ENABLED", False)
        monkeypatch.setattr(cfg, "PREPROCESS_SILENCE_ENABLED", False)

        # Low-amplitude audio so gain actually applies
        audio = (_sine(440, 2.0) * 0.02).astype(np.float32)
        silence = _silence(3.0).astype(np.float32)
        full = np.concatenate([audio, silence, audio])
        original_len = len(full)

        # Act
        result = preprocess(full, SR)

        # Assert — denoise not called, length preserved (no silence compress),
        # gain boosted amplitude
        mock_denoise.assert_not_called()
        assert len(result) == original_len
        orig_rms = float(np.sqrt(np.mean(full ** 2)))
        new_rms = float(np.sqrt(np.mean(result ** 2)))
        assert new_rms > orig_rms  # gain amplified

    @patch("app.services.audio_preprocessor.denoise")
    def test_denoise_flag_controls_invocation(self, mock_denoise, monkeypatch):
        # Arrange — denoise off, others off
        self._disable_all(monkeypatch)
        audio = _sine(440, 1.0).astype(np.float32)
        mock_denoise.side_effect = lambda a, sr: a.copy()

        # Act
        preprocess(audio, SR)

        # Assert
        mock_denoise.assert_not_called()

        # Flip denoise on → should be called
        from app import config as cfg
        monkeypatch.setattr(cfg, "PREPROCESS_DENOISE_ENABLED", True)
        preprocess(audio, SR)
        mock_denoise.assert_called_once()


# ---------------------------------------------------------------------------
# TestNormalizeGain
# ---------------------------------------------------------------------------

class TestNormalizeGain:
    """normalize_gain() — 글로벌 RMS 정규화 (cap=MAX_GAIN_X=10x)."""

    def test_boosts_quiet_audio_to_target_rms(self):
        # sine RMS = amplitude/sqrt(2). 0.005*sine → RMS≈0.00354
        # gain=28x이지만 MAX_GAIN_X=10x로 제한 → RMS≈0.0354
        audio = _sine(440, 1.0) * 0.005
        result = normalize_gain(audio)
        input_rms = float(np.sqrt(np.mean(audio ** 2)))
        result_rms = float(np.sqrt(np.mean(result ** 2)))
        assert result_rms > input_rms * 9.0  # 9x 이상 부스트 확인 (상한 10x)

    def test_does_not_boost_loud_audio(self):
        # RMS가 이미 TARGET(0.1) 이상이면 변경 없음
        audio = _sine(440, 1.0) * 0.5
        result = normalize_gain(audio)
        np.testing.assert_array_almost_equal(result, audio)

    def test_caps_at_max_gain_x(self, monkeypatch):
        # MAX_GAIN_X=5일 때 극단적으로 조용한 오디오에 5x 이상 부스트 안됨
        from app import config as cfg
        monkeypatch.setattr(cfg, "MAX_GAIN_X", 5.0)
        audio = _sine(440, 1.0) * 0.001
        result = normalize_gain(audio)
        input_rms = float(np.sqrt(np.mean(audio ** 2)))
        result_rms = float(np.sqrt(np.mean(result ** 2)))
        assert result_rms <= input_rms * 5.0 * 1.01  # 5x 이상 부스트 없음

    def test_does_not_mutate_input(self):
        audio = _sine(440, 1.0) * 0.01
        original = audio.copy()
        normalize_gain(audio)
        np.testing.assert_array_equal(audio, original)

    def test_handles_silence(self):
        audio = _silence(1.0)
        result = normalize_gain(audio)
        np.testing.assert_array_equal(result, audio)

    def test_output_clipped_within_bounds(self):
        audio = _sine(440, 1.0) * 0.001
        result = normalize_gain(audio)
        assert float(np.max(np.abs(result))) <= 1.0


# ---------------------------------------------------------------------------
# TestLocalNormalizeGain
# ---------------------------------------------------------------------------

class TestLocalNormalizeGain:
    """local_normalize_gain() — 슬라이딩 윈도우 로컬 게인 (cap=LOCAL_MAX_GAIN_X=30x).

    핵심 케이스:
    1. 매우 조용한 오디오가 글로벌 10x 이후에도 부족하면 로컬 30x로 추가 부스트
    2. 이미 충분히 큰 구간은 감쇠 없음 (부스트 전용)
    3. 노이즈만 있는 무음 구간 과도 증폭 없음
    4. LOCAL_MAX_GAIN_X=10x이면 MAX_GAIN_X=10x와 동일 동작
    """

    def test_boosts_very_quiet_audio_beyond_global_cap(self, monkeypatch):
        # 글로벌 10x 후에도 RMS가 낮은 경우 로컬 30x가 추가 부스트
        from app import config as cfg
        monkeypatch.setattr(cfg, "LOCAL_MAX_GAIN_X", 30.0)

        # RMS 0.001 → 글로벌 10x → RMS ≈ 0.01 (여전히 낮음)
        # local 30x → 해당 윈도우 RMS ≈ 0.1 도달 가능
        audio = _sine(440, 2.0) * 0.001
        globally_boosted = normalize_gain(audio)  # 10x 후
        result = local_normalize_gain(globally_boosted, SR)

        result_rms = float(np.sqrt(np.mean(result ** 2)))
        boosted_rms = float(np.sqrt(np.mean(globally_boosted ** 2)))
        assert result_rms > boosted_rms  # 로컬 부스트가 추가로 작동

    def test_local_cap_higher_than_global_cap(self, monkeypatch):
        # LOCAL_MAX_GAIN_X=30이 MAX_GAIN_X=10보다 높으므로 더 많은 부스트 가능
        from app import config as cfg
        monkeypatch.setattr(cfg, "MAX_GAIN_X", 10.0)
        monkeypatch.setattr(cfg, "LOCAL_MAX_GAIN_X", 30.0)

        very_quiet = _sine(440, 2.0) * 0.001
        result_local30 = local_normalize_gain(very_quiet.copy(), SR)

        monkeypatch.setattr(cfg, "LOCAL_MAX_GAIN_X", 10.0)
        result_local10 = local_normalize_gain(very_quiet.copy(), SR)

        rms_30 = float(np.sqrt(np.mean(result_local30 ** 2)))
        rms_10 = float(np.sqrt(np.mean(result_local10 ** 2)))
        assert rms_30 >= rms_10  # 30x cap이 더 높은 RMS 도달

    def test_does_not_attenuate_loud_segments(self, monkeypatch):
        # 이미 충분히 큰 구간(RMS >= TARGET)은 gain=1.0 유지
        from app import config as cfg
        monkeypatch.setattr(cfg, "LOCAL_MAX_GAIN_X", 30.0)

        audio = _sine(440, 2.0) * 0.5  # RMS ≈ 0.35 (TARGET=0.1보다 큼)
        result = local_normalize_gain(audio, SR)

        # 감쇠 없음 — 출력이 입력보다 작아선 안 됨
        result_rms = float(np.sqrt(np.mean(result ** 2)))
        input_rms = float(np.sqrt(np.mean(audio ** 2)))
        assert result_rms >= input_rms * 0.99

    def test_output_clipped_within_bounds(self, monkeypatch):
        from app import config as cfg
        monkeypatch.setattr(cfg, "LOCAL_MAX_GAIN_X", 30.0)

        audio = _sine(440, 2.0) * 0.001
        result = local_normalize_gain(audio, SR)
        assert float(np.max(np.abs(result))) <= 1.0

    def test_does_not_mutate_input(self, monkeypatch):
        from app import config as cfg
        monkeypatch.setattr(cfg, "LOCAL_MAX_GAIN_X", 30.0)

        audio = _sine(440, 2.0) * 0.005
        original = audio.copy()
        local_normalize_gain(audio, SR)
        np.testing.assert_array_equal(audio, original)

    def test_handles_very_short_audio(self, monkeypatch):
        # 500ms 윈도우보다 짧으면 그대로 반환
        from app import config as cfg
        monkeypatch.setattr(cfg, "LOCAL_MAX_GAIN_X", 30.0)

        audio = _sine(440, 0.2)  # 200ms < 500ms window
        result = local_normalize_gain(audio, SR)
        np.testing.assert_array_equal(result, audio)

    def test_silence_segments_not_over_amplified(self, monkeypatch):
        # 진짜 무음 구간(rms < 1e-7)은 gain=1.0 유지 — 폭발 방지
        from app import config as cfg
        monkeypatch.setattr(cfg, "LOCAL_MAX_GAIN_X", 30.0)

        silence = _silence(1.0)
        result = local_normalize_gain(silence, SR)

        # 무음은 무음으로 유지
        result_rms = float(np.sqrt(np.mean(result ** 2)))
        assert result_rms < 1e-5

    def test_quiet_speech_followed_by_silence(self, monkeypatch):
        # 조용한 발화 + 무음 구간 혼합 — 발화 구간만 부스트되어야 함
        from app import config as cfg
        monkeypatch.setattr(cfg, "LOCAL_MAX_GAIN_X", 30.0)

        speech = _sine(440, 1.0) * 0.002
        silence = _silence(1.0)
        audio = np.concatenate([speech, silence]).astype(np.float32)

        result = local_normalize_gain(audio, SR)

        speech_rms_out = float(np.sqrt(np.mean(result[:SR] ** 2)))
        silence_rms_out = float(np.sqrt(np.mean(result[SR:] ** 2)))

        assert speech_rms_out > silence_rms_out * 10  # 발화 구간이 무음보다 월등히 큼


# ---------------------------------------------------------------------------
# TestGainPipelineIntegration  (normalize_gain → local_normalize_gain 연계)
# ---------------------------------------------------------------------------

class TestGainPipelineIntegration:
    """글로벌 → 로컬 게인 순차 적용 통합 케이스."""

    def test_very_quiet_audio_reaches_adequate_rms(self, monkeypatch):
        # RMS 0.001 오디오가 파이프라인 후 VAD 감지 가능 수준(≥0.05)에 도달
        from app import config as cfg
        monkeypatch.setattr(cfg, "MAX_GAIN_X", 10.0)
        monkeypatch.setattr(cfg, "LOCAL_MAX_GAIN_X", 30.0)

        audio = _sine(440, 2.0) * 0.001
        step1 = normalize_gain(audio)
        step2 = local_normalize_gain(step1, SR)

        final_rms = float(np.sqrt(np.mean(step2 ** 2)))
        assert final_rms >= 0.05  # silero VAD가 감지할 수 있는 수준

    def test_local30_outperforms_local10_on_quiet_audio(self, monkeypatch):
        # LOCAL_MAX_GAIN_X=30이 10일 때보다 조용한 오디오를 더 잘 끌어올림
        from app import config as cfg
        monkeypatch.setattr(cfg, "MAX_GAIN_X", 10.0)

        audio = _sine(440, 2.0) * 0.001
        step1 = normalize_gain(audio)

        monkeypatch.setattr(cfg, "LOCAL_MAX_GAIN_X", 30.0)
        result_30 = local_normalize_gain(step1.copy(), SR)

        monkeypatch.setattr(cfg, "LOCAL_MAX_GAIN_X", 10.0)
        result_10 = local_normalize_gain(step1.copy(), SR)

        rms_30 = float(np.sqrt(np.mean(result_30 ** 2)))
        rms_10 = float(np.sqrt(np.mean(result_10 ** 2)))
        assert rms_30 > rms_10


# ---------------------------------------------------------------------------
# ⑤ Loudness Normalization (LUFS) — 2026-06-02
# ---------------------------------------------------------------------------

pytest.importorskip("pyloudnorm")

from app.services import audio_preprocessor as _ap  # noqa: E402
from app.services.audio_preprocessor import (  # noqa: E402
    measure_integrated_lufs,
    normalize_loudness_call,
    loudness_gated_local_gain,
)
from app import config as _cfg  # noqa: E402


class TestMeasureLufs:
    def test_short_clip_returns_none(self):
        # < 0.4s → pyloudnorm 최소블록 미달 → None
        assert measure_integrated_lufs(_sine(440, 0.2), SR) is None

    def test_silence_returns_none(self):
        assert measure_integrated_lufs(_silence(2.0), SR) is None

    def test_louder_has_higher_lufs(self):
        quiet = measure_integrated_lufs(_sine(440, 2.0) * 0.05, SR)
        loud = measure_integrated_lufs(_sine(440, 2.0) * 0.5, SR)
        assert quiet is not None and loud is not None
        assert loud > quiet

    def test_pyloudnorm_missing_returns_none_no_raise(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == "pyloudnorm":
                raise ImportError("simulated")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert measure_integrated_lufs(_sine(440, 2.0), SR) is None


class TestNormalizeLoudnessCall:
    def test_byte_identical_when_unmeasurable(self):
        # 무음 → 측정 불가 → 입력 그대로
        sil = _silence(2.0)
        out = normalize_loudness_call(sil, SR, target_lufs=-16.0, peak_ceiling_dbfs=-1.0)
        assert np.array_equal(out, sil)

    def test_uniform_gain_preserves_crest_factor(self):
        # 균일 gain → peak/rms 비(=snr_db) 보존 → quality_grade 중립 (회귀잠금)
        audio = _sine(440, 2.0) * 0.05
        crest_in = float(np.max(np.abs(audio))) / float(np.sqrt(np.mean(audio ** 2)))
        out = normalize_loudness_call(audio, SR, target_lufs=-16.0, peak_ceiling_dbfs=-1.0)
        crest_out = float(np.max(np.abs(out))) / float(np.sqrt(np.mean(out ** 2)))
        assert abs(crest_in - crest_out) < 0.05   # crest factor 보존

    def test_respects_peak_ceiling(self):
        # 큰 신호를 target 으로 올려도 peak ceiling 초과 금지
        audio = _sine(440, 2.0) * 0.3
        out = normalize_loudness_call(audio, SR, target_lufs=0.0, peak_ceiling_dbfs=-1.0)
        ceiling = 10.0 ** (-1.0 / 20.0)
        assert float(np.max(np.abs(out))) <= ceiling + 1e-3

    def test_quiet_call_boosted_toward_target(self):
        audio = _sine(440, 3.0) * 0.02
        out = normalize_loudness_call(audio, SR, target_lufs=-16.0, peak_ceiling_dbfs=-1.0)
        after = measure_integrated_lufs(out, SR)
        # ceiling 에 안 걸리는 한 target 근처(±2 LU). 작은 sine 은 crest 낮아 여유.
        assert after is not None and after > measure_integrated_lufs(audio, SR)

    def test_does_not_mutate_input(self):
        audio = _sine(440, 2.0) * 0.05
        copy = audio.copy()
        normalize_loudness_call(audio, SR, target_lufs=-16.0, peak_ceiling_dbfs=-1.0)
        assert np.array_equal(audio, copy)


class TestLoudnessGatedLocalGain:
    def test_noise_floor_window_not_boosted(self, monkeypatch):
        # 무음 + 미세 white noise → noise floor 근처 → 미부스트(noise breathing 차단, 회귀잠금)
        monkeypatch.setattr(_cfg, "LOCAL_MAX_GAIN_X", 10.0)
        rng = np.random.default_rng(0)
        noise = (rng.standard_normal(SR * 2) * 1e-4).astype(np.float32)
        out = loudness_gated_local_gain(noise, SR)
        # 게이팅으로 RMS 가 크게 증폭되지 않아야(현행 local 은 10x 부스트)
        rms_in = float(np.sqrt(np.mean(noise ** 2)))
        rms_out = float(np.sqrt(np.mean(out ** 2)))
        assert rms_out < rms_in * 2.0   # 거의 미부스트

    def test_real_speech_window_boosted(self, monkeypatch):
        monkeypatch.setattr(_cfg, "LOCAL_MAX_GAIN_X", 10.0)
        # 현실적 신호: 미세노이즈 floor + 조용한 음성 구간(noise floor 대비 충분히 큼).
        # 균일 진폭이면 noise floor==신호라 전부 게이팅되므로(설계대로), 변동을 준다.
        rng = np.random.default_rng(1)
        noise = (rng.standard_normal(SR * 4) * 1e-4).astype(np.float32)
        speech_seg = (_sine(300, 1.5) * 0.03).astype(np.float32)
        signal = noise.copy()
        signal[SR:SR + len(speech_seg)] += speech_seg   # 1~2.5s 구간에 음성
        out = loudness_gated_local_gain(signal, SR)
        # 음성 구간만 부스트되어야 함
        sp_in = float(np.sqrt(np.mean(signal[SR:SR + len(speech_seg)] ** 2)))
        sp_out = float(np.sqrt(np.mean(out[SR:SR + len(speech_seg)] ** 2)))
        assert sp_out > sp_in   # 음성 구간 부스트

    def test_output_within_bounds(self, monkeypatch):
        monkeypatch.setattr(_cfg, "LOCAL_MAX_GAIN_X", 10.0)
        out = loudness_gated_local_gain(_sine(440, 2.0) * 0.1, SR)
        assert float(np.max(np.abs(out))) <= 1.0


class TestPreprocessLoudnessGates:
    def test_loudness_gates_off_no_loudness_step(self, monkeypatch):
        # 게이트 전부 OFF → loudness/local_gain_gated 단계 미적용
        monkeypatch.setattr(_cfg, "CALL_LOUDNESS_NORM_ENABLED", False)
        monkeypatch.setattr(_cfg, "LOUDNESS_LOCAL_GATE_ENABLED", False)
        monkeypatch.setattr(_cfg, "PREPROCESS_GAIN_ENABLED", True)
        monkeypatch.setattr(_cfg, "PREPROCESS_DENOISE_ENABLED", False)
        monkeypatch.setattr(_cfg, "PREPROCESS_DEDUP_ENABLED", False)
        monkeypatch.setattr(_cfg, "PREPROCESS_SILENCE_ENABLED", False)
        audio = _sine(440, 2.0) * 0.05
        # 예외 없이 통과(기존 gain 경로). loudness 미적용이라 출력은 local_normalize_gain 결과.
        out = preprocess(audio.copy(), SR)
        assert out is not None and len(out) == len(audio)

    def test_call_loudness_on_applies_uniform_gain(self, monkeypatch):
        monkeypatch.setattr(_cfg, "CALL_LOUDNESS_NORM_ENABLED", True)
        monkeypatch.setattr(_cfg, "CALL_LOUDNESS_TARGET_LUFS", -16.0)
        monkeypatch.setattr(_cfg, "CALL_LOUDNESS_PEAK_DBFS", -1.0)
        monkeypatch.setattr(_cfg, "LOUDNESS_LOCAL_GATE_ENABLED", False)
        monkeypatch.setattr(_cfg, "PREPROCESS_GAIN_ENABLED", False)  # gain 단계 끄고 loudness만
        monkeypatch.setattr(_cfg, "PREPROCESS_DENOISE_ENABLED", False)
        monkeypatch.setattr(_cfg, "PREPROCESS_DEDUP_ENABLED", False)
        monkeypatch.setattr(_cfg, "PREPROCESS_SILENCE_ENABLED", False)
        audio = (_sine(440, 3.0) * 0.02).astype(np.float32)
        out = preprocess(audio.copy(), SR)
        # loudness 정규화로 음량 상승
        assert float(np.sqrt(np.mean(out ** 2))) > float(np.sqrt(np.mean(audio ** 2)))
