"""Audio preprocessing pipeline — denoise, deduplicate, compress silence.

Applied automatically before WhisperX transcription to improve STT quality.
Pipeline order: denoise → remove_duplicates → compress_silence.

DeepFilterNet은 별도 Python 프로세스(subprocess)에서 실행하여
메인 프로세스의 CUDA allocator와 완전히 격리한다.
"""

import logging
import os
import subprocess
import sys
import time

import numpy as np
from scipy.signal import fftconvolve

from app import config

logger = logging.getLogger(__name__)

# DeepFilterNet 상주 워커 프로세스 (파일 기반 통신)
_df_process: subprocess.Popen | None = None
_DF_WORKER_SCRIPT: str | None = None
_DF_WORK_DIR: str | None = None


def load_df_model() -> None:
    """DeepFilterNet 상주 워커 프로세스를 시작한다.

    파일 기반 프로토콜: 메인 프로세스가 input.raw를 쓰고 request 파일을 생성하면
    워커가 처리 후 output.raw를 쓰고 done 파일을 생성한다.
    파이프 deadlock 없이 대용량 오디오를 안전하게 전달할 수 있다.
    """
    global _DF_WORKER_SCRIPT, _df_process, _DF_WORK_DIR

    script = '''\
"""DeepFilterNet CPU-only daemon worker — 파일 기반 통신."""
import os, sys, time
os.environ["DF_DEVICE"] = "cpu"

import numpy as np
import torch

# Phase 1: monkey-patch로 init_df()가 CPU에서 모델 로딩하도록 강제
import df.modules
import df.enhance
_cpu = torch.device("cpu")
_cpu_fn = lambda: _cpu
df.modules.get_device = _cpu_fn
df.enhance.get_device = _cpu_fn
# enhance/init_df 등 함수 내부의 로컬 get_device 참조까지 교체
for _name, _obj in vars(df.enhance).items():
    if callable(_obj) and hasattr(_obj, "__globals__"):
        _obj.__globals__["get_device"] = _cpu_fn

from df.enhance import enhance, init_df

_model, _state, _ = init_df()
_model = _model.cpu()

work_dir = sys.argv[1]
open(os.path.join(work_dir, "ready"), "w").close()

while True:
    req_path = os.path.join(work_dir, "request")
    while not os.path.exists(req_path):
        time.sleep(0.01)
    os.remove(req_path)

    input_path = os.path.join(work_dir, "input.raw")
    output_path = os.path.join(work_dir, "output.raw")

    try:
        # input.raw가 완전히 쓰여질 때까지 대기
        for _ in range(100):
            if os.path.exists(input_path) and os.path.getsize(input_path) > 0:
                break
            time.sleep(0.01)
        audio = np.fromfile(input_path, dtype=np.float32).copy()
        with torch.no_grad():
            tensor = torch.from_numpy(audio).float().unsqueeze(0).contiguous()
            enhanced = enhance(_model, _state, tensor)
        result = enhanced.numpy() if hasattr(enhanced, "numpy") else np.array(enhanced)
        if result.ndim == 2:
            result = result.squeeze(0)
        result.astype(np.float32).tofile(output_path)
    except Exception as e:
        open(output_path, "wb").close()
        sys.stderr.write(f"ERROR: {e}\\n")
        sys.stderr.flush()

    open(os.path.join(work_dir, "done"), "w").close()
'''
    work_dir = config.TEMP_DIR / "df_worker"
    work_dir.mkdir(parents=True, exist_ok=True)
    _DF_WORK_DIR = str(work_dir)

    script_path = work_dir / "denoise_daemon.py"
    script_path.write_text(script)
    script_path.chmod(0o600)
    _DF_WORKER_SCRIPT = str(script_path)

    # 이전 시그널 파일 정리
    for f in ("ready", "request", "done", "input.raw", "output.raw"):
        p = work_dir / f
        if p.exists():
            p.unlink()

    # 상주 프로세스 시작 (stderr → 로그 파일)
    env = {**os.environ, "DF_DEVICE": "cpu"}
    stderr_log = open(str(work_dir / "daemon.log"), "w")
    _df_process = subprocess.Popen(
        [sys.executable, _DF_WORKER_SCRIPT, _DF_WORK_DIR],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=stderr_log,
        env=env,
    )

    # ready 시그널 대기
    ready_path = work_dir / "ready"
    deadline = time.time() + 60
    while time.time() < deadline:
        if ready_path.exists():
            ready_path.unlink()
            logger.info("DeepFilterNet 상주 워커 시작 완료 (PID %d)", _df_process.pid)
            return
        if _df_process.poll() is not None:
            logger.warning("DeepFilterNet 워커 종료됨 (exit %d)", _df_process.returncode)
            return
        time.sleep(0.1)
    logger.warning("DeepFilterNet 워커 시작 타임아웃 (60초)")


def _ensure_worker() -> bool:
    """워커가 살아있는지 확인하고 죽었으면 재시작. 사용 가능하면 True."""
    global _df_process
    if _df_process is None or _DF_WORK_DIR is None:
        return False
    # work_dir이 삭제됐으면 (/dev/shm 정리 등) 재시작
    if not os.path.isdir(_DF_WORK_DIR):
        logger.warning("DeepFilterNet work_dir 소멸 — 재시작")
        load_df_model()
    elif _df_process.poll() is not None:
        logger.warning("DeepFilterNet 워커 죽음 (exit %d) — 재시작", _df_process.returncode)
        load_df_model()
    return _df_process is not None and _df_process.poll() is None


# ---------------------------------------------------------------------------
# ① Noise Reduction (DeepFilterNet — 상주 subprocess, 파일 기반)
# ---------------------------------------------------------------------------

def denoise(audio: np.ndarray, sr: int) -> np.ndarray:
    """DeepFilterNet으로 배경 잡음을 제거한다.

    상주 워커와 파일 기반으로 통신하여 파이프 deadlock을 방지한다.
    input.raw → request 시그널 → 워커 처리 → output.raw → done 시그널.
    """
    from pathlib import Path

    if not _ensure_worker():
        logger.warning("DeepFilterNet 워커 미초기화 — denoise 건너뜀")
        return audio

    work_dir = Path(_DF_WORK_DIR)

    try:
        input_path = work_dir / "input.raw"
        output_path = work_dir / "output.raw"
        request_path = work_dir / "request"
        done_path = work_dir / "done"

        # done 시그널 정리 (이전 요청의 잔여)
        done_path.unlink(missing_ok=True)

        # 입력 쓰기
        audio.astype(np.float32).tofile(str(input_path))

        # request 시그널 생성 → 워커가 처리 시작
        request_path.touch()

        # done 시그널 대기
        audio_duration = len(audio) / sr
        timeout = max(60, audio_duration * 2 + 30)
        deadline = time.time() + timeout

        while time.time() < deadline:
            if done_path.exists():
                done_path.unlink(missing_ok=True)
                # 결과 읽기
                if output_path.exists() and output_path.stat().st_size > 0:
                    enhanced = np.fromfile(str(output_path), dtype=np.float32).copy()
                    output_path.unlink(missing_ok=True)
                    input_path.unlink(missing_ok=True)
                    return enhanced
                logger.warning("DeepFilterNet 워커 출력 비어있음 — denoise 건너뜀")
                return audio
            if _df_process and _df_process.poll() is not None:
                logger.warning("DeepFilterNet 워커 처리 중 죽음 — denoise 건너뜀")
                return audio
            time.sleep(0.05)

        logger.warning("DeepFilterNet 워커 타임아웃 (%.0fs) — denoise 건너뜀", timeout)
        return audio

    except Exception as e:
        logger.warning("DeepFilterNet 실패 — denoise 건너뜀: %s", e)
        return audio


# ---------------------------------------------------------------------------
# ② Duplicate Removal (cross-correlation)
# ---------------------------------------------------------------------------

def remove_duplicates(audio: np.ndarray, sr: int) -> np.ndarray:
    """오디오 내 반복 구간을 교차 상관으로 탐지하여 제거한다.

    슬라이딩 윈도우 방식으로 각 윈도우를 최대 MAX_DEDUP_LOOKAHEAD개의 뒷구간과만
    비교하여 O(n²) → O(n*K)로 최적화.
    무음 구간은 비교에서 제외하여 오탐을 방지한다.
    두 번째(뒤쪽) 구간을 제거하고 첫 번째를 유지한다.
    """
    window_samples = int(config.DUPLICATE_WINDOW_SEC * sr)
    if len(audio) < window_samples * 2:
        return audio

    # 윈도우 분할 + RMS 에너지 계산
    hop = window_samples
    windows = []
    for start in range(0, len(audio) - window_samples + 1, hop):
        segment = audio[start:start + window_samples]
        rms = np.sqrt(np.mean(segment ** 2))
        windows.append((start, segment, rms))

    # 제거 대상 윈도우 인덱스
    remove_set: set[int] = set()

    for i in range(len(windows)):
        if i in remove_set:
            continue
        start_i, seg_i, rms_i = windows[i]
        if rms_i < config.SILENCE_RMS_THRESHOLD:
            continue

        norm_i = np.linalg.norm(seg_i)
        if norm_i < 1e-10:
            continue

        # 슬라이딩 윈도우: i 이후 최대 K개 윈도우와만 비교
        max_j = min(i + config.MAX_DEDUP_LOOKAHEAD + 1, len(windows))
        for j in range(i + 1, max_j):
            if j in remove_set:
                continue
            start_j, seg_j, rms_j = windows[j]
            if rms_j < config.SILENCE_RMS_THRESHOLD:
                continue

            norm_j = np.linalg.norm(seg_j)
            if norm_j < 1e-10:
                continue

            corr = fftconvolve(seg_i, seg_j[::-1], mode="full")
            max_corr = np.max(corr) / (norm_i * norm_j)

            if max_corr > config.DUPLICATE_CORR_THRESHOLD:
                remove_set.add(j)
                logger.info(
                    "중복 감지: %.1fs~%.1fs ≈ %.1fs~%.1fs (corr=%.3f)",
                    start_i / sr, (start_i + window_samples) / sr,
                    start_j / sr, (start_j + window_samples) / sr,
                    max_corr,
                )

    if not remove_set:
        return audio

    # 제거 대상이 아닌 구간만 연결
    parts: list[np.ndarray] = []
    prev_end = 0
    removed_ranges = sorted(
        (windows[idx][0], windows[idx][0] + window_samples)
        for idx in remove_set
    )
    for rm_start, rm_end in removed_ranges:
        if rm_start > prev_end:
            parts.append(audio[prev_end:rm_start])
        prev_end = max(prev_end, rm_end)
    if prev_end < len(audio):
        parts.append(audio[prev_end:])

    result = np.concatenate(parts) if parts else np.array([], dtype=audio.dtype)
    logger.info("중복 제거: %d개 구간 제거 (%.1fs → %.1fs, lookahead=%d)",
                len(remove_set), len(audio) / sr, len(result) / sr,
                config.MAX_DEDUP_LOOKAHEAD)
    return result


# ---------------------------------------------------------------------------
# ③ Silence Compression (RMS energy)
# ---------------------------------------------------------------------------

def compress_silence(
    audio: np.ndarray,
    sr: int,
    rms_threshold: float | None = None,
) -> np.ndarray:
    """긴 무음 구간을 SILENCE_COMPRESS_TARGET_SEC(기본 0.5초)로 압축한다.

    20ms 프레임 단위로 RMS 에너지를 계산하여 무음을 탐지한다.
    SILENCE_COMPRESS_MIN_SEC(기본 1.0초) 초과하는 무음만 압축 대상.
    짧은 자연스러운 침묵은 보존한다.

    rms_threshold가 명시되면 해당 값을 사용(denoise 후 동적 하향용),
    None이면 config.SILENCE_RMS_THRESHOLD(기본 0.005)를 사용한다.
    """
    frame_samples = int(config.PREPROCESS_FRAME_MS / 1000 * sr)
    if frame_samples < 1 or len(audio) < frame_samples:
        return audio

    min_silence_samples = int(config.SILENCE_COMPRESS_MIN_SEC * sr)
    target_silence_samples = int(config.SILENCE_COMPRESS_TARGET_SEC * sr)
    threshold = rms_threshold if rms_threshold is not None else config.SILENCE_RMS_THRESHOLD

    # RMS 에너지 계산
    n_frames = len(audio) // frame_samples
    frames = audio[:n_frames * frame_samples].reshape(n_frames, frame_samples)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    is_silent = rms < threshold

    # 무음 구간 탐지
    silence_regions: list[tuple[int, int]] = []
    silence_start = None

    for i, silent in enumerate(is_silent):
        sample_idx = i * frame_samples
        if silent and silence_start is None:
            silence_start = sample_idx
        elif not silent and silence_start is not None:
            silence_end = sample_idx
            duration_samples = silence_end - silence_start
            if duration_samples > min_silence_samples:
                silence_regions.append((silence_start, silence_end))
            silence_start = None

    # 마지막 무음 구간
    if silence_start is not None:
        silence_end = n_frames * frame_samples
        if silence_end - silence_start > min_silence_samples:
            silence_regions.append((silence_start, silence_end))

    if not silence_regions:
        return audio

    # 압축: 긴 무음 → 0.5초 무음으로 교체
    parts: list[np.ndarray] = []
    prev_end = 0
    for start, end in silence_regions:
        parts.append(audio[prev_end:start])
        parts.append(np.zeros(target_silence_samples, dtype=audio.dtype))
        prev_end = end
    if prev_end < len(audio):
        parts.append(audio[prev_end:])

    result = np.concatenate(parts)
    logger.info("무음 압축: %d개 구간 (%.1fs → %.1fs)",
                len(silence_regions), len(audio) / sr, len(result) / sr)
    return result


# ---------------------------------------------------------------------------
# ④ Gain Normalization (RMS-based)
# ---------------------------------------------------------------------------

TARGET_GAIN_RMS = 0.1   # Java AudioProcessor.TARGET_GAIN_RMS와 동일


def normalize_gain(audio: np.ndarray) -> np.ndarray:
    """RMS 기반 게인 정규화. 통화 녹음의 낮은 진폭을 보정한다.

    RMS가 TARGET_GAIN_RMS보다 낮을 때만 게인을 적용하여 볼륨을 끌어올린다.
    이미 충분히 큰 신호는 변경하지 않는다 (클리핑 방지).
    최대 증폭은 config.MAX_GAIN_X(기본 10x)로 제한하여 노이즈 증폭을 억제한다.
    """
    if len(audio) == 0:
        return audio

    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 1e-7:
        return audio

    gain = TARGET_GAIN_RMS / rms
    if gain <= 1.0:
        return audio  # 이미 충분히 큼

    gain = min(gain, config.MAX_GAIN_X)
    logger.info("게인 정규화: rms=%.4f → gain=%.2fx", rms, gain)

    return np.clip(audio * gain, -1.0, 1.0).astype(audio.dtype)


def local_normalize_gain(audio: np.ndarray, sr: int) -> np.ndarray:
    """슬라이딩 윈도우 로컬 게인 정규화.

    글로벌 정규화로 부스트되지 않는 조용한 구간(끝부분 등)을 VAD가 감지할 수 있도록
    500ms 윈도우 단위로 독립적으로 부스트한다. 윈도우 간 선형 보간으로 부드럽게 연결.
    이미 충분히 큰 구간(gain ≤ 1.0)은 건드리지 않는다.
    """
    if len(audio) == 0:
        return audio

    window_samples = int(0.5 * sr)   # 500ms 윈도우
    hop_samples = int(0.1 * sr)      # 100ms hop

    if len(audio) < window_samples:
        return audio

    positions: list[int] = []
    gains: list[float] = []

    for start in range(0, len(audio) - window_samples + 1, hop_samples):
        segment = audio[start:start + window_samples]
        rms = float(np.sqrt(np.mean(segment ** 2)))
        center = start + window_samples // 2

        if rms > 1e-7:
            g = min(TARGET_GAIN_RMS / rms, config.LOCAL_MAX_GAIN_X)
            g = max(g, 1.0)  # 부스트만, 감쇠 없음
        else:
            g = 1.0

        positions.append(center)
        gains.append(g)

    if not gains:
        return audio

    # 윈도우 중심 간 선형 보간으로 연속 gain curve 생성
    gain_curve = np.interp(np.arange(len(audio)), positions, gains).astype(np.float32)

    boosted_count = int(np.sum(gain_curve > 1.01))
    if boosted_count > 0:
        logger.info("로컬 게인 정규화: %d 샘플(%.2fs) 부스트", boosted_count, boosted_count / sr)

    return np.clip(audio * gain_curve, -1.0, 1.0).astype(audio.dtype)


# ---------------------------------------------------------------------------
# ⑤ Loudness Normalization (EBU R128 / LUFS — 통화단위, 2026-06-02)
# ---------------------------------------------------------------------------

# pyloudnorm 최소 블록(400ms). 이보다 짧으면 integrated loudness 측정 불가.
_LUFS_MIN_SEC = 0.4

_pyln_warned = False


def measure_integrated_lufs(audio: np.ndarray, sr: int) -> float | None:
    """ITU-R BS.1770-4(K-weighting + 게이팅) integrated loudness(LUFS) 측정.

    측정 불가 시 None: 길이 <0.4s / 무음 / -inf / config.LUFS_SILENCE_FLOOR 미만 /
    pyloudnorm 미설치·오류. None 이면 호출자가 정규화를 건너뛴다(안전).
    """
    global _pyln_warned
    if audio is None or len(audio) < int(_LUFS_MIN_SEC * sr):
        return None
    try:
        import pyloudnorm as pyln
    except ImportError:
        if not _pyln_warned:
            logger.warning("pyloudnorm 미설치 — LUFS 정규화 건너뜀")
            _pyln_warned = True
        return None
    try:
        meter = pyln.Meter(sr)
        lufs = float(meter.integrated_loudness(np.asarray(audio, dtype=np.float64)))
    except Exception as exc:  # noqa: BLE001 — 측정 실패는 무중단 skip
        logger.warning("LUFS 측정 실패 — 건너뜀: %s", exc)
        return None
    if not np.isfinite(lufs) or lufs < config.LUFS_SILENCE_FLOOR:
        return None
    return lufs


def normalize_loudness_call(
    audio: np.ndarray,
    sr: int,
    target_lufs: float,
    peak_ceiling_dbfs: float,
) -> np.ndarray:
    """통화단위 균일 LUFS 정규화. 전체에 단일 gain 적용(발화간 상대음량 보존).

    측정 불가(measure_integrated_lufs=None) 시 입력을 그대로 반환(byte-identical).
    true-peak ceiling 으로 gain 상한을 걸어 클리핑/품질등급 왜곡을 방지한다.
    균일 gain 은 peak_db·rms_db 를 동일 dB 이동시켜 snr_db(=|peak-rms|) 불변 →
    quality_grade 중립(설계 근거).
    """
    if len(audio) == 0:
        return audio
    lufs = measure_integrated_lufs(audio, sr)
    if lufs is None:
        return audio

    gain = 10.0 ** ((target_lufs - lufs) / 20.0)

    # true-peak ceiling: gain 적용 후 최대 진폭이 ceiling 을 넘지 않도록 상한.
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-9:
        ceiling_amp = 10.0 ** (peak_ceiling_dbfs / 20.0)
        gain = min(gain, ceiling_amp / peak)

    # 선택적 증폭 상한(노이즈 폭주 방어). 0 이면 무제한(ceiling 만).
    if config.LOUDNESS_GAIN_MAX_X > 0:
        gain = min(gain, config.LOUDNESS_GAIN_MAX_X)

    # 감쇠 허용 여부(EBU 정석은 양방향). false 면 부스트 전용.
    if not config.LOUDNESS_ALLOW_ATTENUATE and gain < 1.0:
        gain = 1.0

    if abs(gain - 1.0) < 1e-4:
        return audio

    logger.info("통화단위 LUFS 정규화: %.1f → %.1f LUFS (gain=%.2fx, ceiling=%.1fdBFS)",
                lufs, target_lufs, gain, peak_ceiling_dbfs)
    return np.clip(audio * gain, -1.0, 1.0).astype(audio.dtype)


def loudness_gated_local_gain(audio: np.ndarray, sr: int) -> np.ndarray:
    """noise-floor 게이팅 로컬 게인 — local_normalize_gain 의 noise breathing 대체.

    윈도우(500ms/100ms hop) RMS 가 noise floor*margin 미만이면 gain=1.0(미부스트)
    하여 잡음·숨소리 구간의 과증폭을 차단한다. 진짜 음성 윈도우만 부스트.
    noise floor = max(하위 NOISE_FLOOR_PERCENTILE%ile RMS, 절대 LOUDNESS_GATE_DBFS).
    """
    if len(audio) == 0:
        return audio
    window_samples = int(0.5 * sr)
    hop_samples = int(0.1 * sr)
    if len(audio) < window_samples:
        return audio

    # 윈도우 RMS 수집
    centers: list[int] = []
    rms_list: list[float] = []
    for start in range(0, len(audio) - window_samples + 1, hop_samples):
        seg = audio[start:start + window_samples]
        rms_list.append(float(np.sqrt(np.mean(seg ** 2))))
        centers.append(start + window_samples // 2)
    if not rms_list:
        return audio

    rms_arr = np.asarray(rms_list)
    nonzero = rms_arr[rms_arr > 1e-9]
    floor_pct = float(np.percentile(nonzero, config.NOISE_FLOOR_PERCENTILE)) if len(nonzero) else 0.0
    floor_abs = 10.0 ** (config.LOUDNESS_GATE_DBFS / 20.0)
    noise_floor = max(floor_pct, floor_abs)
    gate = noise_floor * config.LOUDNESS_GATE_MARGIN

    gains: list[float] = []
    gated = 0
    for r in rms_arr:
        if r < gate or r <= 1e-7:
            gains.append(1.0)   # noise floor 근처 → 미부스트(noise breathing 차단)
            if r > 1e-7:
                gated += 1
        else:
            g = min(TARGET_GAIN_RMS / r, config.LOCAL_MAX_GAIN_X)
            gains.append(max(g, 1.0))

    gain_curve = np.interp(np.arange(len(audio)), centers, gains).astype(np.float32)
    boosted = int(np.sum(gain_curve > 1.01))
    logger.info("loudness 게이팅 로컬 게인: 부스트 %d샘플 / 게이팅 %d윈도우 (floor=%.5f)",
                boosted, gated, noise_floor)
    return np.clip(audio * gain_curve, -1.0, 1.0).astype(audio.dtype)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def preprocess(
    audio: np.ndarray,
    sr: int,
    denoise_enabled: bool | None = None,
) -> np.ndarray:
    """4단계 전처리 파이프라인 — 각 단계는 config flag로 독립 제어된다.

    실행 순서: gain_normalize → denoise → deduplicate → compress_silence
    각 단계는 config.PREPROCESS_*_ENABLED 환경변수로 on/off 가능.

    `denoise_enabled` 인자는 레거시 호환용(deprecated). None이면 config 사용.
    """
    original_duration = len(audio) / sr
    result = audio
    timings: dict[str, float] = {}
    applied: list[str] = []

    if config.PREPROCESS_GAIN_ENABLED:
        t = time.time()
        result = normalize_gain(result)
        timings["gain"] = time.time() - t
        applied.append("gain")

        t = time.time()
        if config.LOUDNESS_LOCAL_GATE_ENABLED:
            # noise-floor 게이팅 로컬 게인 (noise breathing 차단). env-gate 기본 OFF.
            result = loudness_gated_local_gain(result, sr)
            timings["local_gain_gated"] = time.time() - t
            applied.append("local_gain_gated")
        else:
            result = local_normalize_gain(result, sr)
            timings["local_gain"] = time.time() - t
            applied.append("local_gain")

    # 인자 denoise_enabled가 명시되면 우선, 없으면 config 사용
    denoise_on = denoise_enabled if denoise_enabled is not None else config.PREPROCESS_DENOISE_ENABLED

    if denoise_on:
        t = time.time()
        result = denoise(result, sr)
        timings["denoise"] = time.time() - t
        applied.append("denoise")

        # Round 3 cascade fix: denoise가 voice RMS를 median 23배 감쇠시키므로 (실측)
        # normalize_gain을 재호출해 진폭을 복원한다. MAX_GAIN_X=10으로 증폭 한계가 걸리므로
        # 완전 복원은 불가하며 silence threshold 하향(아래)과 함께 사용한다.
        # 자세한 내용: uncounted-docs/voice-api/전처리_파이프라인_재활성화.md Round 3
        if config.PREPROCESS_GAIN_ENABLED:
            t = time.time()
            result = normalize_gain(result)
            timings["regain"] = time.time() - t
            applied.append("regain")

    if config.PREPROCESS_DEDUP_ENABLED:
        t = time.time()
        result = remove_duplicates(result, sr)
        timings["dedup"] = time.time() - t
        applied.append("dedup")

    if config.PREPROCESS_SILENCE_ENABLED:
        t = time.time()
        # denoise 후에는 동적 임계값(0.0005)으로 cascade 손실 방지
        silence_threshold = config.SILENCE_RMS_THRESHOLD_DENOISE if denoise_on else None
        result = compress_silence(result, sr, rms_threshold=silence_threshold)
        timings["silence"] = time.time() - t
        applied.append("silence")

    # ⑤ 통화단위 LUFS 정규화 — 모든 단계 끝난 뒤 1회(발화 슬라이스 직전 상태).
    # 균일 gain 이라 모든 발화에 동일 적용 → 발화간 상대음량·다이내믹레인지 보존.
    # env-gate 기본 OFF. denoise+regain 뒤에 위치해 cascade 감쇠도 LUFS 로 정량 복원.
    if config.CALL_LOUDNESS_NORM_ENABLED:
        t = time.time()
        result = normalize_loudness_call(
            result, sr,
            target_lufs=config.CALL_LOUDNESS_TARGET_LUFS,
            peak_ceiling_dbfs=config.CALL_LOUDNESS_PEAK_DBFS,
        )
        timings["loudness"] = time.time() - t
        applied.append("loudness")

    new_duration = len(result) / sr
    reduction = (1 - new_duration / original_duration) * 100 if original_duration > 0 else 0
    timing_str = ", ".join(f"{k}={v:.2f}s" for k, v in timings.items()) if timings else "all disabled"

    logger.info(
        "전처리 완료: %.1fs → %.1fs (%.0f%% 감소) [applied: %s | %s]",
        original_duration, new_duration, reduction,
        ",".join(applied) if applied else "none",
        timing_str,
    )
    return result
