"""하이브리드 발화분리 — 도입부 N초만 NeMo MSDD 고해상도 재분리 후 오버라이트.

배경(2026-06-02 PoC 측정):
  pyannote 3.1 은 통화 도입부의 짧은 turn(0.3~0.45s 의 "여보세요/네/바쁘세요")을
  단일 화자로 뭉쳐 화자가 뒤바뀐다(GT1 4-turn → 1 화자). NeMo MSDD(telephonic,
  multi-scale)는 같은 구간을 정확히 분리한다. 단 전체에 NeMo 를 돌리면 짧은 turn 이
  전체 통계에 묻혀 다시 실패 + VRAM 7.7GB(임계초과). 따라서 도입부 30s 윈도우만
  NeMo 로 재분리하고 그 결과로 word.speaker 를 오버라이트하는 하이브리드가 최적.
  (측정: 10s/20s/전체=실패, 15s/30s/60s=성공. 30s = 정확도·자원 sweet spot.)

ID 통일 (매핑):
  NeMo speaker_0/1 ↔ pyannote SPEAKER_00/01 매핑은 **F0 어쿠스틱 앵커가 메인**이다.
  화자 분리(clustering)는 임베딩/NeMo 가 메인이지만, 두 엔진의 화자 라벨을 잇는 매핑
  단계에서 임베딩 코사인은 신뢰할 수 없다 — pyannote(WeSpeaker)와 NeMo(TitaNet)는
  잠재공간이 다른 별개 모델이라 cross-model 코사인 margin 이 약하고(동성 male×2: 0.235),
  방향이 뒤집힌다(2026-06-02 실측: 코사인=오답/실패 vs F0 앵커=정답). F0 는 모델 무관한
  물리 스칼라(Hz)라 두 엔진이 측정해도 저음/고음 순위가 동일해 rank 매칭이 견고하다.
  우선순위: ①F0 앵커(저음↔저음/고음↔고음) → ②코사인(F0 분리 불가한 혼성 등) → ③overlap.

안전:
  - env gate VOICE_HYBRID_DIAR_ENABLED (기본 false) → 꺼지면 호출 자체 안 함(무회귀).
  - NeMo 서비스 미응답/타임아웃/저신뢰 → 원본 word 유지(fallback, 무중단).
  - DB·오디오 원본 미변경. word.speaker 메모리 오버라이트만.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

WINDOW_SEC_DEFAULT = 30.0
_COSINE_MIN_MARGIN = 0.05   # 코사인 1:1 매핑 최소 margin (이보다 작으면 모호 → overlap 백업)
_F0_MIN_DIFF_HZ = 8.0       # 두 화자 F0 차이가 이보다 작으면 앵커 신뢰 불가 → 코사인 위임


def is_enabled() -> bool:
    return os.environ.get("VOICE_HYBRID_DIAR_ENABLED", "false").strip().lower() == "true"


def _window_sec() -> float:
    try:
        return float(os.environ.get("VOICE_HYBRID_DIAR_WINDOW_SEC", str(WINDOW_SEC_DEFAULT)))
    except ValueError:
        return WINDOW_SEC_DEFAULT


def _nemo_endpoint() -> str:
    return os.environ.get("VOICE_HYBRID_NEMO_ENDPOINT", "http://localhost:8009/api/diarize/intro")


def _cos(a: list[float] | None, b: list[float] | None) -> float | None:
    if not a or not b:
        return None
    import numpy as np
    va = np.asarray(a, dtype="float32")
    vb = np.asarray(b, dtype="float32")
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-12 or nb < 1e-12:
        return None
    return float(np.dot(va, vb) / (na * nb))


def _map_by_cosine(
    pyannote_embeddings: dict[str, list[float]],
    nemo_embeddings: dict[str, list[float]],
) -> dict[str, str] | None:
    """NeMo spk → pyannote spk 코사인 1:1 매핑. 모호/비대칭이면 None(백업 위임).

    **mutual best match(양방향 일치)** 를 요구한다 — NeMo n 의 best 가 pyannote p 이고
    동시에 p 의 best 가 n 이어야 한다. 단방향 best 만으로 매핑하면 pyannote 임베딩이
    약할 때(도입부 단일화자 등) 방향이 뒤집힌 오배정(AB→BA)을 검출하지 못한다
    (2026-06-02 진단: 정답지 없이 오배정을 잡는 유일한 무감독 신호). 비대칭이면 None.
    """
    if not pyannote_embeddings or not nemo_embeddings:
        return None

    def best_match(emb: list[float], pool: dict[str, list[float]]) -> tuple[str, float, float] | None:
        sims = sorted(
            ((k, _cos(emb, e)) for k, e in pool.items()),
            key=lambda x: (x[1] is not None, x[1] or -1.0),
            reverse=True,
        )
        sims = [(k, s) for k, s in sims if s is not None]
        if not sims:
            return None
        best_k, best_s = sims[0]
        second_s = sims[1][1] if len(sims) > 1 else -1.0
        return best_k, best_s, second_s

    mapping: dict[str, str] = {}
    used_py: set[str] = set()
    for n_spk, n_emb in nemo_embeddings.items():
        fwd = best_match(n_emb, pyannote_embeddings)
        if fwd is None:
            return None
        best_p, best_s, second_s = fwd
        if best_s - second_s < _COSINE_MIN_MARGIN:
            return None  # forward margin 모호
        # 역방향: pyannote best_p 의 best 가 다시 n_spk 인가 (mutual best match)
        bwd = best_match(pyannote_embeddings[best_p], nemo_embeddings)
        if bwd is None or bwd[0] != n_spk:
            return None  # 비대칭 → 오배정 의심
        if bwd[1] - bwd[2] < _COSINE_MIN_MARGIN:
            return None  # backward margin 모호
        if best_p in used_py:
            return None  # 1:1 깨짐
        mapping[n_spk] = best_p
        used_py.add(best_p)
    return mapping


def _cosine_margin(
    pyannote_embeddings: dict[str, list[float]],
    nemo_embeddings: dict[str, list[float]],
) -> float:
    """매핑의 최소 코사인 margin(best-2nd). 작으면 임베딩 신뢰 불가. 측정 불가면 0."""
    if not pyannote_embeddings or not nemo_embeddings:
        return 0.0
    margins: list[float] = []
    for n_emb in nemo_embeddings.values():
        sims = sorted(
            (s for s in (_cos(n_emb, p) for p in pyannote_embeddings.values()) if s is not None),
            reverse=True,
        )
        if len(sims) < 2:
            return 0.0
        margins.append(sims[0] - sims[1])
    return min(margins) if margins else 0.0


def _map_by_f0(
    pyannote_f0: dict[str, float],
    nemo_f0: dict[str, float],
) -> dict[str, str] | None:
    """어쿠스틱 앵커(F0/피치) 기반 NeMo→pyannote 매핑.

    임베딩 코사인이 약한(두 male 화자 등) 경우의 백업/검증. 성대 물리구조상 평균 피치는
    화자별로 결정적 차이가 있으므로, F0 오름차순 순위가 같은 화자끼리 매핑한다
    (저음↔저음, 고음↔고음). 2화자 한정. F0 부족/차이 미미(<F0_MIN_DIFF)면 None.

    2026-06-02 실측: 이 세션 두 male 화자 F0 차이 13~24Hz로 임베딩보다 안정적 분리.
    """
    if len(pyannote_f0) != 2 or len(nemo_f0) != 2:
        return None
    py_sorted = sorted(pyannote_f0.items(), key=lambda x: x[1])   # 저음→고음
    ne_sorted = sorted(nemo_f0.items(), key=lambda x: x[1])
    # 각 쪽 두 화자 F0 차이가 너무 작으면 신뢰 불가
    if (py_sorted[1][1] - py_sorted[0][1]) < _F0_MIN_DIFF_HZ:
        return None
    if (ne_sorted[1][1] - ne_sorted[0][1]) < _F0_MIN_DIFF_HZ:
        return None
    # 순위 매칭: 저음 NeMo → 저음 pyannote, 고음 → 고음
    return {
        ne_sorted[0][0]: py_sorted[0][0],
        ne_sorted[1][0]: py_sorted[1][0],
    }


def _map_by_overlap(
    pyannote_turns: list[tuple[float, float, str]],
    nemo_turns: list[dict],
    window_limit: float,
) -> dict[str, str]:
    """백업: 시간 overlap 면적 기반 NeMo spk → pyannote spk 매핑.

    pyannote 도입부가 단일화자로 뭉친 경우 1:1 이 안 될 수 있으므로 코사인 실패시만.
    """
    overlap: dict[tuple[str, str], float] = {}
    for p_st, p_ed, p_spk in pyannote_turns:
        if p_st >= window_limit:
            continue
        p_ed_clip = min(p_ed, window_limit)
        for n in nemo_turns:
            inter_st = max(p_st, n["start"])
            inter_ed = min(p_ed_clip, n["end"])
            if inter_st < inter_ed:
                key = (p_spk, n["nemo_spk"])
                overlap[key] = overlap.get(key, 0.0) + (inter_ed - inter_st)
    mapping: dict[str, str] = {}
    used_py: set[str] = set()
    for (p_spk, n_spk), _ov in sorted(overlap.items(), key=lambda x: x[1], reverse=True):
        if n_spk not in mapping and p_spk not in used_py:
            mapping[n_spk] = p_spk
            used_py.add(p_spk)
    return mapping


def _f0_medians_from_segments(
    audio, sample_rate: int, by_spk: dict[str, list[tuple[float, float]]],
) -> dict[str, float]:
    """화자별 발화구간(최대 누적 20s)에서 F0 median. rank 비교용이라 절대값보다 순위가 중요.

    stt_processor._speaker_f0_medians 와 동일 파라미터(librosa.pyin, C2~C7). 두 측정의
    방법이 달라도 저음/고음 순위는 보존되므로 NeMo·pyannote 측 매칭에 안전하다.
    """
    try:
        import numpy as np
        import librosa  # type: ignore
    except ImportError:
        return {}
    out: dict[str, float] = {}
    for spk, segs in by_spk.items():
        ordered = sorted(segs, key=lambda x: x[1] - x[0], reverse=True)
        parts, total = [], 0.0
        for s, e in ordered:
            parts.append(audio[int(s * sample_rate):int(e * sample_rate)])
            total += e - s
            if total >= 20.0:
                break
        if not parts:
            continue
        chunk = np.concatenate(parts)
        if len(chunk) < sample_rate * 0.3:
            continue
        try:
            f0, _, _ = librosa.pyin(
                chunk.astype("float32"),
                fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
                sr=sample_rate,
            )
            valid = f0[~np.isnan(f0)]
            if len(valid) >= 10:
                out[spk] = float(np.median(valid))
        except Exception:  # noqa: BLE001 — F0 실패는 앵커 미사용으로 폴백
            continue
    return out


def _nemo_f0_from_audio(audio_path: str, nemo_turns: list[dict]) -> dict[str, float]:
    """NeMo 화자별 F0 median — audio_path 를 nemo_turn 구간으로 잘라 측정(앵커 메인 입력).

    nemo_f0 는 NeMo turn 이 나온 뒤에야 산출 가능해 caller 가 미리 못 넘긴다. 여기서
    audio_path(원본)를 직접 읽어 측정한다. F0 는 진폭(gain)에 불변이라 전처리본/원본 차이가
    순위에 영향 없음. 실패 시 빈 dict(코사인/overlap 으로 폴백).
    """
    try:
        import soundfile as sf  # type: ignore
    except ImportError:
        return {}
    try:
        audio, sr = sf.read(audio_path)
    except Exception:  # noqa: BLE001
        return {}
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    by_spk: dict[str, list[tuple[float, float]]] = {}
    for n in nemo_turns:
        try:
            by_spk.setdefault(n["nemo_spk"], []).append((float(n["start"]), float(n["end"])))
        except (KeyError, TypeError, ValueError):
            continue
    return _f0_medians_from_segments(audio, int(sr), by_spk)


def _call_nemo(audio_path: str, window_sec: float) -> dict | None:
    """NeMo 마이크로서비스 호출. 실패 시 None(fallback)."""
    try:
        import requests
    except ImportError:
        logger.warning("[hybrid_diar] requests 미설치 — fallback")
        return None
    try:
        resp = requests.post(
            _nemo_endpoint(),
            json={"audio_path": audio_path, "window_seconds": window_sec},
            timeout=float(os.environ.get("VOICE_HYBRID_NEMO_TIMEOUT", "60")),
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001 — 무중단 fallback 이 목적
        logger.warning("[hybrid_diar] NeMo 서비스 호출 실패 — fallback: %s", type(exc).__name__)
        return None


def _pyannote_turns_from_segments(segments: list[dict]) -> list[tuple[float, float, str]]:
    """segments → (start,end,speaker) turn 리스트 (overlap 백업 매핑용)."""
    turns: list[tuple[float, float, str]] = []
    for seg in segments:
        spk = seg.get("speaker")
        if spk and seg.get("start") is not None and seg.get("end") is not None:
            turns.append((float(seg["start"]), float(seg["end"]), spk))
    return turns


def apply_hybrid_intro(
    result: dict,
    audio_path: str,
    pyannote_embeddings: dict[str, list[float]] | None = None,
    *,
    window_sec: float | None = None,
    pyannote_f0: dict[str, float] | None = None,
    nemo_f0: dict[str, float] | None = None,
    overlap_regions: list[tuple[float, float]] | None = None,
) -> dict:
    """도입부 N초 word.speaker 를 NeMo 결과로 오버라이트한 새 result 반환.

    Args:
        result: whisperx/raw_direct 출력 (segments[].words[].speaker 포함).
        audio_path: 원본 오디오 경로 (NeMo 서비스가 직접 읽음).
        pyannote_embeddings: {pyannote_spk: 256-dim} 화자 임베딩(F0 불가 시 코사인 서브).
        pyannote_f0: {pyannote_spk: F0median Hz} 어쿠스틱 앵커 메인(caller 가 측정해 전달).
        nemo_f0: {nemo_spk: F0median Hz} — 보통 None(내부에서 audio_path 로 측정). 테스트 주입용.
        window_sec: 도입부 윈도우(기본 env/30s).
        overlap_regions: [(start,end)] 동시발화(cross-talk) 구간. 이 구간 word 는 단일 화자로
            오버라이트하지 않고 원본 유지한다 — 중첩은 두 화자 모두에 존재해야 하므로(중첩 처리
            트랙 위임), exclusive NeMo 매핑이 한쪽을 뺏어 timeline 정합을 깨면 안 된다. None=무시.

    Returns:
        새 result dict. 게이트 OFF/NeMo 실패/매핑 실패 시 입력을 그대로 반환(무변경).
    """
    if not is_enabled():
        return result
    win = window_sec if window_sec is not None else _window_sec()

    nemo = _call_nemo(audio_path, win)
    if not nemo or nemo.get("status") != "success" or not nemo.get("turns"):
        return result
    nemo_turns: list[dict] = nemo["turns"]
    nemo_embeddings: dict[str, list[float]] = nemo.get("embeddings") or {}

    segments = result.get("segments") or []
    pyannote_turns = _pyannote_turns_from_segments(segments)

    # ── ID 매핑 (역할 배정) ──
    # 화자 분리(clustering)는 NeMo/임베딩이 메인이지만, "speaker_0/1 이 누구(self/other)냐"
    # 의 역할 배정은 **F0 어쿠스틱 앵커가 메인**이다. 임베딩 코사인은 동성(male×2) 화자에서
    # 방향이 뒤집히는 시한폭탄이라 매핑 단계에선 신뢰할 수 없다(2026-06-02 실측: F0 차이
    # 13~24Hz 로 안정 분리 vs 코사인 margin 0.15 로 뒤집힘). 우선순위:
    #   1) F0 앵커 (저음↔저음, 고음↔고음)  ← 메인
    #   2) 코사인 (F0 불가 시: 여성/혼성 등 F0 분리 가능한 경우만)
    #   3) overlap 백업
    # NeMo F0 는 turn 이 나온 뒤에야 측정 가능 → 내부 산출(테스트는 nemo_f0 직접 주입).
    nemo_f0_eff = nemo_f0
    if nemo_f0_eff is None:
        nemo_f0_eff = nemo.get("f0") or _nemo_f0_from_audio(audio_path, nemo_turns)

    f0_map = _map_by_f0(pyannote_f0 or {}, nemo_f0_eff or {})
    cos_map = _map_by_cosine(pyannote_embeddings or {}, nemo_embeddings)

    mapping = None
    map_src = ""
    if f0_map is not None:
        mapping, map_src = f0_map, "f0_anchor"        # 메인: F0 물리 앵커
        if cos_map and cos_map != f0_map:
            margin = _cosine_margin(pyannote_embeddings or {}, nemo_embeddings)
            logger.info(
                "[hybrid_diar] 코사인↔F0 불일치 — F0 앵커 채택(코사인 margin=%.3f 약/뒤집힘 차단)",
                margin,
            )
    elif cos_map:
        mapping, map_src = cos_map, "cosine"          # F0 불가 시만 코사인(서브)
    if mapping is None:
        mapping = _map_by_overlap(pyannote_turns, nemo_turns, win)
        map_src = "overlap"
    if not mapping:
        logger.warning("[hybrid_diar] ID 매핑 실패 — fallback")
        return result

    # ── 매핑 완전성 가드 ──
    # pyannote 가 도입부를 단일화자로 뭉치면 NeMo 화자 중 일부가 매핑 안 돼(예: speaker_0
    # 누락) 그 word 들이 오버라이트 안 되고 원본(단일화자)으로 남아 GT1 이 깨진다(2026-06-02
    # 진단). 매핑 안 된 NeMo 화자는 이미 매핑된 pyannote 라벨과 겹치지 않는 신규 라벨로
    # 발급해 모든 NeMo 화자가 서로 다른 라벨을 갖도록 보장한다.
    all_nemo_spk = {n["nemo_spk"] for n in nemo_turns}
    used_labels = set(mapping.values())
    unmapped = [s for s in all_nemo_spk if s not in mapping]
    if unmapped:
        # 기존 pyannote 화자 풀에서 미사용 라벨 우선, 없으면 SPEAKER_NN 신규 발급
        existing_pool = sorted({spk for _, _, spk in pyannote_turns})
        idx = 0
        for n_spk in sorted(unmapped):
            label = None
            for cand in existing_pool:
                if cand not in used_labels:
                    label = cand
                    break
            if label is None:
                while f"SPEAKER_{idx:02d}" in used_labels:
                    idx += 1
                label = f"SPEAKER_{idx:02d}"
            mapping[n_spk] = label
            used_labels.add(label)
        logger.info("[hybrid_diar] 미매핑 NeMo 화자 %d개 신규 라벨 발급: %s", len(unmapped), unmapped)

    # ── 하드 오버라이트 (도입부 윈도우 내 word 만) ──
    def nemo_spk_at(t: float) -> str | None:
        for n in nemo_turns:
            if n["start"] <= t <= n["end"]:
                return n["nemo_spk"]
        return None

    regions = overlap_regions or ()

    def in_overlap(t: float) -> bool:
        # 중첩 탐지구간 [start,end] 안의 word 는 단일 화자 오버라이트 대상에서 제외
        # (동시발화는 두 화자 모두에 존재해야 함 — 중첩 트랙이 별도 처리).
        return any(os_ <= t <= oe for os_, oe in regions)

    overwritten = 0
    skipped_overlap = 0
    new_segments: list[dict] = []
    for seg in segments:
        new_seg = dict(seg)
        words = seg.get("words") or []
        new_words = []
        for wd in words:
            nw = dict(wd)
            ws = nw.get("start")
            if ws is not None and ws < win:
                if in_overlap(float(ws)):
                    skipped_overlap += 1            # 중첩 구간 — 원본 유지(timeline 보존)
                else:
                    n_spk = nemo_spk_at(float(ws))
                    if n_spk is not None and n_spk in mapping:
                        if nw.get("speaker") != mapping[n_spk]:
                            nw["speaker"] = mapping[n_spk]
                            nw["speaker_source"] = "hybrid_nemo_intro"
                            overwritten += 1
                    # NeMo 가 침묵으로 본 word 는 원본 유지(fallback)
            new_words.append(nw)
        if new_words:
            new_seg["words"] = new_words
        new_segments.append(new_seg)

    logger.info(
        "[hybrid_diar] 도입부 오버라이트 완료 win=%.0fs map=%s overwritten=%d overlap_skip=%d",
        win, map_src, overwritten, skipped_overlap,
    )
    out = dict(result)
    out["segments"] = new_segments
    return out
