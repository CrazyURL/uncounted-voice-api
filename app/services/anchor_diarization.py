"""보이스프린트 앵커링 화자분리 보정 (TS-VAD 경량).

배경(2026-06-02 실증, 세션 93c28f57 male×2):
  두 화자가 음향적으로 비슷하면(본문 F0 차 5~8Hz) 비감독 클러스터링(pyannote/NeMo
  전체통화)이 붕괴해 발화에 여러 화자가 뭉친다. 도입부 30s 만 NeMo 가 깨끗이 분리
  (F0 차 23Hz)되므로, 도입부에서 화자별 음성지문(anchor)을 고정하고 전체 통화 단어를
  두 지문과의 코사인으로 **1:2 지도분류**(클러스터링 X)한다. 글로벌 SOTA TS-VAD 의
  경량 구현.

  실측: 뭉침 0(전부 화자순수), 도입부 5/5, 본문 화자교대 92.2% vs GT(시간규칙으론
  불가능했던 상대 짧은 끼어들기까지 목소리로 정확 분류).

파이프라인 위치: diarization(word.speaker 부착) 직후, utterance 분할 **전**.
  hybrid_diarization(도입부 한정)과 달리 **전체 통화** word.speaker 를 보정한다.

안전:
  - env gate VOICE_ANCHOR_DIAR_ENABLED (기본 false) → 꺼지면 호출 자체 안 함(무회귀).
  - NeMo 미응답 / 앵커 구축 실패 / 도입부 F0 미분리 → 원본 result 반환(fallback).
  - DB·오디오 원본 미변경. word.speaker 메모리 오버라이트만.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable

from app.services import hybrid_diarization as hd

logger = logging.getLogger(__name__)

INTRO_WINDOW_SEC_DEFAULT = 30.0
_ANCHOR_MIN_SEG_SEC = 0.6      # 앵커용 도입부 구간 최소 길이
_INTRO_F0_MIN_DIFF_HZ = 8.0    # 도입부 두 화자 F0 차가 이보다 작으면 앵커 신뢰불가 → fallback
_EMB_MIN_SEC = 0.7             # 임베딩 최소 오디오 길이(짧으면 중심확장)
_SMOOTH_MARGIN = 0.10          # 이보다 낮은 margin 만 스무딩 후보
_SMOOTH_GAP = 0.3             # 앞뒤 갭이 이보다 작으면(흐름 중) 글리치로 간주
# NeMo 도입부 enclosed-interjection 교정: 도입부(<intro_window) word 가 **다른 화자의 NeMo
# turn 을 완전히 감싸면**(미스바운드 끼어들기, 예: 0.88~1.13s '상대 여보세요'가 '본인' 단어
# 경계 안에 enclosed) 그 화자로 교정. **본문 미적용**(시간가드 — 본문은 NeMo 도 붕괴 →
# 95.9% 정합 오염 방지). 전환(단어 끝이 다음 화자로 넘침: 네·말씀하세요)은 enclosed 아님→보존.
# margin 임계 방식은 라이브(denoise) 오디오에서 margin 변동으로 불발 → enclosed 가 견고.
_ENCLOSED_MIN_SEC = 0.12   # enclosed turn 최소 길이(노이즈 블립 제외)
# 기존 컨벤션: 본인=저음=SPEAKER_01, 상대=고음=SPEAKER_00
_LABEL_LOW = "SPEAKER_01"
_LABEL_HIGH = "SPEAKER_00"

# Text-Informed Diarization(도입부 룰베이스 강제 분할): 주파수로 안 갈리는 도입부 인사 뭉침을
# Whisper 텍스트+타임스탬프로 우회. 한 run(같은 화자 라벨)에 [호출어]+[응답어]가 함께 있으면
# (=뭉침. 정상 분리되면 다른 라벨이라 한 run 에 공존 불가 → self-targeting) 마지막 호출어와
# 첫 응답어 사이에서 강제 분할. GPU/VRAM 0, 결정적.
_TEXT_SPLIT_INTRO_SEC = 5.0
_CALL_WORDS = ("여보세요", "안녕하세요")
_RESP_WORDS = ("네", "넵", "넹", "예", "옙", "어", "응")
# 역할 방향: True=호출부→상대(SP00)/응답부→본인(SP01)(대표님 룰). 통화별 발신/수신에 따라 플립.
_TEXT_SPLIT_CALL_IS_OTHER = True

EmbedFn = Callable[[Any, int], Any]  # (audio_1d, sr) -> 정규화 벡터(np) 또는 None


def is_enabled() -> bool:
    return os.environ.get("VOICE_ANCHOR_DIAR_ENABLED", "false").strip().lower() == "true"


def _intro_window_sec() -> float:
    try:
        return float(os.environ.get("VOICE_ANCHOR_INTRO_WINDOW_SEC", str(INTRO_WINDOW_SEC_DEFAULT)))
    except ValueError:
        return INTRO_WINDOW_SEC_DEFAULT


def _default_embed_fn() -> EmbedFn:
    """SpeakerEmbeddingModel 기반 임베딩 함수(lazy). L2 정규화 벡터 반환."""
    from app.services.speaker_embedding import SpeakerEmbeddingModel, EmbeddingUnavailable
    import numpy as np

    model = SpeakerEmbeddingModel()

    def _embed(seg, sr: int):
        r = model.extract_embedding(seg, sr)
        if isinstance(r, EmbeddingUnavailable):
            return None
        v = np.asarray(r, dtype="float32")
        n = float(np.linalg.norm(v))
        return (v / n) if n > 1e-9 else None

    return _embed


def _embed_span(audio, sample_rate: int, start: float, end: float, embed: EmbedFn):
    """[start,end] 구간 임베딩. 짧으면 중심확장(_EMB_MIN_SEC)."""
    s, e = start, end
    if e - s < _EMB_MIN_SEC:
        c = (s + e) / 2.0
        s, e = max(0.0, c - _EMB_MIN_SEC / 2), c + _EMB_MIN_SEC / 2
    seg = audio[int(s * sample_rate):int(e * sample_rate)]
    if len(seg) == 0:
        return None
    return embed(seg, sample_rate)


def _build_anchors(audio, sample_rate, intro_turns, embed: EmbedFn):
    """도입부 NeMo turn + F0 로 화자별 앵커(평균 임베딩) 2개 구축.

    저음 화자 → SPEAKER_01, 고음 → SPEAKER_00. F0 차가 작으면(_INTRO_F0_MIN_DIFF)
    분리 신뢰불가로 None 반환(상위에서 fallback). 2화자 아니어도 None.

    Returns: (anchors, nemo_label_map) 또는 None. nemo_label_map={nemo_spk: SPEAKER_label}
    (도입부 타이브레이커가 NeMo 화자를 SPEAKER 라벨로 변환할 때 사용).
    """
    import numpy as np

    by_spk: dict[str, list[tuple[float, float]]] = {}
    for t in intro_turns:
        try:
            by_spk.setdefault(t["nemo_spk"], []).append((float(t["start"]), float(t["end"])))
        except (KeyError, TypeError, ValueError):
            continue
    if len(by_spk) != 2:
        return None

    f0 = hd._f0_medians_from_segments(audio, sample_rate, by_spk)
    if len(f0) != 2:
        return None
    (lo_spk, lo_f0), (hi_spk, hi_f0) = sorted(f0.items(), key=lambda x: x[1])
    if (hi_f0 - lo_f0) < _INTRO_F0_MIN_DIFF_HZ:
        logger.info("[anchor_diar] 도입부 F0 차 %.1fHz < %.1f — 앵커 신뢰불가, fallback",
                    hi_f0 - lo_f0, _INTRO_F0_MIN_DIFF_HZ)
        return None

    label_of = {lo_spk: _LABEL_LOW, hi_spk: _LABEL_HIGH}
    logger.info("[anchor_diar] 도입부 F0 %s → label_of %s (저음 %s=%.0fHz→%s)",
                {k: round(v, 1) for k, v in f0.items()}, label_of, lo_spk, lo_f0, _LABEL_LOW)
    anchors: dict[str, Any] = {}
    for spk, segs in by_spk.items():
        embs = []
        for s, e in segs:
            if e - s >= _ANCHOR_MIN_SEG_SEC:
                v = _embed_span(audio, sample_rate, s, e, embed)
                if v is not None:
                    embs.append(v)
        if not embs:
            return None
        a = np.mean(embs, axis=0)
        n = float(np.linalg.norm(a))
        if n < 1e-9:
            return None
        anchors[label_of[spk]] = a / n
    return (anchors, label_of) if len(anchors) == 2 else None


def _classify_words(words, anchors, audio, sample_rate, embed: EmbedFn):
    """각 word 를 두 앵커와의 코사인으로 분류. (label, margin) 리스트 반환(원본 미변경)."""
    import numpy as np

    a_lo, a_hi = anchors[_LABEL_LOW], anchors[_LABEL_HIGH]
    out: list[tuple[str | None, float]] = []
    for w in words:
        s, e = w.get("start"), w.get("end")
        if s is None:
            out.append((None, 0.0))
            continue
        v = _embed_span(audio, sample_rate, float(s), float(e if e is not None else s), embed)
        if v is None:
            out.append((None, 0.0))
            continue
        c_lo = float(np.dot(v, a_lo))
        c_hi = float(np.dot(v, a_hi))
        label = _LABEL_LOW if c_lo >= c_hi else _LABEL_HIGH
        out.append((label, abs(c_lo - c_hi)))
    return out


def _smooth(flat_words, labels):
    """gap-aware 스무딩: 양옆 같은화자 + 저margin + 침묵 미둘러쌈(흐름 중 글리치)만 뒤집음.

    진짜 짧은턴(앞뒤 갭으로 둘러싸임)은 보존한다(여보세요#2 margin 0.005 도 갭이 있어 유지).
    labels 를 in-place 로 고치지 않고 새 리스트 반환.
    """
    roles = [lab for lab, _ in labels]
    margins = [m for _, m in labels]
    n = len(roles)
    fixed = list(roles)
    for i in range(1, n - 1):
        if roles[i] is None or roles[i - 1] is None or roles[i + 1] is None:
            continue
        if roles[i] != roles[i - 1] or roles[i - 1] != roles[i + 1]:
            # roles[i] 가 양옆과 다른 고립점만 후보
            if roles[i] != roles[i - 1] and roles[i - 1] == roles[i + 1] and margins[i] < _SMOOTH_MARGIN:
                gap_before = float(flat_words[i]["start"]) - float(flat_words[i - 1].get("end") or flat_words[i - 1]["start"])
                gap_after = float(flat_words[i + 1]["start"]) - float(flat_words[i].get("end") or flat_words[i]["start"])
                if gap_before < _SMOOTH_GAP and gap_after < _SMOOTH_GAP:
                    fixed[i] = roles[i - 1]
    return fixed


def _pin_body_by_f0(words, roles, audio, sample_rate, intro_sec):
    """**본문(>=intro_sec)** 라벨을 F0 로 도입부 convention(저음=SP01)에 일치시킨다.

    도입부는 NeMo 로 이미 결정적 배정(저음=SP01)됐으므로 건드리지 않고, 임베딩이 본문 전체를
    반전시켰을 때(유사 목소리)만 **본문 라벨만** 스왑해 도입부와 통일한다. 본문 그룹이 2개
    아니거나 F0 측정 불가면 무변경.
    """
    body_idx = [
        i for i, w in enumerate(words)
        if w.get("start") is not None and float(w["start"]) >= intro_sec and roles[i] is not None
    ]
    by_label: dict[str, list] = {}
    for i in body_idx:
        w = words[i]
        by_label.setdefault(roles[i], []).append((float(w["start"]), float(w.get("end") or w["start"])))
    if set(by_label.keys()) != {_LABEL_LOW, _LABEL_HIGH}:
        return roles
    f0 = hd._f0_medians_from_segments(audio, sample_rate, by_label)
    lo, hi = f0.get(_LABEL_LOW), f0.get(_LABEL_HIGH)
    logger.info("[anchor_diar] 본문 F0 핀 측정: SP01=%s SP00=%s body_words=%d",
                None if lo is None else round(lo, 1), None if hi is None else round(hi, 1), len(body_idx))
    if lo is None or hi is None:
        return roles
    if lo > hi:   # 본문 SP01 그룹이 더 고음(=상대) → 본문 반전 → 본문만 스왑
        logger.info("[anchor_diar] 본문 F0 핀 스왑(본문 반전 교정): 본문 SP01=%.0fHz SP00=%.0fHz", lo, hi)
        swap = {_LABEL_LOW: _LABEL_HIGH, _LABEL_HIGH: _LABEL_LOW}
        out = list(roles)
        for i in body_idx:
            out[i] = swap.get(roles[i], roles[i])
        return out
    return roles


def _nemo_enclosed_other(nemo_turns, start: float, end: float, current_label, nemo_label_map):
    """word [start,end] 안에 완전히 enclosed 된 '다른 화자' NeMo turn 의 라벨(없으면 None).

    여보세요#2(상대 turn 0.88~1.13 이 본인 단어 0.73~1.38 안에 enclosed)는 잡고, 전환
    (말씀하세요: 다음화자 turn 이 단어 끝 밖으로 넘침)은 enclosed 가 아니라 안 잡는다.
    """
    for t in nemo_turns:
        try:
            ts, te = float(t["start"]), float(t["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if ts >= start and te <= end and (te - ts) >= _ENCLOSED_MIN_SEC:
            lab = nemo_label_map.get(t.get("nemo_spk"))
            if lab is not None and lab != current_label:
                return lab
    return None


def _nemo_spk_at_start(nemo_turns, t: float):
    """word 시작점 t 의 NeMo 화자(turn 포함). 갭이면 가장 가까운 turn. 음성 onset 은
    임베딩보다 신뢰 가능 — 도입부 직접배정의 기준."""
    for tt in nemo_turns:
        try:
            if float(tt["start"]) <= t <= float(tt["end"]):
                return tt["nemo_spk"]
        except (KeyError, TypeError, ValueError):
            continue
    best, bd = None, 1e9
    for tt in nemo_turns:
        try:
            d = min(abs(t - float(tt["start"])), abs(t - float(tt["end"])))
        except (KeyError, TypeError, ValueError):
            continue
        if d < bd:
            bd, best = d, tt.get("nemo_spk")
    return best


def _nemo_intro_assign(words, roles, nemo_turns, nemo_label_map, intro_sec):
    """도입부(<intro_sec) word 를 **NeMo 로 직접 배정**(임베딩 무시): 시작점 turn 화자 +
    enclosed 끼어들기 보정. 유사 목소리로 임베딩이 못 가르는 도입부 단발성 발화를 물리적
    NeMo VAD/타임스탬프로 결정적 배정. 본문은 미적용(시간가드).

    실측 검증: 시작점+enclosed = 도입부 6/6 결정적(여보세요#1=시작본인, 여보세요#2=enclosed상대,
    네=시작본인, 바쁘세요=시작상대, 말씀하세요=시작본인). **스무딩 이후** 적용(최종).
    """
    if not nemo_turns or not nemo_label_map:
        return roles
    intro_dbg = [
        (round(float(t["start"]), 2), round(float(t.get("end", t["start"])), 2), t.get("nemo_spk"))
        for t in nemo_turns if _safe_float(t.get("start")) is not None and float(t["start"]) < intro_sec
    ]
    logger.info("[anchor_diar] 도입부 NeMo turns(<%.0fs): %s", intro_sec, intro_dbg[:14])

    out = list(roles)
    changed = 0
    for i, w in enumerate(words):
        s = w.get("start")
        if s is None or roles[i] is None:
            continue
        s = float(s)
        if s >= intro_sec:
            continue  # 시간 가드: 본문 미적용
        e = float(w.get("end") or s)
        # enclosed 전용: word 가 다른 화자 NeMo turn 을 완전히 감싸면(미스바운드 끼어들기,
        # 예 여보세요#2) 그 화자로. 시작점 배정은 두 여보세요를 같은 화자로 뭉치므로 안 씀.
        encl = _nemo_enclosed_other(nemo_turns, s, e, out[i], nemo_label_map)
        if encl is not None and encl != out[i]:
            out[i] = encl
            changed += 1
    logger.info("[anchor_diar] 도입부 enclosed 끼어들기 교정 %d개", changed)
    return out


def _safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _text_informed_intro_split(words, roles):
    """도입부(<_TEXT_SPLIT_INTRO_SEC) 같은화자 run 에 [호출어]+[응답어]가 뭉쳐 있으면
    (주파수로 안 갈린 예외) Whisper 텍스트로 강제 분할: 마지막 호출어까지=호출부, 그 뒤=응답부.

    self-targeting: 정상 분리 시 호출/응답이 다른 라벨이라 한 run 에 공존 못 함 → 뭉친 도입부만
    발동(false positive 거의 0). 결정적·CPU. 역할은 _TEXT_SPLIT_CALL_IS_OTHER 로 방향 결정.
    """
    other, owner = (_LABEL_HIGH, _LABEL_LOW) if _TEXT_SPLIT_CALL_IS_OTHER else (_LABEL_LOW, _LABEL_HIGH)
    out = list(roles)
    n = len(words)

    def _wtext(idx):
        return (words[idx].get("word") or "").strip()

    def _is_call(idx):
        return any(k in _wtext(idx) for k in _CALL_WORDS)

    def _is_resp(idx):
        return any(k in _wtext(idx) for k in _RESP_WORDS)

    i = 0
    while i < n:
        s = words[i].get("start")
        if s is None or float(s) >= _TEXT_SPLIT_INTRO_SEC or out[i] is None:
            i += 1
            continue
        # 도입부 같은라벨 연속 run [i..j]
        j = i
        while (j + 1 < n and out[j + 1] == out[i] and words[j + 1].get("start") is not None
               and float(words[j + 1]["start"]) < _TEXT_SPLIT_INTRO_SEC):
            j += 1
        idxs = list(range(i, j + 1))
        if any(_is_call(k) for k in idxs) and any(_is_resp(k) for k in idxs):
            last_call = max((k for k in idxs if _is_call(k)), default=None)
            first_resp = next((k for k in idxs if k > last_call and _is_resp(k)), None) if last_call is not None else None
            if first_resp is not None:
                for k in range(i, first_resp):
                    out[k] = other     # 호출부
                for k in range(first_resp, j + 1):
                    out[k] = owner     # 응답부
                logger.info("[anchor_diar] text-informed split: 호출부[%d..%d]→%s, 응답부[%d..%d]→%s",
                            i, first_resp - 1, other, first_resp, j, owner)
        i = j + 1
    return out


def apply_anchor_diarization(
    result: dict,
    audio_path: str,
    audio: Any,
    sample_rate: int,
    *,
    embed_fn: EmbedFn | None = None,
    intro_window_sec: float | None = None,
) -> dict:
    """전체 통화 word.speaker 를 앵커 1:2 분류로 보정한 새 result 반환.

    Args:
        result: diarization 출력 (segments[].words[].speaker, start, end).
        audio_path: 원본 오디오 경로 (NeMo 서비스가 직접 읽음, 도입부 분리용).
        audio: 1-D float32 PCM (임베딩/분류용).
        sample_rate: 샘플레이트.
        embed_fn: (audio_1d, sr)->정규화벡터. None 이면 SpeakerEmbeddingModel lazy.
        intro_window_sec: 도입부 윈도우(기본 env/30s).

    Returns:
        새 result. 게이트 OFF/NeMo 실패/앵커 실패 시 입력 그대로(무변경).
    """
    if not is_enabled():
        return result
    win = intro_window_sec if intro_window_sec is not None else _intro_window_sec()

    nemo = hd._call_nemo(audio_path, win)
    if not nemo or nemo.get("status") != "success" or not nemo.get("turns"):
        logger.info("[anchor_diar] NeMo 도입부 미응답 — fallback")
        return result

    embed = embed_fn or _default_embed_fn()
    built = _build_anchors(audio, sample_rate, nemo["turns"], embed)
    if built is None:
        return result  # 앵커 구축 실패 → 무변경
    anchors, nemo_label_map = built

    segments = result.get("segments") or []
    flat_words = [w for seg in segments for w in (seg.get("words") or []) if w.get("start") is not None]
    flat_words.sort(key=lambda w: float(w["start"]))
    if not flat_words:
        return result

    labels = _classify_words(flat_words, anchors, audio, sample_rate, embed)
    smoothed = _smooth(flat_words, labels)
    # ① 전체통화 F0 핀(저음=본인=SP01) — convention 먼저 통일(enclosed 가 nemo_label_map 기준과
    #    같은 라벨 체계서 동작하도록).
    smoothed = _pin_body_by_f0(flat_words, smoothed, audio, sample_rate, 0.0)
    # ② 도입부 enclosed 끼어들기 교정 — 여보세요#2(상대 turn 을 감싼 본인 단어)를 상대로 분리.
    #    시작점 직접배정은 두 여보세요를 뭉치므로 enclosed 전용만. (확률적이지만 좋은 run 에서 분리)
    smoothed = _nemo_intro_assign(flat_words, smoothed, nemo["turns"], nemo_label_map, win)
    # ③ Text-Informed 강제 분할 — ①②로도 안 갈린 도입부 인사 뭉침을 Whisper 텍스트로 결정적
    #    분할(호출부/응답부). self-targeting 이라 정상 도입부는 무영향. 비결정성 최종 차단.
    smoothed = _text_informed_intro_split(flat_words, smoothed)

    # 원본 word 객체 id -> 새 라벨(분류된 것만). 원본 미변경, 새 result 구성.
    new_label_by_id = {
        id(w): lab for w, lab in zip(flat_words, smoothed) if lab is not None
    }

    overwritten = 0
    new_segments = []
    for seg in segments:
        new_seg = dict(seg)
        new_words = []
        for w in (seg.get("words") or []):
            nw = dict(w)
            lab = new_label_by_id.get(id(w))
            if lab is not None and nw.get("speaker") != lab:
                nw["speaker"] = lab
                nw["speaker_source"] = "anchor_diar"
                overwritten += 1
            new_words.append(nw)
        if new_words:
            new_seg["words"] = new_words
        new_segments.append(new_seg)

    confident = sum(1 for lab, m in labels if lab is not None and m >= 0.02)
    logger.info(
        "[anchor_diar] 앵커 분류 완료 words=%d overwritten=%d confident=%d/%d",
        len(flat_words), overwritten, confident, len(labels),
    )
    out = dict(result)
    out["segments"] = new_segments
    return out
