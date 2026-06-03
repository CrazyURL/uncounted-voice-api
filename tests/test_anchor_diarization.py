"""보이스프린트 앵커링 화자분리 보정 단위 테스트.

핵심 검증:
  - 게이트 OFF(기본): 무변경(무회귀)
  - gap-aware 스무딩: 흐름 중 글리치만 뒤집고 갭으로 둘러싸인 짧은턴은 보존
  - 코사인 1:2 분류
  - 앵커 구축 가드(2화자 아님 / F0 미분리 → None)
  - end-to-end: NeMo·F0 mock + 가짜 embed 로 collapse 된 라벨 보정

embed/F0/NeMo 미호출 — monkeypatch + 가짜 embed_fn(네트워크·모델 0).
"""
import numpy as np

from app.services import anchor_diarization as ad
from app.services import hybrid_diarization as hd


# ── 게이트 ──────────────────────────────────────────────────────────────────

def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("VOICE_ANCHOR_DIAR_ENABLED", raising=False)
    assert ad.is_enabled() is False
    result = {"segments": [{"words": [{"word": "x", "start": 0.1, "end": 0.5, "speaker": "SPEAKER_00"}]}]}
    out = ad.apply_anchor_diarization(result, "/tmp/x.wav", np.zeros(16000, dtype="float32"), 16000)
    assert out is result   # 무변경


# ── gap-aware 스무딩 ────────────────────────────────────────────────────────

def test_smooth_flips_midflow_glitch():
    # 양옆 SP01, 가운데 SP00 저margin + 앞뒤 갭<0.3(흐름 중) → SP01 로 뒤집힘
    words = [{"start": 1.0, "end": 1.2}, {"start": 1.25, "end": 1.4}, {"start": 1.45, "end": 1.6}]
    labels = [("SPEAKER_01", 0.5), ("SPEAKER_00", 0.01), ("SPEAKER_01", 0.5)]
    assert ad._smooth(words, labels) == ["SPEAKER_01", "SPEAKER_01", "SPEAKER_01"]


def test_smooth_preserves_gap_bounded_short_turn():
    # 여보세요#2 패턴: 저margin 이지만 뒤에 0.36s 갭 → 진짜 짧은턴으로 보존
    words = [{"start": 0.05, "end": 0.55}, {"start": 0.73, "end": 1.40}, {"start": 1.76, "end": 3.10}]
    labels = [("SPEAKER_01", 0.4), ("SPEAKER_00", 0.005), ("SPEAKER_01", 0.4)]
    assert ad._smooth(words, labels)[1] == "SPEAKER_00"   # 보존


def test_smooth_keeps_high_margin_turn():
    # margin 이 충분히 크면(>=0.10) 갭 없어도 보존(확신 있는 턴)
    words = [{"start": 1.0, "end": 1.2}, {"start": 1.25, "end": 1.4}, {"start": 1.45, "end": 1.6}]
    labels = [("SPEAKER_01", 0.5), ("SPEAKER_00", 0.4), ("SPEAKER_01", 0.5)]
    assert ad._smooth(words, labels)[1] == "SPEAKER_00"


# ── 코사인 1:2 분류 ─────────────────────────────────────────────────────────

def _sign_embed(seg, sr):
    """가짜 embed: 구간 평균 부호로 화자 구분(+→SP01쪽 [1,0], -→SP00쪽 [0,1])."""
    m = float(np.mean(seg)) if len(seg) else 0.0
    return np.array([1.0, 0.0], dtype="float32") if m >= 0 else np.array([0.0, 1.0], dtype="float32")


def test_classify_words_by_cosine():
    anchors = {"SPEAKER_01": np.array([1.0, 0.0], dtype="float32"),
               "SPEAKER_00": np.array([0.0, 1.0], dtype="float32")}
    audio = np.concatenate([np.ones(16000, dtype="float32"), -np.ones(16000, dtype="float32")])
    words = [{"start": 0.1, "end": 0.6}, {"start": 1.1, "end": 1.6}]
    labs = ad._classify_words(words, anchors, audio, 16000, _sign_embed)
    assert labs[0][0] == "SPEAKER_01"   # +구간
    assert labs[1][0] == "SPEAKER_00"   # -구간


# ── 앵커 구축 가드 ──────────────────────────────────────────────────────────

def test_build_anchors_none_when_not_two_speakers():
    turns = [{"start": 0.0, "end": 1.0, "nemo_spk": "speaker_0"}]
    assert ad._build_anchors(np.ones(16000, dtype="float32"), 16000, turns, _sign_embed) is None


def test_build_anchors_none_when_f0_too_close(monkeypatch):
    monkeypatch.setattr(hd, "_f0_medians_from_segments",
                        lambda a, sr, by: {"speaker_1": 120.0, "speaker_0": 123.0})  # 3Hz<8
    turns = [{"start": 0.0, "end": 1.0, "nemo_spk": "speaker_1"},
             {"start": 1.0, "end": 2.0, "nemo_spk": "speaker_0"}]
    audio = np.concatenate([np.ones(16000, dtype="float32"), -np.ones(16000, dtype="float32")])
    assert ad._build_anchors(audio, 16000, turns, _sign_embed) is None


# ── end-to-end (NeMo·F0 mock) ───────────────────────────────────────────────

def test_apply_corrects_collapsed_labels(monkeypatch):
    monkeypatch.setenv("VOICE_ANCHOR_DIAR_ENABLED", "true")
    # 도입부 NeMo: speaker_1(저음/본인) [0-1], speaker_0(고음/상대) [1-2]
    monkeypatch.setattr(hd, "_call_nemo", lambda path, win: {
        "status": "success",
        "turns": [{"start": 0.0, "end": 1.0, "nemo_spk": "speaker_1"},
                  {"start": 1.0, "end": 2.0, "nemo_spk": "speaker_0"}],
    })
    monkeypatch.setattr(hd, "_f0_medians_from_segments",
                        lambda a, sr, by: {"speaker_1": 106.0, "speaker_0": 130.0})  # 24Hz 차
    # audio: [0-1]=+1(본인), [1-2]=-1(상대), [2-3]=+1(본인), [3-4]=-1(상대)
    audio = np.concatenate([np.ones(16000, "float32"), -np.ones(16000, "float32"),
                            np.ones(16000, "float32"), -np.ones(16000, "float32")])
    # 원본: 전부 SPEAKER_00 으로 뭉침(collapse)
    result = {"segments": [{
        "start": 0.0, "end": 4.0, "speaker": "SPEAKER_00",
        "words": [
            {"word": "본인1", "start": 2.1, "end": 2.6, "speaker": "SPEAKER_00"},
            {"word": "상대1", "start": 3.1, "end": 3.6, "speaker": "SPEAKER_00"},
        ],
    }]}
    # intro_window=1.0 → 단어(2.1·3.1s)는 본문 → anchor 임베딩 분류 경로 테스트
    out = ad.apply_anchor_diarization(result, "/tmp/x.wav", audio, 16000, embed_fn=_sign_embed,
                                      intro_window_sec=1.0)
    w = out["segments"][0]["words"]
    assert w[0]["speaker"] == "SPEAKER_01" and w[0]["speaker_source"] == "anchor_diar"  # +구간→본인
    assert w[1]["speaker"] == "SPEAKER_00"   # -구간→상대
    # 원본 불변(immutability)
    assert result["segments"][0]["words"][0]["speaker"] == "SPEAKER_00"


# ── 도입부 NeMo enclosed-interjection 교정 (시간가드) ────────────────────────

_NEMO_TURNS = [
    {"start": 0.0, "end": 0.88, "nemo_spk": "speaker_1"},
    {"start": 0.88, "end": 1.13, "nemo_spk": "speaker_0"},   # 상대 끼어들기(여보세요#2 안에 enclosed)
    {"start": 1.12, "end": 2.17, "nemo_spk": "speaker_1"},
    {"start": 2.62, "end": 3.85, "nemo_spk": "speaker_0"},   # 네(1.76-3.10) 밖으로 넘침
    {"start": 4.30, "end": 5.13, "nemo_spk": "speaker_1"},
    {"start": 5.58, "end": 7.21, "nemo_spk": "speaker_0"},   # 말씀하세요(4.42-6.83) 밖으로 넘침
]
_NMAP = {"speaker_1": "SPEAKER_01", "speaker_0": "SPEAKER_00"}


def test_enclosed_detects_interjection():
    # 여보세요#2(0.73-1.38)가 상대 turn(0.88-1.13)을 완전히 감쌈 → 상대
    assert ad._nemo_enclosed_other(_NEMO_TURNS, 0.73, 1.38, "SPEAKER_01", _NMAP) == "SPEAKER_00"


def test_enclosed_ignores_transition_words():
    # 네(1.76-3.10): 상대 turn(2.62-3.85)이 단어 끝 밖 → enclosed 아님 → None
    assert ad._nemo_enclosed_other(_NEMO_TURNS, 1.76, 3.10, "SPEAKER_01", _NMAP) is None
    # 말씀하세요(4.42-6.83): 상대 turn(5.58-7.21) 밖으로 넘침 → None
    assert ad._nemo_enclosed_other(_NEMO_TURNS, 4.42, 6.83, "SPEAKER_01", _NMAP) is None


def test_enclosed_same_speaker_none():
    # 감싸는 turn 이 이미 같은 화자면 변경 없음(None)
    assert ad._nemo_enclosed_other(_NEMO_TURNS, 0.73, 1.38, "SPEAKER_00", _NMAP) is None


# ── 도입부 NeMo 직접배정 ─────────────────────────────────────────────────────

def test_nemo_spk_at_start():
    assert ad._nemo_spk_at_start(_NEMO_TURNS, 0.05) == "speaker_1"   # 여보세요#1
    assert ad._nemo_spk_at_start(_NEMO_TURNS, 3.24) == "speaker_0"   # 바쁘세요
    assert ad._nemo_spk_at_start(_NEMO_TURNS, 1.76) == "speaker_1"   # 네


def test_intro_assign_leaves_non_enclosing_unchanged():
    # enclosed 전용: 다른화자 turn 을 감싸지 않는 단어(여보세요#1·네·말씀하세요)는 anchor 라벨 유지
    words = [{"start": 0.05, "end": 0.55}, {"start": 1.76, "end": 3.10}, {"start": 4.42, "end": 6.83}]
    roles = ["SPEAKER_00", "SPEAKER_00", "SPEAKER_00"]
    out = ad._nemo_intro_assign(words, roles, _NEMO_TURNS, _NMAP, 30.0)
    assert out == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_00"]  # 무변경(enclose 안 함)


def test_intro_assign_enclosed_override():
    # 여보세요#2(0.73-1.38): 시작점 본인이나 상대 turn enclosed → 상대
    out = ad._nemo_intro_assign([{"start": 0.73, "end": 1.38}], ["SPEAKER_01"], _NEMO_TURNS, _NMAP, 30.0)
    assert out[0] == "SPEAKER_00"


def test_intro_assign_skips_body_time_guard():
    # 본문(60>=30): NeMo 직접배정 미적용(본문은 anchor 유지)
    body_turns = [{"start": 60.0, "end": 61.0, "nemo_spk": "speaker_0"}]
    out = ad._nemo_intro_assign([{"start": 60.2, "end": 60.5}], ["SPEAKER_01"], body_turns, _NMAP, 30.0)
    assert out[0] == "SPEAKER_01"


# ── Text-Informed 도입부 강제 분할 ──────────────────────────────────────────

def test_text_informed_split_breaks_bundle():
    # 도입부 뭉침 "여보세요? 여보세요? 네."(전부 SP01) → 호출부→상대SP00, 응답부(네)→본인SP01
    words = [{"start": 0.05, "end": 0.55, "word": "여보세요?"},
             {"start": 0.73, "end": 1.40, "word": "여보세요?"},
             {"start": 1.76, "end": 3.10, "word": "네."}]
    out = ad._text_informed_intro_split(words, ["SPEAKER_01", "SPEAKER_01", "SPEAKER_01"])
    assert out == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01"]


def test_text_informed_split_skips_call_only():
    # 응답어 없으면 분할 안 함(혼자 여보세요만)
    words = [{"start": 0.05, "end": 0.55, "word": "여보세요?"},
             {"start": 0.73, "end": 1.40, "word": "여보세요?"}]
    out = ad._text_informed_intro_split(words, ["SPEAKER_01", "SPEAKER_01"])
    assert out == ["SPEAKER_01", "SPEAKER_01"]


def test_text_informed_split_skips_body():
    # 본문(>=5s)은 미적용
    words = [{"start": 60.0, "end": 60.5, "word": "여보세요?"},
             {"start": 60.6, "end": 61.0, "word": "네."}]
    out = ad._text_informed_intro_split(words, ["SPEAKER_01", "SPEAKER_01"])
    assert out == ["SPEAKER_01", "SPEAKER_01"]


def test_apply_fallback_when_nemo_fails(monkeypatch):
    monkeypatch.setenv("VOICE_ANCHOR_DIAR_ENABLED", "true")
    monkeypatch.setattr(hd, "_call_nemo", lambda path, win: None)
    result = {"segments": [{"words": [{"word": "x", "start": 0.1, "end": 0.5, "speaker": "SPEAKER_00"}]}]}
    out = ad.apply_anchor_diarization(result, "/tmp/x.wav", np.zeros(16000, "float32"), 16000, embed_fn=_sign_embed)
    assert out is result   # 무변경
