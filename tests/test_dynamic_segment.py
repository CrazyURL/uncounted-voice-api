"""동적 윈도우 세그먼테이션 (env-gated) 단위 테스트.

검증:
  - 화자 변화에서만 분리(침묵 무시) — 같은 화자 연속 유지
  - 장기 발화 동적 분할: SOFT(15s) 후 문장종결(.?!) / 침묵 GAP(0.4s) / HARD(30s) 강제
  - 게이트 OFF(기본): 기존 경로(무회귀)
"""
from app import config
from app.services import utterance_segmenter as us


def _w(spk, s, e, text="x"):
    return {"speaker": spk, "start": s, "end": e, "word": text}


# ── 화자변화 only 분리 ──────────────────────────────────────────────────────

def test_split_by_speaker_only_ignores_silence():
    # 같은 화자 + 4s 침묵 갭 → 하나의 발화로 유지(침묵에 안 쪼갬)
    words = [_w("A", 0.0, 1.0), _w("A", 5.0, 6.0)]
    raw = us._split_by_speaker_only(words)
    assert len(raw) == 1
    assert len(raw[0].words) == 2


def test_split_by_speaker_only_splits_on_speaker_change():
    words = [_w("A", 0.0, 1.0), _w("B", 1.1, 2.0), _w("A", 2.1, 3.0)]
    raw = us._split_by_speaker_only(words)
    assert [r.speaker_id for r in raw] == ["A", "B", "A"]


# ── 동적 윈도우 분할 ────────────────────────────────────────────────────────

def _cfg(monkeypatch, soft=15.0, hard=30.0, gap=0.4):
    monkeypatch.setattr(config, "DYNAMIC_SEGMENT_SOFT_SEC", soft)
    monkeypatch.setattr(config, "DYNAMIC_SEGMENT_HARD_SEC", hard)
    monkeypatch.setattr(config, "DYNAMIC_SEGMENT_GAP_SEC", gap)


def test_dynamic_short_run_not_split(monkeypatch):
    _cfg(monkeypatch)
    u = us._RawUtterance("A", [_w("A", 0.0, 5.0), _w("A", 5.1, 10.0)])  # 10s < soft
    assert len(us._split_one_dynamic(u)) == 1


def test_dynamic_split_at_sentence_end_after_soft(monkeypatch):
    _cfg(monkeypatch)
    # 0~16s 한 단어(긴), 그 뒤 "끝." 문장종결, 그 뒤 "다음" → soft 후 종결서 분할
    words = [_w("A", 0.0, 16.0, "계속"), _w("A", 16.1, 17.0, "끝."), _w("A", 17.1, 20.0, "다음")]
    out = us._split_one_dynamic(us._RawUtterance("A", words))
    assert len(out) == 2
    assert out[0].words[-1]["word"] == "끝."


def test_dynamic_split_at_silence_gap_after_soft(monkeypatch):
    _cfg(monkeypatch)
    # soft 후 0.5s 갭(>=0.4)에서 분할
    words = [_w("A", 0.0, 16.0, "계속"), _w("A", 16.5, 18.0, "다음")]  # gap 0.5
    out = us._split_one_dynamic(us._RawUtterance("A", words))
    assert len(out) == 2


def test_dynamic_no_split_before_soft(monkeypatch):
    _cfg(monkeypatch)
    # 문장종결이 있어도 SOFT(15s) 전이면 안 쪼갬(단문 보존)
    words = [_w("A", 0.0, 2.0, "안녕."), _w("A", 2.1, 4.0, "반가워.")]
    assert len(us._split_one_dynamic(us._RawUtterance("A", words))) == 1


def test_dynamic_hard_ceiling_under_30(monkeypatch):
    _cfg(monkeypatch)
    # 종결·갭 없이 35s 연속(빠른 말) → HARD 직전 강제, 각 chunk < 30s
    words = [_w("A", float(i), float(i) + 0.9, "어") for i in range(0, 35)]  # gap 0.1, no 종결
    out = us._split_one_dynamic(us._RawUtterance("A", words))
    assert len(out) >= 2
    for o in out:
        dur = o.words[-1]["end"] - o.words[0]["start"]
        assert dur < 30.0


# ── segment() 통합 (게이트) ─────────────────────────────────────────────────

def test_segment_dynamic_mode(monkeypatch):
    monkeypatch.setattr(config, "DYNAMIC_SEGMENT_ENABLED", True)
    _cfg(monkeypatch)
    # A(침묵갭 포함) → B. A 두 단어는 한 발화, B 별도.
    words = [_w("A", 0.0, 1.0), _w("A", 5.0, 6.0), _w("B", 6.5, 7.0)]
    out = us.segment(words, 10.0)
    assert [b.speaker_id for b in out] == ["A", "B"]


def test_default_mode_silence_split_preserved(monkeypatch):
    # 기본 모드(게이트 OFF): 동일화자라도 침묵 갭에서 분할(각 5s≥MIN 이라 merge 안 됨) → 2개.
    monkeypatch.setattr(config, "DYNAMIC_SEGMENT_ENABLED", False)
    words = [_w("A", 0.0, 5.0), _w("A", 6.0, 11.0)]
    out = us.segment(words, 12.0)
    assert len(out) == 2   # 기존 동작(침묵 분할) 유지


def test_dynamic_keeps_same_speaker_across_silence(monkeypatch):
    # 동적 모드: 같은 침묵 갭이어도 같은 화자면 1개 유지(11s<15s soft → 분할 안 함).
    monkeypatch.setattr(config, "DYNAMIC_SEGMENT_ENABLED", True)
    _cfg(monkeypatch)
    words = [_w("A", 0.0, 5.0), _w("A", 6.0, 11.0)]
    out = us.segment(words, 12.0)
    assert len(out) == 1
