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
    monkeypatch.setattr(hd, "_call_nemo", lambda p, w, num_speakers=2: None)
    r = _result()
    assert nf.apply_nemo_full_diarization(r, "/x.wav", np.zeros(16000), 16000, 100.0) is r


def test_not_two_speakers_fallback(monkeypatch):
    monkeypatch.setenv("VOICE_NEMO_FULL_DIAR_ENABLED", "true")
    monkeypatch.setattr(hd, "_call_nemo", lambda p, w, num_speakers=2: {
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
    monkeypatch.setattr(hd, "_call_nemo", lambda p, w, num_speakers=2: {"status": "success", "turns": turns})
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
    monkeypatch.setattr(hd, "_call_nemo", lambda p, w, num_speakers=2: {"status": "success", "turns": turns})
    monkeypatch.setattr(hd, "_f0_medians_from_segments",
                        lambda a, sr, by: {"speaker_0": 150.0, "speaker_1": 153.0})  # 차 3Hz < 8
    r = _result()
    assert nf.apply_nemo_full_diarization(r, "/x.wav", np.zeros(16000), 16000, 100.0) is r


def _ivr_result():
    """IVR(0~3s 기계음) + 상담사·고객(본문) 시나리오 word 결과."""
    return {"segments": [{"words": [
        {"word": "안내", "start": 0.5, "end": 1.0, "speaker": "SPEAKER_00"},   # IVR
        {"word": "말씀", "start": 1.5, "end": 2.0, "speaker": "SPEAKER_00"},   # IVR
        {"word": "여보세요", "start": 5.0, "end": 5.8, "speaker": "SPEAKER_00"},  # 상담사
        {"word": "네", "start": 6.0, "end": 6.4, "speaker": "SPEAKER_00"},      # 고객
    ]}]}


def test_ivr_excluded_then_two_speakers(monkeypatch):
    """IVR 클러스터(도입부+후반 미등장) 제외 후 남은 2명 매핑, IVR word=None."""
    monkeypatch.setenv("VOICE_NEMO_FULL_DIAR_ENABLED", "true")
    monkeypatch.setenv("VOICE_NEMO_IVR_EXCLUDE_ENABLED", "true")
    # NeMo 자동추정 3명: ivr(0~2.5, 후반 미등장), 상담사(5~5.8), 고객(6~10)
    turns = [
        {"nemo_spk": "speaker_2", "start": 0.0, "end": 2.5},    # IVR
        {"nemo_spk": "speaker_0", "start": 5.0, "end": 5.8},    # 상담사(저음)
        {"nemo_spk": "speaker_1", "start": 6.0, "end": 10.0},   # 고객(고음)
    ]
    seen = {}

    def _capture(p, w, num_speakers=2):
        seen["num_speakers"] = num_speakers
        return {"status": "success", "turns": turns}

    monkeypatch.setattr(hd, "_call_nemo", _capture)
    monkeypatch.setattr(hd, "_f0_medians_from_segments",
                        lambda a, sr, by: {"speaker_0": 100.0, "speaker_1": 180.0})
    out = nf.apply_nemo_full_diarization(_ivr_result(), "/x.wav", np.zeros(48000 * 4), 16000, 10.0)
    assert seen["num_speakers"] is None  # 자동추정 요청 확인
    spk = [w["speaker"] for seg in out["segments"] for w in seg["words"]]
    # IVR word(0.5,1.5)=IVR라벨, 여보세요(5.0)=speaker_0→SP01, 네(6.0)=speaker_1→SP00
    assert spk == [nf._IVR_LABEL, nf._IVR_LABEL, ad._LABEL_LOW, ad._LABEL_HIGH]
    srcs = [w.get("speaker_source") for seg in out["segments"] for w in seg["words"]]
    assert srcs[0] == "nemo_full_ivr" and srcs[1] == "nemo_full_ivr"


def test_ivr_exclude_off_three_speakers_fallback(monkeypatch):
    """IVR 제외 OFF 면 3명 발견 시 기존대로 fallback(2 아님)."""
    monkeypatch.setenv("VOICE_NEMO_FULL_DIAR_ENABLED", "true")
    monkeypatch.setenv("VOICE_NEMO_IVR_EXCLUDE_ENABLED", "false")
    turns = [
        {"nemo_spk": "speaker_2", "start": 0.0, "end": 2.5},
        {"nemo_spk": "speaker_0", "start": 5.0, "end": 5.8},
        {"nemo_spk": "speaker_1", "start": 6.0, "end": 10.0},
    ]
    monkeypatch.setattr(hd, "_call_nemo",
                        lambda p, w, num_speakers=2: {"status": "success", "turns": turns})
    r = _ivr_result()
    assert nf.apply_nemo_full_diarization(r, "/x.wav", np.zeros(48000), 16000, 10.0) is r


def test_ivr_two_clusters_both_excluded(monkeypatch):
    """실측 회귀(eb34a6a): IVR 가 2개 클러스터로 쪼개지고 두 번째가 통화 중간(t>0)에
    시작해도, last_end 만으로 둘 다 제외 → 남은 2명(상담사·고객) 매핑 성공."""
    monkeypatch.setenv("VOICE_NEMO_FULL_DIAR_ENABLED", "true")
    monkeypatch.setenv("VOICE_NEMO_IVR_EXCLUDE_ENABLED", "true")
    # dur=308 기준 실측 분포 모사(비겹침 turn): IVR 2클러스터(last_end<154 둘 다),
    # 상담사·고객은 본문 후반까지 등장(last_end≥154).
    turns = [
        {"nemo_spk": "ivr_a", "start": 0.0, "end": 40.0},      # IVR last 40
        {"nemo_spk": "ivr_b", "start": 44.8, "end": 111.6},    # 중간 시작 IVR, last 111.6
        {"nemo_spk": "agent", "start": 112.0, "end": 130.0},
        {"nemo_spk": "cust", "start": 130.0, "end": 150.0},
        {"nemo_spk": "agent", "start": 200.0, "end": 210.0},   # agent last 210
        {"nemo_spk": "cust", "start": 250.0, "end": 306.7},    # cust last 306.7
    ]
    monkeypatch.setattr(hd, "_call_nemo",
                        lambda p, w, num_speakers=2: {"status": "success", "turns": turns})
    monkeypatch.setattr(hd, "_f0_medians_from_segments",
                        lambda a, sr, by: {"agent": 110.0, "cust": 175.0})
    r = {"segments": [{"words": [
        {"word": "안내", "start": 50.0, "end": 50.5, "speaker": "SPEAKER_00"},    # ivr_b
        {"word": "번호", "start": 205.0, "end": 205.5, "speaker": "SPEAKER_00"},  # agent
        {"word": "네", "start": 260.0, "end": 260.5, "speaker": "SPEAKER_00"},    # cust
    ]}]}
    out = nf.apply_nemo_full_diarization(r, "/x.wav", np.zeros(48000 * 4), 16000, 308.0)
    spk = [w["speaker"] for seg in out["segments"] for w in seg["words"]]
    # 50s=ivr_b→IVR라벨, 205s=agent(저음)→SP01, 260s=cust(고음)→SP00
    assert spk[0] == nf._IVR_LABEL     # IVR word → IVR 라벨 보존
    assert spk[1] == ad._LABEL_LOW     # agent(저음)=SPEAKER_01
    assert spk[2] == ad._LABEL_HIGH    # cust(고음)=SPEAKER_00


def test_ivr_removed_f0_close_keeps_separation(monkeypatch):
    """IVR 제거로 2화자 확보 후 F0 가 가까워도(<8Hz) fallback 하지 않고 분리 보존.
    실측 회귀(eb34a6a: 상담사·고객 F0 차 5.5Hz)."""
    monkeypatch.setenv("VOICE_NEMO_FULL_DIAR_ENABLED", "true")
    monkeypatch.setenv("VOICE_NEMO_IVR_EXCLUDE_ENABLED", "true")
    turns = [
        {"nemo_spk": "ivr", "start": 0.0, "end": 100.0},       # last 100 < 154 → 제외
        {"nemo_spk": "agent", "start": 112.0, "end": 130.0},
        {"nemo_spk": "cust", "start": 130.0, "end": 200.0},
        {"nemo_spk": "agent", "start": 200.0, "end": 240.0},   # agent last 240 → 사람
        {"nemo_spk": "cust", "start": 240.0, "end": 306.0},    # cust last 306 → 사람
    ]
    monkeypatch.setattr(hd, "_call_nemo",
                        lambda p, w, num_speakers=2: {"status": "success", "turns": turns})
    # F0 차 5.5Hz < 8.0 (가까움)
    monkeypatch.setattr(hd, "_f0_medians_from_segments",
                        lambda a, sr, by: {"agent": 150.0, "cust": 155.5})
    r = {"segments": [{"words": [
        {"word": "안내", "start": 50.0, "end": 50.5, "speaker": "SPEAKER_00"},   # ivr→None
        {"word": "번호", "start": 120.0, "end": 120.5, "speaker": "SPEAKER_00"}, # agent(저음)→SP01
        {"word": "네", "start": 250.0, "end": 250.5, "speaker": "SPEAKER_00"},   # cust(고음)→SP00
    ]}]}
    out = nf.apply_nemo_full_diarization(r, "/x.wav", np.zeros(48000 * 4), 16000, 308.0)
    assert out is not r  # fallback 아님(분리 보존)
    spk = [w["speaker"] for seg in out["segments"] for w in seg["words"]]
    assert spk[0] == nf._IVR_LABEL
    assert spk[1] == ad._LABEL_LOW    # agent(저음 150)=SP01
    assert spk[2] == ad._LABEL_HIGH   # cust(고음 155.5)=SP00


def test_ivr_cutoff_nulls_residual_in_human_cluster(monkeypatch):
    """도입부 cutoff: 사람 클러스터(통화 전체 존속)에 흡수된 잔여 IVR 음성도 제거.
    실측 회귀(eb34a6a: speaker_0 가 IVR 구간 4.9~112 에도 turn 보유)."""
    monkeypatch.setenv("VOICE_NEMO_FULL_DIAR_ENABLED", "true")
    monkeypatch.setenv("VOICE_NEMO_IVR_EXCLUDE_ENABLED", "true")
    turns = [
        {"nemo_spk": "ivr", "start": 0.0, "end": 110.0},        # IVR last 110 → cutoff=110
        {"nemo_spk": "agent", "start": 30.0, "end": 60.0},      # 사람인데 IVR 구간에 turn 보유
        {"nemo_spk": "agent", "start": 120.0, "end": 240.0},    # last 240 → 사람
        {"nemo_spk": "cust", "start": 240.0, "end": 306.0},     # last 306 → 사람
    ]
    monkeypatch.setattr(hd, "_call_nemo",
                        lambda p, w, num_speakers=2: {"status": "success", "turns": turns})
    monkeypatch.setattr(hd, "_f0_medians_from_segments",
                        lambda a, sr, by: {"agent": 110.0, "cust": 175.0})
    r = {"segments": [{"words": [
        {"word": "안내", "start": 45.0, "end": 45.5, "speaker": "SPEAKER_00"},   # agent turn(30~60)이나 <cutoff → None
        {"word": "번호", "start": 200.0, "end": 200.5, "speaker": "SPEAKER_00"}, # agent(저음)→SP01
        {"word": "네", "start": 250.0, "end": 250.5, "speaker": "SPEAKER_00"},   # cust(고음)→SP00
    ]}]}
    out = nf.apply_nemo_full_diarization(r, "/x.wav", np.zeros(48000 * 4), 16000, 308.0)
    spk = [w["speaker"] for seg in out["segments"] for w in seg["words"]]
    assert spk[0] == nf._IVR_LABEL     # 도입부(45s<110) 잔여 IVR → IVR 라벨(사람 클러스터 소속이어도)
    assert spk[1] == ad._LABEL_LOW     # agent
    assert spk[2] == ad._LABEL_HIGH    # cust


def test_ivr_cutoff_nulls_wordless_intro_segment(monkeypatch):
    """word 타임스탬프 없는 도입부 segment 도 segment-level 로 IVR 제외(꼬리 멘트)."""
    monkeypatch.setenv("VOICE_NEMO_FULL_DIAR_ENABLED", "true")
    monkeypatch.setenv("VOICE_NEMO_IVR_EXCLUDE_ENABLED", "true")
    turns = [
        {"nemo_spk": "ivr", "start": 0.0, "end": 110.0},        # cutoff=110
        {"nemo_spk": "agent", "start": 120.0, "end": 240.0},
        {"nemo_spk": "cust", "start": 240.0, "end": 306.0},
    ]
    monkeypatch.setattr(hd, "_call_nemo",
                        lambda p, w, num_speakers=2: {"status": "success", "turns": turns})
    monkeypatch.setattr(hd, "_f0_medians_from_segments",
                        lambda a, sr, by: {"agent": 110.0, "cust": 175.0})
    # 첫 segment = word 없는 도입부 IVR 꼬리(start 43 < 110), 둘째 = 본문(words 보유)
    r = {"segments": [
        {"start": 43.0, "end": 44.0, "text": "드리겠습니다.", "speaker": "SPEAKER_00"},
        {"start": 200.0, "end": 205.0, "speaker": "SPEAKER_00",
         "words": [{"word": "번호", "start": 200.0, "end": 200.5, "speaker": "SPEAKER_00"}]},
    ]}
    out = nf.apply_nemo_full_diarization(r, "/x.wav", np.zeros(48000 * 4), 16000, 308.0)
    seg0 = out["segments"][0]
    assert seg0.get("speaker") == nf._IVR_LABEL             # word 없는 도입부 → IVR 라벨
    assert seg0.get("speaker_source") == "nemo_full_ivr"
    # 본문 word 는 정상 매핑
    assert out["segments"][1]["words"][0]["speaker"] == ad._LABEL_LOW


def test_ivr_intro_already_none_word_relabeled_to_ivr(monkeypatch):
    """도입부 단어가 pyannote 단계서 이미 speaker=None 이어도 IVR 라벨로 재배정한다
    (None 으로 남기지 않음 → 잔여 IVR 꼬리도 IVR 화자로 일관 보존)."""
    monkeypatch.setenv("VOICE_NEMO_FULL_DIAR_ENABLED", "true")
    monkeypatch.setenv("VOICE_NEMO_IVR_EXCLUDE_ENABLED", "true")
    turns = [
        {"nemo_spk": "ivr", "start": 0.0, "end": 110.0},        # cutoff=110
        {"nemo_spk": "agent", "start": 120.0, "end": 240.0},
        {"nemo_spk": "cust", "start": 240.0, "end": 306.0},
    ]
    monkeypatch.setattr(hd, "_call_nemo",
                        lambda p, w, num_speakers=2: {"status": "success", "turns": turns})
    monkeypatch.setattr(hd, "_f0_medians_from_segments",
                        lambda a, sr, by: {"agent": 110.0, "cust": 175.0})
    r = {"segments": [{"words": [
        {"word": "자막", "start": 104.0, "end": 104.5, "speaker": None},   # 도입부+이미 None
        {"word": "번호", "start": 200.0, "end": 200.5, "speaker": "SPEAKER_00"},
    ]}]}
    out = nf.apply_nemo_full_diarization(r, "/x.wav", np.zeros(48000 * 4), 16000, 308.0)
    w0 = out["segments"][0]["words"][0]
    assert w0["speaker"] == nf._IVR_LABEL                 # 이미 None 이어도 IVR 라벨로 재배정
    assert w0.get("speaker_source") == "nemo_full_ivr"


def test_ivr_exclude_no_ivr_match_fallback(monkeypatch):
    """3명이지만 IVR 패턴(후반 미등장) 없으면 제외 0 → 2 아님 fallback(보수적)."""
    monkeypatch.setenv("VOICE_NEMO_FULL_DIAR_ENABLED", "true")
    monkeypatch.setenv("VOICE_NEMO_IVR_EXCLUDE_ENABLED", "true")
    # 셋 다 통화 후반(>5s)까지 등장 → IVR 아님
    turns = [
        {"nemo_spk": "speaker_2", "start": 0.0, "end": 9.0},
        {"nemo_spk": "speaker_0", "start": 5.0, "end": 9.5},
        {"nemo_spk": "speaker_1", "start": 6.0, "end": 10.0},
    ]
    monkeypatch.setattr(hd, "_call_nemo",
                        lambda p, w, num_speakers=2: {"status": "success", "turns": turns})
    r = _ivr_result()
    assert nf.apply_nemo_full_diarization(r, "/x.wav", np.zeros(48000), 16000, 10.0) is r
