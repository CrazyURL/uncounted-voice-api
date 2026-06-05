# -*- coding: utf-8 -*-
"""IVR(비대화 화자) role 제외 회귀 — IVR 이 최장 발화여도 self 오선택 안 됨."""
import numpy as np

from app.services import speaker_analysis_service as sa


def _segs():
    # IVR 이 가장 길다(150s) — 사람(각 50s)보다 길어도 self 가 되면 안 됨.
    return [
        {"speaker": "SPEAKER_IVR", "start": 0.0, "end": 150.0},
        {"speaker": "SPEAKER_00", "start": 150.0, "end": 200.0},
        {"speaker": "SPEAKER_01", "start": 200.0, "end": 250.0},
    ]


def test_is_human_helper():
    assert sa._is_human("SPEAKER_00")
    assert sa._is_human("SPEAKER_01")
    assert not sa._is_human("SPEAKER_IVR")
    assert not sa._is_human(None)


def test_ivr_not_selected_as_self_even_if_longest():
    sr = 16000
    audio = np.zeros(sr * 251, dtype=np.float32)
    res = sa.analyze_speakers(
        audio, sr, _segs(),
        pre_mask_texts_by_speaker={},
        reference_embedding=None,      # 임베딩 없음 → 최장발화 휴리스틱 경로
        embedding_model=None,
    )
    # IVR 은 절대 self 가 아니어야 한다(최장이라도).
    assert res["SPEAKER_IVR"].speaker_role == "other"
    assert res["SPEAKER_IVR"].speaker_relation is None  # 관계 추정도 제외
    # self 는 사람 화자 중 하나.
    selfs = [l for l, r in res.items() if r.speaker_role == "self"]
    assert selfs and all(sa._is_human(l) for l in selfs)
