"""
emotion-only 모델(dialog_act_head 없는 heads.pt) 호환성 테스트.

커버:
- emotion-only heads.pt 로드 시 dialog_act_head=None graceful
- predict()에서 dialog_act=None 반환 (crash 없음)
- 기존 multi-head heads.pt 경로 회귀 없음
"""
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn


HIDDEN = 256


def _make_fake_encoder():
    cfg = types.SimpleNamespace(hidden_size=HIDDEN)
    enc = MagicMock()
    enc.config = cfg
    enc.eval.return_value = enc

    def _call(**kwargs):
        bs = kwargs["input_ids"].shape[0]
        return types.SimpleNamespace(
            last_hidden_state=torch.randn(bs, 5, HIDDEN)
        )

    enc.side_effect = _call
    return enc


def _heads_emotion_only() -> dict:
    head = nn.Linear(HIDDEN, 3)
    return {"emotion_head": head.state_dict()}


def _heads_multi() -> dict:
    e_head = nn.Linear(HIDDEN, 3)
    d_head = nn.Linear(HIDDEN, 15)
    return {"emotion_head": e_head.state_dict(), "dialog_act_head": d_head.state_dict()}


def _tokenizer_mock():
    tok = MagicMock()
    tok.return_value = {
        "input_ids": torch.zeros(2, 5, dtype=torch.long),
        "attention_mask": torch.ones(2, 5),
    }
    return tok


@pytest.fixture()
def svc():
    from app.services.auto_label_service import AutoLabelService
    return AutoLabelService()


@pytest.fixture()
def model_dir(tmp_path):
    d = tmp_path / "v_test"
    (d / "encoder").mkdir(parents=True)
    (d / "tokenizer").mkdir()
    (d / "heads.pt").write_bytes(b"")
    return d


def test_emotion_only_load_dialog_act_head_is_none(svc, model_dir):
    """emotion-only heads.pt 로드 시 _dialog_act_head=None."""
    with (
        patch("app.services.auto_label_service._resolve_current_model_path", return_value=model_dir),
        patch("transformers.AutoModel.from_pretrained", return_value=_make_fake_encoder()),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=_tokenizer_mock()),
        patch("torch.load", return_value=_heads_emotion_only()),
    ):
        svc._try_load()

    assert svc._dialog_act_head is None
    assert svc._emotion_head is not None
    assert svc.is_available()


def test_emotion_only_predict_returns_dialog_act_none(svc, model_dir):
    """emotion-only 모델 predict() 결과 dialog_act=None, crash 없음."""
    with (
        patch("app.services.auto_label_service._resolve_current_model_path", return_value=model_dir),
        patch("transformers.AutoModel.from_pretrained", return_value=_make_fake_encoder()),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=_tokenizer_mock()),
        patch("torch.load", return_value=_heads_emotion_only()),
    ):
        svc._try_load()

    results = svc.predict(["문장 하나", "문장 둘"])

    assert len(results) == 2
    for r in results:
        assert r.dialog_act is None, f"emotion-only 결과에 dialog_act가 None이 아님: {r.dialog_act}"
        assert r.emotion in ("긍정", "중립", "부정")


def test_multi_head_load_dialog_act_head_preserved(svc, model_dir):
    """multi-head heads.pt 로드 시 dialog_act_head 유지 (회귀 없음)."""
    with (
        patch("app.services.auto_label_service._resolve_current_model_path", return_value=model_dir),
        patch("transformers.AutoModel.from_pretrained", return_value=_make_fake_encoder()),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=_tokenizer_mock()),
        patch("torch.load", return_value=_heads_multi()),
    ):
        svc._try_load()

    assert svc._dialog_act_head is not None
    assert svc._emotion_head is not None
    assert svc.is_available()
