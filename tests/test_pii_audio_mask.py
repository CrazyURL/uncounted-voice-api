# -*- coding: utf-8 -*-
"""오디오 PII 비프 — CORE 라벨 필터 + worker 게이트."""
from app.pii_masker import CORE_PII_LABELS


class TestCorePiiLabels:
    def test_includes_high_precision(self):
        for lbl in ["전화번호", "주민등록번호", "카드번호", "계좌번호", "이메일", "IP주소"]:
            assert lbl in CORE_PII_LABELS

    def test_excludes_name_and_extended(self):
        # 이름·extended(numeric_sensitive 등)는 오디오 비프 대상 아님
        assert "이름" not in CORE_PII_LABELS
        assert "numeric_sensitive_like" not in CORE_PII_LABELS

    def test_core_filter_drops_non_core(self):
        ranges = [
            (0.0, 1.0, "전화번호"),
            (1.0, 2.0, "numeric_sensitive_like"),
            (2.0, 3.0, "이름"),
            (3.0, 4.0, "주민등록번호"),
        ]
        beep = [r for r in ranges if r[2] in CORE_PII_LABELS]
        assert [r[2] for r in beep] == ["전화번호", "주민등록번호"]


class TestWorkerGate:
    def test_mask_audio_pii_off_by_default(self, monkeypatch):
        from app import config
        monkeypatch.setattr(config, "PII_AUDIO_MASK_ENABLED", False)
        from app.worker import build_submit_params
        assert "mask_audio_pii" not in build_submit_params()

    def test_mask_audio_pii_on_when_gated(self, monkeypatch):
        from app import config
        monkeypatch.setattr(config, "PII_AUDIO_MASK_ENABLED", True)
        from app.worker import build_submit_params
        p = build_submit_params()
        assert p.get("mask_audio_pii") == "true"
        assert p.get("pii_intervals_only") == "true"  # 메타데이터는 항상
