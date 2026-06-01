"""PR-γ' — relation_inference (Ollama LLM 관계추론) + fallback 회귀 테스트.

검증:
  - 게이트 OFF(기본): LLM 미호출, 기존 호칭 정규식 동작 그대로 (무회귀)
  - 게이트 ON + Ollama 정상: LLM 결과 우선
  - 게이트 ON + Ollama 실패/저신뢰/비정상라벨: 정규식 fallback
  - "여보세요" 오탐: 정규식은 배우자(기존 버그) → LLM 은 직장동료(정정)
  - 기존 가족 통화(엄마/자기야): 정규식 fallback 시 정확 분류 유지

외부 Ollama 미접속 — urllib 을 monkeypatch 로 가짜 응답 주입(네트워크 0).
"""
import json
import os
from io import BytesIO

import pytest

from app.services import relation_inference as ri
from app.services.speaker_analysis_service import (
    _detect_relation,
    _detect_relation_by_salutation,
)


# ── 게이트 ─────────────────────────────────────────────────────────────────

def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("RELATION_INFER_OLLAMA_ENABLED", raising=False)
    assert ri.is_enabled() is False
    # 게이트 OFF → infer 는 호출 즉시 None
    assert ri.infer("아무 텍스트") is None


def test_enabled_flag(monkeypatch):
    monkeypatch.setenv("RELATION_INFER_OLLAMA_ENABLED", "true")
    assert ri.is_enabled() is True
    monkeypatch.setenv("RELATION_INFER_OLLAMA_ENABLED", "false")
    assert ri.is_enabled() is False


# ── build_dialogue ──────────────────────────────────────────────────────────

def test_build_dialogue_includes_both_speakers():
    d = ri.build_dialogue({"SPEAKER_00": ["여보세요", "네"], "SPEAKER_01": ["안녕하세요"]})
    assert "SPEAKER_00:" in d and "SPEAKER_01:" in d
    assert "여보세요" in d and "안녕하세요" in d


def test_build_dialogue_skips_empty():
    d = ri.build_dialogue({"SPEAKER_00": ["", "  "], "SPEAKER_01": ["내용"]})
    assert "내용" in d
    assert d.count("SPEAKER_00") == 0  # 빈 줄만 있는 화자는 미포함


# ── Ollama mock 주입 ────────────────────────────────────────────────────────

def _mock_ollama(monkeypatch, response_obj):
    """urllib.request.urlopen 을 가짜 Ollama 응답으로 대체."""
    body = json.dumps({"response": json.dumps(response_obj)}).encode("utf-8")

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return body

    def fake_urlopen(req, timeout=120):
        return _Resp()

    monkeypatch.setattr(ri.urllib.request, "urlopen", fake_urlopen)


def test_infer_returns_llm_result_when_enabled(monkeypatch):
    monkeypatch.setenv("RELATION_INFER_OLLAMA_ENABLED", "true")
    _mock_ollama(monkeypatch, {"relation": "직장동료", "confidence": 0.9})
    out = ri.infer("SPEAKER_00: 여보세요\nSPEAKER_01: AD 쪽 붙어야 작동합니다")
    assert out == ("직장동료", 0.9)


def test_infer_low_confidence_returns_none(monkeypatch):
    monkeypatch.setenv("RELATION_INFER_OLLAMA_ENABLED", "true")
    monkeypatch.setenv("RELATION_INFER_MIN_CONFIDENCE", "0.70")
    _mock_ollama(monkeypatch, {"relation": "직장동료", "confidence": 0.4})
    assert ri.infer("짧은 대화") is None  # 저신뢰 → fallback 신호


def test_infer_invalid_label_returns_none(monkeypatch):
    monkeypatch.setenv("RELATION_INFER_OLLAMA_ENABLED", "true")
    _mock_ollama(monkeypatch, {"relation": "외계인", "confidence": 0.99})
    assert ri.infer("대화") is None  # 라벨 화이트리스트 외 → fallback


def test_infer_network_failure_returns_none(monkeypatch):
    monkeypatch.setenv("RELATION_INFER_OLLAMA_ENABLED", "true")

    def boom(req, timeout=120):
        raise OSError("connection refused")

    monkeypatch.setattr(ri.urllib.request, "urlopen", boom)
    assert ri.infer("대화") is None  # 네트워크 실패 → fallback


# ── _detect_relation: LLM 우선 + 정규식 fallback ────────────────────────────

def test_detect_relation_fallback_when_gate_off(monkeypatch):
    # 게이트 OFF → 기존 정규식대로. "여보세요"는 정규식상 배우자(기존 버그 유지=무회귀)
    monkeypatch.setenv("RELATION_INFER_OLLAMA_ENABLED", "false")
    other = ["여보세요 여보세요 네 바쁘세요?"]
    all_by = {"SPEAKER_00": other, "SPEAKER_01": ["AD 붙어야 작동"]}
    # 기존 동작 보존 확인: 정규식이 '여보' 매칭 → 배우자
    assert _detect_relation_by_salutation(other) == "배우자"
    assert _detect_relation(other, all_by) == "배우자"  # 게이트 OFF라 정규식과 동일


def test_detect_relation_llm_corrects_yeoboseyo(monkeypatch):
    # 게이트 ON + Ollama 가 직장동료 → "여보세요→배우자" 오탐 정정
    monkeypatch.setenv("RELATION_INFER_OLLAMA_ENABLED", "true")
    _mock_ollama(monkeypatch, {"relation": "직장동료", "confidence": 0.9})
    other = ["여보세요 여보세요 네 바쁘세요?"]
    all_by = {"SPEAKER_00": other, "SPEAKER_01": ["AD 붙어야 작동합니다"]}
    assert _detect_relation(other, all_by) == "직장동료"


def test_detect_relation_llm_fail_falls_back_to_salutation(monkeypatch):
    # 게이트 ON 이지만 Ollama 실패 → 정규식 fallback (가족 통화 정확 분류 유지)
    monkeypatch.setenv("RELATION_INFER_OLLAMA_ENABLED", "true")

    def boom(req, timeout=120):
        raise OSError("refused")

    monkeypatch.setattr(ri.urllib.request, "urlopen", boom)
    other = ["엄마 어디야"]
    all_by = {"SPEAKER_00": other, "SPEAKER_01": ["응 집이야"]}
    assert _detect_relation(other, all_by) == "부모"  # 정규식 fallback


def test_detect_relation_family_regression_gate_off(monkeypatch):
    # 회귀: 게이트 OFF 에서 가족 호칭 정확 분류 유지
    monkeypatch.setenv("RELATION_INFER_OLLAMA_ENABLED", "false")
    assert _detect_relation(["자기야 사랑해"], None) == "배우자"
    assert _detect_relation(["엄마 나야"], None) == "부모"
    assert _detect_relation(["형 어디야"], None) == "형제자매"
