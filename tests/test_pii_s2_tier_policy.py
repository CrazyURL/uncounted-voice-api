"""PR-S2A: auto_confirmed tier 정책 변경 검증.

정책:
  - rigid(주민/운전면허/여권/카드/이메일) → auto_confirmed 유지
  - loose(IP주소/계좌번호) → needs_human_decision 강등
  - 전화번호: 구분자(하이픈/공백/점) 형식만 auto_confirmed, 붙여쓰기 raw 숫자열은 needs_human
  - 음성 전사형 전화(한글 숫자어)는 강한 discriminator hint 로 auto_confirmed 유지

불변(가장 중요): mask_pii(납품 마스킹)는 tier 를 보지 않으므로 본 정책 변경에 영향받지 않는다.

안전: 원문/matched_text/snippet 미출력. detect-batch 응답은 type/offset/tier 만.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.pii_confidence import score_candidates
from app.pii_masker import detect_pii_spans, mask_pii
from app.routers import pii as pii_router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(pii_router.router)
    with TestClient(app) as c:
        yield c


def _tier_of(cands, pii_type):
    xs = [c for c in cands if c["type"] == pii_type]
    return xs[0]["confidence_tier"] if xs else None


def _batch(client, text):
    resp = client.post("/api/v1/pii/detect-batch",
                       json={"items": [{"utterance_id": "u", "text": text}]})
    assert resp.status_code == 200
    return resp.json()["results"][0]["candidates"]


# ── detect-batch tier (정책 핵심) ───────────────────────────────────
class TestDetectBatchTierPolicy:
    def test_ip_is_needs_human(self, client):
        cands = _batch(client, "장비 192.168.0.1 확인")
        assert _tier_of(cands, "IP주소") == "needs_human_decision"

    def test_account_is_needs_human(self, client):
        cands = _batch(client, "입금 35219008123456 계좌")
        assert _tier_of(cands, "계좌번호") == "needs_human_decision"

    def test_account_010_14digit_is_needs_human(self, client):
        # PR-S1 dedup 으로 14자리 010 숫자열은 계좌로 단일화 → 정책상 needs_human.
        cands = _batch(client, "입금 01012345678901 계좌")
        assert _tier_of(cands, "계좌번호") == "needs_human_decision"
        assert _tier_of(cands, "전화번호") is None  # 전화 substring 후보 없음(dedup)

    def test_delimited_phone_is_auto_confirmed(self, client):
        cands = _batch(client, "번호 010-1234-5678 입니다")
        assert _tier_of(cands, "전화번호") == "auto_confirmed"

    def test_raw_digit_phone_is_needs_human(self, client):
        # 붙여쓰기 11자리(구분자 없음) → 검수.
        cands = _batch(client, "번호 01099998888 임")
        assert _tier_of(cands, "전화번호") == "needs_human_decision"

    def test_email_is_auto_confirmed(self, client):
        cands = _batch(client, "메일 test@example.com 입니다")
        assert _tier_of(cands, "이메일") == "auto_confirmed"

    def test_resident_is_auto_confirmed(self, client):
        cands = _batch(client, "주민 900101-1234567 입니다")
        assert _tier_of(cands, "주민등록번호") == "auto_confirmed"


# ── detect_pii_spans 전화 형식 hint ─────────────────────────────────
class TestPhoneFormatHint:
    def test_delimited_phone_gets_hint(self):
        spans = detect_pii_spans("번호 010-1234-5678 임")
        ph = [s for s in spans if s["type"] == "전화번호"]
        assert len(ph) == 1
        assert ph[0].get("high_precision_pattern") is True

    def test_raw_phone_no_hint(self):
        spans = detect_pii_spans("번호 01012345678 임")
        ph = [s for s in spans if s["type"] == "전화번호"]
        assert len(ph) == 1
        assert "high_precision_pattern" not in ph[0]


# ── mask_pii 불변 (tier 변경이 마스킹에 새지 않음) ──────────────────
class TestMaskPiiUnchanged:
    def test_ip_still_masked(self):
        assert "***.***.***.***" in mask_pii("서버 192.168.0.1 접속")["masked_text"]

    def test_phone_still_masked(self):
        assert "010-****-5678" in mask_pii("번호 010-1234-5678 임")["masked_text"]

    def test_account_still_masked(self):
        r = mask_pii("입금 35219008123456 계좌")
        assert "35219008123456" not in r["masked_text"]
        assert any(d["type"] == "계좌번호" for d in r["pii_detected"])

    def test_mask_counts_unchanged(self):
        # tier 와 무관하게 탐지·마스킹 건수는 PR-S1 과 동일해야 한다.
        r = mask_pii("IP 192.168.0.1 폰 010-1234-5678 메일 a@b.com")
        types = {d["type"] for d in r["pii_detected"]}
        assert {"IP주소", "전화번호", "이메일"} <= types
        assert r["total_masked"] >= 3


# ── 기존 auto_confirmed 7건 shape 기준 tier 재분류 (PR-S2 적용 후) ───
# PR-S1 후 잔존 5건: IP 4 + 계좌 1. PR-S2 적용 시 전부 needs_human.
class TestExistingFiveAfterPolicy:
    def test_remaining_five_all_needs_human(self):
        spans = [
            {"type": "IP주소", "char_start": 5, "char_end": 19, "matched_text": "x"},
            {"type": "IP주소", "char_start": 3, "char_end": 16, "matched_text": "x"},
            {"type": "IP주소", "char_start": 0, "char_end": 13, "matched_text": "x"},
            {"type": "IP주소", "char_start": 8, "char_end": 21, "matched_text": "x"},
            {"type": "계좌번호", "char_start": 5, "char_end": 19, "matched_text": "x"},
        ]
        out = score_candidates(spans)
        tiers = [c["confidence_tier"] for c in out]
        assert tiers == ["needs_human_decision"] * 5
        assert all(c["confidence_tier"] != "auto_confirmed" for c in out)
