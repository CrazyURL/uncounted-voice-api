"""POST /api/v1/pii/detect-batch 라우트 테스트 (PII-1A).

기존 detect_pii_spans 를 호출하고 pii_confidence 로 tier 를 합성한 후보를
utterance 단위로 반환한다. 응답에 원문 span text 는 절대 포함되지 않는다.

라우트를 격리해서 마운트하므로 app.main(GPU/whisperx 체인) 을 끌어오지 않는다.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import pii as pii_router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(pii_router.router)
    with TestClient(app) as c:
        yield c


@pytest.mark.unit
def test_detect_batch_phone_auto_confirmed(client):
    resp = client.post(
        "/api/v1/pii/detect-batch",
        json={"items": [{"utterance_id": "u1", "text": "제 번호는 010-1234-5678입니다."}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    results = body["results"]
    assert len(results) == 1
    assert results[0]["utterance_id"] == "u1"
    cands = results[0]["candidates"]
    assert len(cands) == 1
    assert cands[0]["type"] == "전화번호"
    assert cands[0]["confidence_tier"] == "auto_confirmed"


@pytest.mark.unit
def test_detect_batch_name_needs_human(client):
    resp = client.post(
        "/api/v1/pii/detect-batch",
        json={"items": [{"utterance_id": "u2", "text": "홍길동 씨를 만났습니다."}]},
    )
    assert resp.status_code == 200
    cands = resp.json()["results"][0]["candidates"]
    names = [c for c in cands if c["type"] == "이름"]
    assert len(names) == 1
    assert names[0]["confidence_tier"] == "needs_human_decision"


@pytest.mark.unit
def test_detect_batch_response_excludes_raw_text(client):
    """응답 어디에도 원문 PII 문자열이 들어가면 안 된다."""
    secret = "010-1234-5678"
    resp = client.post(
        "/api/v1/pii/detect-batch",
        json={"items": [{"utterance_id": "u3", "text": f"번호 {secret} 입니다"}]},
    )
    assert resp.status_code == 200
    # 직렬화 전체 문자열에 원문 PII 가 없어야 한다.
    assert secret not in resp.text
    cands = resp.json()["results"][0]["candidates"]
    for c in cands:
        assert "matched_text" not in c
        assert set(c.keys()) == {
            "type",
            "char_start",
            "char_end",
            "confidence",
            "high_precision_pattern",
            "confidence_tier",
        }


@pytest.mark.unit
def test_detect_batch_offsets_point_into_text(client):
    text = "이메일은 test@example.com 입니다"
    resp = client.post(
        "/api/v1/pii/detect-batch",
        json={"items": [{"utterance_id": "u4", "text": text}]},
    )
    cands = resp.json()["results"][0]["candidates"]
    # name-masking on 이면 "이메일"이 이름 오탐으로 같이 잡힐 수 있다(threshold 튜닝 대상).
    # 이메일 후보만 골라 offset 정확성을 검증한다.
    email = next(c for c in cands if c["type"] == "이메일")
    # offset 은 항상 반환 (None 아님)
    assert isinstance(email["char_start"], int)
    assert isinstance(email["char_end"], int)
    # offset 이 실제 텍스트 구간을 가리킨다 (서버는 원문 미반환, 클라이언트만 확인)
    assert text[email["char_start"]:email["char_end"]] == "test@example.com"


@pytest.mark.unit
def test_detect_batch_empty_text_yields_no_candidates(client):
    resp = client.post(
        "/api/v1/pii/detect-batch",
        json={"items": [{"utterance_id": "u5", "text": "안녕하세요 반갑습니다"}]},
    )
    assert resp.status_code == 200
    assert resp.json()["results"][0]["candidates"] == []


@pytest.mark.unit
def test_detect_batch_multiple_items(client):
    resp = client.post(
        "/api/v1/pii/detect-batch",
        json={
            "items": [
                {"utterance_id": "a", "text": "010-1111-2222"},
                {"utterance_id": "b", "text": "그냥 인사말입니다"},
            ]
        },
    )
    results = {r["utterance_id"]: r["candidates"] for r in resp.json()["results"]}
    assert len(results["a"]) == 1
    assert results["b"] == []
