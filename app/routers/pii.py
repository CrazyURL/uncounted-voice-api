"""PII 후보 탐지 배치 라우트 (PII-1A).

기존 pii_masker.detect_pii_spans 를 단일 소스로 호출하고, pii_confidence 로
confidence tier 를 합성해 utterance 단위 후보를 반환한다.

⚠️ 안전 계약 (강제):
  - 응답에 원문 span text(matched_text) 를 절대 포함하지 않는다.
  - type / confidence / confidence_tier / high_precision_pattern / char_start / char_end 만 반환.
  - char_start/char_end 는 internal review 포인터이며 외부 export 대상이 아니다.

이름(ambiguous) 후보를 큐에 띄우기 위해 enable_name_masking=True 로 호출한다.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.pii_confidence import score_candidates
from app.pii_masker import detect_pii_spans

router = APIRouter(prefix="/api/v1", tags=["pii"])


class DetectBatchItem(BaseModel):
    utterance_id: str
    text: str


class DetectBatchRequest(BaseModel):
    items: list[DetectBatchItem] = Field(default_factory=list)
    # PII 검수는 이름 후보도 사람 판단 큐로 보내야 하므로 기본 True.
    enable_name_masking: bool = True


class PiiCandidateOut(BaseModel):
    type: str
    char_start: int
    char_end: int
    confidence: float
    high_precision_pattern: bool
    confidence_tier: str


class DetectBatchResultItem(BaseModel):
    utterance_id: str
    candidates: list[PiiCandidateOut]


class DetectBatchResponse(BaseModel):
    results: list[DetectBatchResultItem]


@router.post("/pii/detect-batch", response_model=DetectBatchResponse)
def detect_batch(req: DetectBatchRequest) -> DetectBatchResponse:
    """발화 텍스트 배치를 받아 PII 후보(원문 미포함)를 반환한다."""
    results: list[DetectBatchResultItem] = []
    for item in req.items:
        spans = detect_pii_spans(item.text, enable_name_masking=req.enable_name_masking)
        scored = score_candidates(spans)  # matched_text 제거됨
        results.append(
            DetectBatchResultItem(
                utterance_id=item.utterance_id,
                candidates=[PiiCandidateOut(**c) for c in scored],
            )
        )
    return DetectBatchResponse(results=results)
