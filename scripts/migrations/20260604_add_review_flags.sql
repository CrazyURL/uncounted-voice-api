-- Migration: utterances soft-flag (human review) columns
-- Date: 2026-06-04
-- Track: PII/품질 플래그 로드맵 T2 — 모든 소프트플래그의 공통 착지점
--        (호격/Nim-Guard/반복환각/저신뢰/욕설·혐오/기업기밀 등 12종+)
--
-- SAFETY: purely ADDITIVE + NULLABLE. The running GPU worker does not write
-- these columns until REVIEW_FLAGS_ENABLED is turned on (worker only includes
-- the key when stt_processor produces it — same gating as the overlap columns,
-- 20260602). Applying this on the live (dev=live) DB has zero impact on
-- in-flight processing.
--
-- HOW TO APPLY: run in the Supabase SQL editor (voice-api has no direct DB
-- connection string / no local migration runner). Mirror into the canonical
-- migrations repo (uncounted-api) afterwards.
--
-- Semantics:
--   review_flags          [{"type": "<flag>", "severity": "low|med|high",
--                            "detail": "<사람이 읽는 근거>", "span": [start,end]?}]
--                          마스킹이 아님 — 사람 검수 대기 신호. PII 자동마스킹([PII_*])과 구분.
--                          type 예: vocative(호격), object_nim(사물+님 확신형환각),
--                                   repetition(반복/루프 환각), low_confidence(웅얼거림),
--                                   profanity(욕설·혐오), corp_secret(기업기밀/상표).
--   review_priority_score  정수(높을수록 우선). 레드큐 정렬 키.
--                          = Σ severity 가중치 (low=1, med=3, high=5) — worker 산출.

ALTER TABLE utterances
    ADD COLUMN IF NOT EXISTS review_flags          jsonb,
    ADD COLUMN IF NOT EXISTS review_priority_score integer;

-- 레드큐 1차 쿼리 = `WHERE review_priority_score > 0 ORDER BY review_priority_score DESC`.
-- 부분 인덱스로 플래그 없는 대다수 발화를 스캔에서 제외(프리미엄 경로 유지).
CREATE INDEX IF NOT EXISTS idx_utterances_review_priority
    ON utterances (review_priority_score DESC)
    WHERE review_priority_score IS NOT NULL AND review_priority_score > 0;

COMMENT ON COLUMN utterances.review_flags IS
    '사람 검수 대기 소프트플래그 배열(마스킹 아님). type/severity/detail. PII 자동마스킹과 구분. 로드맵 T2.';
COMMENT ON COLUMN utterances.review_priority_score IS
    'Σ severity 가중치(low1/med3/high5). 레드큐 정렬 키. 0/NULL = 검수불요.';

-- ROLLBACK (if ever needed):
-- DROP INDEX IF EXISTS idx_utterances_review_priority;
-- ALTER TABLE utterances
--   DROP COLUMN IF EXISTS review_flags,
--   DROP COLUMN IF EXISTS review_priority_score;
