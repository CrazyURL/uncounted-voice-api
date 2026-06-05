-- Migration: utterances MOS(통화품질) 컬럼
-- Date: 2026-06-05
-- Track: 개선 로드맵 #2 — non-intrusive MOS (POLQA/PESQ 는 intrusive=원본필요→우리 불가)
--
-- SAFETY: ADDITIVE + NULLABLE. float 컬럼이라 CHECK 제약 없음(라벨 enum 사고 회피).
-- MOS 백필(scripts/analysis/mos_backfill.py, CPU)이 채운다. STT 파이프라인 무관.
--
-- HOW TO APPLY: Supabase SQL editor.
--
-- Semantics:
--   mos_score   torchaudio SQUIM subjective MOS 추정(1–5, non-intrusive). 4.0+ toll / 3.5+ 양호.
--   mos_pesq    SQUIM objective PESQ 추정(1–4.5, MOS-LQO, reference-free).
--   mos_method  산출 방법 버전(예: "squim_v1").

ALTER TABLE utterances
    ADD COLUMN IF NOT EXISTS mos_score  double precision,
    ADD COLUMN IF NOT EXISTS mos_pesq   double precision,
    ADD COLUMN IF NOT EXISTS mos_method text,
    -- 로드맵 #3: true noise SNR(세그먼탈). 기존 snr_db=crest factor(동적범위)와 별개 지표.
    ADD COLUMN IF NOT EXISTS true_snr_db double precision;

COMMENT ON COLUMN utterances.mos_score IS
    'non-intrusive MOS 추정(torchaudio SQUIM subjective, 1-5). POLQA/PESQ intrusive 대체. 로드맵 #2.';
COMMENT ON COLUMN utterances.mos_pesq IS
    'SQUIM objective PESQ 추정(1-4.5, MOS-LQO, reference-free).';

-- ROLLBACK:
-- ALTER TABLE utterances DROP COLUMN IF EXISTS mos_score, DROP COLUMN IF EXISTS mos_pesq, DROP COLUMN IF EXISTS mos_method;
