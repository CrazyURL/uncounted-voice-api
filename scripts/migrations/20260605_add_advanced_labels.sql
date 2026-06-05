-- Migration: 고차 라벨 컬럼 (V-A 감정 / 노이즈 카테고리 / 다차원 dialog_act)
-- Date: 2026-06-05
-- Track: 빅테크 고차 라벨 — EmotionML V-A, 채널/음향환경, ISO 24617-2 다차원
--
-- SAFETY: ADDITIVE + NULLABLE. float/jsonb — CHECK 제약 없음(라벨 enum 사고 회피).
-- 백필(scripts/analysis/*_backfill.py)이 모델로 채움. 재처리 완료 후 실행(GPU 경합 회피).
--
-- HOW TO APPLY: Supabase SQL editor.

ALTER TABLE utterances
    -- V-A 감정 (audeering wav2vec2-dim, EmotionML 차원). 카테고리(emotion)에 2D 추가.
    ADD COLUMN IF NOT EXISTS emotion_valence   double precision,  -- 부정(-1)~긍정(+1)
    ADD COLUMN IF NOT EXISTS emotion_arousal   double precision,  -- 차분(0)~흥분(1)
    ADD COLUMN IF NOT EXISTS emotion_dominance double precision,
    -- 노이즈 카테고리 (음향장면: clean/babble/street/static 등). STT 강건성 학습용.
    ADD COLUMN IF NOT EXISTS noise_category    text,
    ADD COLUMN IF NOT EXISTS noise_confidence  double precision,
    -- 다차원 dialog_act (ISO 24617-2: communicative_function + dimension). 단일 dialog_act 보강.
    ADD COLUMN IF NOT EXISTS dialog_act_dims   jsonb;

COMMENT ON COLUMN utterances.emotion_valence IS 'EmotionML Valence(-1~+1). audeering wav2vec2-dim.';
COMMENT ON COLUMN utterances.noise_category IS '음향장면(clean/babble/street/static 등). STT robustness 메타.';
COMMENT ON COLUMN utterances.dialog_act_dims IS 'ISO 24617-2 다차원 화행 {communicative_function, dimension}. LLM 산출.';
