-- Migration: utterances cross-talk (overlap) metadata columns
-- Date: 2026-06-02
-- Track: Phase 2 Task 5 (화자중첩 탐지/플래그)
--
-- SAFETY: purely ADDITIVE + NULLABLE. The running GPU worker does not read or
-- write these columns until OVERLAP_DETECTION_ENABLED is turned on, so applying
-- this on the live (dev=live) DB has zero impact on in-flight processing.
--
-- HOW TO APPLY: run in the Supabase SQL editor (voice-api has no direct DB
-- connection string / no local migration runner). Mirror into the canonical
-- migrations repo (uncounted-api) afterwards.
--
-- Semantics:
--   is_overlapping     true if any >=cutoff cross-talk region intersects this utterance.
--                      Distinct from existing `interruption_flag` (barge-in / turn-taking);
--                      this column = simultaneous speech (>=2 speakers active).
--   overlap_count      number of (clipped) overlap regions inside the utterance.
--   overlap_total_sec  summed clipped overlap duration (seconds).
--   overlap_ratio      overlap_total_sec / utterance duration (0..1).
--   overlap_intervals  [{"start_sec":..,"end_sec":..}, ...] clipped to utterance bounds.

ALTER TABLE utterances
    ADD COLUMN IF NOT EXISTS is_overlapping    boolean,
    ADD COLUMN IF NOT EXISTS overlap_count     integer,
    ADD COLUMN IF NOT EXISTS overlap_total_sec double precision,
    ADD COLUMN IF NOT EXISTS overlap_ratio     double precision,
    ADD COLUMN IF NOT EXISTS overlap_intervals jsonb;

-- Buyer filter `WHERE is_overlapping = false` is the primary query path.
-- Partial index keeps the premium (non-overlap) scan fast.
CREATE INDEX IF NOT EXISTS idx_utterances_is_overlapping
    ON utterances (is_overlapping);

COMMENT ON COLUMN utterances.is_overlapping IS
    'Cross-talk present (>=2 speakers simultaneously, >=cutoff). Premium tier = false. See Task 5.';
COMMENT ON COLUMN utterances.overlap_ratio IS
    'Overlap duration / utterance duration (0..1). Pricing/quality signal — count alone is insufficient.';

-- ROLLBACK (if ever needed):
-- DROP INDEX IF EXISTS idx_utterances_is_overlapping;
-- ALTER TABLE utterances
--   DROP COLUMN IF EXISTS is_overlapping,
--   DROP COLUMN IF EXISTS overlap_count,
--   DROP COLUMN IF EXISTS overlap_total_sec,
--   DROP COLUMN IF EXISTS overlap_ratio,
--   DROP COLUMN IF EXISTS overlap_intervals;
