-- Migration: data lineage / provenance audit trail (Phase 3 Task 8)
-- Date: 2026-06-03
--
-- SAFETY: purely ADDITIVE. New table + one nullable column on utterances.
-- The worker writes here only when LINEAGE_TRACKING_ENABLED=true (default OFF),
-- so applying this on live (dev=live) has zero impact on in-flight processing.
-- No formal FK constraint on utterances (avoids full-table validation lock on a
-- live table); latest_lineage_run_id is a LOGICAL fk to lineage_runs.id.
--
-- HOW TO APPLY: Supabase SQL editor. Mirror into canonical migrations repo after.
--
-- Granularity: ONE row per session processing RUN (not per utterance) — model
-- versions / gates / pipeline SHA are uniform across a session's run, so this
-- stays small (sessions << utterances). On reprocess, a NEW run row is appended
-- (append-only audit); utterances.latest_lineage_run_id repoints to the newest.

CREATE TABLE IF NOT EXISTS lineage_runs (
    id                BIGSERIAL PRIMARY KEY,
    session_id        text NOT NULL,                 -- matches utterances.session_id (16-hex, NOT uuid)
    task_id           text,
    created_at        timestamptz NOT NULL DEFAULT now(),
    pipeline_git_sha  text,                          -- voice-api repo commit at processing time
    pipeline_version  text,                          -- config.PIPELINE_VERSION (e.g. v2-largev3-int8)
    service_version   text,                          -- config.VERSION
    model_versions    jsonb,                         -- {stt, diarization, align, auto_label, pii_mask, ...}
    gate_states       jsonb,                         -- {gain, denoise, silence, speaker_mapping, overlap, loudness, hybrid_diar, ...}
    audio_info        jsonb                          -- {duration_sec, source_key, ...} (optional)
);

CREATE INDEX IF NOT EXISTS idx_lineage_runs_session
    ON lineage_runs (session_id, created_at DESC);

-- Denormalized current-pointer on utterances for fast "latest lineage" lookup
-- (avoids window-function scan of the append-only table at export time).
ALTER TABLE utterances
    ADD COLUMN IF NOT EXISTS latest_lineage_run_id bigint;

CREATE INDEX IF NOT EXISTS idx_utterances_lineage_run
    ON utterances (latest_lineage_run_id);

COMMENT ON TABLE lineage_runs IS
    'Append-only provenance audit: one row per session processing run. Forward-only (historical runs not reconstructable). See Phase 3 Task 8.';
COMMENT ON COLUMN utterances.latest_lineage_run_id IS
    'Logical FK -> lineage_runs.id of the most recent processing run for this utterance (denormalized for fast export).';

-- ROLLBACK:
-- DROP INDEX IF EXISTS idx_utterances_lineage_run;
-- ALTER TABLE utterances DROP COLUMN IF EXISTS latest_lineage_run_id;
-- DROP TABLE IF EXISTS lineage_runs;
