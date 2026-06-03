"""Data lineage / provenance capture (Phase 3 Task 8).

Builds ONE provenance record per session processing run: pipeline git SHA,
model versions, gate states, audio info. Pure metadata (no ML, no I/O except a
one-time cached `git rev-parse`). The worker persists this into lineage_runs and
repoints utterances.latest_lineage_run_id when LINEAGE_TRACKING_ENABLED=true.

Forward-only: captures the run that is happening now. Historical runs are not
reconstructable (the gates/models used then are unknown).
"""
from __future__ import annotations

import logging
import os
import subprocess
from typing import Any

from app import config

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_git_sha_cache: str | None = None


def capture_git_sha() -> str:
    """voice-api repo의 현재 커밋(short SHA). 1회 캡처 후 캐시. 실패 시 'unknown'."""
    global _git_sha_cache
    if _git_sha_cache is not None:
        return _git_sha_cache
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT, capture_output=True, text=True, timeout=5,
        )
        _git_sha_cache = out.stdout.strip() or "unknown"
    except Exception as e:  # noqa: BLE001 — git 미가용 환경에서도 무중단
        logger.debug("lineage: git sha 캡처 실패: %s", e)
        _git_sha_cache = "unknown"
    return _git_sha_cache


def _gate_states() -> dict[str, Any]:
    """현재 파이프라인 게이트/모드 상태. 없는 플래그는 getattr 기본값으로 안전."""
    g = lambda name, default=None: getattr(config, name, default)  # noqa: E731
    return {
        "preprocess_gain": g("PREPROCESS_GAIN_ENABLED"),
        "preprocess_denoise": g("PREPROCESS_DENOISE_ENABLED"),
        "preprocess_silence": g("PREPROCESS_SILENCE_ENABLED"),
        "preprocess_dedup": g("PREPROCESS_DEDUP_ENABLED"),
        "speaker_mapping_mode": g("SPEAKER_MAPPING_MODE"),
        "overlap_detection": g("OVERLAP_DETECTION_ENABLED"),
        "hybrid_diarization": g("VOICE_HYBRID_DIAR_ENABLED"),
        # PR #29 미머지 환경에서도 안전(없으면 None)
        "loudness_norm": g("CALL_LOUDNESS_NORM_ENABLED"),
        "loudness_local_gate": g("LOUDNESS_LOCAL_GATE_ENABLED"),
    }


def _model_versions(job_result: dict) -> dict[str, Any]:
    """모델 버전 집합. run-level 동일값은 config, 가공산출 버전은 job_result에서."""
    utts = job_result.get("utterances") or []
    auto_label_ver = next(
        (u.get("auto_label_model_version") for u in utts if u.get("auto_label_model_version")),
        None,
    )
    return {
        "stt": getattr(config, "MODEL_SIZE", None),
        "compute_type": getattr(config, "COMPUTE_TYPE", None),
        "diarization": getattr(config, "DIARIZATION_MODEL", None),
        "align": "whisperx-align",
        "auto_label": auto_label_ver,
        "pii_mask": job_result.get("pii_mask_version") or getattr(config, "PII_MASK_VERSION", None),
    }


def build_run_record(session_id: str, task_id: str, job_result: dict) -> dict[str, Any]:
    """세션 처리 run 1건의 lineage 레코드(dict). DB insert 페이로드."""
    duration = job_result.get("duration_seconds")
    return {
        "session_id": session_id,
        "task_id": task_id,
        "pipeline_git_sha": capture_git_sha(),
        "pipeline_version": getattr(config, "PIPELINE_VERSION", None),
        "service_version": getattr(config, "VERSION", None),
        "model_versions": _model_versions(job_result),
        "gate_states": _gate_states(),
        "audio_info": {"duration_sec": duration} if duration is not None else {},
    }
