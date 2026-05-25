"""STT 발화분리(segmentation) 품질 진단 — read-only.

목적:
    /admin/calls 에서 발화 단위가 과분할되는 문제를 수치로 진단하고, Segmenter v2
    (app.services.utterance_segmenter_v2)를 메모리상으로만 적용했을 때의 감소율을
    추정한다(--dry-run-v2). DB 는 절대 변경하지 않는다.

안전 계약 (강제):
    - DB write 금지. 본 스크립트는 SELECT 만 한다 (insert/update/upsert/delete 호출 없음).
    - 원문 transcript_text / 개인정보를 export 하거나 로그로 출력하지 않는다.
      → transcript_text 는 이 프로세스 메모리에서만 길이·종결성·키워드 카운트 계산에 쓰이고
        어디에도 원문 그대로 기록되지 않는다.
    - 결과물은 집계 수치 + 익명 패턴 카운트만. 개별 발화 원문 없음.
    - 재처리·모델 교체·prod 반영 금지. dry-run 은 메모리상 추정일 뿐 DB 미반영.

사용법 (uncounted-voice-api 디렉토리에서, .env.live 자격증명 필요):
    source .env.live && python scripts/analysis/stt_segmentation_audit.py --dry-run-v2
    python scripts/analysis/stt_segmentation_audit.py --max-sessions 60 --out logs/seg_audit.json

⚠️ 운영 DB 읽기 — 승인된 진단 목적에만 사용.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.services.korean_sentence_ending import (  # noqa: E402
    ends_with_connective,
    ends_with_particle,
    ends_with_sentence_ending,
)
from app.services.utterance_segmenter_v2 import merge_v2  # noqa: E402

# 납품 후보로 보는 동의 상태
DELIVERY_CONSENT = ("approved", "both_agreed")

# 도메인 오인식 진단 키워드 (익명 카운트만 — 원문 미출력).
# 정오답이 아니라 '등장 빈도'를 보고 hotword/사전 후보를 판단하기 위함.
DOMAIN_KEYWORDS = (
    "수석", "선생님", "DLP", "호스트", "원격", "메일", "점검", "기준",
    "네트워크", "보안", "방화벽", "서버", "계정", "로그", "백업", "포트",
)

SHORT_FRAG_SEC = 2.0
SHORT_FRAG_WORDS = 5


# ─────────────────────────── DB (read-only) ───────────────────────────

def _load_supabase():
    """환경변수 우선, 없으면 repo .env.live → ../uncounted-api/.env 순으로 자격증명을 읽는다."""
    from supabase import create_client

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

    for env_path in (_REPO_ROOT / ".env.live", _REPO_ROOT.parent / "uncounted-api" / ".env"):
        if url and key:
            break
        if not env_path.is_file():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("SUPABASE_URL=") and not url:
                url = line.split("=", 1)[1].strip().strip("\"'")
            elif line.startswith(("SUPABASE_SERVICE_KEY=", "SUPABASE_SERVICE_ROLE_KEY=")) and not key:
                key = line.split("=", 1)[1].strip().strip("\"'")

    if not (url and key):
        sys.exit("SUPABASE_URL / SERVICE KEY 를 찾을 수 없습니다 (env 또는 .env.live).")
    return create_client(url, key)


def _fetch_done_sessions(client, limit: int) -> list[dict]:
    """gpu_upload_status='done' 세션을 최신순으로 읽는다 (id, consent_status, utterance_count)."""
    resp = (
        client.table("sessions")
        .select("id, consent_status, utterance_count, updated_at")
        .eq("gpu_upload_status", "done")
        .order("updated_at", desc=True)
        .limit(limit)
        .execute()
    )
    return resp.data or []


def _fetch_utterances(client, session_id: str) -> list[dict]:
    """한 세션의 utterance 를 sequence_order 순으로 페이지네이션해 읽는다."""
    rows: list[dict] = []
    page, frm = 1000, 0
    while True:
        resp = (
            client.table("utterances")
            .select(
                "session_id, sequence_order, start_sec, end_sec, duration_sec, "
                "transcript_text, transcript_words, speaker_id, pii_intervals, numeric_patterns"
            )
            .eq("session_id", session_id)
            .order("sequence_order", desc=False)
            .range(frm, frm + page - 1)
            .execute()
        )
        data = resp.data or []
        rows.extend(data)
        if len(data) < page:
            break
        frm += page
    return rows


# ─────────────────────────── metrics ───────────────────────────

def _duration(u: dict) -> float:
    s, e = _f(u.get("start_sec")), _f(u.get("end_sec"))
    return max(0.0, e - s)


def _word_count(u: dict) -> int:
    words = u.get("transcript_words")
    if isinstance(words, list) and words:
        return len(words)
    return len((u.get("transcript_text") or "").split())


def _char_len(u: dict) -> int:
    return len((u.get("transcript_text") or "").replace(" ", ""))


def _is_short_fragment(u: dict) -> bool:
    return _duration(u) < SHORT_FRAG_SEC or _word_count(u) < SHORT_FRAG_WORDS


def _bucket_duration(d: float) -> str:
    if d <= 0.0:
        return "0s"
    if d < 0.5:
        return "<0.5s"
    if d < 1.0:
        return "<1s"
    if d < 2.0:
        return "<2s"
    if d <= 10.0:
        return "2-10s"
    return ">10s"


def _new_metrics() -> dict:
    return {
        "utterances": 0,
        "duration_buckets": Counter(),
        "text_len": Counter(),       # 1-2글자 / 3-5글자 / 1단어 / <5단어
        "adjacency": Counter(),      # same_speaker_gap_le_0.4/0.8/1.2, speaker_change
        "finality": Counter(),       # ending / connective / particle / other
        "domain_keywords": Counter(),
        "pii_boundary_splits": 0,
        "short_fragments": 0,
        # 같은 화자 gap<=0.8 인접쌍에서 병합 차단 사유 분해
        "merge_blocking": Counter(),
    }


def _accumulate(m: dict, utts: list[dict]) -> None:
    m["utterances"] += len(utts)
    for u in utts:
        d = _duration(u)
        m["duration_buckets"][_bucket_duration(d)] += 1
        if _is_short_fragment(u):
            m["short_fragments"] += 1

        chars, words = _char_len(u), _word_count(u)
        if 1 <= chars <= 2:
            m["text_len"]["1-2글자"] += 1
        if 3 <= chars <= 5:
            m["text_len"]["3-5글자"] += 1
        if words == 1:
            m["text_len"]["1단어"] += 1
        if words < 5:
            m["text_len"]["<5단어"] += 1

        text = u.get("transcript_text") or ""
        if ends_with_sentence_ending(text):
            m["finality"]["ending"] += 1
        elif ends_with_connective(text):
            m["finality"]["connective"] += 1
        elif ends_with_particle(text):
            m["finality"]["particle"] += 1
        else:
            m["finality"]["other"] += 1

        for kw in DOMAIN_KEYWORDS:
            if kw in text:
                m["domain_keywords"][kw] += 1

    # 인접쌍 + PII 경계 분리
    for a, b in zip(utts, utts[1:]):
        if a.get("speaker_id") != b.get("speaker_id"):
            m["adjacency"]["speaker_change"] += 1
            continue
        gap = _f(b.get("start_sec")) - _f(a.get("end_sec"))
        if gap <= 0.4:
            m["adjacency"]["same_speaker_gap_le_0.4"] += 1
        if gap <= 0.8:
            m["adjacency"]["same_speaker_gap_le_0.8"] += 1
        if gap <= 1.2:
            m["adjacency"]["same_speaker_gap_le_1.2"] += 1
        if _pii_straddles(a, b):
            m["pii_boundary_splits"] += 1

        # 같은 화자 gap<=0.8 쌍의 병합 가능성/차단 사유 (정책 튜닝 근거)
        if gap <= 0.8:
            left_short = _is_short_fragment(a)
            right_short = _is_short_fragment(b)
            left_ending = ends_with_sentence_ending(a.get("transcript_text") or "")
            if left_short and not left_ending:
                m["merge_blocking"]["forward_mergeable(left_short,no_ending)"] += 1
            elif left_short and left_ending:
                m["merge_blocking"]["blocked_by_left_ending"] += 1
            elif not left_short and right_short and not left_ending:
                # 미완결 긴 발화 + 짧은 연속 조각 → 양방향 정책이면 병합 가능 (forward-only가 놓침)
                m["merge_blocking"]["missed_bidir_incomplete_left"] += 1
            elif not left_short and right_short and left_ending:
                # 완결된 긴 발화 뒤 짧은 조각 → 별개 문장, 병합 안 함이 맞음
                m["merge_blocking"]["missed_complete_left(keep)"] += 1
            else:
                m["merge_blocking"]["both_long"] += 1


def _pii_straddles(a: dict, b: dict) -> bool:
    """PII interval 이 a 우측 끝과 b 좌측 시작에 동시에 닿으면 경계 분리로 본다."""
    eps = 0.25
    a_end, b_start = _f(a.get("end_sec")), _f(b.get("start_sec"))
    ai = a.get("pii_intervals") or []
    bi = b.get("pii_intervals") or []
    right = any(_f(iv.get("endSec", iv.get("end"))) >= a_end - eps for iv in ai)
    left = any(_f(iv.get("startSec", iv.get("start"))) <= b_start + eps for iv in bi)
    return right and left


def _f(v) -> float:
    return float(v) if isinstance(v, (int, float)) else 0.0


def _finalize(m: dict) -> dict:
    """Counter 를 일반 dict 로 + 비율 추가."""
    n = max(1, m["utterances"])
    return {
        "utterances": m["utterances"],
        "short_fragment_count": m["short_fragments"],
        "short_fragment_pct": round(100.0 * m["short_fragments"] / n, 1),
        "duration_buckets": dict(m["duration_buckets"]),
        "text_len": dict(m["text_len"]),
        "adjacency": dict(m["adjacency"]),
        "finality": dict(m["finality"]),
        "finality_pct": {k: round(100.0 * v / n, 1) for k, v in m["finality"].items()},
        "domain_keywords": dict(m["domain_keywords"]),
        "pii_boundary_splits": m["pii_boundary_splits"],
        "merge_blocking": dict(m["merge_blocking"]),
    }


# ─────────────────────────── dry-run v2 ───────────────────────────

def _to_units(utts: list[dict]) -> list[dict]:
    return [
        {
            "start_sec": _f(u.get("start_sec")),
            "end_sec": _f(u.get("end_sec")),
            "speaker_id": u.get("speaker_id", "SPEAKER_00"),
            "transcript_text": u.get("transcript_text") or "",
            "word_count": _word_count(u),
            "pii_intervals": u.get("pii_intervals") or [],
            "numeric_patterns": u.get("numeric_patterns") or [],
        }
        for u in utts
    ]


def _dry_run_v2(utts: list[dict], bidirectional: bool = False) -> list[dict]:
    """세션 단위로 merge_v2 적용 (DB 미반영, 메모리 전용)."""
    return merge_v2(_to_units(utts), bidirectional=bidirectional)


def _short_and_ending(utts: list[dict]) -> tuple[int, int]:
    short = sum(1 for u in utts if _is_short_fragment(u))
    ending = sum(1 for u in utts if ends_with_sentence_ending(u.get("transcript_text") or ""))
    return short, ending


# ─────────────────────────── main ───────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="STT segmentation 품질 진단 (read-only)")
    parser.add_argument("--max-sessions", type=int, default=60,
                        help="분석할 done 세션 최대 수")
    parser.add_argument("--dry-run-v2", action="store_true",
                        help="Segmenter v2 메모리 적용 후 감소율 추정")
    parser.add_argument("--v2-bidirectional", action="store_true",
                        help="dry-run 시 bidirectional 병합(권장) 사용")
    parser.add_argument("--out", default=None, help="집계 JSON 저장 경로")
    args = parser.parse_args()

    client = _load_supabase()
    sessions = _fetch_done_sessions(client, args.max_sessions)
    print(f"[fetch] done 세션 {len(sessions)}개 로드")

    overall = _new_metrics()
    cat_delivery = _new_metrics()
    cat_pii = _new_metrics()
    cat_short = _new_metrics()

    session_summaries: list[dict] = []
    dry = {"before_utt": 0, "after_utt": 0, "before_short": 0, "after_short": 0,
           "before_ending": 0, "after_ending": 0}

    for s in sessions:
        sid = s["id"]
        utts = _fetch_utterances(client, sid)
        if not utts:
            continue

        _accumulate(overall, utts)

        short_n = sum(1 for u in utts if _is_short_fragment(u))
        short_ratio = short_n / len(utts)
        has_pii = any((u.get("pii_intervals") or u.get("numeric_patterns")) for u in utts)
        is_delivery = s.get("consent_status") in DELIVERY_CONSENT

        if is_delivery:
            _accumulate(cat_delivery, utts)
        if has_pii:
            _accumulate(cat_pii, utts)
        if short_ratio >= 0.5:
            _accumulate(cat_short, utts)

        if args.dry_run_v2:
            merged = _dry_run_v2(utts, bidirectional=args.v2_bidirectional)
            b_short, b_end = _short_and_ending(utts)
            a_short, a_end = _short_and_ending(merged)
            dry["before_utt"] += len(utts)
            dry["after_utt"] += len(merged)
            dry["before_short"] += b_short
            dry["after_short"] += a_short
            dry["before_ending"] += b_end
            dry["after_ending"] += a_end

        session_summaries.append({
            "session_id": sid,
            "consent_status": s.get("consent_status"),
            "utterances": len(utts),
            "short_ratio_pct": round(100.0 * short_ratio, 1),
            "has_pii": has_pii,
        })

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sessions_analyzed": len(session_summaries),
        "thresholds": {
            "short_fragment_sec": SHORT_FRAG_SEC,
            "short_fragment_words": SHORT_FRAG_WORDS,
        },
        "overall": _finalize(overall),
        "by_category": {
            "delivery_candidates": _finalize(cat_delivery),
            "pii_flag": _finalize(cat_pii),
            "short_heavy(>=50%)": _finalize(cat_short),
        },
        "sessions": session_summaries,
    }

    if args.dry_run_v2 and dry["before_utt"]:
        report["dry_run_v2"] = {
            "before_utterances": dry["before_utt"],
            "after_utterances": dry["after_utt"],
            "utterance_reduction_pct": round(
                100.0 * (dry["before_utt"] - dry["after_utt"]) / dry["before_utt"], 1),
            "before_short_fragments": dry["before_short"],
            "after_short_fragments": dry["after_short"],
            "short_fragment_reduction_pct": round(
                100.0 * (dry["before_short"] - dry["after_short"]) / max(1, dry["before_short"]), 1),
            "before_ending_pct": round(100.0 * dry["before_ending"] / dry["before_utt"], 1),
            "after_ending_pct": round(100.0 * dry["after_ending"] / max(1, dry["after_utt"]), 1),
        }

    _print_summary(report)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[out] 집계 JSON 저장: {out_path}")


def _print_summary(report: dict) -> None:
    o = report["overall"]
    print(f"\n===== STT Segmentation Audit ({report['sessions_analyzed']} sessions) =====")
    print(f"총 발화: {o['utterances']}  | short 단편: {o['short_fragment_count']} ({o['short_fragment_pct']}%)")
    print(f"duration 분포: {o['duration_buckets']}")
    print(f"text length: {o['text_len']}")
    print(f"인접쌍: {o['adjacency']}")
    print(f"종결성(%): {o['finality_pct']}")
    print(f"PII 경계 분리(straddle): {o['pii_boundary_splits']}")
    print(f"병합 차단 사유(same-spk gap<=0.8): {o['merge_blocking']}")
    print(f"도메인 키워드 빈도: {o['domain_keywords']}")
    if "dry_run_v2" in report:
        d = report["dry_run_v2"]
        print("\n----- Segmenter v2 dry-run (메모리 추정, DB 미반영) -----")
        print(f"발화 수: {d['before_utterances']} → {d['after_utterances']} "
              f"(-{d['utterance_reduction_pct']}%)")
        print(f"short 단편: {d['before_short_fragments']} → {d['after_short_fragments']} "
              f"(-{d['short_fragment_reduction_pct']}%)")
        print(f"종결어미 끝 비율: {d['before_ending_pct']}% → {d['after_ending_pct']}%")


if __name__ == "__main__":
    main()
