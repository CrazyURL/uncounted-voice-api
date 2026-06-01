# -*- coding: utf-8 -*-
"""PR-γ' — peer 단위 longest-call relation 추론 (READ-ONLY, DB write 없음).

설계 (2026-06-02, "분석 batch만" 승인):
  1. peers 109건 그룹화 (이미 sessions.peer_id 1308 연결)
  2. peer 별 가장 긴 통화 1건 선택 (duration DESC)
  3. 그 통화 transcript 로 relation 추정 — Ollama qwen2.5:7b
  4. confidence < THRESHOLD 이면 다음 긴 통화 추가 (confidence ladder, 최대 3건)
  5. 결과를 '제안값'으로 리포트만 (DB write 0 — peers/session_speakers 무변경)

⚠️ 본 스크립트는 DB write 를 하지 않는다. 기존 관계모델 v2(propagation gate /
   override lock)와의 정합·통일 적용은 별도 승인 트랙. 휴리스틱 "여보세요→배우자"
   오탐의 정정 가능성을 LLM 추론으로 '측정'만 한다.

PII 보호: raw transcript / 실명 / 전화번호를 stdout·리포트에 출력하지 않는다.
  relation 레이블·confidence·세션수·해시 prefix 만.

사용:
  python scripts/analysis/peer_relation_infer.py --limit 5      # 표본 추론
  python scripts/analysis/peer_relation_infer.py --peer <id>    # 단건
  python scripts/analysis/peer_relation_infer.py --all          # 109 peer 전수 (write 없음)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

CONFIDENCE_THRESHOLD = 0.70
MAX_LADDER = 3            # confidence ladder 최대 통화 수


def _load_env() -> dict:
    e: dict[str, str] = {}
    path = "/home/gdash/project/Uncounted-root/uncounted-voice-api/.env.dev"
    for ln in open(path, encoding="utf-8"):
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1)
            e[k] = v
    return e


def _sb(env: dict):
    from supabase import create_client
    return create_client(env["SUPABASE_URL"], env["SUPABASE_SERVICE_KEY"])


def infer_relation(transcript: str) -> dict:
    """Ollama 로 relation 추정. app.services.relation_inference 코어를 재사용.

    batch 는 게이트와 무관하게 항상 추론하므로 RELATION_INFER_OLLAMA_ENABLED 를
    프로세스 내에서 강제 on 한 뒤 service.infer 를 호출한다(서버 env 무변경 —
    이 스크립트 프로세스 한정). threshold 미만이어도 raw 값을 보고용으로 받기
    위해 min_confidence 를 0 으로 둔다.
    """
    os.environ["RELATION_INFER_OLLAMA_ENABLED"] = "true"
    os.environ.setdefault("RELATION_INFER_MIN_CONFIDENCE", "0.0")
    sys.path.insert(0, "/home/gdash/project/Uncounted-root/uncounted-voice-api")
    from app.services import relation_inference as ri
    out = ri.infer(transcript)
    if out is None:
        return {"relation": "UNKNOWN", "confidence": 0.0, "reason": "no_result"}
    return {"relation": out[0], "confidence": out[1], "reason": ""}


def _session_transcript(sb, session_id: str, max_chars: int = 6000) -> str:
    """발화 transcript 를 화자 라벨과 함께 결합 (마스킹된 텍스트 사용)."""
    rows = (sb.table("utterances")
            .select("sequence_order,speaker_id,transcript_text")
            .eq("session_id", session_id).order("sequence_order").execute().data)
    lines = []
    for r in rows:
        spk = r.get("speaker_id") or "?"
        txt = (r.get("transcript_text") or "").strip()
        if txt:
            lines.append(f"{spk}: {txt}")
    joined = "\n".join(lines)
    return joined[:max_chars]


def infer_peer(sb, peer_id: str) -> dict:
    """peer 의 longest-call 부터 confidence ladder 로 relation 추정 (write 없음)."""
    sessions = (sb.table("sessions")
                .select("id,duration,utterance_count")
                .eq("peer_id", peer_id)
                .order("duration", desc=True).execute().data)
    sessions = [s for s in sessions if (s.get("utterance_count") or 0) > 0]
    if not sessions:
        return {"peer_id": peer_id[:8], "relation": "UNKNOWN", "confidence": 0.0,
                "calls_used": 0, "total_calls": 0, "reason": "no_utterances"}

    acc_transcript = ""
    result = {"relation": "UNKNOWN", "confidence": 0.0, "reason": ""}
    used = 0
    for s in sessions[:MAX_LADDER]:
        t = _session_transcript(sb, s["id"])
        if not t:
            continue
        acc_transcript = (acc_transcript + "\n\n" + t).strip() if acc_transcript else t
        used += 1
        result = infer_relation(acc_transcript[:8000])
        if result["confidence"] >= CONFIDENCE_THRESHOLD:
            break
    return {
        "peer_id": peer_id[:8],
        "relation": result["relation"],
        "confidence": round(result["confidence"], 2),
        "reason": result["reason"],
        "calls_used": used,
        "total_calls": len(sessions),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--peer", help="단일 peer id")
    ap.add_argument("--limit", type=int, default=5, help="표본 peer 수")
    ap.add_argument("--all", action="store_true", help="전 peer (write 없음)")
    ap.add_argument("--out", default="/tmp/peer_relation_infer.json")
    args = ap.parse_args()

    env = _load_env()
    sb = _sb(env)

    if args.peer:
        peers = [{"id": args.peer}]
    else:
        q = sb.table("peers").select("id,relationship,call_count,total_duration")
        peers = q.execute().data
        peers.sort(key=lambda p: (p.get("total_duration") or 0), reverse=True)
        if not args.all:
            peers = peers[: args.limit]

    results = []
    for i, p in enumerate(peers):
        r = infer_peer(sb, p["id"])
        results.append(r)
        print(f"[{i+1}/{len(peers)}] peer={r['peer_id']} "
              f"relation={r['relation']} conf={r['confidence']} "
              f"calls={r['calls_used']}/{r['total_calls']} ({r['reason']})")

    json.dump(results, open(args.out, "w"), ensure_ascii=False)
    # 분포 요약 (write 없음 — 제안값 리포트)
    from collections import Counter
    dist = Counter(r["relation"] for r in results)
    print(f"\n=== 제안 relation 분포 (DB write 없음): {dict(dist)} ===")
    print(f"평균 confidence: {round(sum(r['confidence'] for r in results)/max(len(results),1),2)}")
    print(f"저신뢰(<{CONFIDENCE_THRESHOLD}) peer: {sum(1 for r in results if r['confidence']<CONFIDENCE_THRESHOLD)}")
    print(f"saved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
