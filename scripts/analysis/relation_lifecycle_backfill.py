# -*- coding: utf-8 -*-
"""관계 lifecycle backfill — 비파괴 적용.

per-call: 교차검증(교사 데모션) → session_speakers.speaker_relation=null(강등분).
per-peer: consolidate → peers.relationship 갱신(positive만), 충돌→rel_source=conflict_review,
          무신호→보존(덮어쓰기 금지).

  PYTHONPATH=. python3 scripts/analysis/relation_lifecycle_backfill.py            # dry-run
  PYTHONPATH=. python3 scripts/analysis/relation_lifecycle_backfill.py --apply    # 적용
"""
import argparse, json, urllib.request, collections, os
from app.services.relation_crossvalidate import crossvalidate_relation, consolidate_peer_relation

ap = argparse.ArgumentParser()
ap.add_argument("--apply", action="store_true", help="DB write 적용(없으면 dry-run)")
args = ap.parse_args()

env = {}
for ln in open(".env.dev"):
    if "=" in ln and not ln.startswith("#"):
        k, v = ln.split("=", 1); env[k] = v.strip().strip('"')
U = env["SUPABASE_URL"]; K = env["SUPABASE_SERVICE_KEY"]
H = {"apikey": K, "Authorization": "Bearer " + K}


def ga(p):
    o, off = [], 0
    while True:
        u = U + "/rest/v1/" + p + ("&" if "?" in p else "?") + "limit=1000&offset=%d" % off
        rows = json.load(urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=60)); o += rows
        if len(rows) < 1000:
            break
        off += 1000
    return o


def patch(path, body):
    req = urllib.request.Request(
        U + "/rest/v1/" + path, data=json.dumps(body).encode(),
        headers={**H, "Content-Type": "application/json", "Prefer": "return=minimal"}, method="PATCH")
    return urllib.request.urlopen(req, timeout=20).status


sessions = ga("sessions?select=id,peer_id&peer_id=not.is.null")
sess_peer = {s["id"]: s["peer_id"] for s in sessions}
ss = ga("session_speakers?select=session_id,speaker_relation,speaker_role&speaker_role=eq.other")
rel_by_sess = {r["session_id"]: r["speaker_relation"] for r in ss}
segs = ga("session_segments?select=session_id,topic")
topics_by_sess = collections.defaultdict(set)
for s in segs:
    if s.get("topic"):
        topics_by_sess[s["session_id"]].add(s["topic"])
peers_now = {p["id"]: p.get("relationship") for p in ga("peers?select=id,relationship")}

peer_sessions = collections.defaultdict(list)
for sid, pid in sess_peer.items():
    peer_sessions[pid].append(sid)

demote_calls = []      # (session_id) 강등 대상
peer_updates = []      # (pid, relationship, conf, source)
peer_flags = []        # (pid) 충돌 검수
n_demote = n_pos = n_conflict = n_nosig = 0

for pid, sids in peer_sessions.items():
    per_call = []
    for sid in sids:
        rel = rel_by_sess.get(sid)
        if not rel:
            per_call.append(None); continue
        validated, reason = crossvalidate_relation(rel, topics_by_sess.get(sid))
        if validated is None:
            demote_calls.append(sid); n_demote += 1
        per_call.append(validated)
    prel, conf, src, reason = consolidate_peer_relation(per_call)
    if src in ("cross_call_consistent", "cross_call_dominant", "single_call"):
        peer_updates.append((pid, prel, conf, src)); n_pos += 1
    elif src == "conflict":
        peer_flags.append(pid); n_conflict += 1
    else:  # no_signal → 보존
        n_nosig += 1

print("=== 관계 lifecycle %s ===" % ("APPLY" if args.apply else "DRY-RUN"))
print("  per-call 교사 데모션:", n_demote, "통화")
print("  peer 관계 갱신(positive):", n_pos, "| 충돌→검수플래그:", n_conflict, "| 무신호 보존:", n_nosig)

if args.apply:
    ok = err = 0
    for sid in demote_calls:
        try:
            patch("session_speakers?session_id=eq.%s&speaker_role=eq.other" % sid, {"speaker_relation": None}); ok += 1
        except Exception as e:
            err += 1
    for pid, prel, conf, src in peer_updates:
        try:
            patch("peers?id=eq.%s" % pid, {"relationship": prel, "rel_confidence": conf, "rel_source": "relation_lifecycle"}); ok += 1
        except Exception:
            err += 1
    for pid in peer_flags:
        try:
            patch("peers?id=eq.%s" % pid, {"rel_source": "conflict_review"}); ok += 1
        except Exception:
            err += 1
    print("  적용: write 성공 %d | 실패 %d" % (ok, err))
else:
    print("  (dry-run — write 0. --apply 로 적용)")
