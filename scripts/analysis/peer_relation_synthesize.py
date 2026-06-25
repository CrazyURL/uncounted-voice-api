# -*- coding: utf-8 -*-
"""peer 단위 관계 종합확정 — 호칭우선 샘플 + 침투성 규칙 (2026-06-17).

PR-γ'(peer_relation_infer.py, longest-call)의 후속·교정판.
검증으로 밝혀진 사실:
  - 가족 호칭(오빠/아빠/언니…)은 마스킹 안 됨 → 전사문에 그대로 살아있음.
  - longest-call 샘플은 '업무 통화'만 잡아 가족 호칭을 놓침 → 형제자매 오판/누락.
  - 해결 = 호칭 빈도 높은 통화를 우선 샘플 + 침투성 규칙(공적관계는 사적 가족대화 불가).

검증 결과(2026-06-17): 문소라→형제자매(0.95) / 파파→가족·부모 / 고객→UNKNOWN(오탐0).

peer당 1개 정답을 확정해 peers.relationship 1로우에 새김(관계 단일정본).
세부(형제자매/부모) 불확실시 대분류 "가족"으로(틀린 세부 금지, 대표님 원칙).

사용:
  python peer_relation_synthesize.py                 # dry-run 전수(write 0)
  python peer_relation_synthesize.py --peer <id>     # 단건 dry-run
  python peer_relation_synthesize.py --apply         # conf>=GATE write
"""
from __future__ import annotations
import argparse, json, os, sys, urllib.request, urllib.parse
from collections import defaultdict, Counter

REPO = "/home/gdash/project/Uncounted-root/uncounted-voice-api"
GATE = 0.80                       # --apply 신뢰 게이트
SAMPLE_CALLS = 12                 # peer당 샘플 통화 수
PER_CALL_CHARS = 700
OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b-instruct-q4_K_M"
KIN = ["오빠","누나","형","언니","동생","엄마","아빠","어머니","아버지","엄빠",
       "조카","며느리","손주","사위","이모","고모","삼촌","외삼촌",
       "여보","자기야","남편","와이프","신랑","애기아빠","애기엄마"]

RULE = """반드시 한국어로만 답하라(중국어 절대 금지).
같은 상대방과의 여러 통화를 종합해 '최종 관계' 1개를 확정한다.
가족 호칭(오빠/누나/형/언니/동생/엄마/아빠 등)은 마스킹 안 돼 그대로 보인다. [PII_이름]은 실제 이름이다.

[침투성 원칙 — 핵심]
- 공적관계(고객/거래처/직장동료/직장상사)는 사적 가족 대화를 절대 못 가진다. 사적단서 0이면 비즈니스 라벨 유지.
- 사적관계(부모/형제자매/배우자/자녀/연인/친구)는 업무 대화도 가능.
- 단 1건이라도 공적관계 불가능한 사적단서가 명확하면, 나머지가 전부 업무여도 사적관계로 확정.

[가족 세부 판별 — 확실할 때만]
- 형제자매: 상대가 나를 오빠/누나/형/언니로 부르거나 내가 상대를 그렇게 부름. 같은 부모 공유. 한쪽 자녀=상대 조카.
- 부모: 상대가 나를 '아들/딸'로 부름. 며느리/손주 언급.
- 자녀: 내가 상대를 '아들/딸'로 부름.
- 배우자: 여보/자기/남편/와이프/신랑 호칭.
→ '엄마/딸' 단순 언급만으로 부모 단정 금지. 세부 불확실하면 그냥 "가족"(틀린 세부보다 맞는 대분류).

taxonomy 1개만: 부모/형제자매/배우자/자녀/연인/친구/가족/직장상사/직장동료/거래처/고객/UNKNOWN
JSON만 출력: {"relationship":"<라벨>","confidence":0~1,"is_specific":true/false,"decisive_evidence":"<호칭/단서>","reason":"<한국어1문장>"}"""


def _env() -> dict:
    e = {}
    for ln in open(os.path.join(REPO, ".env.dev"), encoding="utf-8"):
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1); e[k] = v.strip().strip('"').strip("'")
    return e


ENV = _env()
U = ENV["SUPABASE_URL"]; K = ENV["SUPABASE_SERVICE_KEY"]
H = {"apikey": K, "Authorization": "Bearer " + K}


def sb_get(q):
    return json.loads(urllib.request.urlopen(
        urllib.request.Request(U + "/rest/v1/" + q, headers=H), timeout=40).read())


def sb_patch(table, idval, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        U + f"/rest/v1/{table}?id=eq.{idval}", data=data, method="PATCH",
        headers={**H, "Content-Type": "application/json", "Prefer": "return=minimal"})
    return urllib.request.urlopen(req, timeout=30).getcode()


def peer_sample_calls(peer_id):
    """호칭 빈도 높은 통화 우선 SAMPLE_CALLS개 (없으면 길이순)."""
    sess = sb_get(f"sessions?select=id,duration&peer_id=eq.{peer_id}&limit=120")
    if not sess:
        return []
    ids = [s["id"] for s in sess]
    dur = {s["id"]: (s.get("duration") or 0) for s in sess}
    by_sess = defaultdict(list)
    for i in range(0, len(ids), 30):
        il = "(" + ",".join(f'"{x}"' for x in ids[i:i+30]) + ")"
        rows = sb_get(f"utterances?select=session_id,start_ms,transcript_text"
                      f"&session_id=in.{urllib.parse.quote(il)}"
                      f"&order=start_ms.asc&limit=4000")
        for r in rows:
            t = (r.get("transcript_text") or "").strip()
            if t:
                by_sess[r["session_id"]].append(t)
    scored = []
    for sid, parts in by_sess.items():
        txt = " ".join(parts)
        kc = sum(txt.count(k) for k in KIN)
        scored.append((kc, dur.get(sid, 0), txt[:PER_CALL_CHARS]))
    # 호칭 빈도 desc, 동률이면 길이 desc
    scored.sort(key=lambda x: (-x[0], -x[1]))
    return [t for _, _, t in scored[:SAMPLE_CALLS]]


def infer(calls):
    body = RULE + "\n\n" + "\n---\n".join(f"[통화{i+1}] {c}" for i, c in enumerate(calls))
    payload = json.dumps({"model": MODEL, "prompt": body, "stream": False,
                          "options": {"temperature": 0.1, "num_ctx": 12288}}).encode()
    raw = urllib.request.urlopen(urllib.request.Request(
        OLLAMA, data=payload, headers={"Content-Type": "application/json"}),
        timeout=300).read()
    resp = json.loads(raw).get("response", "")
    # ```json fence / 본문에서 첫 {...} 추출
    s = resp.find("{"); e = resp.rfind("}")
    if s == -1 or e == -1:
        return {"relationship": "UNKNOWN", "confidence": 0.0, "is_specific": False,
                "decisive_evidence": "parse_fail", "reason": resp[:80]}
    try:
        return json.loads(resp[s:e+1])
    except Exception:
        return {"relationship": "UNKNOWN", "confidence": 0.0, "is_specific": False,
                "decisive_evidence": "parse_fail", "reason": resp[s:s+80]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peer")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--out", default="/tmp/peer_relation_synth.json")
    args = ap.parse_args()

    if args.peer:
        peers = sb_get(f"peers?select=id,relationship,rel_source,call_count&id=eq.{args.peer}")
    else:
        peers = sb_get("peers?select=id,relationship,rel_source,call_count&order=call_count.desc")

    results = []
    for i, p in enumerate(peers):
        calls = peer_sample_calls(p["id"])
        if not calls:
            r = {"relationship": "UNKNOWN", "confidence": 0.0, "is_specific": False,
                 "decisive_evidence": "no_transcript", "reason": ""}
        else:
            r = infer(calls)
        prop = r.get("relationship", "UNKNOWN")
        conf = float(r.get("confidence", 0) or 0)
        cur = p.get("relationship")
        change = (prop != cur and prop != "UNKNOWN" and conf >= GATE)
        rec = {"id": p["id"], "cur": cur, "cur_src": p.get("rel_source"),
               "prop": prop, "conf": round(conf, 2),
               "is_specific": r.get("is_specific"), "calls": len(calls),
               "ev": str(r.get("decisive_evidence", ""))[:40], "change": change}
        results.append(rec)
        flag = "★CHG" if change else ("·gate" if conf < GATE and prop != "UNKNOWN" else "")
        print(f"[{i+1}/{len(peers)}] {p['id'][:8]} {cur}→{prop} conf={conf:.2f} "
              f"spec={r.get('is_specific')} calls={len(calls)} {flag} ({rec['ev']})")

        if args.apply and change:
            code = sb_patch("peers", p["id"], {
                "relationship": prop, "rel_confidence": round(conf, 2),
                "rel_source": "llm_permeability"})
            print(f"        APPLIED rc={code}")

    json.dump(results, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n=== 제안 분포: {dict(Counter(r['prop'] for r in results))}")
    print(f"=== 변경대상(conf>={GATE}, !=UNKNOWN, !=현재): {sum(1 for r in results if r['change'])}건")
    print(f"=== 저신뢰(<{GATE}, non-UNKNOWN): {sum(1 for r in results if r['conf']<GATE and r['prop']!='UNKNOWN')}건")
    print(f"saved: {args.out}  (apply={args.apply})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
