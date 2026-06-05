# -*- coding: utf-8 -*-
"""평가 하베스트 1단계: 대표 샘플 N통화 선정(층화 — 길이×품질).

frozen gold set 은 "전부 테스트"가 아니라 "대표 샘플"로 다양성을 담는다.
길이(≤200s NeMo-full / >200s anchor) × 품질등급(A/B/C) 층으로 비례 선정.
출력: eval_harness/sample_manifest.json (세션 id/seq/dur/grade).

사용: PYTHONPATH=. python3 eval_harness/select_sample.py [--n 30]
"""
import os, json, urllib.request, argparse
from collections import defaultdict
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=30); args = ap.parse_args()

env = {}
for ln in open(os.path.join(os.path.dirname(__file__), "../.env.dev")):
    if "=" in ln and not ln.startswith("#"):
        k, v = ln.split("=", 1); env[k] = v.strip().strip('"')
U = env["SUPABASE_URL"]; K = env["SUPABASE_SERVICE_KEY"]
H = {"apikey": K, "Authorization": "Bearer " + K}
def GET(p): return json.load(urllib.request.urlopen(urllib.request.Request(U + "/rest/v1/" + p, headers=H), timeout=60))

rows = []
off = 0
while True:
    r = GET(f"sessions?select=id,session_seq,duration&gpu_upload_status=eq.done&consent_status=eq.both_agreed&raw_audio_url=not.is.null&order=id.asc&limit=1000&offset={off}")
    if not r: break
    rows += [x for x in r if x.get("duration")]
    if len(r) < 1000: break
    off += 1000

# 층: (길이 bucket) × (대표 품질) — quality_tier 없으면 utterance grade 대신 길이만.
def lbucket(d): return "short(<=200)" if d <= 200 else "long(>200)"
strata = defaultdict(list)
for x in rows:
    strata[lbucket(x["duration"])].append(x)

# 각 층에서 길이 분위로 고르게 추출
pick = []
per = max(1, args.n // max(1, len(strata)))
import bisect
for st, items in strata.items():
    items.sort(key=lambda x: x["duration"])
    k = min(per, len(items))
    for i in range(k):
        idx = int((i + 0.5) * len(items) / k)
        pick.append(items[min(idx, len(items)-1)])
# 중복 제거
seen = set(); pick = [p for p in pick if not (p["id"] in seen or seen.add(p["id"]))][:args.n]

out = os.path.join(os.path.dirname(__file__), "sample_manifest.json")
json.dump({"n": len(pick), "sessions": pick}, open(out, "w"), ensure_ascii=False, indent=2)
print(f"선정 {len(pick)}통화 → {out}")
for st in strata: print(f"  {st}: 후보 {len(strata[st])}")
print("길이:", sorted(p["duration"] for p in pick))
