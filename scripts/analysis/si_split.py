# -*- coding: utf-8 -*-
"""Speaker-Independent train/val/test split (빅테크 데이터누수 방지 절대원칙).

원칙: 같은 화자(identity)가 두 split 에 동시 존재 금지. union-find 로 화자공유 세션을
한 클러스터로 묶고, 클러스터 단위로 70/15/15 배분(클러스터는 절대 쪼개지 않음).
화자 identity = {owner=user_id, peer=session.pid(있으면)}.

⚠️ 현 데이터 한계: owner 1명·pid null → 모든 세션이 1 클러스터 → SI-split 불가(degenerate).
   실유저 다수 + peer 링크(pid) 채워지면 자동 동작(프로덕션-ready). 이 스크립트가 그 알고리즘.
출력: scripts/analysis/si_split_manifest.json

사용: PYTHONPATH=. python3 scripts/analysis/si_split.py
"""
import os, json, urllib.request
env = {}
for ln in open(os.path.join(os.path.dirname(__file__), "../../.env.dev")):
    if "=" in ln and not ln.startswith("#"):
        k, v = ln.split("=", 1); env[k] = v.strip().strip('"')
U = env["SUPABASE_URL"]; K = env["SUPABASE_SERVICE_KEY"]
H = {"apikey": K, "Authorization": "Bearer " + K}
def GET(p): return json.load(urllib.request.urlopen(urllib.request.Request(U + "/rest/v1/" + p, headers=H), timeout=60))

sessions = []
off = 0
while True:
    r = GET(f"sessions?select=id,user_id,pid&gpu_upload_status=eq.done&consent_status=eq.both_agreed&raw_audio_url=not.is.null&order=id.asc&limit=1000&offset={off}")
    if not r: break
    sessions += r
    if len(r) < 1000: break
    off += 1000

# union-find: 화자 identity 노드. 세션은 owner·peer 노드를 잇는다.
parent = {}
def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(a, b):
    parent[find(a)] = find(b)

for s in sessions:
    ids = []
    if s.get("user_id"): ids.append("owner:" + s["user_id"])
    if s.get("pid"): ids.append("peer:" + str(s["pid"]))
    if not ids:  # 화자 미상 → 세션 고유노드
        ids = ["sess:" + s["id"]]
    for i in ids[1:]:
        union(ids[0], i)
    s["_root"] = find(ids[0])

# 클러스터 = 같은 root
from collections import defaultdict
clusters = defaultdict(list)
for s in sessions:
    clusters[s["_root"]].append(s["id"])
cl = sorted(clusters.values(), key=len, reverse=True)

# 클러스터 단위 70/15/15 (greedy: 큰 것부터 가장 모자란 split 에)
split = {"train": [], "val": [], "test": []}
target = {"train": 0.70, "val": 0.15, "test": 0.15}
total = len(sessions)
for c in cl:
    # 현재 비율 대비 가장 부족한 split 에 배정
    need = {k: target[k] * total - len(split[k]) for k in split}
    pick = max(need, key=need.get)
    split[pick] += c

degenerate = len(cl) < 3
out = {
    "_note": "Speaker-Independent: 같은 화자가 두 split 에 없음. 클러스터(화자공유) 단위 분할.",
    "clusters": len(cl),
    "largest_cluster": len(cl[0]) if cl else 0,
    "degenerate": degenerate,
    "degenerate_reason": "클러스터<3 (단일 owner·peer미링크 → 데이터누수 방지 분할 불가. 실유저 다수 필요)" if degenerate else None,
    "split_counts": {k: len(v) for k, v in split.items()},
    "split": {k: v for k, v in split.items()},
}
p = os.path.join(os.path.dirname(__file__), "si_split_manifest.json")
json.dump(out, open(p, "w"), ensure_ascii=False)
print(f"세션 {total} / 화자 클러스터 {len(cl)} (최대 {out['largest_cluster']})")
print(f"split: {out['split_counts']}")
if degenerate:
    print(f"⚠️ DEGENERATE — {out['degenerate_reason']}")
    print("   알고리즘은 프로덕션-ready. 실유저 다수+pid 링크 채워지면 자동 SI-split.")
else:
    print(f"✅ SI-split OK → {p}")
