# -*- coding: utf-8 -*-
"""Demographics & Diversity 요약 (ISO/IEC TR 24028 bias 메타) — 빅테크 납품 필수 동봉.

session_speakers(역할/성별/연령)를 데이터셋 전체로 롤업. 역할(self=owner/other=peer)별로도
분리해 *구조*를 투명하게(단일 owner=self 반복 vs 다양한 peer) 노출.
출력: scripts/analysis/dataset_demographics.json (납품 dataset_summary 에 병합 가능).

사용: PYTHONPATH=. python3 scripts/analysis/demographics_report.py
"""
import os, json, urllib.request, collections
env = {}
for ln in open(os.path.join(os.path.dirname(__file__), "../../.env.dev")):
    if "=" in ln and not ln.startswith("#"):
        k, v = ln.split("=", 1); env[k] = v.strip().strip('"')
U = env["SUPABASE_URL"]; K = env["SUPABASE_SERVICE_KEY"]
H = {"apikey": K, "Authorization": "Bearer " + K}
def GET(p): return json.load(urllib.request.urlopen(urllib.request.Request(U + "/rest/v1/" + p, headers=H), timeout=60))

# 납품대상(both_agreed) 세션의 화자만
sids = set()
off = 0
while True:
    r = GET(f"sessions?select=id&gpu_upload_status=eq.done&consent_status=eq.both_agreed&raw_audio_url=not.is.null&order=id.asc&limit=1000&offset={off}")
    if not r: break
    sids.update(x["id"] for x in r)
    if len(r) < 1000: break
    off += 1000

rows = []
off = 0
while True:
    r = GET(f"session_speakers?select=session_id,speaker_role,speaker_gender,speaker_voice_age_range&order=id.asc&limit=1000&offset={off}")
    if not r: break
    rows += [x for x in r if x.get("session_id") in sids]
    if len(r) < 1000: break
    off += 1000

def dist(items):
    c = collections.Counter(items); n = sum(c.values())
    return {k: {"count": v, "pct": round(100 * v / n, 1)} for k, v in c.most_common() if k is not None}, n

def pct_only(items):
    c = collections.Counter(x for x in items if x is not None); n = sum(c.values())
    return {k: round(100 * v / n, 1) for k, v in c.most_common()}

report = {
    "_note": "ISO/IEC TR 24028 다양성/편향 메타. 역할별 분리로 구조 투명화(self=수집자 반복 / other=상대 다양).",
    "speaker_records": len(rows),
    "overall": {
        "gender_pct": pct_only(x.get("speaker_gender") for x in rows),
        "age_pct": pct_only(x.get("speaker_voice_age_range") for x in rows),
    },
    "by_role": {},
}
for role in ("self", "other"):
    rr = [x for x in rows if x.get("speaker_role") == role]
    report["by_role"][role] = {
        "n": len(rr),
        "gender_pct": pct_only(x.get("speaker_gender") for x in rr),
        "age_pct": pct_only(x.get("speaker_voice_age_range") for x in rr),
    }

out = os.path.join(os.path.dirname(__file__), "dataset_demographics.json")
json.dump(report, open(out, "w"), ensure_ascii=False, indent=2)
print(f"화자 레코드 {len(rows)} → {out}")
print("전체 성별:", report["overall"]["gender_pct"])
print("전체 연령:", report["overall"]["age_pct"])
print("self(수집자):", report["by_role"]["self"]["gender_pct"], report["by_role"]["self"]["age_pct"])
print("other(상대):", report["by_role"]["other"]["gender_pct"], report["by_role"]["other"]["age_pct"])
