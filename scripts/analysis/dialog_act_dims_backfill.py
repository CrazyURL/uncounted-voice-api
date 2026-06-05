# -*- coding: utf-8 -*-
"""다차원 dialog_act 백필 (ISO 24617-2) — Ollama qwen2.5 LLM.

단일 dialog_act 를 communicative_function + dimension 다차원으로 보강.
⚠️ Ollama 가 GPU 사용 → 재처리 완료 후 실행(경합 회피). migration 20260605 선적용.

사용: PYTHONPATH=. python3 scripts/analysis/dialog_act_dims_backfill.py [--limit N] [--apply]
"""
import os, json, urllib.request, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--apply", action="store_true")
args = ap.parse_args()
env = {}
for ln in open(os.path.join(os.path.dirname(__file__), "../../.env.dev")):
    if "=" in ln and not ln.startswith("#"):
        k, v = ln.split("=", 1); env[k] = v.strip().strip('"')
U = env["SUPABASE_URL"]; K = env["SUPABASE_SERVICE_KEY"]
H = {"apikey": K, "Authorization": "Bearer " + K, "Content-Type": "application/json"}
def GET(p): return json.load(urllib.request.urlopen(urllib.request.Request(U + "/rest/v1/" + p, headers=H), timeout=60))
def PATCH(p, b): return urllib.request.urlopen(urllib.request.Request(U + "/rest/v1/" + p, data=json.dumps(b).encode(), method="PATCH", headers=H), timeout=20).status

OLLAMA = "http://localhost:11434/api/generate"
DIMS = "Task, AutoFeedback, AlloFeedback, TurnManagement, TimeManagement, OwnCommunicationManagement, Discourse, SocialObligation"
PROMPT = (
    "다음 한국어 통화 발화를 ISO 24617-2 화행으로 분류해 JSON만 출력.\n"
    f'dimension 은 [{DIMS}] 중 하나.\n'
    'communicative_function 예: Question, Answer, Inform, Request, Suggest, Agreement, Greeting, Thanking, Apology 등.\n'
    '형식: {{"communicative_function": "...", "dimension": "..."}}\n발화: "{text}"\nJSON:'
)
def classify(text):
    body = {"model": "qwen2.5:7b-instruct-q4_K_M", "prompt": PROMPT.format(text=text[:300]),
            "stream": False, "format": "json", "options": {"temperature": 0}}
    r = urllib.request.urlopen(urllib.request.Request(OLLAMA, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}), timeout=60)
    resp = json.load(r).get("response", "{}")
    d = json.loads(resp)
    return {"communicative_function": d.get("communicative_function"), "dimension": d.get("dimension")}

lim = args.limit if args.limit else 500
done = 0; off = 0
while True:
    rows = GET(f"utterances?select=id,transcript_text&transcript_text=not.is.null&dialog_act_dims=is.null&order=id.asc&limit={lim}" + (f"&offset={off}" if not args.limit else ""))
    if not rows: break
    for u in rows:
        t = (u.get("transcript_text") or "").strip()
        if not t: continue
        try:
            dims = classify(t)
            if args.apply: PATCH(f"utterances?id=eq.{u['id']}", {"dialog_act_dims": dims})
            done += 1
            if done <= 3: print(f"  예: {t[:30]} → {dims}")
        except Exception: pass
    if args.limit or len(rows) < lim: break
    off += lim
print(f"{'적용' if args.apply else 'dry-run'} {done}건")
