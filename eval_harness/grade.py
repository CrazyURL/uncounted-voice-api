# -*- coding: utf-8 -*-
"""평가 하베스트 3단계: grader — 고정 gold에 현재 파이프라인 출력을 자동 채점.

reviewed=true 인 gold 만 채점. 메트릭:
  - WER (jiwer): gold text_gold vs 현재 DB transcript_text (PII 정규화).
  - DER (pyannote.metrics): gold speaker_gold vs 파이프라인 speaker (발화 시간구간).
  - PII recall: 가린수 / (가린수 + 놓친수[pii_missed]).
출력: 점수표(세션별 + 종합).

사용: PYTHONPATH=. python3 eval_harness/grade.py
"""
import os, json, glob, re, urllib.request
import numpy as np
import jiwer
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

env = {}
for ln in open(os.path.join(os.path.dirname(__file__), "../.env.dev")):
    if "=" in ln and not ln.startswith("#"):
        k, v = ln.split("=", 1); env[k] = v.strip().strip('"')
U = env["SUPABASE_URL"]; K = env["SUPABASE_SERVICE_KEY"]
H = {"apikey": K, "Authorization": "Bearer " + K}
def GET(p): return json.load(urllib.request.urlopen(urllib.request.Request(U + "/rest/v1/" + p, headers=H), timeout=60))

def norm_text(s):
    s = re.sub(r"\[PII_[^\]]*\]", "<PII>", s or "")  # PII 토큰 공통화(STT 정확도 = 비-PII)
    s = re.sub(r"[^0-9A-Za-z가-힣<> ]", " ", s)
    return " ".join(s.split())

base = os.path.dirname(__file__)
golds = []
for f in sorted(glob.glob(os.path.join(base, "gold", "*.json"))):
    g = json.load(open(f, encoding="utf-8"))
    if any(u.get("reviewed") for u in g["utterances"]):
        golds.append(g)
if not golds:
    print("채점 대상 없음 (gold 에 reviewed=true 발화 없음 — 검수 후 실행)"); raise SystemExit

der_metric = DiarizationErrorRate()
print(f"{'세션':>8} {'발화':>5} {'WER':>7} {'DER':>7} {'PIIrecall':>9}")
agg = {"wer_n": 0, "wer_e": 0, "pii_m": 0, "pii_miss": 0, "ders": []}
for g in golds:
    sid = g["session_id"]
    us = GET(f"utterances?session_id=eq.{sid}&select=sequence_order,start_sec,end_sec,speaker_id,transcript_text&order=sequence_order")
    by_seq = {u["sequence_order"]: u for u in us}
    gold_txt = " ".join(norm_text(u["text_gold"]) for u in g["utterances"] if u.get("reviewed"))
    hyp_txt = " ".join(norm_text((by_seq.get(u["seq"]) or {}).get("transcript_text", "")) for u in g["utterances"] if u.get("reviewed"))
    # WER
    try:
        meas = jiwer.compute_measures(gold_txt, hyp_txt)
        wer = meas["wer"]; agg["wer_e"] += meas["substitutions"]+meas["deletions"]+meas["insertions"]; agg["wer_n"] += meas["hits"]+meas["substitutions"]+meas["deletions"]
    except Exception:
        wer = float("nan")
    # DER (발화 시간구간 기준)
    ref = Annotation(); hyp = Annotation()
    for u in g["utterances"]:
        if not u.get("reviewed"): continue
        st, en = u.get("start"), u.get("end")
        pu = by_seq.get(u["seq"])
        if st is None or en is None or en <= st: continue
        ref[Segment(float(st), float(en))] = str(u.get("speaker_gold"))
        if pu: hyp[Segment(float(st), float(en))] = str(pu.get("speaker_id"))
    try:
        der = der_metric(ref, hyp); agg["ders"].append(der)
    except Exception:
        der = float("nan")
    # PII recall
    m = sum(u.get("pii_masked_count", 0) for u in g["utterances"] if u.get("reviewed"))
    miss = sum(len(u.get("pii_missed", [])) for u in g["utterances"] if u.get("reviewed"))
    agg["pii_m"] += m; agg["pii_miss"] += miss
    rec = (m / (m + miss)) if (m + miss) else 1.0
    nrev = sum(1 for u in g["utterances"] if u.get("reviewed"))
    print(f"{g['session_seq']:>8} {nrev:>5} {wer*100:>6.1f}% {der*100:>6.1f}% {rec*100:>8.1f}%")

print("\n=== 종합 ===")
WER = agg["wer_e"]/max(1, agg["wer_n"])
DER = float(np.mean(agg["ders"])) if agg["ders"] else float("nan")
REC = agg["pii_m"]/max(1, agg["pii_m"]+agg["pii_miss"])
print(f"WER {WER*100:.1f}% | DER {DER*100:.1f}% | PII recall {REC*100:.1f}% ({agg['pii_m']}가림/{agg['pii_miss']}놓침)")
print("바: WER 텔레포니 10-20% / DER 5-15% / PII recall ≥98%")
