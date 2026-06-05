# -*- coding: utf-8 -*-
"""평가 하베스트 2단계: 주석 부트스트랩 — 검수자가 빈손이 아니라 *교정만* 하도록 미리 채움.

각 샘플 세션의 DB 출력(STT/화자/PII후보)을 gold 템플릿으로 생성.
검수자는 text_gold(STT 오류 교정)·speaker_gold(화자 확인)·pii_missed(놓친 PII 추가)만 수정 후
reviewed=true. → frozen gold.
출력: eval_harness/gold/<seq>.json

사용: PYTHONPATH=. python3 eval_harness/bootstrap_annotations.py
"""
import os, json, urllib.request, re
from app.ner_guard.detector import detect_name_hits

env = {}
for ln in open(os.path.join(os.path.dirname(__file__), "../.env.dev")):
    if "=" in ln and not ln.startswith("#"):
        k, v = ln.split("=", 1); env[k] = v.strip().strip('"')
U = env["SUPABASE_URL"]; K = env["SUPABASE_SERVICE_KEY"]
H = {"apikey": K, "Authorization": "Bearer " + K}
def GET(p): return json.load(urllib.request.urlopen(urllib.request.Request(U + "/rest/v1/" + p, headers=H), timeout=60))

base = os.path.dirname(__file__)
manifest = json.load(open(os.path.join(base, "sample_manifest.json"), encoding="utf-8"))
gold_dir = os.path.join(base, "gold"); os.makedirs(gold_dir, exist_ok=True)

_PHONE = re.compile(r"01[016-9][-\s]?\d{3,4}[-\s]?\d{4}|\d{6}[-\s]?[1-4]\d{6}")

made = 0
for s in manifest["sessions"]:
    sid = s["id"]
    us = GET(f"utterances?session_id=eq.{sid}&select=sequence_order,start_sec,end_sec,speaker_id,transcript_text,storage_path,pii_intervals&order=sequence_order&limit=1000")
    utts = []
    for u in us:
        txt = u.get("transcript_text") or ""
        # PII 후보(검수자 확인용): 이미 마스킹된 [PII_*] + 놓친 의심(사전 풀네임·원시번호)
        masked = len(re.findall(r"\[PII_", txt))
        suspect = [h.text for h in detect_name_hits(txt) if h.kind == "full"] + _PHONE.findall(txt)
        utts.append({
            "seq": u["sequence_order"],
            "start": u.get("start_sec"), "end": u.get("end_sec"),
            "audio_ref": u.get("storage_path"),
            "speaker_pipeline": u.get("speaker_id"),
            "stt_pipeline": txt,
            # ↓ 검수자 교정 대상
            "text_gold": txt,                 # STT 오류만 교정(PII 토큰 [PII_*]는 유지)
            "speaker_gold": u.get("speaker_id"),  # 화자 맞으면 그대로, 틀리면 수정
            "pii_masked_count": masked,       # 파이프라인이 가린 수(참고)
            "pii_missed": suspect,            # ★검수자: 파이프라인이 *놓친* PII만 여기 추가(자동후보 미리채움)
            "reviewed": False,
        })
    gold = {
        "session_id": sid, "session_seq": s["session_seq"], "duration": s["duration"],
        "_instructions": "text_gold=STT오류 교정([PII_*]유지) / speaker_gold=화자확인 / pii_missed=놓친PII만 / reviewed=true",
        "utterances": utts,
    }
    json.dump(gold, open(os.path.join(gold_dir, f"{s['session_seq']}.json"), "w"), ensure_ascii=False, indent=2)
    made += 1
print(f"부트스트랩 {made} 세션 → eval_harness/gold/*.json (검수자 교정 대기)")
