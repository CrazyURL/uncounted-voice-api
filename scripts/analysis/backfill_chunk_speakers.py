# -*- coding: utf-8 -*-
"""chunk 갭 세션 session_speakers 백필 (재처리 없이). 저장 발화 duration→self/other.
self gender=프로필, other=null(자가신고로). source='chunk_backfill_heuristic'."""
import os, urllib.request, json, urllib.parse, sys
from collections import defaultdict
REPO="/home/gdash/project/Uncounted-root/uncounted-voice-api"
e={}
for l in open(os.path.join(REPO,".env.dev"),encoding='utf-8'):
    l=l.strip()
    if '=' in l and not l.startswith('#'):
        k,v=l.split('=',1); e[k]=v.strip().strip('"').strip("'")
U=e['SUPABASE_URL']; K=e['SUPABASE_SERVICE_KEY']; H={'apikey':K,'Authorization':'Bearer '+K}
APPLY="--apply" in sys.argv
def getall(q):
    out=[]; off=0
    while True:
        p=json.loads(urllib.request.urlopen(urllib.request.Request(U+'/rest/v1/'+q+f"&limit=1000&offset={off}",headers=H),timeout=40).read())
        out+=p
        if len(p)<1000: break
        off+=1000
    return out
def get(q): return json.loads(urllib.request.urlopen(urllib.request.Request(U+'/rest/v1/'+q,headers=H),timeout=40).read())
def post(rows):
    d=json.dumps(rows).encode()
    r=urllib.request.Request(U+"/rest/v1/session_speakers",data=d,method='POST',
        headers={**H,'Content-Type':'application/json','Prefer':'return=minimal'})
    return urllib.request.urlopen(r,timeout=30).getcode()
GMAP={"남성":"male","여성":"female","논바이너리":"non_binary"}
prof={p['user_id']:GMAP.get(p.get('gender')) for p in get("users_profile?select=user_id,gender")}
spk=set(x['session_id'] for x in getall("session_speakers?select=session_id"))
gap=[s for s in getall("sessions?select=id,user_id&utterance_count=gt.0") if s['id'] not in spk]
print(f"갭 세션(speaker행 없음): {len(gap)}")
made=0; rows_all=[]
for s in gap:
    ut=getall(f"utterances?select=speaker_id,start_ms,end_ms&session_id=eq.{s['id']}")
    dur=defaultdict(float)
    for u in ut:
        lbl=u.get('speaker_id')
        if lbl: dur[lbl]+= max(0,(u.get('end_ms') or 0)-(u.get('start_ms') or 0))
    if not dur: continue
    self_lbl=max(dur,key=dur.get)
    sg=prof.get(s['user_id'])
    for lbl in dur:
        rows_all.append({"session_id":s['id'],"speaker_label":lbl,
            "speaker_role":"self" if lbl==self_lbl else "other",
            "speaker_role_source":"chunk_backfill_heuristic",
            "speaker_gender": sg if lbl==self_lbl else None,
            "speaker_voice_age_range":None,"speaker_speech_age_range":None,"speaker_relation":None})
    made+=1
print(f"생성할 session_speakers 행: {len(rows_all)} ({made}세션, 세션당 평균 {round(len(rows_all)/max(made,1),1)}화자)")
print("샘플:", rows_all[:2])
if not APPLY:
    print("[DRY] --apply 로 적재"); sys.exit(0)
ok=0
for i in range(0,len(rows_all),50):
    c=post(rows_all[i:i+50]); ok+= (c in(200,201,204))
print(f"적재 완료 배치ok={ok}, 총행={len(rows_all)}")
