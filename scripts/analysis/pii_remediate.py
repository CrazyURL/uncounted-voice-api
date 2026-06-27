# -*- coding: utf-8 -*-
"""발견된 PII 누출 remediate — 마스킹본 transcript_text의 정형 PII를 토큰으로 치환."""
import os, re, urllib.request, json, sys
REPO="/home/gdash/project/Uncounted-root/uncounted-voice-api"
e={}
for l in open(os.path.join(REPO,".env.dev"),encoding='utf-8'):
    l=l.strip()
    if '=' in l and not l.startswith('#'):
        k,v=l.split('=',1); e[k]=v.strip().strip('"').strip("'")
U=e['SUPABASE_URL']; K=e['SUPABASE_SERVICE_KEY']; H={'apikey':K,'Authorization':'Bearer '+K}
APPLY="--apply" in sys.argv
# (정규식, 토큰) — 고신뢰만
SUBS=[
 (re.compile(r'\d{6}[-\s]?[1-4]\d{6}'),'[PII_주민번호]'),
 (re.compile(r'01[016789][-\s.]?\d{3,4}[-\s.]?\d{4}'),'[PII_전화번호]'),
 (re.compile(r'0(2|3[1-3]|4[1-4]|5[1-5]|6[1-4])[-\s.]?\d{3,4}[-\s.]?\d{4}'),'[PII_전화번호]'),
 (re.compile(r'[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}'),'[PII_이메일]'),
 (re.compile(r'\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}'),'[PII_카드번호]'),
 (re.compile(r'문소라|김기웅|강기원|노종윤|문식환'),'[PII_이름]'),
]
def patch(uid,text):
    d=json.dumps({"transcript_text":text}).encode()
    r=urllib.request.Request(U+f"/rest/v1/utterances?id=eq.{uid}",data=d,method='PATCH',
        headers={**H,'Content-Type':'application/json','Prefer':'return=minimal'})
    return urllib.request.urlopen(r,timeout=30).getcode()
changed=0; off=0; rows_changed=[]
while True:
    q=f"utterances?select=id,transcript_text&transcript_text=not.is.null&order=id&limit=1000&offset={off}"
    rows=json.loads(urllib.request.urlopen(urllib.request.Request(U+'/rest/v1/'+q,headers=H),timeout=60).read())
    for r in rows:
        t=r.get('transcript_text') or ''; nt=t
        for pat,tok in SUBS: nt=pat.sub(tok,nt)
        if nt!=t:
            rows_changed.append((r['id'],t,nt)); changed+=1
            if APPLY: patch(r['id'],nt)
    if len(rows)<1000: break
    off+=1000
print(f"remediate 대상 발화: {changed}건")
for uid,t,nt in rows_changed[:6]:
    print(f"  {uid[:8]}: ...{re.sub(chr(92)+'d','#',t[:60])}... → 마스킹")
print("APPLY" if APPLY else "[DRY] --apply 로 적용")
