# -*- coding: utf-8 -*-
"""PII 마스킹 자동 감사 — 마스킹본(transcript_text)에 정형 PII가 살아있으면 누출.
사람검수(주관적 이름/맥락)는 제외, 정규식으로 잡히는 고위험 정형 PII 전수 스캔."""
import os, re, urllib.request, json
from collections import Counter
REPO="/home/gdash/project/Uncounted-root/uncounted-voice-api"
e={}
for l in open(os.path.join(REPO,".env.dev"),encoding='utf-8'):
    l=l.strip()
    if '=' in l and not l.startswith('#'):
        k,v=l.split('=',1); e[k]=v.strip().strip('"').strip("'")
U=e['SUPABASE_URL']; K=e['SUPABASE_SERVICE_KEY']; H={'apikey':K,'Authorization':'Bearer '+K}

PII = {
 "휴대폰": re.compile(r'01[016789][-\s.]?\d{3,4}[-\s.]?\d{4}'),
 "주민번호": re.compile(r'\d{6}[-\s]?[1-4]\d{6}'),
 "카드번호": re.compile(r'\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4}'),
 "이메일": re.compile(r'[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}'),
 "일반전화": re.compile(r'0(2|3[1-3]|4[1-4]|5[1-5]|6[1-4])[-\s.]?\d{3,4}[-\s.]?\d{4}'),
 "긴숫자(계좌의심)": re.compile(r'\b\d{11,16}\b'),
}
# 알려진 실명(호칭 아님) 누출 체크
NAMES=["문소라","김기웅","강기원","노종윤"]

leak=Counter(); samples={}; tok=Counter(); name_leak=Counter()
total=0; off=0
while True:
    q=f"utterances?select=transcript_text&transcript_text=not.is.null&order=id&limit=1000&offset={off}"
    rows=json.loads(urllib.request.urlopen(urllib.request.Request(U+'/rest/v1/'+q,headers=H),timeout=60).read())
    for r in rows:
        t=r.get('transcript_text') or ''
        total+=1
        for tk in re.findall(r'\[PII_[^\]]*\]', t): tok[tk]+=1
        for name,pat in PII.items():
            m=pat.findall(t)
            if m:
                leak[name]+=len(m)
                if name not in samples: samples[name]=re.sub(r'\d',"#",m[0])  # 숫자 가린 샘플
        for nm in NAMES:
            if nm in t: name_leak[nm]+=1
    if len(rows)<1000: break
    off+=1000
print(f"=== 전수 발화 {total}건 PII 감사 ===")
print("정형 PII 누출(마스킹 실패):", dict(leak) if leak else "0 (없음)")
if samples: print("  누출 샘플(숫자가림):", samples)
print("알려진 실명 누출:", dict(name_leak) if name_leak else "0 (전부 마스킹됨)")
print("마스킹 토큰 분포(상위):", dict(tok.most_common(10)))
