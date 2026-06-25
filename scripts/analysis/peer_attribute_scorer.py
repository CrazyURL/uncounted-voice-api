# -*- coding: utf-8 -*-
"""peer 속성 다층 증거 스코어러 (2026-06-25). 음향 폐기, 텍스트+메타만.

검증된 4층 (각 층 테스트 통과):
  ① 메타/연락처명  — 직급·회사·기관·기업번호→업무, 가족호칭→가족 (94% coverage, 화계와 100% 일치)
  ② 화계(존/반말)  — 상호반말→가족, 상호존대→업무 (가족 8~19% vs 업무 62~95%, 겹침0)
  ③ 호칭(역할귀속) — 상대→나 오빠/누나/형/언니 (DIRECT vocative만, 조사붙은 3인칭 제외)
  ④ conflict-stop — 층 불일치시 자동확정 금지→human (3인칭 호칭 오염 차단)

상태: PEER_STRONG(2층+합의→auto후보) / WEAK(1층) / CONFLICT(불일치→human) / UNKNOWN.
성별: 가족 호칭에서만 파생(gender_source=relation_derived). 업무=null(human-lock, 날조금지).
출력: JSON + 요약. --apply 는 안전정책으로 relationship 만 write(gender/age 컬럼은 dev PC 후).

사용: python peer_attribute_scorer.py [--apply]
"""
from __future__ import annotations
import os, re, json, sys, urllib.request, urllib.parse
from collections import Counter

REPO = "/home/gdash/project/Uncounted-root/uncounted-voice-api"
_env = {}
for _l in open(os.path.join(REPO, ".env.dev"), encoding="utf-8"):
    _l = _l.strip()
    if "=" in _l and not _l.startswith("#"):
        _k, _v = _l.split("=", 1); _env[_k] = _v.strip().strip('"').strip("'")
U = _env["SUPABASE_URL"]; K = _env["SUPABASE_SERVICE_KEY"]
H = {"apikey": K, "Authorization": "Bearer " + K}

def get(q):
    return json.loads(urllib.request.urlopen(urllib.request.Request(U + "/rest/v1/" + q, headers=H), timeout=40).read())
def patch(pid, payload):
    return urllib.request.urlopen(urllib.request.Request(
        U + f"/rest/v1/peers?id=eq.{pid}", data=json.dumps(payload).encode(), method="PATCH",
        headers={**H, "Content-Type": "application/json", "Prefer": "return=minimal"}), timeout=30).status

# ── 층 ①: 메타/연락처명 ──
TITLE_RE = re.compile(r'^통화\s?녹음\s+(.+)_\d{6}_\d{6}$')
BIZ_T = re.compile(r'(사장|대표|이사|부장|차장|과장|대리|주임|책임|팀장|실장|원장|센터장|점장|지점장|소장|매니저|기사|선임|수석|상무|전무|회장|국장|본부장|위원|박사|교수|선생)')
BIZ_O = re.compile(r'(\(주\)|㈜|주식회사|회사|센터|지점|영업|콜센터|고객센터|상담|은행|증권|보험|카드|캐피탈|병원|의원|약국|공단|공사|진흥원|협회|재단|학원|마트|백화점|텔레콤|통신|관리사무소|구청|시청|주민센터|우체국|택배|물류|SK|KT|유플러스|삼성|현대|국민|신한|우리|하나|농협|기업|카카오|쿠팡|배민)')
KIN = re.compile(r'(아빠|엄마|아버지|어머니|형|누나|오빠|언니|이모|고모|삼촌|할머니|할아버지|장모|장인|처제|매형|와이프|남편|여보|파파|마마)')
def _phone(r): return bool(re.match(r'^[\d+][\d+\-\s().]*$', r)) and len(re.sub(r'\D', '', r)) >= 7
def _corp(r): return re.sub(r'\D', '', r)[:4] in ('1577','1588','1599','1600','1644','1666','1670','1688','1899','1855')
def meta_layer(title):
    m = TITLE_RE.match((title or '').strip())
    if not m: return None
    cid = m.group(1).strip()
    if _phone(cid): return '업무' if _corp(cid) else None
    if KIN.search(cid): return '가족'
    if BIZ_T.search(cid) or BIZ_O.search(cid): return '업무'
    return None

# ── 층 ②③: 화계 + 호칭 (utterance 1회 fetch 공유) ──
JOND = re.compile(r'(요|니다|니까|세요|십시오|어요|아요|에요|예요|네요|데요|까요|을게요|습니다|드려요|드립니다)[\s.?!]*$')
BAN = re.compile(r'(잖아|거든|는데|냐|자|야|지|어|아|네|군|구나|걸|래|대|는다|ㄴ다|다)[\s.?!]*$')
def _lvl(t):
    t = (t or '').strip()
    if JOND.search(t): return '존대'
    if BAN.search(t): return '반말'
    return None
PART = r'(?!가|이|은|는|을|를|한테|에게|께|의|도|만|와|과|랑|보다|처럼|부터|까지|들|네)'
def _voc(text, term):
    return sum(1 for m in re.finditer(re.escape(term), text) if re.match(PART, text[m.end():m.end()+2]))
def hwagye_hochik(pid):
    sess = get(f"sessions?select=id&peer_id=eq.{pid}&utterance_count=gt.5&limit=12")
    sids = [s['id'] for s in sess]
    c = Counter(); p2o = Counter(); o2p = Counter()
    for i in range(0, len(sids), 20):
        if not sids[i:i+20]: break
        il = "(" + ",".join(f'"{x}"' for x in sids[i:i+20]) + ")"
        sp = get(f"session_speakers?select=session_id,speaker_label,speaker_role&session_id=in.{urllib.parse.quote(il)}&limit=400")
        role = {(r['session_id'], r['speaker_label']): r.get('speaker_role') for r in sp}
        ut = get(f"utterances?select=session_id,speaker_id,transcript_text&session_id=in.{urllib.parse.quote(il)}&limit=3000")
        for u in ut:
            r = role.get((u['session_id'], u.get('speaker_id'))); t = u.get('transcript_text') or ''
            lv = _lvl(t)
            if r in ('self', 'other') and lv: c[lv] += 1
            if r == 'other':
                for term in ('오빠','누나','형','언니'): p2o[term] += _voc(t, term)
            elif r == 'self':
                for term in ('아빠','엄마','아버지','어머니'): o2p[term] += _voc(t, term)
    tot = c['존대'] + c['반말']
    hw = None; hw_conf = 0.0
    if tot >= 10:
        jr = c['존대'] / tot
        if jr >= 0.55: hw, hw_conf = '업무', round(jr, 2)
        elif jr < 0.4: hw, hw_conf = '가족', round(1 - jr, 2)
    return hw, hw_conf, p2o, o2p

def score_peer(pid, title, cur_rel, cur_src):
    mc = meta_layer(title)
    hw, hw_conf, p2o, o2p = hwagye_hochik(pid)
    # 호칭 → 가족 카테고리 + 구체관계/성별
    # 형제신호(상대→나 오빠/누나/형/언니) vs 부모신호(나→상대 아빠/엄마). 공유부모 3인칭 노이즈 때문에
    # 구체관계는 한쪽 신호만 명확할 때만 확정. 둘 다 강하면 구체=CONFLICT→가족(카테고리)+human.
    rel = gen = gsrc = None; hoc = 0; ho = None; spec_conflict = False
    sib = {'오빠': ('female', p2o['오빠']), '누나': ('male', p2o['누나']),
           '언니': ('female', p2o['언니']), '형': ('male', p2o['형'])}
    sib_term = max(sib, key=lambda k: sib[k][1])
    sib_n = sib[sib_term][1]
    par_n = o2p['아빠'] + o2p['아버지'] + o2p['엄마'] + o2p['어머니']
    par_gen = 'male' if (o2p['아빠'] + o2p['아버지']) >= (o2p['엄마'] + o2p['어머니']) else 'female'
    if sib_n >= 2 or par_n >= 2:
        ho = '가족'
        if sib_n >= 2 and par_n >= 2:
            spec_conflict = True; hoc = max(sib_n, par_n)   # 구체 충돌 → 가족만, 구체 human
        elif sib_n >= 2:
            rel, gen, gsrc, hoc = '형제자매', sib[sib_term][0], 'relation_derived', sib_n
        else:
            rel, gen, gsrc, hoc = '부모', par_gen, 'relation_derived', par_n
    # 합의/상태
    votes = [x for x in (mc, hw, ho) if x]
    if not votes:
        state, cat = 'UNKNOWN', None
    elif len(set(votes)) > 1:
        state, cat = 'CONFLICT', None
    elif len(votes) >= 2:
        state, cat = 'PEER_STRONG', votes[0]
    else:
        state, cat = 'WEAK', votes[0]
    # 신뢰도
    if cat == '가족' and rel:
        conf = round(0.5 * (hw_conf or 0.8) + 0.5 * min(1.0, 0.6 + hoc * 0.04), 2)
    elif cat == '업무':
        conf = round(hw_conf or (0.7 if mc == '업무' else 0.0), 2)
    else:
        conf = 0.0
    return {
        "id": pid, "category": cat, "relationship": rel, "gender": gen, "gender_source": gsrc,
        "confidence": conf, "state": state, "spec_conflict": spec_conflict,
        "layers": {"meta": mc, "hwagye": hw, "hochik": ho},
        "cur_rel": cur_rel, "cur_src": cur_src,
    }

HUMAN_SRC = ('ground_truth_consented', 'human', 'human_locked', 'manual_correction')

def main():
    apply = "--apply" in sys.argv
    peers = get("peers?select=id,relationship,rel_source&order=call_count.desc")
    results = []
    for p in peers:
        t = get(f"sessions?select=title&peer_id=eq.{p['id']}&title=not.is.null&limit=1")
        r = score_peer(p['id'], t[0]['title'] if t else None, p.get('relationship'), p.get('rel_source'))
        results.append(r)
    json.dump(results, open("/tmp/peer_attr_scores.json", "w"), ensure_ascii=False, indent=1)
    print("=== 상태 분포 ===", dict(Counter(r['state'] for r in results)))
    print("=== 카테고리(확정) ===", dict(Counter(r['category'] for r in results if r['category'])))
    fam = [r for r in results if r['state'] == 'PEER_STRONG' and r['category'] == '가족' and r['relationship']]
    famc = [r for r in results if r['category'] == '가족' and r.get('spec_conflict')]
    print(f"=== 가족 구체관계 자동확정(깨끗한 단일신호) {len(fam)}건 ===")
    for r in fam: print(f"  {r['id'][:8]} {r['relationship']}/{r['gender']} conf={r['confidence']} (현재 {r['cur_rel']})")
    print(f"=== 가족 카테고리만 auto·구체는 human(신호충돌) {len(famc)}건 ===")
    for r in famc: print(f"  {r['id'][:8]} 가족(구체 human) (현재 {r['cur_rel']}/{r['cur_src']})")
    print(f"=== CONFLICT(human) {sum(1 for r in results if r['state']=='CONFLICT')} · WEAK {sum(1 for r in results if r['state']=='WEAK')} · UNKNOWN {sum(1 for r in results if r['state']=='UNKNOWN')}")

    if not apply:
        print("\n[DRY-RUN] write 없음. --apply 로 안전정책 적용. (gender/age 는 peers 컬럼 신설 후)")
        return 0
    # 안전정책: PEER_STRONG·가족·구체관계만, human-lock 보존, conflict/weak/unknown skip
    wrote = 0
    for r in fam:
        if r['cur_src'] in HUMAN_SRC:
            print(f"  skip {r['id'][:8]} (human-lock 보존)"); continue
        if r['relationship'] == r['cur_rel']:
            continue
        c = patch(r['id'], {"relationship": r['relationship'], "rel_confidence": r['confidence'], "rel_source": "multilayer_v1"})
        wrote += c in (200, 204)
        print(f"  write {r['id'][:8]} → {r['relationship']} rc={c}")
    print(f"\n적용: 가족 구체관계 {wrote}건 (업무/conflict/weak/human-lock 미적용). gender/age=peers컬럼 신설 후 적재.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
