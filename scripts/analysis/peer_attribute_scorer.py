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

# gender canonical = 앱 정본 한국어(users_profile·peers 089 union CHECK). 영문 파생값을 한국어로 정규화해 emit.
GEN_EN2KO = {'male': '남성', 'female': '여성', 'non_binary': '논바이너리'}


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
        "id": pid, "category": cat, "relationship": rel, "gender": GEN_EN2KO.get(gen), "gender_source": gsrc,
        "confidence": conf, "state": state, "spec_conflict": spec_conflict,
        "layers": {"meta": mc, "hwagye": hw, "hochik": ho},
        "cur_rel": cur_rel, "cur_src": cur_src,
    }

HUMAN_SRC = ('ground_truth_consented', 'human', 'human_locked', 'manual_correction')

# ── 관계 교차검증: peer_stated 자가신고(ground truth·값불변) vs 호칭·화계 ──
# 스펙: relationship 값은 override_locked 라 절대 안 덮음. attr_state 만 갱신.
#   일치=peer_stated_verified / 신호無=peer_stated_unverified(초기값) / 충돌=peer_stated_flagged.
# 호칭(녹음에 박혀 조작불가)=최강축, 화계(가족반말 vs 업무존대)=보조축. 약하면 unverified(틀린flag<놓침).
FAMILY_RELS = {'부모', '배우자', '형제자매', '자녀', '친구'}
WORK_RELS = {'직장상사', '직장동료', '거래처', '교사', '고객'}


def cross_check_verdict(rel, hw, hw_conf, p2o, o2p):
    sib = p2o['오빠'] + p2o['누나'] + p2o['형'] + p2o['언니']   # peer→owner 형제 호칭
    par = o2p['아빠'] + o2p['엄마'] + o2p['아버지'] + o2p['어머니']  # owner→peer 부모 호칭
    # 최강축 = 구체 친족 호칭(조작불가). 가족관계는 화계로 flag 금지:
    #   부모는 자녀→부모 존대가 정상, 친구는 존/반말 혼재라 화계 대조가 오탐.
    if rel == '형제자매':
        if sib >= 2: return 'peer_stated_verified'
        if par >= 2: return 'peer_stated_flagged'   # 부모 호칭인데 형제자매 신고=충돌
        return 'peer_stated_unverified'
    if rel == '부모':
        if par >= 2: return 'peer_stated_verified'
        if sib >= 2: return 'peer_stated_flagged'
        return 'peer_stated_unverified'             # 존대여도 flag 안 함(부모 존대=정상)
    # 화계축 = 업무관계만 신뢰(존대 규범 뚜렷). 상호반말(가족신호)이면 공식관계와 충돌.
    if rel in WORK_RELS and hw:
        if hw == '업무' and hw_conf >= 0.6: return 'peer_stated_verified'
        if hw == '가족' and hw_conf >= 0.7: return 'peer_stated_flagged'
    return 'peer_stated_unverified'   # 친구/배우자/자녀/기타·신호약함 → 초기값 유지(틀린flag<놓침)


def cross_check(apply):
    peers = get("peers?select=id,display_name,relationship,rel_source,attr_state&rel_source=eq.peer_stated")
    print(f"=== 관계 교차검증(rel_source=peer_stated): {len(peers)}건 ===")
    if not peers:
        print("peer_stated 자가신고 데이터 없음 — 동의페이지 관계 자가신고 적재 후 실행(현재 정상).")
        return 0
    out = Counter(); wrote = 0
    for p in peers:
        rel = p.get('relationship')
        if not rel:
            continue
        hw, hw_conf, p2o, o2p = hwagye_hochik(p['id'])
        v = cross_check_verdict(rel, hw, hw_conf, p2o, o2p)
        out[v] += 1
        sib = p2o['오빠'] + p2o['누나'] + p2o['형'] + p2o['언니']
        par = o2p['아빠'] + o2p['엄마'] + o2p['아버지'] + o2p['어머니']
        print(f"  {p['id'][:8]} rel={rel} hw={hw}({hw_conf}) sib={sib} par={par} → {v}")
        if apply and v != p.get('attr_state'):
            patch(p['id'], {'attr_state': v})   # ★attr_state 만. relationship/override_locked/rel_confidence 불변.
            wrote += 1
    print("판정:", dict(out))
    print(f"attr_state 갱신 {wrote}건 (값·잠금 불변)" if apply else "[DRY] --apply 로 attr_state 갱신")
    return 0


def validate_xcheck():
    # 로직 검증(읽기전용): 기존 라벨 peer(관계 있는 것)를 자가신고로 가정해 판정 확인.
    peers = get("peers?select=id,relationship,rel_source&relationship=not.is.null&limit=200")
    peers = [p for p in peers if p.get('relationship') not in (None, 'UNKNOWN')]
    print(f"=== [검증·읽기전용] 기존 라벨 peer {len(peers)}건에 판정 로직 적용 ===")
    ok = Counter()
    for p in peers:
        hw, hw_conf, p2o, o2p = hwagye_hochik(p['id'])
        v = cross_check_verdict(p['relationship'], hw, hw_conf, p2o, o2p)
        ok[v] += 1
        sib = p2o['오빠'] + p2o['누나'] + p2o['형'] + p2o['언니']; par = o2p['아빠'] + o2p['엄마'] + o2p['아버지'] + o2p['어머니']
        print(f"  {p['id'][:8]} rel={p['relationship']}({p.get('rel_source')}) hw={hw} sib={sib} par={par} → {v}")
    print("판정 분포:", dict(ok))
    return 0

def main():
    if "--cross-check" in sys.argv:
        return cross_check("--apply" in sys.argv)
    if "--validate-xcheck" in sys.argv:
        return validate_xcheck()
    apply = "--apply" in sys.argv
    peers = get("peers?select=id,relationship,rel_source,override_locked&order=call_count.desc")
    results = []
    for p in peers:
        t = get(f"sessions?select=title&peer_id=eq.{p['id']}&title=not.is.null&limit=1")
        r = score_peer(p['id'], t[0]['title'] if t else None, p.get('relationship'), p.get('rel_source'))
        r['override_locked'] = p.get('override_locked')
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
        print("\n[DRY-RUN] write 없음. --apply 로 087 컬럼 적재.")
        return 0
    # 087 컬럼 적재: override_locked=false 만. attr_category/attr_state/gender/gender_source.
    # relationship(legacy)·voice/speech_age 는 미변경(legacy 보존·age 미산출). 가족 깨끗단일신호만 relationship도 갱신.
    wrote = 0; skip_lock = 0
    for r in results:
        if r.get('override_locked'):
            skip_lock += 1; continue
        payload = {"attr_category": r['category'], "attr_state": r['state']}
        if r['gender'] in ('남성', '여성', '논바이너리'):
            payload['gender'] = r['gender']; payload['gender_source'] = r['gender_source']
        # 가족 + 깨끗한 단일 호칭신호일 때만 relationship 갱신(legacy 비-human은 덮어도 무방)
        if r['state'] == 'PEER_STRONG' and r['category'] == '가족' and r['relationship'] and r['cur_src'] not in HUMAN_SRC:
            payload['relationship'] = r['relationship']; payload['rel_confidence'] = r['confidence']; payload['rel_source'] = 'multilayer_v1'
        c = patch(r['id'], payload)
        wrote += c in (200, 204)
    print(f"\n087 적재: {wrote}건 (override_locked skip {skip_lock}). attr_category/attr_state/gender 채움. relationship(legacy)·age 미변경.")
    # 검증
    from collections import Counter as _C
    chk = get("peers?select=attr_category,attr_state&limit=200")
    print("적재 후 attr_category:", dict(_C(x.get('attr_category') for x in chk)))
    print("적재 후 attr_state:", dict(_C(x.get('attr_state') for x in chk)))
    return 0

if __name__ == "__main__":
    sys.exit(main())
