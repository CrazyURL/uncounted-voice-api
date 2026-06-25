# -*- coding: utf-8 -*-
"""sessions.peer_id link-only 복구 (2026-06-17).

배경: sessions.peer_id 전멸(0/1317). peers 109·title 멀쩡 → 링크만 끊김.
복구키 S1 = ops-secrets/peer_hash_secret.env PEER_HASH_SECRET (fp a47993da, 6/14 109peer 생성 secret).
counterpartyKey.ts 알고리즘 복제: identityHash=HMAC_SHA256(secret,"{userId}|{kind}|{normalizedId}").

★안전: title→key→기존 peer.peer_identity_hash 매칭시에만 sessions.peer_id 세팅.
   매칭 없으면 SKIP (신규 peer 생성 절대 안 함 → 993 중복사고 원천차단).

사용:
  python peer_id_link_recovery.py            # dry-run (write 0)
  python peer_id_link_recovery.py --apply    # 매칭 세션 peer_id 세팅
"""
from __future__ import annotations
import argparse, json, re, hmac, hashlib, unicodedata, os, urllib.request, urllib.parse
from collections import Counter

REPO = "/home/gdash/project/Uncounted-root/uncounted-voice-api"
SECRET_FILE = "/home/gdash/.config/ops-secrets/peer_hash_secret.env"


def load_env(p):
    e = {}
    try:
        for ln in open(p, encoding="utf-8"):
            ln = ln.strip()
            if "=" in ln and not ln.startswith("#"):
                k, v = ln.split("=", 1); e[k] = v.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return e


ENV = load_env(os.path.join(REPO, ".env.dev"))
U = ENV["SUPABASE_URL"]; K = ENV["SUPABASE_SERVICE_KEY"]
H = {"apikey": K, "Authorization": "Bearer " + K}
S1 = load_env(SECRET_FILE).get("PEER_HASH_SECRET")
assert S1, "S1 PEER_HASH_SECRET not found"
S1_FP = hashlib.sha256(S1.encode()).hexdigest()[:8]


def sb_get(q):
    return json.loads(urllib.request.urlopen(
        urllib.request.Request(U + "/rest/v1/" + q, headers=H), timeout=60).read())


def sb_patch(idval, peer_id):
    data = json.dumps({"peer_id": peer_id}).encode()
    req = urllib.request.Request(
        U + f"/rest/v1/sessions?id=eq.{idval}", data=data, method="PATCH",
        headers={**H, "Content-Type": "application/json", "Prefer": "return=minimal"})
    return urllib.request.urlopen(req, timeout=30).getcode()


# ── counterpartyKey.ts 복제 ──
TITLE_RE = re.compile(r'^통화\s?녹음\s+(.+)_\d{6}_\d{6}$')
HON = re.compile(r'(선생님|님|씨)$')


def parse_title(t):
    if not t: return None
    m = TITLE_RE.match(t.strip())
    if not m: return None
    r = m.group(1).strip()
    return r or None


def looks_phone(r):
    if not re.match(r'^[\d+][\d+\-\s().]*$', r): return False
    return len(re.sub(r'\D', '', r)) >= 7


def normalize_phone(phone):
    if not phone: return ''
    n = re.sub(r'[^\d+]', '', phone)
    if n.startswith('+82'): n = '0' + n[3:]
    elif n.startswith('82') and len(n) >= 10: n = '0' + n[2:]
    if re.match(r'^1[45678]\d', n): return n
    if not n.startswith('0') and len(n) >= 9: n = '0' + n
    return n


def norm_name(r):
    s = unicodedata.normalize('NFC', r).strip()
    s = re.sub(r'\s+', ' ', s); s = HON.sub('', s).strip()
    return s


def derive_identity(user_id, title):
    raw = parse_title(title)
    if raw is None: return None
    if looks_phone(raw):
        kind, nid = 'title_phone', normalize_phone(raw)
    else:
        kind, nid = 'title_name', norm_name(raw)
    if not nid: return None
    h = hmac.new(S1.encode(), f"{user_id}|{kind}|{nid}".encode(), hashlib.sha256).hexdigest()
    return h, kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    print(f"S1 fp={S1_FP} (expect a47993da)")

    peers = sb_get("peers?select=id,peer_identity_hash,identity_kind")
    pmap = {p["peer_identity_hash"]: p["id"] for p in peers if p.get("peer_identity_hash")}
    print(f"peers={len(peers)}, identity_hash 보유={len(pmap)}")

    sessions = []
    off = 0
    while True:
        page = sb_get(f"sessions?select=id,title,user_id,peer_id&title=not.is.null"
                      f"&order=date.desc&limit=1000&offset={off}")
        sessions.extend(page)
        if len(page) < 1000:
            break
        off += 1000
    print(f"sessions(title 보유)={len(sessions)}")

    plan = []           # (session_id, peer_id)
    stats = Counter()
    kind_link = Counter()
    matched_peers = set()
    for s in sessions:
        d = derive_identity(s.get("user_id"), s.get("title"))
        if d is None:
            stats["unparseable"] += 1; continue
        ih, kind = d
        pid = pmap.get(ih)
        if pid:
            plan.append((s["id"], pid)); stats["would_link"] += 1
            kind_link[kind] += 1; matched_peers.add(pid)
        else:
            stats["no_match_skip"] += 1

    print(f"\n=== dry-run 결과 ===")
    print(f"  링크 예정(would_link): {stats['would_link']}  (kind: {dict(kind_link)})")
    print(f"  매칭없음 skip(신규생성 안함): {stats['no_match_skip']}")
    print(f"  title 파싱불가 skip: {stats['unparseable']}")
    print(f"  커버된 peer 수: {len(matched_peers)}/{len(pmap)}")
    print(f"  ★신규 peer 생성: 0 (link-only)")

    if not args.apply:
        print("\n[DRY-RUN] write 없음. 적용하려면 --apply")
        return 0

    print(f"\n[APPLY] {len(plan)} 세션 peer_id 세팅 시작...")
    ok = 0; fail = 0
    for sid, pid in plan:
        try:
            c = sb_patch(sid, pid)
            if c in (200, 204): ok += 1
            else: fail += 1; print(f"  fail {sid[:8]} rc={c}")
        except Exception as e:
            fail += 1; print(f"  err {sid[:8]} {e}")
    print(f"\n=== APPLY 완료: ok={ok} fail={fail} ===")
    # 검증
    after = sb_get("sessions?select=id&peer_id=not.is.null&title=not.is.null&limit=1")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
