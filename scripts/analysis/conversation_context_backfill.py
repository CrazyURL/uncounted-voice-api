# -*- coding: utf-8 -*-
"""T5 — utterances.conversation_context 결정론 백필 (LLM 없음).

075 마이그(conversation_context JSONB) 적용 전제. 발화별로 4키 객체를 항상 산출한다:
  turn_index      : 세션 내 sequence_order 파생(1-base)
  topic_thread    : start_sec 가 속한 session_segments.topic (없으면 "default_context")
  discourse_role  : utterance_form 에서 결정론 매핑(opening/question/response/closing/
                    backchannel/statement). 없으면 "default_context".
  prev_turn_gist  : 직전 '다른 화자' 발화의 마스킹 transcript_text N자 truncate.
                    없으면 "default_context".

⚠️ 결측 Fallback 강제(대표님 지침): 입력 지표 누락 시 KeyError/빈문자열 양산 금지 →
   누락 필드는 "default_context"(명시 sentinel). 객체는 항상 4키 보유(빈 장부 재발 방지).

가드:
  - Resume: conversation_context IS NULL 인 발화만 픽(이미 채운 행 skip).
  - --eligible: consent_status=both_agreed AND review_status=approved 세션만(비용 한정·안전선#5).
    --session 단건이면 그 세션만(eligible 무관).
  - 무중단 fallback: 행별 try/except, 실패 시 skip(카운트만).
  - PII 안전: 입력은 마스킹된 transcript_text 만. prev_turn_gist 도 마스킹 텍스트 → DB write 전
    PII 정규식 재스캔(전화/주민/카드 등), 매치 시 해당 필드 드롭("default_context"로 치환).
    transcript 본문은 stdout/로그에 찍지 않는다(turn_index/role/topic·구조만).

사용:
  PYTHONPATH=. python3 scripts/analysis/conversation_context_backfill.py --session <id>          # 단건 dry-run
  PYTHONPATH=. python3 scripts/analysis/conversation_context_backfill.py --eligible --limit 500   # 배치
  (--apply 없으면 dry-run, DB write 0)
"""
import argparse
import json
import os
import urllib.request

# ── 상수 ────────────────────────────────────────────────────────────────
DEFAULT = "default_context"          # 결측 sentinel (빈문자열/KeyError 금지)
PREV_GIST_MAX_CHARS = 60             # prev_turn_gist truncate 길이

# discourse_role 결정론 매핑 우선순위(위에서부터 먼저 매치되는 것 채택).
# utterance_form 키: turn_type / utterance_type / is_greeting / is_closing / is_backchannel.


def _load_env() -> dict:
    e: dict[str, str] = {}
    path = os.path.join(os.path.dirname(__file__), "../../.env.dev")
    for ln in open(path, encoding="utf-8"):
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1)
            e[k] = v.strip().strip('"')
    return e


# ── discourse_role 결정론 매핑 ───────────────────────────────────────────
def _discourse_role(form: dict | None) -> str:
    """utterance_form → discourse_role(closed). 누락/빈 dict 면 DEFAULT."""
    if not isinstance(form, dict) or not form:
        return DEFAULT
    if form.get("is_greeting"):
        return "opening"
    if form.get("is_closing"):
        return "closing"
    if form.get("is_backchannel"):
        return "backchannel"
    turn_type = form.get("turn_type")
    if turn_type == "opening":
        return "opening"
    if turn_type == "closing":
        return "closing"
    utt_type = form.get("utterance_type")
    if utt_type == "question":
        return "question"
    if utt_type in ("response", "answer") or form.get("is_short_response"):
        return "response"
    if utt_type == "statement":
        return "statement"
    # turn_type 만 있고 utterance_type 미상 → statement 로 안전 귀결
    if turn_type:
        return "statement"
    return DEFAULT


# ── topic_thread: start_sec 가 속한 세그먼트 topic ───────────────────────
def _topic_thread(start_sec, segments: list[dict]) -> str:
    """start_sec(초) 가 [start_ms,end_ms] 에 드는 첫 세그먼트 topic. 없으면 DEFAULT.

    세그먼트는 시간상 겹칠 수 있어 segment_index 순으로 첫 포함 매치를 채택한다.
    """
    if start_sec is None or not segments:
        return DEFAULT
    try:
        ms = float(start_sec) * 1000.0
    except (TypeError, ValueError):
        return DEFAULT
    for seg in segments:
        s, e = seg.get("start_ms"), seg.get("end_ms")
        if s is None or e is None:
            continue
        if s <= ms <= e:
            topic = seg.get("topic")
            return topic if topic else DEFAULT
    return DEFAULT


# ── prev_turn_gist: 직전 '다른 화자' 발화 truncate ──────────────────────
def _prev_turn_gist(idx: int, rows: list[dict], pii_scan) -> str:
    """rows[idx] 기준 직전 '다른 화자' 발화의 마스킹 text N자. 없으면 DEFAULT.

    PII 재스캔: 마스킹본이라도 잔존 패턴 매치 시 DEFAULT 로 드롭(안전선#4).
    """
    cur_spk = rows[idx].get("speaker_id")
    for j in range(idx - 1, -1, -1):
        prev = rows[j]
        if prev.get("speaker_id") == cur_spk:
            continue
        txt = (prev.get("transcript_text") or "").strip()
        if not txt:
            return DEFAULT
        if pii_scan(txt):           # 잔존 PII → 드롭
            return DEFAULT
        gist = txt[:PREV_GIST_MAX_CHARS]
        return gist if gist else DEFAULT
    return DEFAULT


# ── PII 재스캔(전화/주민/카드 등 고정밀 패턴) ────────────────────────────
def _build_pii_scanner():
    """app.pii_masker 의 고정밀 PII_PATTERNS 재사용. 매치 시 True(=드롭 대상)."""
    try:
        from app.pii_masker import PII_PATTERNS
        patterns = [p for (p, _repl, _label) in PII_PATTERNS]
    except Exception:
        patterns = []

    def scan(text: str) -> bool:
        for p in patterns:
            if p.search(text or ""):
                return True
        return False

    return scan


def build_context(idx: int, rows: list[dict], segments: list[dict], pii_scan) -> dict:
    """발화 1건의 conversation_context 객체(항상 4키)."""
    r = rows[idx]
    seq = r.get("sequence_order")
    turn_index = int(seq) if isinstance(seq, int) or (isinstance(seq, str) and str(seq).isdigit()) else (idx + 1)
    return {
        "turn_index": turn_index,
        "topic_thread": _topic_thread(r.get("start_sec"), segments),
        "discourse_role": _discourse_role(r.get("utterance_form")),
        "prev_turn_gist": _prev_turn_gist(idx, rows, pii_scan),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="", help="단건 session_id (eligible 무관).")
    ap.add_argument("--eligible", action="store_true",
                    help="consent_status=both_agreed AND review_status=approved 세션만.")
    ap.add_argument("--limit", type=int, default=0, help="처리 발화 상한(0=무제한).")
    ap.add_argument("--apply", action="store_true", help="DB write 적용(없으면 dry-run).")
    ap.add_argument("--cpu", action="store_true", help="(LLM 없음 — 무시, 골격 호환용)")
    args = ap.parse_args()

    env = _load_env()
    os.environ.update(env)
    U = env["SUPABASE_URL"]
    K = env["SUPABASE_SERVICE_KEY"]
    H = {"apikey": K, "Authorization": "Bearer " + K, "Content-Type": "application/json"}

    def GET(p):
        return json.load(urllib.request.urlopen(urllib.request.Request(U + "/rest/v1/" + p, headers=H), timeout=60))

    def PATCH(p, b):
        return urllib.request.urlopen(urllib.request.Request(
            U + "/rest/v1/" + p, data=json.dumps(b).encode(), method="PATCH", headers=H), timeout=20).status

    pii_scan = _build_pii_scanner()

    # ── 대상 세션 목록 ──────────────────────────────────────────────────
    if args.session:
        session_ids = [args.session]
    else:
        flt = ""
        if args.eligible:
            flt = "&consent_status=eq.both_agreed&review_status=eq.approved"
        # conversation_context NULL 발화가 1건이라도 있는 세션만 훑기 위해
        # 세션 전수에서 픽(REST 상 distinct 불가 → 세션 단위 순회).
        sess_rows = GET(f"sessions?select=id{flt}&order=id.asc")
        session_ids = [s["id"] for s in sess_rows]

    done = 0
    skipped = 0
    failed = 0
    samples: list[dict] = []
    role_dist: dict[str, int] = {}

    for sid in session_ids:
        # 세션의 모든 발화(순서대로) — prev_turn_gist 계산에 전체 필요.
        rows = GET(
            "utterances?select=id,sequence_order,speaker_id,start_sec,utterance_form,"
            f"transcript_text,conversation_context&session_id=eq.{sid}&order=sequence_order.asc")
        if not rows:
            continue
        segments = GET(
            f"session_segments?select=segment_index,topic,start_ms,end_ms&session_id=eq.{sid}&order=segment_index.asc")

        for idx, r in enumerate(rows):
            if args.limit and done >= args.limit:
                break
            # Resume: 이미 채운 행 skip.
            if r.get("conversation_context") is not None:
                skipped += 1
                continue
            try:
                ctx = build_context(idx, rows, segments, pii_scan)
                role_dist[ctx["discourse_role"]] = role_dist.get(ctx["discourse_role"], 0) + 1
                if args.apply:
                    PATCH(f"utterances?id=eq.{r['id']}", {"conversation_context": ctx})
                done += 1
                if len(samples) < 3:
                    # PII 본문 미출력: prev_turn_gist 는 길이/DEFAULT 여부만 노출.
                    g = ctx["prev_turn_gist"]
                    samples.append({
                        "turn_index": ctx["turn_index"],
                        "topic_thread": ctx["topic_thread"],
                        "discourse_role": ctx["discourse_role"],
                        "prev_turn_gist": (DEFAULT if g == DEFAULT else f"<{len(g)} chars>"),
                        "keys": sorted(ctx.keys()),
                    })
            except Exception:
                failed += 1
        if args.limit and done >= args.limit:
            break

    mode = "적용(apply)" if args.apply else "dry-run"
    print(f"[T5 conversation_context] {mode}: 처리 {done} | skip(이미채움) {skipped} | 실패 {failed}")
    print(f"discourse_role 분포: {role_dist}")
    print("샘플(PII 본문 제외, 구조/라벨만):")
    for s in samples:
        print("  ", json.dumps(s, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
