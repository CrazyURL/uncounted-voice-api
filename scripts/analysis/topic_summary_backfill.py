# -*- coding: utf-8 -*-
"""T4 — sessions.session_topic_summary LLM 백필 (Ollama qwen2.5, 1콜/세션).

세션의 마스킹 transcript 전체를 조립(6000~8000자 truncate)해 Ollama 로 '무엇에 관한
통화인지' 1문장 요약을 생성한다. JSON {topic_summary}. utterance_count<3 세션은 skip
(빈약 요약 방지).

가드:
  - Resume: session_topic_summary IS NULL 인 세션만 픽(이미 채운 세션 skip).
  - --eligible: consent_status=both_agreed AND review_status=approved 세션만(비용 한정·안전선#5).
    --session 단건이면 그 세션만(eligible 무관).
  - 무중단 fallback: 세션별 try/except, Ollama 실패/파싱 실패 시 skip(크래시 금지, 카운트만).
  - PII 안전: 입력은 마스킹된 transcript_text 만(원문 PII 금지). LLM 출력 요약은 DB write 전
    PII 정규식 재스캔(전화/주민/카드 등) → 매치 시 그 세션 write 드롭(안전선#4).
    transcript 본문/요약 원문은 stdout/로그 미출력(요약은 dry-run 검증용으로만 1~2건 노출).
  - 모델명 비노출(안전선#6): 출력 텍스트/필드에 모델명 유입 금지.

사용:
  PYTHONPATH=. python3 scripts/analysis/topic_summary_backfill.py --session <id>        # dry-run 1세션
  PYTHONPATH=. python3 scripts/analysis/topic_summary_backfill.py --eligible --limit 50  # 배치
  (--apply 없으면 dry-run, DB write 0)
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

MAX_TRANSCRIPT_CHARS = 8000          # truncate 상한(6000~8000)
MIN_UTTERANCES = 3                   # 빈약 요약 방지

_PROMPT = """다음은 한국어 통화 녹취록이다(PII 마스킹됨). 이 통화가 '무엇에 관한 통화인지'
1문장으로 요약하라.

규칙:
- 고유명사/숫자가 [PII_*] 로 마스킹되어 있으면 일반명사(예: 어떤 사람, 어떤 번호)로 풀어쓴다.
- 화자가 누구인지(이름/관계) 단정하지 마라.
- 정확히 1문장. 군더더기 없이.

녹취록:
{transcript}

JSON 한 줄로만 답하라. 형식: {{"topic_summary": "<1문장 요약>"}}"""


def _load_env() -> dict:
    e: dict[str, str] = {}
    path = os.path.join(os.path.dirname(__file__), "../../.env.dev")
    for ln in open(path, encoding="utf-8"):
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1)
            e[k] = v.strip().strip('"')
    return e


def _ollama_url() -> str:
    return os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")


def _model() -> str:
    return os.environ.get("RELATION_INFER_MODEL", "qwen2.5:7b-instruct-q4_K_M")


def _build_pii_scanner():
    """app.pii_masker 고정밀 PII_PATTERNS 재사용. 매치 시 True(=요약 드롭)."""
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


def assemble_transcript(GET, session_id: str, max_chars: int = MAX_TRANSCRIPT_CHARS) -> str:
    """마스킹 발화를 화자 라벨과 함께 결합(peer_relation_infer._session_transcript 참고)."""
    rows = GET(
        "utterances?select=sequence_order,speaker_id,transcript_text"
        f"&session_id=eq.{session_id}&order=sequence_order.asc")
    lines = []
    for r in rows:
        spk = r.get("speaker_id") or "?"
        txt = (r.get("transcript_text") or "").strip()
        if txt:
            lines.append(f"{spk}: {txt}")
    return "\n".join(lines)[:max_chars]


def summarize(transcript: str, *, timeout: float = 120.0) -> str | None:
    """transcript → 1문장 요약 or None(실패)."""
    transcript = (transcript or "").strip()
    if not transcript:
        return None
    payload = json.dumps({
        "model": _model(),
        "prompt": _PROMPT.format(transcript=transcript),
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0},
    }).encode("utf-8")
    req = urllib.request.Request(_ollama_url(), data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None
    try:
        parsed = json.loads(body.get("response", "{}"))
    except json.JSONDecodeError:
        return None
    s = parsed.get("topic_summary")
    if not isinstance(s, str):
        return None
    s = s.strip()
    return s or None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="", help="단건 session_id (eligible 무관).")
    ap.add_argument("--eligible", action="store_true",
                    help="consent_status=both_agreed AND review_status=approved 세션만.")
    ap.add_argument("--limit", type=int, default=0, help="처리 세션 상한(0=무제한).")
    ap.add_argument("--apply", action="store_true", help="DB write 적용(없으면 dry-run).")
    ap.add_argument("--cpu", action="store_true",
                    help="설정 시 CUDA 숨김(Ollama 자체 서버 — 골격 호환용).")
    args = ap.parse_args()
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

    # ── 대상 세션(session_topic_summary NULL only = Resume) ────────────────
    if args.session:
        sel = (f"sessions?select=id,utterance_count&id=eq.{args.session}"
               "&session_topic_summary=is.null")
    else:
        flt = ""
        if args.eligible:
            flt = "&consent_status=eq.both_agreed&review_status=eq.approved"
        lim = f"&limit={args.limit}" if args.limit else ""
        sel = (f"sessions?select=id,utterance_count&session_topic_summary=is.null"
               f"{flt}&order=id.asc{lim}")
    sessions = GET(sel)

    done = 0
    skipped_thin = 0       # utterance_count<3
    skipped_pii = 0        # 요약에 PII 잔존 → 드롭
    failed = 0             # Ollama 실패
    samples: list[str] = []

    for s in sessions:
        if args.limit and done >= args.limit:
            break
        if (s.get("utterance_count") or 0) < MIN_UTTERANCES:
            skipped_thin += 1
            continue
        try:
            transcript = assemble_transcript(GET, s["id"])
            if not transcript:
                skipped_thin += 1
                continue
            summary = summarize(transcript)
            if summary is None:
                failed += 1
                continue
            if pii_scan(summary):           # 안전선#4: 요약에 잔존 PII → 드롭
                skipped_pii += 1
                continue
            if args.apply:
                PATCH(f"sessions?id=eq.{s['id']}", {"session_topic_summary": summary})
            done += 1
            if len(samples) < 2:            # dry-run 검증용 1~2건만 노출(요약은 PII 재스캔 통과본).
                samples.append(summary)
        except Exception:
            failed += 1

    mode = "적용(apply)" if args.apply else "dry-run"
    print(f"[T4 topic_summary] {mode}: 처리 {done} | 빈약 skip(<3발화) {skipped_thin} "
          f"| PII드롭 {skipped_pii} | 실패(Ollama) {failed}")
    print("샘플 요약(PII 재스캔 통과본):")
    for s in samples:
        print("  -", s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
