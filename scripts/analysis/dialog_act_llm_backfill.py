# -*- coding: utf-8 -*-
"""T3(임시) — utterances.dialog_act LLM 백필 (Ollama qwen2.5, heuristic_mvp).

발화별 마스킹 transcript_text 를 Ollama 로 화행 분류한다. 출력은 반드시 closed 15종
enum(scripts/prepare_emotion_dataset.py:DIALOG_ACT_LABELS) 중 하나로 강제 — enum 밖이면
'기타'로 매핑(자유생성 금지).

⚠️ provenance 정직: 본 백필은 supervised head 가 아닌 LLM 휴리스틱(MVP)이다.
   label_source='heuristic_mvp', auto_label_model_version='heuristic_mvp' 로 명시 기록해
   향후 supervised 결과와 구분 가능하게 한다(모델명 자체는 비노출 — 안전선#6).

가드:
  - Resume: dialog_act IS NULL 인 발화만 픽(이미 채운 행 skip) → 중단돼도 재실행 시 이어감.
  - --eligible: consent_status=both_agreed AND review_status=approved 세션만(비용 한정·안전선#5).
    --session 단건이면 그 세션만(eligible 무관).
  - 무중단 fallback: 행별 try/except, Ollama 실패/파싱 실패 시 skip(크래시 금지, 카운트만).
  - PII 안전: 입력은 마스킹된 transcript_text 만(원문 PII 금지). dialog_act 는 enum 라벨이라
    PII 유입 불가하나, 방어적으로 본문은 stdout/로그 미출력(라벨·카운트만).
  - --limit 로 배치 제어(발화 단위라 콜 多).

사용:
  PYTHONPATH=. python3 scripts/analysis/dialog_act_llm_backfill.py --session <id> --limit 5   # dry-run
  PYTHONPATH=. python3 scripts/analysis/dialog_act_llm_backfill.py --eligible --limit 500       # 배치
  (--apply 없으면 dry-run, DB write 0)
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

# closed enum 정본 import (자유생성 금지 — enum 밖이면 '기타').
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from scripts.prepare_emotion_dataset import DIALOG_ACT_LABELS  # noqa: E402

_FALLBACK = "기타"
_DIALOG_SET = set(DIALOG_ACT_LABELS)
_METHOD = "heuristic_mvp"            # provenance: supervised head 와 구분
_FIXED_CONF = 0.5                    # LLM 확신도 미회신 시 고정값

_PROMPT = """다음은 한국어 통화 발화 1개다(PII 마스킹됨). 이 발화의 화행(speech act)을
아래 목록 중 정확히 하나로 분류하라. 목록 밖 값은 절대 쓰지 마라.

가능한 화행: {labels}

발화: {text}

JSON 한 줄로만 답하라. 형식: {{"dialog_act": "<화행>", "confidence": <0.0~1.0>}}"""


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


def classify(text: str, *, timeout: float = 60.0) -> tuple[str, float] | None:
    """발화 → (dialog_act∈enum, confidence). 실패 시 None(호출자 skip).

    enum 밖 출력은 '기타'로 강제 매핑. confidence 미회신 시 고정 0.5.
    """
    text = (text or "").strip()
    if not text:
        return None
    prompt = _PROMPT.format(labels=", ".join(DIALOG_ACT_LABELS), text=text)
    payload = json.dumps({
        "model": _model(),
        "prompt": prompt,
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

    label = parsed.get("dialog_act")
    if label not in _DIALOG_SET:        # enum 강제: 밖이면 '기타'
        label = _FALLBACK
    try:
        conf = float(parsed.get("confidence", _FIXED_CONF))
        conf = max(0.0, min(1.0, conf))
    except (TypeError, ValueError):
        conf = _FIXED_CONF
    return label, conf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="", help="단건 session_id (eligible 무관).")
    ap.add_argument("--eligible", action="store_true",
                    help="consent_status=both_agreed AND review_status=approved 세션만.")
    ap.add_argument("--limit", type=int, default=0, help="처리 발화 상한(0=무제한).")
    ap.add_argument("--apply", action="store_true", help="DB write 적용(없으면 dry-run).")
    ap.add_argument("--cpu", action="store_true",
                    help="설정 시 CUDA 숨김(Ollama는 자체 서버라 직접 영향 적음 — 골격 호환용).")
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

    # ── 대상 발화 쿼리(dialog_act NULL only = Resume) ──────────────────────
    if args.session:
        sess_flt = f"&session_id=eq.{args.session}"
    elif args.eligible:
        # 적격 세션 id 선조회 후 in 필터.
        sess_rows = GET("sessions?select=id&consent_status=eq.both_agreed&review_status=eq.approved&order=id.asc")
        ids = [s["id"] for s in sess_rows]
        if not ids:
            print("[T3 dialog_act] eligible 세션 없음 — 대상 0")
            return 0
        sess_flt = "&session_id=in.(" + ",".join(ids) + ")"
    else:
        sess_flt = ""

    lim = args.limit if args.limit else 1000
    rows = GET(
        "utterances?select=id,transcript_text&dialog_act=is.null"
        f"&transcript_text=not.is.null{sess_flt}&order=id.asc&limit={lim}")

    done = 0
    skipped_empty = 0
    failed = 0
    samples: list[dict] = []
    label_dist: dict[str, int] = {}

    for u in rows:
        if args.limit and done >= args.limit:
            break
        txt = (u.get("transcript_text") or "").strip()
        if not txt:
            skipped_empty += 1
            continue
        try:
            res = classify(txt)
            if res is None:             # Ollama 실패/파싱 실패 → skip(크래시 금지)
                failed += 1
                continue
            label, conf = res
            label_dist[label] = label_dist.get(label, 0) + 1
            if args.apply:
                PATCH(f"utterances?id=eq.{u['id']}", {
                    "dialog_act": label,
                    "dialog_act_confidence": round(conf, 3),
                    "label_source": _METHOD,
                    "auto_label_model_version": _METHOD,
                })
            done += 1
            if len(samples) < 3:        # PII 본문 미출력: 라벨/conf 만.
                samples.append({"dialog_act": label, "confidence": round(conf, 3),
                                "in_enum": label in _DIALOG_SET})
        except Exception:
            failed += 1

    mode = "적용(apply)" if args.apply else "dry-run"
    print(f"[T3 dialog_act] {mode}: 처리 {done} | 빈텍스트 skip {skipped_empty} | 실패(Ollama) {failed}")
    print(f"enum 적합: 산출 라벨 전부 15종 closed enum 내 = {all(k in _DIALOG_SET for k in label_dist)}")
    print(f"dialog_act 분포: {label_dist}")
    print("샘플(PII 본문 제외, 라벨/conf 만):")
    for s in samples:
        print("  ", json.dumps(s, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
