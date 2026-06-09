# -*- coding: utf-8 -*-
"""세션 단위 라벨 백필 오케스트레이터 — 4종을 순서대로 best-effort 실행.

신규 세션 처리 직후(worker 인라인) 또는 수동으로 단건 세션의 라벨 4종을 채운다:
  1. dialog_act_llm_backfill.py     (Ollama LLM — 화행)
  2. va_emotion_backfill.py --cpu   (감정 V-A; ⚠️ 반드시 --cpu = voice-api GPU 충돌/OOM 방지)
  3. conversation_context_backfill.py (결정론 — 대화 맥락)
  4. topic_summary_backfill.py      (Ollama LLM — 주제 요약)

각 스크립트를 격리 subprocess(`PYTHONPATH=. ./venv/bin/python3 ...`)로 호출한다.
하나가 실패해도 다음으로 진행(best-effort) — 실패는 로그+카운트만. 끝에 요약 출력.

사용:
  PYTHONPATH=. ./venv/bin/python3 scripts/analysis/run_session_backfills.py \\
      --session <id> --apply --cpu
  (--apply 없으면 각 스크립트가 dry-run, DB write 0)
"""
import argparse
import os
import subprocess
import sys
import time

# (스크립트 파일명, --cpu 전달 여부)
_BACKFILLS: list[tuple[str, bool]] = [
    ("dialog_act_llm_backfill.py", False),
    ("va_emotion_backfill.py", True),       # 반드시 --cpu (GPU 충돌 방지)
    ("conversation_context_backfill.py", False),
    ("topic_summary_backfill.py", False),
]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_PYTHON = os.path.join(_REPO_ROOT, "venv", "bin", "python3")


def _run_one(script: str, session: str, apply: bool, cpu: bool) -> tuple[bool, str]:
    """한 백필 스크립트를 subprocess 로 실행. (성공여부, 메모) 반환. 예외 삼킴."""
    rel = os.path.join("scripts", "analysis", script)
    cmd = [_PYTHON, rel, "--session", session]
    if apply:
        cmd.append("--apply")
    if cpu:
        cmd.append("--cpu")

    env = dict(os.environ)
    env["PYTHONPATH"] = "."

    try:
        proc = subprocess.run(
            cmd,
            cwd=_REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout(1800s)"
    except Exception as e:  # noqa: BLE001 — best-effort, 절대 크래시 금지
        return False, f"spawn-error: {e}"

    if proc.returncode == 0:
        return True, "ok"
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    note = tail[-1] if tail else ""
    return False, f"exit={proc.returncode} {note}"


def main() -> int:
    ap = argparse.ArgumentParser(description="세션 단위 라벨 백필 4종 오케스트레이터")
    ap.add_argument("--session", required=True, help="대상 session_id (필수).")
    ap.add_argument("--apply", action="store_true", help="DB write 적용(없으면 dry-run).")
    ap.add_argument("--cpu", action="store_true",
                    help="va_emotion 에 --cpu 전달(voice-api GPU 충돌/OOM 방지). 권장 ON.")
    args = ap.parse_args()

    session = args.session
    print(f"[orchestrator] session={session} apply={args.apply} cpu={args.cpu}")
    print(f"[orchestrator] python={_PYTHON}")

    results: list[tuple[str, bool, str]] = []
    for script, needs_cpu in _BACKFILLS:
        cpu_flag = bool(args.cpu and needs_cpu)
        t0 = time.time()
        ok, note = _run_one(script, session, args.apply, cpu_flag)
        dt = time.time() - t0
        status = "OK  " if ok else "FAIL"
        print(f"[orchestrator] {status} {script} ({dt:.1f}s) {note}")
        results.append((script, ok, note))

    n_ok = sum(1 for _, ok, _ in results if ok)
    n_fail = len(results) - n_ok
    print(f"[orchestrator] summary: {n_ok} ok, {n_fail} failed (of {len(results)})")
    for script, ok, note in results:
        if not ok:
            print(f"[orchestrator]   - FAILED {script}: {note}")

    # best-effort: 일부 실패해도 0 반환(세션 처리를 막지 않음). 전부 실패 시 1.
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
