"""이름 PII 필터 내부 eval harness (baseline precision/recall + FP 카테고리).

목적:
    pii_candidates 의 관리자 검수 라벨(confirmed/corrected = 진짜 이름, rejected = 오탐)을
    정답으로 삼아, 현재 voice-api 이름 detector(detect_pii_spans)의 baseline
    precision/recall 을 측정하고, 3글자 firehose 경로가 만드는 FP 를 구조 카테고리로 집계한다.
    이후 denylist 보강 / graded confidence 룰 튜닝의 **회귀 게이트**로 재사용한다.

안전 계약 (강제 — migration 076 + 이번 승인 조건):
    - DB write 금지. 본 스크립트는 SELECT 만 한다.
    - 원문 PII / transcript_text / matched_text / char offset 을 export 하거나 로그로 출력하지 않는다.
      → transcript_text 는 이 프로세스 메모리에서만 쓰이고 어디에도 기록되지 않는다.
    - 결과물은 **집계 수치 + FP 구조 카테고리 카운트만**. 개별 토큰/오프셋 없음.
    - denylist 자동 보강 금지 (본 스크립트는 측정만, 룰을 수정하지 않는다).
    - prod 반영 금지.

사용법 (uncounted-voice-api 디렉토리에서):
    # 자격증명: 환경변수(SUPABASE_URL + SUPABASE_SERVICE_KEY) 우선,
    #           없으면 ../uncounted-api/.env 의 SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY 사용.
    python scripts/analysis/eval_name_filter.py
    python scripts/analysis/eval_name_filter.py --report   # .research/ 에 집계 report.md 도 생성

⚠️ baseline 측정 실행은 별도 승인 사안. 무단 실행 금지.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 저장소 루트를 path 에 추가해 app.* import 가능하게 한다 (scripts/analysis/ → 루트는 parents[2]).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.pii_confidence import score_candidates  # noqa: E402
from app.pii_masker import _HONORIFICS, detect_pii_spans  # noqa: E402

NAME_TYPE = "이름"
POSITIVE_DECISIONS = frozenset({"confirmed", "corrected"})
NEGATIVE_DECISIONS = frozenset({"rejected"})

# FP 카테고리 분류·시뮬레이션용 조사/어미 어두 글자 (구조 휴리스틱, 원문 미출력).
# ⚠️ 이 set 기반 "조사 후행 제외"는 detector 에 적용하지 않는다 — 진짜 이름+조사
#    ("김용철이")와 일반어+조사("정상화를")를 구분 못 해 마스킹 recall 을 깬다(아래 분석 결과).
_JOSA_EOMI_HEAD = frozenset(
    "을를이가은는에의도만과와로으한했하해인임라고며면서게지네요죠까란답"
)
_BOUNDARY_CHARS = " \t\n,.:;!?()\"'·…—-~"


def _load_supabase():
    """환경변수 우선, 없으면 ../uncounted-api/.env 에서 자격증명을 읽어 supabase client 생성."""
    from supabase import create_client

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not (url and key):
        env_path = _REPO_ROOT.parent / "uncounted-api" / ".env"
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("SUPABASE_URL=") and not url:
                    url = line.split("=", 1)[1].strip().strip("\"'")
                elif line.startswith("SUPABASE_SERVICE_ROLE_KEY=") and not key:
                    key = line.split("=", 1)[1].strip().strip("\"'")
    if not (url and key):
        sys.exit("SUPABASE_URL / SERVICE KEY 를 찾을 수 없습니다 (env 또는 ../uncounted-api/.env).")
    return create_client(url, key)


def _fetch_name_labels(client) -> list[dict]:
    """predicted_type='이름' 이고 admin_decision 이 기록된 검수 후보를 페이지네이션으로 읽는다."""
    rows: list[dict] = []
    page, frm = 1000, 0
    while True:
        resp = (
            client.table("pii_candidates")
            .select("utterance_id, char_start, char_end, admin_decision")
            .eq("predicted_type", NAME_TYPE)
            .not_.is_("admin_decision", "null")
            .order("id", desc=False)
            .range(frm, frm + page - 1)
            .execute()
        )
        data = resp.data or []
        rows.extend(data)
        if len(data) < page:
            break
        frm += page
    return rows


def _fetch_transcripts(client, utterance_ids: list[str]) -> dict[str, str]:
    """utterance_id → transcript_text 맵 (메모리 전용. 절대 기록/출력하지 않는다)."""
    texts: dict[str, str] = {}
    chunk = 200
    for i in range(0, len(utterance_ids), chunk):
        ids = utterance_ids[i : i + chunk]
        resp = (
            client.table("utterances")
            .select("id, transcript_text")
            .in_("id", ids)
            .execute()
        )
        for r in resp.data or []:
            if r.get("transcript_text"):
                texts[r["id"]] = r["transcript_text"]
    return texts


def _emits_name_at(text: str, char_start: int, char_end: int) -> bool:
    """현재 detector 가 [char_start, char_end) 위치에 이름 span 을 emit 하는가.

    정확 offset 일치 우선, 없으면 구간 overlap 으로 판정(detector 버전 차로 offset 이
    1~2글자 이동했을 가능성 방어).
    """
    spans = detect_pii_spans(text, enable_name_masking=True)
    for sp in spans:
        if sp["type"] != NAME_TYPE:
            continue
        if sp["char_start"] == char_start and sp["char_end"] == char_end:
            return True
        if sp["char_start"] < char_end and char_start < sp["char_end"]:
            return True
    return False


def _josa_rule_keeps(text: str, char_start: int, char_end: int) -> bool:
    """제안 룰(조사/어미 후행 제외)이 이 3글자 span 을 **여전히 이름으로 유지**하는가.

    제안 룰: 성+2글자 직후가 조사/어미로 이어지면 더 긴 단어/활용형 substring 으로 보고 제외.
            단 호칭이 뒤따르면(이름 신호) 유지. bare(경계 후행)는 이번 단계 미적용 → 유지.
    detector 변경 전 read-only 시뮬레이션 전용. 이 룰은 기존 emit 를 더 좁힐 뿐 추가하지 않는다.
    """
    after = text[char_end:]
    after_stripped = after.lstrip()
    for h in _HONORIFICS:
        if after_stripped.startswith(h):
            return True
    nxt = after[0] if after else ""
    if nxt in _JOSA_EOMI_HEAD:
        return False
    return True


def _name_tier_at(text: str, char_start: int, char_end: int) -> str | None:
    """검수 큐 관점: [char_start,char_end) 이름 span 의 confidence_tier 를 반환.

    detect_pii_spans(emit, 마스킹 경로와 동일) → score_candidates(검수 후보 점수) 를 그대로 거친다.
    emit 안 되면 None. graded confidence 효과(weak_trailing → auto_rejected)를 측정하기 위함.
    """
    spans = detect_pii_spans(text, enable_name_masking=True)
    for sp in spans:
        if sp["type"] != NAME_TYPE:
            continue
        hit = (sp["char_start"] == char_start and sp["char_end"] == char_end) or (
            sp["char_start"] < char_end and char_start < sp["char_end"]
        )
        if hit:
            return score_candidates([sp])[0]["confidence_tier"]
    return None


def _categorize_fp(text: str, char_start: int, char_end: int) -> str:
    """FP(rejected 인데 여전히 emit) 를 **구조 카테고리**로만 분류한다. 원문 미반환.

    - honorific_follows : 뒤에 호칭(_HONORIFICS) → 이름 신호 강한데 사람이 rejected (전사 잡음 등)
    - josa_eomi_follows : 뒤 첫 글자가 조사/어미 → 성+2글자가 더 긴 단어/활용형의 일부 (firehose 핵심)
    - bare_boundary     : 뒤가 공백/문장부호/끝 → 문맥 없는 3글자 (애매 일반명사 가능)
    - other             : 그 외
    """
    after = text[char_end:]
    after_stripped = after.lstrip()
    for h in _HONORIFICS:
        if after_stripped.startswith(h):
            return "honorific_follows"
    nxt = after[0] if after else ""
    if nxt in _BOUNDARY_CHARS or nxt == "":
        return "bare_boundary"
    if nxt in _JOSA_EOMI_HEAD:
        return "josa_eomi_follows"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description="이름 PII 필터 baseline eval harness")
    parser.add_argument("--report", action="store_true", help=".research/ 에 집계 report.md 생성")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="조사/어미 후행 제외 룰의 투영 confusion matrix 를 추가 출력 (detector 미변경, read-only)",
    )
    parser.add_argument(
        "--graded",
        action="store_true",
        help="graded confidence 효과(검수 큐 tier 구성) 측정. 마스킹 emit 와 분리해서 본다.",
    )
    args = parser.parse_args()

    client = _load_supabase()
    labels = _fetch_name_labels(client)
    if not labels:
        sys.exit("검수된 이름 후보(admin_decision)가 없습니다.")

    utt_ids = sorted({r["utterance_id"] for r in labels})
    texts = _fetch_transcripts(client, utt_ids)

    tp = fp = fn = tn = 0
    missing_text = 0
    fp_categories: dict[str, int] = {}
    # 시뮬레이션 누적기 (제안 룰은 emit 를 더 좁힐 뿐이므로 손실/제거만 카운트)
    sim_tp_lost = 0  # confirmed 인데 룰이 제외 → recall 회귀 (반드시 0 이어야 게이트 통과)
    sim_fp_removed = 0  # rejected 인데 룰이 제외 → 정상 개선
    sim_fp_removed_cat: dict[str, int] = {}
    # graded confidence 누적기: 검수 큐(needs_human/auto_confirmed) vs 큐 이탈(auto_rejected).
    NEEDS = "needs_human_decision"
    REJECTED_TIER = "auto_rejected"
    g_pos_in_queue = g_pos_dropped = g_neg_in_queue = g_neg_dropped = 0

    for r in labels:
        text = texts.get(r["utterance_id"])
        if text is None:
            missing_text += 1
            continue
        cs, ce, dec = r["char_start"], r["char_end"], r["admin_decision"]
        if cs is None or ce is None:
            missing_text += 1
            continue
        emitted = _emits_name_at(text, cs, ce)
        is_positive = dec in POSITIVE_DECISIONS
        is_negative = dec in NEGATIVE_DECISIONS
        tier = _name_tier_at(text, cs, ce) if (args.graded and emitted) else None
        if is_positive:
            if emitted:
                tp += 1
                if args.simulate and not _josa_rule_keeps(text, cs, ce):
                    sim_tp_lost += 1
                if args.graded:
                    if tier == REJECTED_TIER:
                        g_pos_dropped += 1  # confirmed 인데 큐 이탈 (큐 recall 손실)
                    else:
                        g_pos_in_queue += 1
            else:
                fn += 1
        elif is_negative:
            if emitted:
                fp += 1
                cat = _categorize_fp(text, cs, ce)
                fp_categories[cat] = fp_categories.get(cat, 0) + 1
                if args.simulate and not _josa_rule_keeps(text, cs, ce):
                    sim_fp_removed += 1
                    sim_fp_removed_cat[cat] = sim_fp_removed_cat.get(cat, 0) + 1
                if args.graded:
                    if tier == REJECTED_TIER:
                        g_neg_dropped += 1  # rejected 가 큐 이탈 (정상 개선)
                    else:
                        g_neg_in_queue += 1  # 큐에 남은 FP
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    # ── 보고 (집계 수치 + FP 카테고리만; 원문/offset 없음) ──────────────
    lines = [
        "===== 이름 PII 필터 baseline (검수셋 기준) =====",
        f"라벨 후보: {len(labels)}  (텍스트 누락 제외: {missing_text})",
        f"TP {tp}  FP {fp}  FN {fn}  TN {tn}",
        f"precision {precision:.3f}  recall {recall:.3f}",
        "",
        "[FP 구조 카테고리]",
    ]
    for cat, n in sorted(fp_categories.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {cat:<18} {n}")
    if not fp_categories:
        lines.append("  (없음)")
    report = "\n".join(lines)
    print(report)

    if args.simulate:
        proj_tp = tp - sim_tp_lost
        proj_fn = fn + sim_tp_lost
        proj_fp = fp - sim_fp_removed
        proj_tn = tn + sim_fp_removed
        proj_precision = proj_tp / (proj_tp + proj_fp) if (proj_tp + proj_fp) else 0.0
        proj_recall = proj_tp / (proj_tp + proj_fn) if (proj_tp + proj_fn) else 0.0
        gate_ok = (
            sim_tp_lost == 0
            and proj_recall >= 1.0
            and proj_precision > precision
            and proj_fp < fp
            and sim_fp_removed_cat.get("josa_eomi_follows", 0) > 0
        )
        sim_lines = [
            "",
            "===== [SIMULATE] 조사/어미 후행 제외 룰 투영 (detector 미변경) =====",
            f"TP {proj_tp}  FP {proj_fp}  FN {proj_fn}  TN {proj_tn}",
            f"precision {precision:.3f} → {proj_precision:.3f}   recall {recall:.3f} → {proj_recall:.3f}",
            f"TP 손실(=recall 회귀): {sim_tp_lost}   FP 제거: {sim_fp_removed}",
            "[FP 제거 카테고리]",
        ]
        for cat, n in sorted(sim_fp_removed_cat.items(), key=lambda kv: -kv[1]):
            sim_lines.append(f"  {cat:<18} {n}")
        sim_lines.append(f"회귀 게이트 통과: {'YES' if gate_ok else 'NO'} (TP손실 0 & recall 1.0 & precision↑ & FP↓ & josa제거>0)")
        print("\n".join(sim_lines))

    if args.graded:
        # 마스킹 recall = detect emit (변하지 않음). 검수 큐 = needs_human 에 남은 후보.
        masking_recall = tp / (tp + fn) if (tp + fn) else 0.0
        queue_pos = g_pos_in_queue  # 큐에 남은 confirmed
        queue_neg = g_neg_in_queue  # 큐에 남은 rejected(FP)
        queue_precision = queue_pos / (queue_pos + queue_neg) if (queue_pos + queue_neg) else 0.0
        gate_ok = (
            masking_recall >= 1.0  # 마스킹(탐지) recall 무손실
            and g_pos_dropped == 0  # confirmed 가 큐에서 이탈하지 않음
            and g_neg_dropped > 0  # rejected 중 일부가 큐 이탈(개선)
            and queue_precision > precision  # 큐 precision 이 baseline(0.418) 초과
        )
        g_lines = [
            "",
            "===== [GRADED] graded confidence 검수 큐 효과 (마스킹과 분리) =====",
            f"마스킹 emit recall: {masking_recall:.3f}  (detect_pii_spans 불변 — 프라이버시 무손실)",
            f"큐 잔류 confirmed: {g_pos_in_queue}   큐 이탈 confirmed(auto_rejected): {g_pos_dropped}",
            f"큐 잔류 rejected(FP): {g_neg_in_queue}   큐 이탈 rejected(auto_rejected): {g_neg_dropped}",
            f"검수 큐 precision: {precision:.3f}(baseline) → {queue_precision:.3f}",
            f"게이트 통과: {'YES' if gate_ok else 'NO'} "
            "(마스킹recall 1.0 & confirmed 큐이탈 0 & rejected 큐이탈>0 & 큐precision↑)",
        ]
        print("\n".join(g_lines))

    if args.report:
        out_dir = _REPO_ROOT / ".research" / "pii_name_eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "baseline_report.md").write_text(
            "# 이름 PII 필터 baseline\n\n"
            "> 집계 수치 + FP 구조 카테고리만. 원문/offset 미포함 (076 계약).\n\n"
            "```\n" + report + "\n```\n",
            encoding="utf-8",
        )
        print(f"\nreport: {out_dir / 'baseline_report.md'}")


if __name__ == "__main__":
    main()
