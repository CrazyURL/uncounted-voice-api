# -*- coding: utf-8 -*-
"""관계(relation) 맥락 교차검증 — 주제(topic) + 대화목적(dialog_act)로 후보 관계를 재검토.

원칙(대표 확정, project_relation_label_context_principle):
  - 단일 호칭 1개로 관계 확정 금지.
  - 맥락(주제 + 대화목적)이 관계를 지지하면 유지/복원, 약하거나 모순이면 null.
  - **틀린 관계 > 빈값** → 애매하면 null.

의존성(이제 충족): topic 세그먼트 분류(0.79) + dialog_act 백필.
1차는 주제 게이트(primary), 대화목적은 보강(corroboration). 충분치 않으면 SER(감정) 추가 검토.
"""
from __future__ import annotations
from typing import Iterable, Optional

# ★게이트 비대상(broad): 가족·친구류는 *모든 주제*를 얘기하므로 주제로 강등하지 않는다.
#   (dry-run서 배우자/부모를 건강·직장 주제로 오강등 → 과강등 70% 확인 후 보수화 2026-06-07)
BROAD_RELATIONS = frozenset({
    "친구", "기타", "부모", "자녀", "배우자", "형제자매",
})

# 관계 → 지지 주제. **교사만** 게이트한다(B안, 2026-06-07 실측 결정).
#   근거: 데모션 77건 중 27%만 propagation이 못 잡고, 그마저 고객/직장은 "어떤 주제든 가능"이라
#   게이트가 fragile(고객+건강=정상인데 오강등). 주제에 *강하게* 묶인 관계는 교사(→교육)뿐.
#   나머지 관계는 게이트 안 함(no_rule_keep) → propagation 다수결에 맡긴다.
RELATION_TOPIC_COMPAT: dict[str, frozenset] = {
    "교사": frozenset({"교육/공부", "교육"}),
}

# 거래/업무성 대화목적(9-group). 비율이 높으면 '개인(가족류)' 관계를 약화.
_TRANSACTIONAL_DIALOG_ACTS = frozenset({"질문/확인", "정보", "감사/사과"})
# 개인적(가정/사적) 관계 — 거래성 대화목적이 우세하면 의심 가중
_PERSONAL_RELATIONS = frozenset({"부모", "자녀", "배우자", "형제자매", "교사"})


def crossvalidate_relation(
    relation: Optional[str],
    session_topics: Iterable[str] | None,
    dialog_act_dist: dict[str, int] | None = None,
) -> tuple[Optional[str], str]:
    """관계 후보를 맥락으로 교차검증. 반환 (검증된_관계|None, 사유).

    - broad(친구/기타)·None: 게이트 안 함(그대로).
    - compat 규칙 없는 관계: 보수적으로 유지(모르면 손대지 않음).
    - 주제 지지: 유지(복원 포함). 주제 미지지: null 강등.
    - 대화목적 보강: 미지지 + 거래성 우세면 강등 확신↑(사유에 기록).
    """
    if not relation or relation in BROAD_RELATIONS:
        return relation, "broad_or_none"
    compat = RELATION_TOPIC_COMPAT.get(relation)
    if compat is None:
        return relation, "no_rule_keep"

    topics = set(t for t in (session_topics or []) if t)
    if topics & compat:
        return relation, "topic_supported"

    # 주제 미지지 — 대화목적으로 보강 판단
    da_note = ""
    if dialog_act_dist:
        total = sum(dialog_act_dist.values()) or 1
        trans = sum(v for k, v in dialog_act_dist.items() if k in _TRANSACTIONAL_DIALOG_ACTS)
        trans_ratio = trans / total
        if relation in _PERSONAL_RELATIONS and trans_ratio >= 0.6:
            da_note = f",dialog_act_transactional={trans_ratio:.2f}"

    # 틀린 관계 > 빈값 → null 강등
    return None, f"topic_unsupported(relation={relation},topics={sorted(topics)}{da_note})"


# ── cross-call peer 전파 (관계 lifecycle 2단계) ────────────────────────────
# 같은 peer의 여러 통화 관계(데모션 후)를 묶어 통합. v2 모델 규칙:
#   - 강키(일관+다수 지지): 전파 OK.
#   - 약키(단일 통화만 지지): 자동전파 금지(그 통화엔 유지, 다른 통화로 안 퍼뜨림).
#   - 충돌(통화마다 다른 관계): 전파 보류 → null 유지 + 검수 플래그.
# 互恵 관계(부모↔자녀)는 같은 peer-쌍을 양쪽 시점에서 본 것 → 충돌 아님(호환쌍).
_RECIPROCAL_PAIRS = (frozenset({"부모", "자녀"}),)
_MIN_SUPPORT = 2   # 강키 최소 지지 통화수


def consolidate_peer_relation(
    call_relations: Iterable[Optional[str]],
) -> tuple[Optional[str], float, str, str]:
    """peer의 통화별 관계(데모션 후) → 통합 관계.
    반환 (peer_relation|None, confidence, source, reason).
    """
    from collections import Counter

    valid = [r for r in (call_relations or []) if r and r != "기타"]
    if not valid:
        return None, 0.0, "no_signal", "전 통화 무신호/null"

    cnt = Counter(valid)
    distinct = set(cnt)
    # 互恵쌍은 하나로 본다(부모/자녀 = 같은 관계의 양면)
    for pair in _RECIPROCAL_PAIRS:
        if distinct <= pair and len(distinct) > 1:
            distinct = {sorted(pair)[0]}  # 대표값으로 통일(충돌 아님)

    total = sum(cnt.values())
    if len(distinct) == 1:
        rel = cnt.most_common(1)[0][0]
        if total >= _MIN_SUPPORT:
            conf = round(min(0.95, 0.6 + 0.1 * total), 2)
            return rel, conf, "cross_call_consistent", f"{total}통화 일관 지지"
        # 단일 통화 지지: peer 관계로는 유지(저신뢰), 단 다른 통화로 전파는 안 함(약키)
        return rel, 0.5, "single_call", "단일 통화 지지(전파 안 함, peer값 유지)"

    # 다수결: 互恵 병합 후 최빈값이 2위의 2배 이상이면 '강키(압도적 다수)' → 전파.
    merged = Counter()
    for r, n in cnt.items():
        key = r
        for pair in _RECIPROCAL_PAIRS:
            if r in pair:
                key = sorted(pair)[0]
        merged[key] += n
    ranked = merged.most_common()
    top, n_top = ranked[0]
    n_second = ranked[1][1] if len(ranked) > 1 else 0
    if n_top >= _MIN_SUPPORT and n_top >= 2 * n_second:
        conf = round(min(0.9, 0.5 + n_top / total * 0.4), 2)
        return top, conf, "cross_call_dominant", f"압도적 다수({top} {n_top}/{total})"

    # 진짜 모호 → 보류 + 검수 플래그
    return None, 0.0, "conflict", f"통화간 관계 충돌(모호): {dict(cnt)}"
