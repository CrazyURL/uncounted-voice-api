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

# 주제 무관하게 어떤 대화에서도 가능 → 게이트하지 않음
BROAD_RELATIONS = frozenset({"친구", "기타"})

# 관계 → 그 관계를 '지지'하는 주제(19분류 기준). 교집합 없으면 맥락 미지지 → 강등 후보.
RELATION_TOPIC_COMPAT: dict[str, frozenset] = {
    "교사": frozenset({"교육"}),
    "부모": frozenset({"가족", "건강", "주거와 생활", "교육", "식음료"}),
    "자녀": frozenset({"가족", "건강", "주거와 생활", "교육"}),
    "배우자": frozenset({"가족", "연애/결혼", "주거와 생활", "여행", "식음료"}),
    "형제자매": frozenset({"가족", "주거와 생활", "건강"}),
    "직장상사": frozenset({"회사/아르바이트", "상거래 전반"}),
    "직장동료": frozenset({"회사/아르바이트", "상거래 전반"}),
    "거래처": frozenset({"상거래 전반", "회사/아르바이트"}),
    "고객": frozenset({"상거래 전반"}),
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
