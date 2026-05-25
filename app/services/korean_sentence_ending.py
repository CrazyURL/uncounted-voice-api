"""한국어 문장 종결성 판정 — 규칙 기반 휴리스틱.

형태소 분석기를 두지 않고(저장소 의존성 추가 회피) 마지막 어절의 접미사로
'문장이 종결되었는가'를 근사한다. 발화 병합(Segmenter v2)에서 문장 경계를
보존하고, 진단 스크립트에서 종결성 비율을 집계하는 데 공용으로 쓴다.

휴리스틱이므로 재현율 손실(예: "그래")과 일부 오탐(예: 부사 "다")을 감수한다.
"""

import re

# 마지막 어절에서 종결 판정 전에 제거할 후행 기호 (구두점·인용부호·말줄임표)
_TRAILING_PUNCT = ".,!?\"'~…·:;)]}"

# 고정밀 우선 종결 접미사 (구어체 평서/의문/존댓말).
# 길이 긴 것을 앞에 두어 부분 매칭 모호성을 줄인다.
# 연결어미(예: "-는데", "-니까", "-다가")는 의도적으로 제외한다.
_SENTENCE_ENDINGS: tuple[str, ...] = (
    "습니다", "ㅂ니다", "입니다", "니다",
    "거든요", "잖아요", "는데요", "더라고요", "ㄹ게요", "을게요", "ㄹ까요", "을까요",
    "에요", "예요", "어요", "아요", "여요", "해요", "세요", "게요", "네요", "구요", "군요",
    "나요", "까요", "지요",
    "구나", "더라", "거든",
    "요", "죠", "다", "까", "네", "군",
)

# 마지막 어절이 이 연결어미로 끝나면 '미완결'로 본다 (병합 후보 신호).
_CONNECTIVE_SUFFIXES: tuple[str, ...] = (
    "는데", "은데", "ㄴ데", "니까", "다가", "면서", "는데도", "지만", "거나",
    "고서", "어서", "아서", "여서", "려고", "으려고", "도록", "든지", "처럼",
    "그래서", "그러나", "그리고", "그런데", "또는", "및",
)


def _last_token(text: str) -> str:
    """후행 구두점을 제거한 마지막 공백 단위 어절을 반환."""
    if not text:
        return ""
    cleaned = text.strip().rstrip(_TRAILING_PUNCT).strip()
    if not cleaned:
        return ""
    return cleaned.split()[-1]


def ends_with_sentence_ending(text: str) -> bool:
    """마지막 어절이 문장 종결 접미사로 끝나면 True (휴리스틱).

    연결어미로 끝나면 우선적으로 False를 반환해 미완결로 본다.
    """
    token = _last_token(text)
    if not token:
        return False
    if token.endswith(_CONNECTIVE_SUFFIXES):
        return False
    return token.endswith(_SENTENCE_ENDINGS)


def ends_with_connective(text: str) -> bool:
    """마지막 어절이 연결어미/접속 표현으로 끝나면 True (미완결 신호)."""
    token = _last_token(text)
    if not token:
        return False
    return token.endswith(_CONNECTIVE_SUFFIXES)


# 진단용: 마지막 어절이 조사로 끝나는 경우(불완전 구) 근사 탐지.
_PARTICLE_SUFFIXES: tuple[str, ...] = (
    "은", "는", "이", "가", "을", "를", "에", "에서", "에게", "한테", "께",
    "으로", "로", "와", "과", "랑", "이랑", "의", "도", "만", "까지", "부터",
    "보다", "처럼", "마다", "조차", "마저", "밖에",
)


def ends_with_particle(text: str) -> bool:
    """마지막 어절이 조사로 끝나면 True (불완전 구 근사 — 진단용)."""
    token = _last_token(text)
    if not token:
        return False
    if token.endswith(_SENTENCE_ENDINGS):
        return False
    return token.endswith(_PARTICLE_SUFFIXES)
