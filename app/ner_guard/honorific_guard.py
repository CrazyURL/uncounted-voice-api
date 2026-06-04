# -*- coding: utf-8 -*-
"""Nim-Guard — 사물+'님' 인칭 오용 적출 (확신형 환각 검수 플래그).

통계 신호(prob/energy)가 못 잡는 "공인인증서님"류를 문법 규칙으로 적출(설계 §6).
Kiwi 형태소 분석 필요 — 지연 import, 미설치 시 빈 결과(크래시 금지).
자동수정 아님 → 검수 플래그(red queue)용.
"""
import logging

logger = logging.getLogger(__name__)

# 인간 호칭 allowlist (앞 단어가 이거면 PASS)
HUMAN_TITLES = frozenset(
    "수석 책임 매니저 선생 고객 기사 팀장 사장 사용자 부장 차장 과장 대리 주임 실장 "
    "이사 상무 전무 부회장 대표 원장 교수 담당자 관리자 상담원 센터장 단장 사모 "
    "형 누나 오빠 언니 어머 아버 어머니 아버지 할머니 할아버지 따님 아드".split()
)
# 이름 판정용 성씨(짧은 NNP가 성+이름 패턴이면 사람으로 간주)
_SURNAMES = frozenset(
    "김 이 박 최 정 강 조 윤 장 임 한 오 서 신 권 황 안 송 전 홍 유 고 문 양 손 배 백 허 남 심 노 하 곽 성 차".split()
)

_kiwi = None
_kiwi_failed = False


def _get_kiwi():
    global _kiwi, _kiwi_failed
    if _kiwi is None and not _kiwi_failed:
        try:
            from kiwipiepy import Kiwi
            _kiwi = Kiwi()
        except Exception as e:  # 미설치/로드실패 → 비활성(크래시 금지)
            _kiwi_failed = True
            logger.warning("Nim-Guard 비활성 (kiwipiepy 로드 실패): %s", repr(e)[:80])
    return _kiwi


def _is_person_name(noun: str) -> bool:
    return 2 <= len(noun) <= 4 and noun[0] in _SURNAMES and all("가" <= c <= "힣" for c in noun)


def detect_inanimate_honorific(text: str) -> list[str]:
    """'사물+님' 의심 구절 목록 반환. Kiwi 없으면 []."""
    kiwi = _get_kiwi()
    if kiwi is None or not text:
        return []
    toks = kiwi.tokenize(text)
    flags: list[str] = []
    for i in range(1, len(toks)):
        prev, cur = toks[i - 1], toks[i]
        if cur.form == "님" and cur.tag == "XSN":
            noun, pos = prev.form, prev.tag
            if noun in HUMAN_TITLES:
                continue                                   # 인간 호칭 → PASS
            if pos == "NNP" and _is_person_name(noun):
                continue                                   # 이름+님 → PASS
            if pos in ("NNG", "NNP"):
                flags.append(noun + "님")                  # 사물+님 → FLAG
    return flags
