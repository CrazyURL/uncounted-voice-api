import re
from typing import Optional


# 한국 성씨 목록 (상위 빈도)
KOREAN_SURNAMES = (
    "김", "이", "박", "최", "정", "강", "조", "윤", "장", "임",
    "한", "오", "서", "신", "권", "황", "안", "송", "류", "전",
    "홍", "고", "문", "양", "손", "배", "백", "허", "유", "남",
    "심", "노", "하", "곽", "성", "차", "주", "우", "구", "민",
)

# 성씨와 겹치는 고빈도 일상어 제외 (2글자 접두사)
# 이 접두사로 시작하는 모든 매칭을 건너뛴다.
# 예: "정신" 등록 → "정신", "정신과", "정신적" 모두 건너뜀
_NAME_EXCLUDE_PREFIX = frozenset({
    # 이~
    "이런", "이제", "이거", "이건", "이게", "이걸", "이것", "이날", "이번",
    "이미", "이후", "이전", "이상", "이하", "이유", "이름", "이용", "이동",
    "이야", "이렇", "이리", "이래", "이때", "이내", "이틀", "이해",
    "이어", "이를", "이른", "이룬", "이뤄", "이끌", "이외", "이며", "이면",
    "이라", "이랑", "이요", "이에", "이든", "이니", "이나", "이다", "이도",
    # 정~
    "정말", "정도", "정리", "정보", "정신", "정상", "정확", "정식", "정기",
    "정해", "정한", "정하", "정작", "정오", "정문", "정답", "정비", "정산",
    # 강~
    "강한", "강해", "강화", "강조", "강력", "강당", "강의", "강물", "강변",
    "강남", "강북", "강서", "강동", "강원",
    # 하~
    "하는", "하고", "하면", "하게", "하지", "하자", "하다", "하며", "하니",
    "하여", "하루", "하반", "하나", "하늘", "하얀", "하물", "하필", "하소",
    "하던", "하더", "하도", "하긴", "하기", "하네", "하세",
    "하락", "하루", "하산", "하수", "하위", "하차", "하한",
    # 조~
    "조금", "조용", "조건", "조사", "조치", "조차", "조절", "조만", "조기",
    # 장~
    "장소", "장면", "장기", "장래", "장비", "장점", "장애", "장난", "장마",
    # 한~
    "한번", "한데", "한참", "한편", "한동", "한때", "한층", "한결", "한마",
    "한다", "한두", "한쪽", "한테",
    # 안~
    "안녕", "안전", "안내", "안정", "안쪽", "안에", "안과", "안개",
    "안되", "안돼", "안나", "안해",
    # 오~
    "오늘", "오전", "오후", "오래", "오히", "오직", "오른", "오면", "오고",
    # 서~
    "서로", "서울", "서쪽", "서비", "서류", "서둘",
    # 고~
    "고객", "고민", "고생", "고마", "고장", "고향", "고르", "고른",
    # 송~
    "송도", "송이", "송출",
    # 문~
    "문제", "문의", "문서", "문화", "문자", "문득", "문밖",
    # 남~
    "남자", "남편", "남쪽", "남는", "남기", "남은", "남녀", "남부",
    # 배~
    "배우", "배달", "배경", "배치",
    # 신~
    "신경", "신청", "신용", "신호", "신규", "신발", "신기", "신선",
    # 손~
    "손님", "손잡", "손으", "손해",
    # 백~
    "백만", "백원", "백화",
    # 최~
    "최근", "최고", "최대", "최소", "최종", "최선", "최초", "최저",
    # 윤~
    "윤리",
    # 임~
    "임시", "임대", "임금",
    # 권~
    "권리", "권한",
    # 양~
    "양쪽", "양해",
    # 유~
    "유지", "유리", "유일", "유사", "유명", "유효",
    # 차~
    "차이", "차라", "차량", "차로", "차원", "차례",
    # 주~
    "주로", "주요", "주의", "주변", "주민", "주간", "주말", "주소", "주어",
    # 민~
    "민간", "민원",
    # 성~
    "성격", "성과", "성공", "성장",
    # 구~
    "구체", "구간", "구매", "구역",
    # 전~
    "전화", "전혀", "전체", "전부", "전달", "전문", "전국", "전자", "전기",
    "전환", "전통", "전략", "전날", "전반", "전용", "전망", "전제", "전선",
    # 홍~
    "홍보",
    # 황~
    "황당",
    # 노~
    "노력", "노래",
    # 배~  (보강)
    "배고",
    # 허~
    "허락",
    # 곽~  (드물지만)
    # 우~
    "우리", "우선",
})

# 한국 성씨 정규식 (컴파일됨) — 40개 성씨를 | 로 연결
_SURNAME_PATTERN = re.compile(
    f"({'|'.join(re.escape(s) for s in KOREAN_SURNAMES)})([가-힣]{{1,2}})"
)

# 호칭/직함 (이름 뒤에 올 수 있는 단어)
# 2글자 이름(성+1글자)은 다음 호칭이 와야 PII로 판정.
# 3글자 이름(성+2글자)은 호칭 무관 PII로 판정.
_HONORIFICS = (
    # 존칭 + 직함
    "씨", "님", "선생", "교수", "박사", "사장", "대표", "이사",
    "부장", "차장", "과장", "대리", "사원", "주임", "팀장", "실장",
    "원장", "국장", "처장", "위원", "총장", "학장", "선배", "후배",
    # 일상 호칭 (실제 통화에서 매우 흔함 — 알파 샘플 "김용철 형" 케이스)
    "형", "형님", "누나", "언니", "오빠", "동생",
    "어머니", "아버지", "엄마", "아빠", "할머니", "할아버지",
    "아주머니", "아저씨", "삼촌", "이모", "고모", "외삼촌",
)

# PII 패턴 정의 (순서 중요: 구체적인 패턴이 먼저)
PII_PATTERNS = [
    # 주민등록번호: 900101-1234567
    (
        re.compile(r"(\d{6})\s*[-]\s*([1-4]\d{6})"),
        lambda m: f"{m.group(1)}-*******",
        "주민등록번호",
    ),
    # 운전면허번호: 12-34-567890-12
    (
        re.compile(r"(\d{2})-(\d{2})-(\d{6})-(\d{2})"),
        lambda m: "**-**-******-**",
        "운전면허번호",
    ),
    # 여권번호: M12345678
    (
        re.compile(r"([A-Z])(\d{8})"),
        lambda m: f"{m.group(1)}********",
        "여권번호",
    ),
    # 카드번호: 1234-5678-9012-3456 또는 공백 구분
    (
        re.compile(r"(\d{4})[\s-](\d{4})[\s-](\d{4})[\s-](\d{4})"),
        lambda m: f"{m.group(1)}-****-****-{m.group(4)}",
        "카드번호",
    ),
    # 이메일
    (
        re.compile(r"([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"),
        lambda m: f"{m.group(1)[0]}***@{m.group(2)}",
        "이메일",
    ),
    # 전화번호: 010-1234-5678, 02-123-4567 등
    (
        re.compile(r"(0\d{1,2})[\s.-](\d{3,4})[\s.-](\d{4})"),
        lambda m: f"{m.group(1)}-****-{m.group(3)}",
        "전화번호",
    ),
    # 전화번호 (붙여쓰기): 01012345678 / 0111234567 등 — 010~019 prefix 명시
    # 011~019 는 구형 PCS 번호로 일부 사용자 잔존. 뒷자리 7~8자리 가능.
    (
        re.compile(r"(01[0-9])(\d{3,4})(\d{4})"),
        lambda m: f"{m.group(1)}****{m.group(3)}",
        "전화번호",
    ),
    # 계좌번호: 11~14자리 연속 숫자 (단어 경계, 최소 11자리로 상향)
    (
        re.compile(r"\b(\d{3})(\d{8,11})\b"),
        lambda m: f"{m.group(1)}{'*' * len(m.group(2))}",
        "계좌번호",
    ),
    # IP주소
    (
        re.compile(r"\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b"),
        lambda m: "***.***.***.***",
        "IP주소",
    ),
]


# ── 음성 전사형 PII (STT 한글 숫자어) — PII-1A Category 1 ─────────────────
# 한국어 통화 STT 는 전화번호를 "공일공 일이삼사 오육칠팔" 처럼 한글 숫자어로 전사한다.
# 위 \d 정규식은 이를 못 잡으므로 별도 패스로 탐지한다. 이 패스는 detect_pii_spans 에
# include_spoken_pii=True 일 때만 동작하며, 텍스트/오디오 마스킹 경로(mask_pii 등)는
# 기본값 False 로 호출하므로 영향받지 않는다 (마스킹 freeze).
_SINO_DIGIT = {
    "공": "0", "영": "0", "일": "1", "이": "2", "삼": "3",
    "사": "4", "오": "5", "육": "6", "칠": "7", "팔": "8", "구": "9",
}
_NATIVE_DIGIT = {
    "하나": "1", "둘": "2", "셋": "3", "넷": "4", "다섯": "5",
    "여섯": "6", "일곱": "7", "여덟": "8", "아홉": "9",
}
# 토큰화 우선순위: 2글자 native 를 1글자 sino 보다 먼저 (regex 교체는 순서 우선).
_DIGIT_WORD_ALT = (
    "다섯|여섯|일곱|여덟|아홉|하나|둘|셋|넷|"
    "공|영|일|이|삼|사|오|육|칠|팔|구"
)
_DIGIT_WORD = re.compile(_DIGIT_WORD_ALT)
# 휴대폰 정규화 기준: 01X + 10~11자리. 0(공/영) 으로 시작해야만 매칭.
_SPOKEN_MOBILE_RE = re.compile(r"^01[0-9]\d{7,8}$")
# 휴대폰 자릿수 = 단어 수(각 단어=1자리). 긴 창(11)을 먼저 시도.
_SPOKEN_PHONE_WINDOWS = (11, 10)


def _spoken_word_to_digit(word: str) -> str:
    """한글 숫자어 1단어 → 숫자 1자리."""
    return _NATIVE_DIGIT.get(word) or _SINO_DIGIT[word]


def _spoken_digit_runs(text: str) -> list[list[re.Match]]:
    """텍스트의 숫자어 토큰을 공백으로만 이어진 maximal run 으로 묶는다."""
    runs: list[list[re.Match]] = []
    cur: list[re.Match] = []
    prev_end = -1
    for m in _DIGIT_WORD.finditer(text):
        # 직전 토큰과의 사이가 공백뿐일 때만 같은 run (조사·어미 등이 끼면 분리).
        if cur and text[prev_end:m.start()].strip() == "":
            cur.append(m)
        else:
            if cur:
                runs.append(cur)
            cur = [m]
        prev_end = m.end()
    if cur:
        runs.append(cur)
    return runs


def detect_spoken_phone_spans(text: str) -> list[dict]:
    """한글 숫자어로 전사된 휴대폰 번호를 탐지한다 (offset 보존).

    discriminator (오탐 방지):
      - 숫자어가 공백으로만 이어진 run 안에서 10~11단어 창을 슬라이드
      - 창의 정규화 결과가 01X + 10~11자리(휴대폰)여야 함 → 0(공/영) 시작 강제
    세는 말("하나 둘 셋...")·일상 음절·조사는 0 으로 시작하지 않거나 자릿수가 맞지
    않아 배제된다. 인접 조사("...팔 이에요"의 '이')는 창 밖으로 떨어진다.
    """
    spans: list[dict] = []
    for run in _spoken_digit_runs(text):
        digits = [_spoken_word_to_digit(m.group(0)) for m in run]
        i = 0
        n = len(run)
        while i < n:
            matched = False
            for size in _SPOKEN_PHONE_WINDOWS:
                if i + size > n:
                    continue
                window = "".join(digits[i:i + size])
                if _SPOKEN_MOBILE_RE.match(window):
                    start = run[i].start()
                    end = run[i + size - 1].end()
                    spans.append({
                        "type": "전화번호",
                        "char_start": start,
                        "char_end": end,
                        "matched_text": text[start:end],
                        # per-span tier hint (pii_confidence 가 type 기본값 대신 사용).
                        "confidence": 0.95,
                        "high_precision_pattern": True,
                    })
                    i += size  # 겹치지 않게 창 뒤로 이동
                    matched = True
                    break
            if not matched:
                i += 1
    return spans


def _matches_exclude_prefix(surname: str, given: str) -> bool:
    """성+이름이 제외 목록의 접두사와 일치하는지 확인한다.

    "정신"이 제외 목록에 있으면 "정신", "정신과", "정신적" 모두 제외된다.
    """
    full = surname + given
    # 정확히 일치 (2글자 매칭: 성+1글자)
    if full in _NAME_EXCLUDE_PREFIX:
        return True
    # 접두사 일치 (3글자 매칭: 성+2글자 중 성+첫글자가 제외 목록에 있으면)
    if len(given) >= 2 and (surname + given[0]) in _NAME_EXCLUDE_PREFIX:
        return True
    return False


def _is_likely_name_with_context(
    surname: str, given: str, before: str, after: str
) -> bool:
    """앞뒤 문맥을 포함해 이름 여부를 판단한다."""

    # 1. 제외 목록 (접두사 매칭)
    if _matches_exclude_prefix(surname, given):
        return False

    # 2. 단어 중간에서 매칭된 경우 제외
    #    이름은 공백/문장부호 뒤에 오거나 문장 시작이어야 함
    if before and before[-1] not in " \t\n,.:;!?()\"'·…—-~":
        return False

    # 3. 2글자 이름 (성+1글자): 뒤에 호칭이 올 때만 이름으로 판단
    if len(given) == 1:
        after_stripped = after.lstrip()
        for h in _HONORIFICS:
            if after_stripped.startswith(h):
                return True
        return False

    # 4. 3글자 이름 (성+2글자): 제외 목록을 통과하고 단어 시작이면 이름으로 간주
    return True


# ── 통화 등급 (049 v5 정합) ────────────────────────────────────────────
# 'premium': 양측 개인. enable_name_masking은 호출자 옵션.
# 'standard': 한쪽이 기업/가상번호 (직무 발화자 비식별화 의무, 약관 v1.2 제5조의2 3항).
#             enable_name_masking을 호출자 값과 무관하게 True로 강제.
# 'excluded': 거래 불가 통화 — 본 마스커는 호출되지 않으나, 호출 시 standard와 동일 처리.
CallGrade = str  # Literal['premium', 'standard', 'excluded']


def _resolve_name_masking(
    enable_name_masking: bool, grade: Optional[CallGrade]
) -> bool:
    """STANDARD/EXCLUDED 등급은 enable_name_masking을 강제 True로 박는다.

    약관 v1.2 제5조의2 3항: 직무 발화자 측 실명·소속·직책 등 식별 가능 단어의 강화 마스킹.
    """
    if grade in ("standard", "excluded"):
        return True
    return enable_name_masking


def detect_pii_spans(
    text: str,
    enable_name_masking: bool = False,
    grade: Optional[CallGrade] = None,
    *,
    include_spoken_pii: bool = False,
) -> list[dict]:
    """텍스트에서 PII의 위치(span)를 감지하고 리스트로 반환한다.

    Args:
        text: 입력 텍스트
        enable_name_masking: 이름 마스킹 활성화 여부 (premium 기본 False)
        grade: 통화 등급 ('premium' / 'standard' / 'excluded'). standard/excluded는
               enable_name_masking을 자동 True로 강제.
    """
    enable_name_masking = _resolve_name_masking(enable_name_masking, grade)
    spans = []

    # 1. PII_PATTERNS 감지
    for pattern, _, label in PII_PATTERNS:
        for m in pattern.finditer(text):
            spans.append({
                "type": label,
                "char_start": m.start(),
                "char_end": m.end(),
                "matched_text": m.group(0)
            })

    # 2. 이름 마스킹 감지
    if enable_name_masking:
        for m in _SURNAME_PATTERN.finditer(text):
            s = m.group(1)
            g = m.group(2)
            before = text[:m.start()]
            after = text[m.end():]

            if _is_likely_name_with_context(s, g, before, after):
                spans.append({
                    "type": "이름",
                    "char_start": m.start(),
                    "char_end": m.end(),
                    "matched_text": m.group(0)
                })

    # 3. 음성 전사형 PII (candidate 전용 — 마스킹 경로는 기본 False 로 미동작)
    if include_spoken_pii:
        spans.extend(detect_spoken_phone_spans(text))

    return spans


def mask_pii(
    text: str,
    enable_name_masking: bool = False,
    grade: Optional[CallGrade] = None,
) -> dict:
    """텍스트에서 PII를 마스킹하고 결과를 반환한다.

    Args:
        text: 입력 텍스트
        enable_name_masking: 이름 마스킹 활성화 여부 (premium 기본 False)
        grade: 통화 등급 (049 v5). standard/excluded는 enable_name_masking 자동 True.
    """
    enable_name_masking = _resolve_name_masking(enable_name_masking, grade)
    spans = detect_pii_spans(text, enable_name_masking)

    # 중첩된 span 처리: 시작 위치 순, 길이 역순으로 정렬
    spans.sort(key=lambda x: (x["char_start"], -(x["char_end"] - x["char_start"])))

    # 중첩 제거 (먼저 나온 긴 패턴 우선)
    non_overlapping = []
    last_end = -1
    for span in spans:
        if span["char_start"] >= last_end:
            non_overlapping.append(span)
            last_end = span["char_end"]

    # 역순 치환 (index 보존)
    non_overlapping.sort(key=lambda x: x["char_start"], reverse=True)

    masked_chars = list(text)
    detected_summary = {}

    for span in non_overlapping:
        label = span["type"]
        matched = span["matched_text"]
        start = span["char_start"]
        end = span["char_end"]

        # 치환값 계산
        replacer_val = None
        if label == "이름":
            s = matched[0]
            g = matched[1:]
            replacer_val = f"{s}{'O' * len(g)}"
        else:
            for p, r, l in PII_PATTERNS:
                if l == label:
                    m = p.fullmatch(matched)
                    if m:
                        replacer_val = r(m) if callable(r) else r
                        break
            if not replacer_val:
                replacer_val = "*" * len(matched)

        # 치환 적용
        masked_chars[start:end] = list(replacer_val)

        # 요약 업데이트
        detected_summary[label] = detected_summary.get(label, 0) + 1

    # PII_PATTERNS 선언 순서 + 이름을 마지막에 배치 (중복 라벨 제거, 하위 호환)
    seen = set()
    pattern_order: list[str] = []
    for _, _, label in PII_PATTERNS:
        if label not in seen:
            seen.add(label)
            pattern_order.append(label)
    pattern_order.append("이름")
    pii_detected = [
        {"type": label, "count": detected_summary[label]}
        for label in pattern_order if label in detected_summary
    ]

    return {
        "masked_text": "".join(masked_chars),
        "pii_detected": pii_detected,
        "total_masked": sum(detected_summary.values()),
    }


def mask_segments(
    segments: list[dict],
    enable_name_masking: bool = False,
    grade: Optional[CallGrade] = None,
) -> list[dict]:
    """세그먼트 리스트의 텍스트를 마스킹한다.

    여러 세그먼트에서 같은 PII 유형이 감지되면 count를 합산하여
    유형별 단일 항목으로 반환한다 (응답 스키마 PIIDetectedItem 계약).
    이전에는 list.extend()로 중복 dict 항목이 누적되어 응답에서
    유형이 중복되거나 일부 클라이언트에서 처리 시 마지막 값만 보였음.

    Args:
        segments: 세그먼트 리스트
        enable_name_masking: 이름 마스킹 활성화 여부 (premium 기본 False)
        grade: 통화 등급 (049 v5). standard/excluded는 enable_name_masking 자동 True.
    """
    enable_name_masking = _resolve_name_masking(enable_name_masking, grade)
    type_count_map: dict[str, int] = {}
    for seg in segments:
        result = mask_pii(seg.get("text", ""), enable_name_masking)
        seg["text"] = result["masked_text"]
        for item in result["pii_detected"]:
            t = item["type"]
            type_count_map[t] = type_count_map.get(t, 0) + int(item["count"])

    # PII_PATTERNS 선언 순서 유지 (이름은 항상 마지막) — 응답 일관성
    seen: set[str] = set()
    pattern_order: list[str] = []
    for _, _, label in PII_PATTERNS:
        if label not in seen:
            seen.add(label)
            pattern_order.append(label)
    pattern_order.append("이름")
    return [
        {"type": label, "count": type_count_map[label]}
        for label in pattern_order
        if label in type_count_map
    ]
