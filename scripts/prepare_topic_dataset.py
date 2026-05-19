"""
주제별 텍스트 일상 대화 데이터셋 처리 스크립트

AI허브 '주제별 텍스트 일상 대화' 데이터를 읽어 두 가지 출력을 생성한다:

1. data/topic/train.csv / val.csv  — topic_segmentation_service 학습용 CSV
2. scripts/generated_topic_seeds.json — TOPIC_SEED_PHRASES 보강용 신규 키워드

사용법:
  python scripts/prepare_topic_dataset.py [--dummy] [--output-dir data/topic]
  python scripts/prepare_topic_dataset.py --suggest-seeds  # 시드 키워드만 출력

  --dummy        실제 데이터 없이 더미 데이터로 테스트
  --output-dir   출력 디렉토리 (기본: data/topic)
  --suggest-seeds  시드 키워드 분석 후 JSON 출력 (CSV 생성 생략)
  --top-n N      주제별 상위 N개 키워드 추출 (기본: 20)

환경 변수:
  DATASET_AIHUB_TOPIC_DIALOG_DIR   AI허브 주제별 텍스트 일상 대화 경로
                                   → JSON 파일이 있는 최상위 폴더

JSON 구조 (AI허브 표준):
  {"data": [
    {
      "topic": "건강/의료",
      "conversations": [
        {"text": "발화 텍스트", "speaker": "A"},
        ...
      ]
    }
  ]}

출력 CSV 컬럼:
  text, topic, source

topic 라벨은 TOPIC_SEED_PHRASES 의 30개 고정 키에 최대한 매핑.
매핑 안 되는 주제는 "기타" 처리.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 고정 30개 주제 — topic_segmentation_service.py 와 동기화
# ---------------------------------------------------------------------------

CANONICAL_TOPICS = [
    "건강/의료", "식사/음식", "날씨/계절", "직장/업무", "가족",
    "여행/외출", "쇼핑/소비", "금융/돈", "교육/공부", "취미/여가",
    "연애/관계", "집/생활", "교통/이동", "뉴스/시사", "스포츠",
    "IT/기술", "문화/예술", "반려동물", "종교/신앙", "환경/자연",
    "경조사", "친목/모임", "갈등/고민", "감사/칭찬", "안부/인사",
    "계획/약속", "추억/과거", "자녀양육", "부동산", "기타",
]

# 데이터셋의 다양한 주제명 → canonical 매핑
TOPIC_ALIAS_MAP: dict[str, str] = {
    # 건강/의료
    "건강": "건강/의료", "의료": "건강/의료", "병원": "건강/의료",
    "건강의료": "건강/의료", "건강 의료": "건강/의료",
    # 식사/음식
    "음식": "식사/음식", "식사": "식사/음식", "요리": "식사/음식",
    "맛집": "식사/음식", "음식음료": "식사/음식",
    # 날씨
    "날씨": "날씨/계절", "계절": "날씨/계절", "기후": "날씨/계절",
    # 직장/업무
    "직장": "직장/업무", "업무": "직장/업무", "일": "직장/업무",
    "회사": "직장/업무", "직업": "직장/업무",
    # 가족
    "가족": "가족", "육아": "자녀양육", "자녀": "자녀양육",
    # 여행
    "여행": "여행/외출", "여행외출": "여행/외출",
    # 쇼핑
    "쇼핑": "쇼핑/소비", "소비": "쇼핑/소비",
    # 금융
    "금융": "금융/돈", "경제": "금융/돈", "돈": "금융/돈",
    # 교육
    "교육": "교육/공부", "공부": "교육/공부", "학업": "교육/공부",
    # 취미
    "취미": "취미/여가", "여가": "취미/여가", "레저": "취미/여가",
    # 연애
    "연애": "연애/관계", "관계": "연애/관계",
    # 집
    "주거": "집/생활", "집": "집/생활",
    # 교통
    "교통": "교통/이동", "이동": "교통/이동",
    # 뉴스
    "뉴스": "뉴스/시사", "시사": "뉴스/시사", "정치": "뉴스/시사",
    # 스포츠
    "스포츠": "스포츠",
    # IT
    "it": "IT/기술", "기술": "IT/기술", "디지털": "IT/기술",
    # 문화
    "문화": "문화/예술", "예술": "문화/예술", "엔터테인먼트": "문화/예술",
    # 반려동물
    "반려동물": "반려동물", "동물": "반려동물",
    # 종교
    "종교": "종교/신앙",
    # 환경
    "환경": "환경/자연", "자연": "환경/자연",
    # 경조사
    "경조사": "경조사", "결혼": "경조사",
    # 친목
    "친목": "친목/모임", "모임": "친목/모임",
    # 갈등
    "갈등": "갈등/고민", "고민": "갈등/고민",
    # 감사
    "감사": "감사/칭찬", "칭찬": "감사/칭찬",
    # 안부
    "안부": "안부/인사", "인사": "안부/인사",
    # 계획
    "계획": "계획/약속", "약속": "계획/약속",
    # 추억
    "추억": "추억/과거", "과거": "추억/과거",
    # 부동산
    "부동산": "부동산",
}


def _map_topic(raw: str) -> str:
    normalized = raw.strip().lower().replace(" ", "")
    # exact match with canonical
    for canon in CANONICAL_TOPICS:
        if normalized == canon.lower().replace(" ", "").replace("/", ""):
            return canon
    # alias map (case-insensitive)
    for alias, canon in TOPIC_ALIAS_MAP.items():
        if alias.lower() in normalized or normalized in alias.lower():
            return canon
    return "기타"


# ---------------------------------------------------------------------------
# 더미 데이터
# ---------------------------------------------------------------------------

def make_dummy_data(n: int = 300) -> list[dict]:
    templates = [
        ("오늘 병원에 다녀왔어요 검사 결과가 좋게 나왔어요", "건강/의료"),
        ("저녁에 삼겹살 먹으러 갈 건데 같이 갈래요?", "식사/음식"),
        ("요즘 날씨가 너무 더워서 에어컨 없이는 못 살아요", "날씨/계절"),
        ("오늘 회의가 세 개나 있어서 너무 바빠요", "직장/업무"),
        ("부모님께서 올해 환갑이셔서 여행 계획 중이에요", "가족"),
        ("다음 달에 제주도 여행 가기로 했어요", "여행/외출"),
        ("세일 기간에 옷을 많이 샀는데 카드값이 걱정이에요", "쇼핑/소비"),
        ("요즘 주식이 많이 올라서 기분이 좋아요", "금융/돈"),
        ("내일 수학 시험인데 공부가 잘 안돼요", "교육/공부"),
        ("주말에 등산을 다녀왔는데 너무 힘들었어요", "취미/여가"),
        ("남자친구랑 사소한 일로 다퉜는데 어떡하죠", "연애/관계"),
        ("전세 계약이 곧 끝나서 새 집을 알아보고 있어요", "집/생활"),
        ("출퇴근 시간에 버스가 너무 막혀서 힘들어요", "교통/이동"),
        ("오늘 뉴스에서 선거 결과 봤어요?", "뉴스/시사"),
        ("어제 야구 경기 봤어요? 우리팀이 이겼어요", "스포츠"),
        ("새 스마트폰 샀는데 기능이 너무 좋아요", "IT/기술"),
        ("요즘 드라마가 너무 재밌어서 매일 봐요", "문화/예술"),
        ("강아지가 아파서 동물병원에 다녀왔어요", "반려동물"),
        ("이번 주말에 결혼식이 세 개나 있어요", "경조사"),
        ("오래된 친구들이랑 오랜만에 모임을 가졌어요", "친목/모임"),
        ("요즘 회사에서 스트레스를 많이 받고 있어요", "갈등/고민"),
        ("도와줘서 정말 고마워요 덕분에 해결됐어요", "감사/칭찬"),
        ("오랜만이에요 요즘 어떻게 지내요?", "안부/인사"),
        ("다음 주에 여행 계획이 있어서 기대돼요", "계획/약속"),
        ("어릴 때 여기서 많이 놀았던 기억이 나요", "추억/과거"),
    ]
    rows = []
    for i in range(n):
        text, topic = templates[i % len(templates)]
        rows.append({"text": f"{text} {i}", "topic": topic, "source": "dummy"})
    return rows


# ---------------------------------------------------------------------------
# 실제 데이터셋 로더
# ---------------------------------------------------------------------------

def load_aihub_topic_dialog(base_dir: Path) -> list[dict]:
    """AI허브 주제별 텍스트 일상 대화 로더

    구조 A (data 래퍼):
      {"data": [{"topic": "...", "conversations": [{"text": "..."}, ...]}]}

    구조 B (dialogues 래퍼):
      {"dialogues": [{"topic": "...", "utterances": [{"text": "..."}, ...]}]}

    구조 C (리스트):
      [{"topic": "...", "text": "..."}, ...]
    """
    rows = []

    for json_file in base_dir.rglob("*.json"):
        try:
            with json_file.open(encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("주제대화 JSON 읽기 실패: %s — %s", json_file, e)
            continue

        # 구조 C — 발화 리스트 직접
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                text = (item.get("text") or item.get("utterance") or "").strip()
                raw_topic = (item.get("topic") or item.get("category") or "").strip()
                if text and raw_topic:
                    rows.append({
                        "text": text,
                        "topic": _map_topic(raw_topic),
                        "source": "aihub_topic_dialog",
                    })
            continue

        if not isinstance(data, dict):
            continue

        # 구조 A: data 래퍼
        for item in data.get("data", []):
            raw_topic = (item.get("topic") or item.get("category") or "").strip()
            for utt in item.get("conversations", []) or item.get("utterances", []):
                text = (utt.get("text") or utt.get("utterance") or "").strip()
                if text and raw_topic:
                    rows.append({
                        "text": text,
                        "topic": _map_topic(raw_topic),
                        "source": "aihub_topic_dialog",
                    })

        # 구조 B: dialogues 래퍼
        for item in data.get("dialogues", []):
            raw_topic = (item.get("topic") or item.get("category") or "").strip()
            for utt in item.get("utterances", []):
                text = (utt.get("text") or utt.get("utterance") or "").strip()
                if text and raw_topic:
                    rows.append({
                        "text": text,
                        "topic": _map_topic(raw_topic),
                        "source": "aihub_topic_dialog",
                    })

    logger.info("AI허브 주제대화: %d건 로드 (from %s)", len(rows), base_dir)
    return rows


# ---------------------------------------------------------------------------
# 시드 키워드 추출
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "이", "가", "을", "를", "은", "는", "와", "과", "에", "의", "도",
    "에서", "으로", "로", "이고", "이다", "있다", "없다", "하다", "한다",
    "그", "그리고", "그런데", "하지만", "그래서", "또", "더", "아", "어",
    "네", "예", "아니요", "어떤", "어떻게", "왜", "무슨", "저는", "나는",
    "우리", "제가", "저도", "것", "거", "것이", "게", "그게", "이게",
    "좀", "너무", "정말", "진짜", "많이", "같아", "같이", "뭐", "잖아",
}

_TOKEN_RE = re.compile(r"[가-힣a-zA-Z0-9]+")


def extract_seed_keywords(rows: list[dict], top_n: int = 20) -> dict[str, list[str]]:
    """주제별 발화에서 TF-IDF 근사 키워드 추출 (단순 빈도 기반)"""
    by_topic: dict[str, list[str]] = defaultdict(list)
    global_freq: Counter[str] = Counter()
    topic_freq: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        tokens = [t for t in _TOKEN_RE.findall(row["text"]) if t not in _STOPWORDS and len(t) >= 2]
        topic = row["topic"]
        by_topic[topic].extend(tokens)
        for t in tokens:
            global_freq[t] += 1
            topic_freq[topic][t] += 1

    total_docs = max(len(rows), 1)
    result: dict[str, list[str]] = {}
    for topic, freq in topic_freq.items():
        # TF-IDF 근사: topic 내 빈도 / 전체 빈도
        scored = []
        for word, cnt in freq.items():
            tf = cnt / max(len(by_topic[topic]), 1)
            idf = total_docs / max(global_freq[word], 1)
            scored.append((word, tf * idf))
        scored.sort(key=lambda x: -x[1])
        result[topic] = [w for w, _ in scored[:top_n]]

    return result


# ---------------------------------------------------------------------------
# 파이프라인
# ---------------------------------------------------------------------------

def dedup(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    result = []
    for r in rows:
        if r["text"] not in seen:
            seen.add(r["text"])
            result.append(r)
    return result


def split_train_val(rows: list[dict], val_ratio: float = 0.2) -> tuple[list[dict], list[dict]]:
    random.shuffle(rows)
    split = int(len(rows) * (1 - val_ratio))
    return rows[:split], rows[split:]


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "topic", "source"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("저장: %s (%d건)", path, len(rows))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="주제별 일상 대화 데이터셋 처리")
    parser.add_argument("--dummy", action="store_true", help="더미 데이터로 테스트")
    parser.add_argument("--output-dir", default="data/topic", help="출력 디렉토리")
    parser.add_argument("--suggest-seeds", action="store_true", help="시드 키워드만 분석 후 JSON 출력")
    parser.add_argument("--top-n", type=int, default=20, help="주제별 상위 키워드 수")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)

    if args.dummy:
        logger.info("더미 모드")
        all_rows = make_dummy_data(300)
    else:
        all_rows = []
        if d := os.environ.get("DATASET_AIHUB_TOPIC_DIALOG_DIR"):
            all_rows += load_aihub_topic_dialog(Path(d))
        else:
            logger.error("DATASET_AIHUB_TOPIC_DIALOG_DIR 미설정. --dummy 또는 환경변수를 확인하세요.")
            raise SystemExit(1)

    all_rows = dedup(all_rows)
    logger.info("중복 제거 후: %d건", len(all_rows))

    if args.suggest_seeds:
        seeds = extract_seed_keywords(all_rows, top_n=args.top_n)
        seeds_path = Path("scripts/generated_topic_seeds.json")
        seeds_path.write_text(json.dumps(seeds, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("시드 키워드 저장: %s", seeds_path)
        logger.info("topic_segmentation_service.py 의 TOPIC_SEED_PHRASES 에 병합하세요.")
        return

    train, val = split_train_val(all_rows)
    write_csv(train, output_dir / "train.csv")
    write_csv(val, output_dir / "val.csv")

    # 시드 키워드도 함께 생성
    seeds = extract_seed_keywords(all_rows, top_n=args.top_n)
    seeds_path = output_dir / "suggested_topic_seeds.json"
    seeds_path.write_text(json.dumps(seeds, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("시드 키워드 제안: %s", seeds_path)
    logger.info("완료 — train=%d, val=%d", len(train), len(val))


if __name__ == "__main__":
    main()
