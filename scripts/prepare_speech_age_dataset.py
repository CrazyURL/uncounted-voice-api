"""
말투 연령 데이터셋 준비 스크립트

AI허브 "연령대별 특징적 발화" 데이터셋을 읽어 KcELECTRA 말투 연령 헤드 학습용
train/val CSV 를 생성한다.

사용법:
  python scripts/prepare_speech_age_dataset.py [--dummy] [--output-dir data/speech_age]

  --dummy      실제 데이터셋 없이 더미 데이터로 동작 (테스트용)
  --output-dir 출력 디렉토리 (기본: data/speech_age)
  --no-balance 클래스 균형 언더샘플링 생략

환경 변수 (--dummy 없이 사용 시):
  DATASET_AIHUB_SPEECH_AGE_DIR    AI허브 연령대별 특징적 발화 데이터셋 경로
                                  → JSON/CSV 파일이 있는 최상위 폴더

출력:
  data/speech_age/train.csv  — text,age_group,source
  data/speech_age/val.csv
  data/speech_age/dataset_stats.json

연령 그룹 4종:
  20대 | 30대 | 40대 | 50대+

AI허브 연령대별 특징적 발화 지원 구조:

  구조 A — 발화 단위 JSON 리스트:
    [
      { "text": "발화 텍스트", "age": "20대" },
      ...
    ]

  구조 B — 대화 단위 JSON:
    {
      "utterances": [
        { "text": "발화 텍스트", "speaker_age": "30대" },
        ...
      ]
    }

  구조 C — CSV 파일:
    text,age_group
    "발화 텍스트",20대
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

AGE_GROUPS = ["20대", "30대", "40대", "50대+"]

# AI허브 원본 레이블 → 통합 레이블 정규화
AGE_LABEL_MAP: dict[str, str] = {
    "20대": "20대", "20s": "20대", "20": "20대",
    "30대": "30대", "30s": "30대", "30": "30대",
    "40대": "40대", "40s": "40대", "40": "40대",
    "50대": "50대+", "50s": "50대+", "50": "50대+",
    "60대": "50대+", "60s": "50대+", "60": "50대+",
    "70대": "50대+", "70대 이상": "50대+",
    "50대+": "50대+", "50대이상": "50대+",
}


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# 더미 데이터
# ---------------------------------------------------------------------------

def make_dummy_data(n: int = 600) -> list[dict]:
    templates = [
        # 20대 특징: 줄임말, 이모지 대신 감탄사, 격식 없음
        ("야 진짜 완전 웃겼어 ㅋㅋ 거기서 그러면 어떡해", "20대"),
        ("오늘 알바 진짜 힘들었는데 그냥 집 가고 싶다", "20대"),
        ("야 그 영화 봤어? 대박 쩔었잖아", "20대"),
        ("졸업하고 취업 준비하는 게 생각보다 빡세다", "20대"),
        # 30대 특징: 일상·육아·직장, 보통 구어체
        ("요즘 야근이 너무 많아서 몸이 힘들어요", "30대"),
        ("아이 유치원 보내고 나서 잠깐 쉬는 시간이 소중하더라고요", "30대"),
        ("이번에 팀장이 바뀌면서 분위기가 달라졌어요", "30대"),
        ("집 대출 이자가 올라서 부담이 커졌어요", "30대"),
        # 40대 특징: 존댓말 혼용, 현실적 고민
        ("요즘 무릎이 좀 안 좋아서 병원을 다니고 있습니다", "40대"),
        ("애들 교육비가 장난이 아니에요 진짜로", "40대"),
        ("부모님 건강이 걱정되는 나이가 됐어요", "40대"),
        ("직장 생활 이십 년 넘으니까 슬슬 지치는 것 같아요", "40대"),
        # 50대+ 특징: 격식체, 경험 중심
        ("젊을 때 좀 더 저축을 했어야 했는데 아쉽습니다", "50대+"),
        ("손주 보는 재미로 사는 것 같습니다", "50대+"),
        ("건강이 최고라는 걸 이제야 알겠어요", "50대+"),
        ("요즘 스마트폰 쓰기가 영 어렵더라고요", "50대+"),
    ]
    rows = []
    for i in range(n):
        tpl = templates[i % len(templates)]
        rows.append({
            "text": f"{tpl[0]} {i}",
            "age_group": tpl[1],
            "source": "dummy",
        })
    return rows


# ---------------------------------------------------------------------------
# 실제 데이터셋 로더
# ---------------------------------------------------------------------------

def _normalize_age(raw: str) -> str | None:
    raw = raw.strip()
    normalized = AGE_LABEL_MAP.get(raw)
    if normalized:
        return normalized
    # 숫자 접두사 시도 (예: "20대 초반" → "20대")
    for prefix, label in AGE_LABEL_MAP.items():
        if raw.startswith(prefix):
            return label
    return None


def load_speech_age_json(base_dir: Path) -> list[dict]:
    """JSON 파일 기반 연령대별 발화 로더 (구조 A/B 모두 지원)"""
    rows = []
    unknown_labels: set[str] = set()

    for json_file in base_dir.rglob("*.json"):
        try:
            with json_file.open(encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("연령대 발화 JSON 읽기 실패: %s — %s", json_file, e)
            continue

        utterances_raw: list[dict] = []

        if isinstance(data, list):
            # 구조 A: 발화 리스트
            utterances_raw = data
        elif isinstance(data, dict):
            # 구조 B: 대화 단위
            utterances_raw = (
                data.get("utterances")
                or data.get("data")
                or data.get("dataset")
                or []
            )
            # 중첩된 경우 풀기
            flattened: list[dict] = []
            for item in utterances_raw:
                if isinstance(item, dict) and "utterances" in item:
                    flattened.extend(item["utterances"])
                elif isinstance(item, dict):
                    flattened.append(item)
            utterances_raw = flattened

        for utt in utterances_raw:
            if not isinstance(utt, dict):
                continue
            text = (
                utt.get("text") or utt.get("utterance")
                or utt.get("content") or utt.get("발화") or ""
            ).strip()
            raw_age = (
                utt.get("age") or utt.get("speaker_age") or utt.get("age_group")
                or utt.get("연령대") or utt.get("AgeGroup") or ""
            ).strip()
            if not text or not raw_age:
                continue
            age_group = _normalize_age(raw_age)
            if age_group is None:
                unknown_labels.add(raw_age)
                continue
            rows.append({"text": text, "age_group": age_group, "source": "aihub_speech_age"})

    if unknown_labels:
        logger.warning("연령대 발화 — 매핑 없는 연령 레이블 (스킵): %s", unknown_labels)

    logger.info("JSON 로드 완료: %d건 (from %s)", len(rows), base_dir)
    return rows


def load_speech_age_csv(base_dir: Path) -> list[dict]:
    """CSV 파일 기반 연령대별 발화 로더 (구조 C)"""
    rows = []
    unknown_labels: set[str] = set()

    for csv_file in base_dir.rglob("*.csv"):
        try:
            with csv_file.open(encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = (row.get("text") or row.get("발화") or "").strip()
                    raw_age = (
                        row.get("age_group") or row.get("age") or row.get("연령대") or ""
                    ).strip()
                    if not text or not raw_age:
                        continue
                    age_group = _normalize_age(raw_age)
                    if age_group is None:
                        unknown_labels.add(raw_age)
                        continue
                    rows.append({
                        "text": text,
                        "age_group": age_group,
                        "source": "aihub_speech_age",
                    })
        except Exception as e:
            logger.warning("연령대 발화 CSV 읽기 실패: %s — %s", csv_file, e)

    if unknown_labels:
        logger.warning("연령대 발화 CSV — 매핑 없는 레이블 (스킵): %s", unknown_labels)

    logger.info("CSV 로드 완료: %d건 (from %s)", len(rows), base_dir)
    return rows


def load_speech_age(base_dir: Path) -> list[dict]:
    json_rows = load_speech_age_json(base_dir)
    csv_rows = load_speech_age_csv(base_dir)
    return json_rows + csv_rows


# ---------------------------------------------------------------------------
# 전처리 파이프라인
# ---------------------------------------------------------------------------

def dedup(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    result = []
    for r in rows:
        h = sha256_of(r["text"])
        if h not in seen:
            seen.add(h)
            result.append(r)
    return result


def balance_undersample(rows: list[dict]) -> list[dict]:
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r["age_group"]].append(r)

    counts = {k: len(v) for k, v in by_class.items()}
    min_count = min(counts.values())
    logger.info("클래스별 건수: %s", counts)
    logger.info("언더샘플링 목표: %d건/클래스", min_count)

    result = []
    for cls_rows in by_class.values():
        random.shuffle(cls_rows)
        result.extend(cls_rows[:min_count])
    random.shuffle(result)
    return result


def split_train_val(rows: list[dict], val_ratio: float = 0.2) -> tuple[list[dict], list[dict]]:
    random.shuffle(rows)
    split = int(len(rows) * (1 - val_ratio))
    return rows[:split], rows[split:]


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "age_group", "source"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("저장: %s (%d건)", path, len(rows))


def write_stats(train: list[dict], val: list[dict], path: Path) -> None:
    def count_by(rows: list[dict], key: str) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for r in rows:
            counts[r[key]] += 1
        return dict(counts)

    stats = {
        "total_train": len(train),
        "total_val": len(val),
        "age_groups": AGE_GROUPS,
        "train_age_dist": count_by(train, "age_group"),
        "val_age_dist": count_by(val, "age_group"),
        "train_source_dist": count_by(train, "source"),
    }
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("통계: %s", path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="말투 연령 데이터셋 준비 스크립트")
    parser.add_argument("--dummy", action="store_true", help="더미 데이터로 테스트")
    parser.add_argument("--output-dir", default="data/speech_age", help="출력 디렉토리")
    parser.add_argument("--no-balance", action="store_true", help="클래스 균형 조정 생략")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)

    if args.dummy:
        logger.info("더미 모드: 실제 데이터셋 없이 테스트 데이터 생성")
        all_rows = make_dummy_data(n=600)
    else:
        base_dir = os.environ.get("DATASET_AIHUB_SPEECH_AGE_DIR")
        if not base_dir:
            logger.error(
                "DATASET_AIHUB_SPEECH_AGE_DIR 미설정. 환경변수를 설정하거나 --dummy 를 사용하세요."
            )
            raise SystemExit(1)

        all_rows = load_speech_age(Path(base_dir))
        if not all_rows:
            logger.error("로드된 데이터 없음. 데이터셋 경로를 확인하거나 --dummy 를 사용하세요.")
            raise SystemExit(1)

    logger.info("원본 합계: %d건", len(all_rows))
    deduped = dedup(all_rows)
    logger.info("중복 제거 후: %d건", len(deduped))

    if args.no_balance:
        balanced = deduped
        logger.info("균형 조정 생략")
    else:
        balanced = balance_undersample(deduped)
        logger.info("균형 조정 후: %d건", len(balanced))

    train, val = split_train_val(balanced)

    write_csv(train, output_dir / "train.csv")
    write_csv(val, output_dir / "val.csv")
    write_stats(train, val, output_dir / "dataset_stats.json")

    logger.info("완료 — train=%d, val=%d", len(train), len(val))


if __name__ == "__main__":
    main()
