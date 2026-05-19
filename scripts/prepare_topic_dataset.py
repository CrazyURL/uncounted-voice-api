"""
주제별 텍스트 일상 대화 데이터셋 준비 스크립트

AI허브 주제별 텍스트 일상 대화 데이터(020)에서 topic + speech_act CSV를 생성.
Training 폴더 → train.csv, Validation 폴더 → val.csv (별도 분할 없음).

사용법:
  python scripts/prepare_topic_dataset.py [--analyze-only] [--dummy] [--output-dir data/topic]

  --analyze-only  label_analysis.json 만 생성하고 종료
  --dummy         더미 데이터로 동작 (테스트용, 데이터셋 불필요)
  --output-dir    topic CSV 출력 디렉토리 (기본: data/topic)
  --sa-output-dir speech_act CSV 출력 디렉토리 (기본: data/speech_act)
  --seed          랜덤 시드 (기본: 42)

환경 변수:
  TOPIC_DATASET_PATH    AI허브 주제별 텍스트 일상 대화 최상위 폴더

출력:
  data/topic/train.csv          — text, topic, topic_group, source
  data/topic/val.csv
  data/topic/label_analysis.json
  data/speech_act/train.csv     — text, speech_act, speech_act_group, source
  data/speech_act/val.csv

라벨 체계:
  topic    = annotations.subject  (20-class 실제 주제 라벨: 식음료, 교통, ...)
  item.category = 항상 "일상대화"로 dataset_category로만 보존
  "상거래전반" → "상거래 전반" 자동 정규화

speech_act:
  raw 세부값과 group(단언/지시/표현/언약) 둘 다 저장.
  1차 학습은 speech_act_group(4-class) 우선.

내부 학습용 source 컬럼.
외부 export 시 label_origin/method로 일반화 필요.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import zipfile
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SPEECH_ACT_PATTERN = re.compile(r'\((\w+)\)\s+(.+)')

# Internal training only.
# Never export this raw source value to external dataset ZIP.
# External export must map it to label_origin/method.
SOURCE = "aihub_topic"

# 20 topics → 6 groups
TOPIC_TO_GROUP: dict[str, str] = {
    "가족": "인간관계",
    "연애/결혼": "인간관계",
    "건강": "생활",
    "미용": "생활",
    "반려동물": "생활",
    "식음료": "생활",
    "주거와 생활": "생활",
    "게임": "문화/여가",
    "방송/연예": "문화/여가",
    "영화/만화": "문화/여가",
    "스포츠/레저": "문화/여가",
    "여행": "문화/여가",
    "계절/날씨": "사회/환경",
    "교통": "사회/환경",
    "사회이슈": "사회/환경",
    "타 국가 이슈": "사회/환경",
    "교육": "교육/군대",
    "군대": "교육/군대",
    "상거래 전반": "경제/직장",
    "회사/아르바이트": "경제/직장",
}

# 오타 정규화 맵
TOPIC_NORMALIZATION: dict[str, str] = {
    "상거래전반": "상거래 전반",
}


def _normalize_topic(raw: str) -> str:
    """오타/표기 변형 정규화"""
    return TOPIC_NORMALIZATION.get(raw.strip(), raw.strip())


def _normalize_speech_act(raw: str) -> tuple[str, str]:
    """(그룹) 상세행위 → (speech_act, speech_act_group)"""
    m = SPEECH_ACT_PATTERN.match(raw.strip())
    if m:
        return raw.strip(), m.group(1)
    if raw.strip() in ("N/A", "", "턴토크 사인(관습적 반응)"):
        return raw.strip(), "기타"
    return raw.strip(), "기타"


def _is_split_path(path: Path, split: str) -> bool:
    """Training/Validation 경로를 path.parts 기준으로 robust하게 판정.
    파일명에 의존하지 않음 — Validation 아래 TL_ 파일 오분류 방지.
    """
    parts = [p.lower() for p in path.parts]
    if split == "train":
        return any("training" in p or p in ("train", "tr") for p in parts)
    if split == "val":
        return any("validation" in p or "valid" in p or p in ("val", "vl") for p in parts)
    return False


def _iter_labeled_jsons(zip_path: Path):
    """라벨 ZIP 내부 JSON 파일들을 순회하며 (text, topic, dataset_category, speech_act, sa_group) yield.

    topic    = annotations.subject  (실제 학습 라벨)
    dataset_category = item.category (항상 "일상대화", 보존용)
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        json_names = [n for n in zf.namelist() if n.endswith(".json")]
        for jname in json_names:
            try:
                with zf.open(jname) as f:
                    data = json.loads(f.read().decode("utf-8", errors="replace"))
                for item in data.get("info", []):
                    dataset_category = (item.get("category") or "").strip()
                    ann = item.get("annotations", {})
                    # subject is the real topic label (20 classes: 식음료, 교통, ...)
                    subject = (ann.get("subject") or "").strip()
                    topic = _normalize_topic(subject)
                    if not topic:
                        continue
                    for line in ann.get("lines", []):
                        text = (line.get("norm_text") or line.get("text", "")).strip()
                        if not text or len(text) < 2:
                            continue
                        sa_raw = line.get("speechAct", "N/A")
                        speech_act, sa_group = _normalize_speech_act(sa_raw)
                        yield text, topic, dataset_category, speech_act, sa_group
            except Exception as e:
                logger.debug("Skip %s: %s", jname, e)


def _collect_split(base_dir: Path, split: str) -> list[dict]:
    """Training or Validation 폴더 내 모든 라벨 ZIP을 처리."""
    label_zips = [
        p for p in base_dir.rglob("*.zip")
        if _is_split_path(p, split)
    ]
    if not label_zips:
        logger.warning("No label ZIPs found for split=%s under %s", split, base_dir)
    else:
        logger.info("Found %d label ZIPs for split=%s", len(label_zips), split)

    rows = []
    for zp in sorted(label_zips):
        for text, topic, dataset_cat, speech_act, sa_group in _iter_labeled_jsons(zp):
            rows.append({
                "text": text,
                "topic": topic,
                "topic_group": TOPIC_TO_GROUP.get(topic, "기타"),
                "speech_act": speech_act,
                "speech_act_group": sa_group,
                "source": SOURCE,
            })

    if split == "val" and not rows:
        logger.warning("No validation data collected; val.csv will be empty")

    return rows


def _make_dummy_rows(n: int = 100) -> tuple[list[dict], list[dict]]:
    topics = list(TOPIC_TO_GROUP.keys())
    speech_acts = [
        "(단언) 주장하기", "(지시) 질문하기",
        "(표현) 긍정감정 표현하기", "(언약) 약속하기(제3자와)/(개인적 수준)",
    ]
    rows = []
    for i in range(n):
        topic = topics[i % len(topics)]
        sa_raw = speech_acts[i % len(speech_acts)]
        speech_act, sa_group = _normalize_speech_act(sa_raw)
        rows.append({
            "text": f"더미 텍스트 샘플 {i}",
            "topic": topic,
            "topic_group": TOPIC_TO_GROUP.get(topic, "기타"),
            "speech_act": speech_act,
            "speech_act_group": sa_group,
            "source": "dummy",
        })
    mid = int(n * 0.8)
    return rows[:mid], rows[mid:]


def _write_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    logger.info("Wrote %d rows → %s", len(rows), path)


def _build_analysis(train_rows: list[dict], val_rows: list[dict],
                    raw_topic_counts: Counter | None = None) -> dict:
    all_rows = train_rows + val_rows
    topic_counts = Counter(r["topic"] for r in all_rows)
    group_counts = Counter(r["topic_group"] for r in all_rows)
    sa_counts = Counter(r["speech_act"] for r in all_rows)
    sa_group_counts = Counter(r["speech_act_group"] for r in all_rows)
    unmapped_topics = sorted({r["topic"] for r in all_rows if r["topic"] not in TOPIC_TO_GROUP})
    unmapped_count = sum(topic_counts.get(t, 0) for t in unmapped_topics)
    unmapped_ratio = round(unmapped_count / max(1, len(all_rows)), 4)

    result = {
        "total_samples": len(all_rows),
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "categories": dict(topic_counts),
        "category_groups": dict(group_counts),
        "speech_acts": dict(sa_counts.most_common(30)),
        "speech_act_groups": dict(sa_group_counts),
        "unmapped_topics": unmapped_topics,
        "unmapped_topic_ratio": unmapped_ratio,
        "normalization_report": TOPIC_NORMALIZATION,
    }
    if raw_topic_counts:
        result["topic_raw_values"] = dict(raw_topic_counts)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--output-dir", default="data/topic")
    parser.add_argument("--sa-output-dir", default="data/speech_act")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out_topic = Path(args.output_dir)
    out_sa = Path(args.sa_output_dir)

    if args.dummy:
        logger.info("Dummy mode: generating 100 samples")
        train_rows, val_rows = _make_dummy_rows(100)
        analysis = _build_analysis(train_rows, val_rows)
    else:
        dataset_path = os.environ.get("TOPIC_DATASET_PATH", "")
        if not dataset_path:
            raise EnvironmentError("TOPIC_DATASET_PATH not set")
        base = Path(dataset_path)
        if not base.exists():
            raise FileNotFoundError(f"TOPIC_DATASET_PATH not found: {base}")

        logger.info("Loading training data from %s", base)
        train_rows = _collect_split(base, "train")
        logger.info("Loading validation data from %s", base)
        val_rows = _collect_split(base, "val")

        if not train_rows:
            logger.error("No training data collected — check dataset structure")
            raise SystemExit(1)

        # Collect raw (pre-normalization) topic distribution for analysis
        raw_topic_counts: Counter = Counter()
        for r in train_rows + val_rows:
            # We normalized already; track via normalization report
            pass

        analysis = _build_analysis(train_rows, val_rows)

    out_topic.mkdir(parents=True, exist_ok=True)
    analysis_path = out_topic / "label_analysis.json"
    analysis_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2))
    logger.info("label_analysis.json → %s  (%d total)", analysis_path, analysis["total_samples"])
    logger.info("topic classes: %d", len(analysis["categories"]))
    if analysis["unmapped_topics"]:
        logger.warning("unmapped topic ratio: %.1f%%  unmapped: %s",
                       analysis["unmapped_topic_ratio"] * 100, analysis["unmapped_topics"])

    if args.analyze_only:
        logger.info("--analyze-only: done")
        return

    topic_fields = ["text", "topic", "topic_group", "source"]
    _write_csv(train_rows, out_topic / "train.csv", topic_fields)
    _write_csv(val_rows, out_topic / "val.csv", topic_fields)

    sa_fields = ["text", "speech_act", "speech_act_group", "source"]
    _write_csv(train_rows, out_sa / "train.csv", sa_fields)
    _write_csv(val_rows, out_sa / "val.csv", sa_fields)

    logger.info("Done. topic=%s  speech_act=%s", out_topic, out_sa)


if __name__ == "__main__":
    main()
