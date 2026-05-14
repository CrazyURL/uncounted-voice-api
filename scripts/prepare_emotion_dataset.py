"""
감정 데이터셋 병합 스크립트

7개 한국어 감정 데이터셋을 읽어 통합 train/val CSV 를 생성한다.

사용법:
  python scripts/prepare_emotion_dataset.py [--dummy] [--output-dir data/emotion]

  --dummy     실제 데이터셋 없이 더미 데이터로 동작 (테스트용)
  --output-dir  출력 디렉토리 (기본: data/emotion)

환경 변수 (--dummy 없이 사용 시 필수):
  DATASET_AIHUB_EMOTION_DIR       AI허브 감정분류 데이터셋 경로
  DATASET_AIHUB_SINGLE_DIR        AI허브 단발성 대화 데이터셋 경로
  DATASET_NIKL_DIR                NIKL 감정 데이터셋 경로
  DATASET_KEMDY_DIR               KEMDy 데이터셋 경로
  DATASET_AIHUB_DIALOG_DIR        AI허브 감성 대화 말뭉치 경로 (선택)
  DATASET_AIHUB_FREE_DIALOG_DIR   AI허브 감정이 태깅된 자유대화(성인) 경로 (선택)
                                  — zip.part0 파일들이 있는 폴더 지정
  DATASET_DIALECT_DIR             AI허브 중·노년층 한국어 방언 데이터 경로 (선택)
                                  — 강원도/경상도 JSON 라벨링 데이터 폴더

통합 감정 7종 (세부감정):
  기쁨 | 놀람 | 슬픔 | 분노 | 불안 | 당황 | 중립

상위 카테고리 파생 (API 응답 시 계산):
  긍정 ← 기쁨, 놀람
  부정 ← 슬픔, 분노, 불안, 당황
  중립 ← 중립
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

# ---------------------------------------------------------------------------
# 통합 레이블 정의
# ---------------------------------------------------------------------------

# 세부감정 7종 (모델 학습 타겟)
UNIFIED_EMOTIONS = ["기쁨", "놀람", "슬픔", "분노", "불안", "당황", "중립"]

# 7종 → 상위 카테고리 3종 파생 매핑 (모델 미학습, API 계산값)
EMOTION_TO_CATEGORY: dict[str, str] = {
    "기쁨": "긍정",
    "놀람": "긍정",
    "슬픔": "부정",
    "분노": "부정",
    "불안": "부정",
    "당황": "부정",
    "중립": "중립",
}

# AI허브 감정분류 원본 7종 → 통합 7종
AIHUB_EMOTION_MAP: dict[str, str] = {
    "기쁨": "기쁨",
    "놀람": "놀람",
    "슬픔": "슬픔",
    "공포": "불안",
    "역겨움": "분노",
    "분노": "분노",
    "중립": "중립",
}

# AI허브 감성 대화 말뭉치 6종 → 통합 7종
AIHUB_DIALOG_EMOTION_MAP: dict[str, str] = {
    "기쁨": "기쁨",
    "당황": "당황",
    "분노": "분노",
    "불안": "불안",
    "상처": "슬픔",
    "슬픔": "슬픔",
}

# 자유대화(성인) VerifyEmotionTarget 7종 → 통합 7종
FREE_DIALOG_EMOTION_MAP: dict[str, str] = {
    "기쁨": "기쁨",
    "사랑스러움": "기쁨",
    "없음": "중립",
    "놀라움": "놀람",
    "화남": "분노",
    "슬픔": "슬픔",
    "두려움": "불안",
}

# dialog_act 기존 15종
DIALOG_ACT_LABELS = [
    "진술", "질문", "요청", "감사", "인사", "사과",
    "동의", "반대", "확인", "부정", "응답", "제안",
    "명령", "감탄", "기타",
]

# 방언 데이터셋 발화의도 → dialog_act 매핑
DIALECT_INTENT_MAP: dict[str, str] = {
    "화자의견": "진술",
    "화자느낌": "감탄",
    "사실묘사": "진술",
    "질문": "질문",
    "명령": "명령",
    "부탁/요청/제안": "요청",
    "기타": "기타",
}


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# 더미 데이터 생성
# ---------------------------------------------------------------------------

def make_dummy_data(n: int = 700) -> list[dict]:
    templates = [
        ("오늘 정말 기분이 좋아요 설레네요", "기쁨", "진술"),
        ("이게 무슨 뜻인가요?", "중립", "질문"),
        ("너무 화가 나서 참을 수가 없어요", "분노", "감탄"),
        ("감사합니다 덕분에 살았어요", "기쁨", "감사"),
        ("이 제품을 환불하고 싶습니다", "당황", "요청"),
        ("오늘 날씨가 흐리네요", "중립", "진술"),
        ("정말 슬프고 괴롭습니다", "슬픔", "진술"),
        ("좋아요 그렇게 하겠습니다", "기쁨", "동의"),
        ("아니요 그건 아닌 것 같아요", "중립", "반대"),
        ("네 알겠습니다", "중립", "응답"),
        ("어떻게 해야 할지 너무 불안해요", "불안", "진술"),
        ("갑자기 그런 일이 생기다니 놀랐어요", "놀람", "진술"),
    ]
    rows = []
    for i in range(n):
        tpl = templates[i % len(templates)]
        rows.append({
            "text": f"{tpl[0]} {i}",
            "emotion": tpl[1],
            "dialog_act": tpl[2],
            "source": "dummy",
        })
    return rows


# ---------------------------------------------------------------------------
# 실제 데이터셋 로더
# ---------------------------------------------------------------------------

def load_aihub_emotion(base_dir: Path) -> list[dict]:
    """AI허브 감정분류 데이터셋 (JSON 형식)"""
    rows = []
    for json_file in base_dir.rglob("*.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            utterances = data.get("utterances", data.get("data", []))
            for item in utterances:
                text = item.get("transcript", item.get("text", "")).strip()
                emotion_raw = item.get("emotion", "")
                emotion = AIHUB_EMOTION_MAP.get(emotion_raw)
                if text and emotion:
                    rows.append({
                        "text": text,
                        "emotion": emotion,
                        "dialog_act": "기타",
                        "source": "aihub_emotion",
                    })
        except Exception as e:
            logger.warning("AI허브 감정분류 파일 읽기 실패: %s — %s", json_file, e)
    logger.info("AI허브 감정분류: %d건 로드", len(rows))
    return rows


def load_aihub_single(base_dir: Path) -> list[dict]:
    """AI허브 단발성 대화 감정 데이터셋"""
    rows = []
    for json_file in base_dir.rglob("*.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            for item in data if isinstance(data, list) else data.get("data", []):
                text = item.get("발화", item.get("utterance", "")).strip()
                emotion_raw = item.get("감정", item.get("emotion", ""))
                emotion = AIHUB_EMOTION_MAP.get(emotion_raw)
                if text and emotion:
                    rows.append({
                        "text": text,
                        "emotion": emotion,
                        "dialog_act": "기타",
                        "source": "aihub_single",
                    })
        except Exception as e:
            logger.warning("AI허브 단발성 파일 읽기 실패: %s — %s", json_file, e)
    logger.info("AI허브 단발성: %d건 로드", len(rows))
    return rows


def load_nikl(base_dir: Path) -> list[dict]:
    """NIKL 감정 데이터셋"""
    rows = []
    for json_file in base_dir.rglob("*.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            for doc in data.get("document", []):
                for utt in doc.get("utterance", []):
                    text = utt.get("form", "").strip()
                    emotion_raw = utt.get("emotion", {}).get("type", "")
                    emotion = AIHUB_EMOTION_MAP.get(emotion_raw, "중립")
                    if text:
                        rows.append({
                            "text": text,
                            "emotion": emotion,
                            "dialog_act": "기타",
                            "source": "nikl",
                        })
        except Exception as e:
            logger.warning("NIKL 파일 읽기 실패: %s — %s", json_file, e)
    logger.info("NIKL: %d건 로드", len(rows))
    return rows


def load_aihub_dialog(base_dir: Path) -> list[dict]:
    """AI허브 감성 대화 말뭉치 (원천데이터 Excel 형식)

    폴더 구조:
      base_dir/
        Training_*/원천데이터/*.zip  (감성대화말뭉치_Training.xlsx 포함)
        Validation_*/원천데이터/*.zip

    열: 연령, 성별, 상황키워드, 신체질환, 감정_대분류, 감정_소분류,
         사람문장1, 시스템문장1, 사람문장2, 시스템문장2, 사람문장3, 시스템문장3
    """
    try:
        import io
        import openpyxl
        import zipfile as _zipfile
    except ImportError:
        logger.warning("openpyxl 미설치 — 감성 대화 말뭉치 건너뜀. pip install openpyxl")
        return []

    rows = []
    for zip_path in base_dir.rglob("*.zip"):
        try:
            zf = _zipfile.ZipFile(zip_path)
            xlsx_names = [n for n in zf.namelist() if n.endswith(".xlsx")]
            if not xlsx_names:
                continue
            with zf.open(xlsx_names[0]) as f:
                wb = openpyxl.load_workbook(io.BytesIO(f.read()), read_only=True)
            ws = wb.active
            headers: tuple | None = None
            for row in ws.iter_rows(values_only=True):
                if headers is None:
                    headers = row
                    continue
                if not any(row):
                    continue
                row_dict = dict(zip(headers, row))
                emotion_raw = str(row_dict.get("감정_대분류") or "").strip()
                emotion = AIHUB_DIALOG_EMOTION_MAP.get(emotion_raw)
                if not emotion:
                    continue
                for col in ("사람문장1", "사람문장2", "사람문장3"):
                    text = str(row_dict.get(col) or "").strip()
                    if text:
                        rows.append({
                            "text": text,
                            "emotion": emotion,
                            "dialog_act": "기타",
                            "source": "aihub_dialog",
                        })
        except Exception as e:
            logger.warning("감성 대화 말뭉치 파일 읽기 실패: %s — %s", zip_path, e)

    logger.info("AI허브 감성 대화 말뭉치: %d건 로드", len(rows))
    return rows


def load_aihub_free_dialog(base_dir: Path) -> list[dict]:
    """AI허브 감정이 태깅된 자유대화(성인) 라벨링 데이터 (134-1)

    폴더 구조:
      base_dir/
        **/*.zip.part0   — 실제 zip 파일 (확장자만 .part0)

    JSON 구조:
      { "Conversation": [
          { "Text": "...",
            "VerifyEmotionTarget": "기쁨"|"없음"|"놀라움"|"화남"|"슬픔"|"두려움"|"사랑스러움",
            "VerifyEmotionCategory": "긍정"|"중립"|"부정",  # 상위 카테고리 (미사용)
            ... }
      ] }

    VerifyEmotionTarget (세부감정 7종) 을 사용하여 통합 7종으로 매핑.
    """
    rows = []
    for part_path in base_dir.rglob("*.part0"):
        try:
            import zipfile as _zipfile
            with _zipfile.ZipFile(part_path) as zf:
                json_names = [n for n in zf.namelist() if n.endswith(".json")]
                for jname in json_names:
                    try:
                        with zf.open(jname) as jf:
                            data = json.loads(jf.read().decode("utf-8"))
                        for utt in data.get("Conversation", []):
                            text = utt.get("Text", "").strip()
                            emotion_raw = utt.get("VerifyEmotionTarget", "").strip()
                            emotion = FREE_DIALOG_EMOTION_MAP.get(emotion_raw)
                            if text and emotion:
                                rows.append({
                                    "text": text,
                                    "emotion": emotion,
                                    "dialog_act": "기타",
                                    "source": "aihub_free_dialog",
                                })
                    except Exception as e:
                        logger.warning("자유대화 JSON 읽기 실패: %s/%s — %s", part_path, jname, e)
        except Exception as e:
            logger.warning("자유대화 zip 읽기 실패: %s — %s", part_path, e)
    logger.info("AI허브 감정이 태깅된 자유대화(성인): %d건 로드", len(rows))
    return rows


def load_kemdy(base_dir: Path) -> list[dict]:
    """KEMDy 데이터셋 (CSV 형식, valence/arousal → 세부감정 매핑)

    valence + arousal 조합으로 7종 세부감정 근사:
      valence > 0.5, arousal high  → 기쁨
      valence > 0.5, arousal low   → 중립
      valence < -0.5, arousal high → 분노
      valence < -0.5, arousal low  → 슬픔
      그 외                        → 중립
    """
    rows = []
    for csv_file in base_dir.rglob("*.csv"):
        try:
            with csv_file.open(encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = row.get("Transcript", row.get("transcript", "")).strip()
                    try:
                        valence = float(row.get("Valence", row.get("valence", 0)))
                    except ValueError:
                        valence = 0.0
                    try:
                        arousal = float(row.get("Arousal", row.get("arousal", 0)))
                    except ValueError:
                        arousal = 0.0

                    if valence > 0.5:
                        emotion = "기쁨" if arousal > 0 else "중립"
                    elif valence < -0.5:
                        emotion = "분노" if arousal > 0 else "슬픔"
                    else:
                        emotion = "중립"

                    if text:
                        rows.append({
                            "text": text,
                            "emotion": emotion,
                            "dialog_act": "기타",
                            "source": "kemdy",
                        })
        except Exception as e:
            logger.warning("KEMDy 파일 읽기 실패: %s — %s", csv_file, e)
    logger.info("KEMDy: %d건 로드", len(rows))
    return rows


def load_dialect(base_dir: Path) -> list[dict]:
    """AI허브 중·노년층 한국어 방언 데이터 (강원도, 경상도)

    폴더 구조:
      base_dir/
        **/*.json   — 라벨링 JSON 파일

    JSON 구조 (발화 단위):
      {
        "transcription": {
          "standard": "표준어 대응표현",  ← 텍스트 입력으로 사용
          "sentences": [{ "sentenceId": 1, ... }]
        },
        "annotation": {
          "intents": [{ "sentenceId": 1, "tagType": "화자의견" }],
          "emotions": [{ "sentenceId": 1, "tagType": "..." }]  ← 실제 값 확인 필요
        }
      }

    주의:
      - emotions[].tagType 의 실제 값 목록은 스키마에 미열거.
        샘플 다운로드 후 unique 값 확인 → DIALECT_EMOTION_MAP 에 추가 필요.
      - 현재는 발화의도(intents) → dialog_act 매핑만 적용.
        감정 레이블은 알려진 값만 수용하고 미매핑 값은 건너뜀.
      - transcription.standard 없는 경우 transcription.pronunciation 사용.
    """
    # 방언 데이터셋 감정 태그 → 통합 7종
    # TODO: 샘플 데이터 확인 후 실제 값으로 채울 것
    DIALECT_EMOTION_MAP: dict[str, str] = {
        # 일반적으로 사용되는 후보값 (확인 전 추정)
        "기쁨": "기쁨",
        "슬픔": "슬픔",
        "분노": "분노",
        "놀람": "놀람",
        "불안": "불안",
        "당황": "당황",
        "중립": "중립",
        "없음": "중립",
        "긍정": "기쁨",   # 3-class로 나온 경우 fallback
        "부정": "슬픔",
    }

    rows = []
    unknown_emotions: set[str] = set()

    for json_file in base_dir.rglob("*.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            transcription = data.get("transcription", {})
            annotation = data.get("annotation", {})

            # 표준어 텍스트 추출 (문장 단위 sentences 우선, 없으면 전체 standard)
            sentences_map: dict[int, str] = {}
            for sent in transcription.get("sentences", []):
                sid = sent.get("sentenceId")
                text = (sent.get("standard") or sent.get("pronunciation") or "").strip()
                if sid and text:
                    sentences_map[sid] = text

            # 발화의도 매핑
            intent_map: dict[int, str] = {}
            for intent in annotation.get("intents", []):
                sid = intent.get("sentenceId")
                tag = intent.get("tagType", "")
                if sid:
                    intent_map[sid] = DIALECT_INTENT_MAP.get(tag, "기타")

            # 감정 매핑
            emotion_map: dict[int, str] = {}
            for emo in annotation.get("emotions", []):
                sid = emo.get("sentenceId")
                tag = emo.get("tagType", "").strip()
                if sid and tag:
                    mapped = DIALECT_EMOTION_MAP.get(tag)
                    if mapped:
                        emotion_map[sid] = mapped
                    else:
                        unknown_emotions.add(tag)

            # 레코드 생성 (문장 단위)
            for sid, text in sentences_map.items():
                dialog_act = intent_map.get(sid, "기타")
                emotion = emotion_map.get(sid)
                if text and emotion:
                    rows.append({
                        "text": text,
                        "emotion": emotion,
                        "dialog_act": dialog_act,
                        "source": "dialect",
                    })
                elif text and dialog_act != "기타":
                    # 감정 레이블 없어도 dialog_act 만 있으면 추가 (emotion=중립 기본값)
                    rows.append({
                        "text": text,
                        "emotion": "중립",
                        "dialog_act": dialog_act,
                        "source": "dialect_intent_only",
                    })
        except Exception as e:
            logger.warning("방언 파일 읽기 실패: %s — %s", json_file, e)

    if unknown_emotions:
        logger.warning(
            "방언 데이터셋 미매핑 감정 태그 (DIALECT_EMOTION_MAP 에 추가 필요): %s",
            sorted(unknown_emotions),
        )
    logger.info("중·노년층 방언 데이터셋: %d건 로드", len(rows))
    return rows


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
    """클래스 균형 언더샘플링 — 최소 클래스 크기에 맞춤"""
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r["emotion"]].append(r)

    min_count = min(len(v) for v in by_class.values())
    logger.info("클래스별 건수: %s", {k: len(v) for k, v in by_class.items()})
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
        writer = csv.DictWriter(f, fieldnames=["text", "emotion", "dialog_act", "source"])
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
        "emotion_labels": UNIFIED_EMOTIONS,
        "emotion_to_category": EMOTION_TO_CATEGORY,
        "train_emotion_dist": count_by(train, "emotion"),
        "val_emotion_dist": count_by(val, "emotion"),
        "train_source_dist": count_by(train, "source"),
    }
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("통계: %s", path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="감정 데이터셋 병합 스크립트")
    parser.add_argument("--dummy", action="store_true", help="더미 데이터로 테스트")
    parser.add_argument("--output-dir", default="data/emotion", help="출력 디렉토리")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)

    if args.dummy:
        logger.info("더미 모드: 실제 데이터셋 없이 테스트 데이터 생성")
        all_rows = make_dummy_data(n=700)
    else:
        all_rows = []
        if d := os.environ.get("DATASET_AIHUB_EMOTION_DIR"):
            all_rows += load_aihub_emotion(Path(d))
        else:
            logger.warning("DATASET_AIHUB_EMOTION_DIR 미설정 — AI허브 감정분류 건너뜀")

        if d := os.environ.get("DATASET_AIHUB_SINGLE_DIR"):
            all_rows += load_aihub_single(Path(d))
        else:
            logger.warning("DATASET_AIHUB_SINGLE_DIR 미설정 — AI허브 단발성 건너뜀")

        if d := os.environ.get("DATASET_NIKL_DIR"):
            all_rows += load_nikl(Path(d))
        else:
            logger.warning("DATASET_NIKL_DIR 미설정 — NIKL 건너뜀")

        if d := os.environ.get("DATASET_KEMDY_DIR"):
            all_rows += load_kemdy(Path(d))
        else:
            logger.warning("DATASET_KEMDY_DIR 미설정 — KEMDy 건너뜀")

        if d := os.environ.get("DATASET_AIHUB_DIALOG_DIR"):
            all_rows += load_aihub_dialog(Path(d))
        else:
            logger.info("DATASET_AIHUB_DIALOG_DIR 미설정 — 감성 대화 말뭉치 건너뜀 (선택)")

        if d := os.environ.get("DATASET_AIHUB_FREE_DIALOG_DIR"):
            all_rows += load_aihub_free_dialog(Path(d))
        else:
            logger.info("DATASET_AIHUB_FREE_DIALOG_DIR 미설정 — 자유대화(성인) 건너뜀 (선택)")

        if d := os.environ.get("DATASET_DIALECT_DIR"):
            all_rows += load_dialect(Path(d))
        else:
            logger.info("DATASET_DIALECT_DIR 미설정 — 방언 데이터셋 건너뜀 (선택)")

        if not all_rows:
            logger.error("로드된 데이터 없음. 환경변수를 확인하거나 --dummy 를 사용하세요.")
            raise SystemExit(1)

    logger.info("원본 합계: %d건", len(all_rows))
    deduped = dedup(all_rows)
    logger.info("중복 제거 후: %d건", len(deduped))
    balanced = balance_undersample(deduped)
    logger.info("균형 조정 후: %d건", len(balanced))
    train, val = split_train_val(balanced)

    write_csv(train, output_dir / "train.csv")
    write_csv(val, output_dir / "val.csv")
    write_stats(train, val, output_dir / "dataset_stats.json")

    logger.info("완료 — train=%d, val=%d", len(train), len(val))


if __name__ == "__main__":
    main()
