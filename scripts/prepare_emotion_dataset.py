"""
감정 데이터셋 병합 스크립트

AI허브 감성대화 + 자유대화(성인/청소년) + 공감형 대화 4개 데이터셋을 읽어
통합 train/val CSV 를 생성한다.
연령별 발화 데이터셋은 별도 age_speech/ CSV 로 출력한다.

사용법:
  python scripts/prepare_emotion_dataset.py [--dummy] [--output-dir data/emotion]

  --dummy     실제 데이터셋 없이 더미 데이터로 동작 (테스트용)
  --output-dir  출력 디렉토리 (기본: data/emotion)
  --age-output-dir  말투 연령 출력 디렉토리 (기본: data/age_speech)
  --no-balance  클래스 균형 언더샘플링 생략 (중립 비율 보존)

환경 변수 (--dummy 없이 사용 시):
  DATASET_AIHUB_DIALOG_DIR        AI허브 감성 대화 말뭉치 (018.감성대화) 경로
                                  → Training+Validation ZIP 파일이 있는 최상위 폴더
  DATASET_AIHUB_FREE_ADULT_DIR    AI허브 자유대화(성인) (134-1) 경로
                                  → TL_*.zip 파일이 있는 최상위 폴더
  DATASET_AIHUB_FREE_TEEN_DIR     AI허브 자유대화(청소년) (134-2) 경로 (선택)
  EMPATHY_DATASET_PATH            공감형 대화 데이터셋 경로 (선택)
                                  → TL_*.zip 파일이 있는 최상위 폴더
  AGE_SPEECH_DATASET_PATH         연령별 특징 발화 데이터셋 경로 (선택)
                                  → TL_*.zip 파일이 있는 최상위 폴더

출력:
  data/emotion/train.csv    — text,emotion,dialog_act,source
  data/emotion/val.csv
  data/emotion/dataset_stats.json
  data/age_speech/train.csv — text,age_range,source  (AGE_SPEECH_DATASET_PATH 설정 시)
  data/age_speech/val.csv

감정 라벨 3종:
  긍정 | 중립 | 부정

018.감성대화 E-코드 매핑:
  E10–E59 → 부정  (불안/슬픔/분노/당황/상처 계열 50종)
  E60–E69 → 긍정  (기쁨 계열 10종)
  중립 없음 (해당 데이터셋 특성상 제외)

134-1/134-2 자유대화:
  VerifyEmotionCategory 필드 직접 사용 (긍정/중립/부정)

공감형 대화:
  info.speaker_emotion 기준:
  기쁨/설렘/사랑/행복/... → 긍정
  슬픔/분노/혐오/공포/불안/... → 부정
  그 외 → 중립

연령별 발화 (speech_age):
  Category 기준: 10대/20대 → 20s, 30대 → 30s, 40대 → 40s, 50대이상 → 50s+
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import zipfile
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 감정 레이블 정의
# ---------------------------------------------------------------------------

EMOTION_LABELS = ["긍정", "중립", "부정"]

# 018.감성대화 E-코드 → 3-class 매핑
def _ecode_to_emotion(ecode: str) -> str | None:
    """E10-E59 → 부정, E60-E69 → 긍정, 나머지 → None (스킵)"""
    try:
        n = int(ecode.lstrip("E"))
        if 10 <= n <= 59:
            return "부정"
        if 60 <= n <= 69:
            return "긍정"
    except (ValueError, AttributeError):
        pass
    return None

# 134-1/134-2 VerifyEmotionCategory 허용 값
VALID_FREE_DIALOG_CATEGORIES = {"긍정", "중립", "부정"}

# dialog_act 기존 15종 (감정 데이터에서는 "기타" 기본값)
DIALOG_ACT_LABELS = [
    "진술", "질문", "요청", "감사", "인사", "사과",
    "동의", "반대", "확인", "부정", "응답", "제안",
    "명령", "감탄", "기타",
]

# 공감형 대화 감정 키워드 → 3-class 매핑
_EMPATHY_POSITIVE = {"기쁨", "설렘", "사랑", "행복", "즐거움", "희망", "만족", "감사", "편안"}
_EMPATHY_NEGATIVE = {"슬픔", "분노", "혐오", "공포", "불안", "우울", "절망", "외로움", "당황", "상처"}

# 연령별 발화 Category → age_range 매핑 (4-class: 20s/30s/40s/50s+)
_AGE_CATEGORY_MAP: dict[str, str] = {
    "10대": "20s",    # 10대는 30세 미만으로 20s에 합산
    "20대": "20s",
    "30대": "30s",
    "40대": "40s",
    "50대이상": "50s+",
}


def _empathy_emotion_to_3class(emotion_str: str) -> str:
    """공감형 대화 감정 문자열 → 긍정/중립/부정"""
    e = emotion_str.strip()
    if e in _EMPATHY_POSITIVE:
        return "긍정"
    if e in _EMPATHY_NEGATIVE:
        return "부정"
    return "중립"


def _age_group_to_range(age_group_str: str) -> str:
    """AgeGroup 숫자 문자열 또는 Category 문자열 → 20s/30s/40s/50s+/unknown"""
    s = age_group_str.strip()
    # Category 형식: "10대", "20대", ...
    if s in _AGE_CATEGORY_MAP:
        return _AGE_CATEGORY_MAP[s]
    # AgeGroup 숫자 형식: "20", "30", ...
    try:
        n = int(s)
        if n < 30:
            return "20s"
        if n < 40:
            return "30s"
        if n < 50:
            return "40s"
        return "50s+"
    except (ValueError, TypeError):
        pass
    return "unknown"


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# 공통 ZIP JSON 이터레이터
# ---------------------------------------------------------------------------

def _iter_json_data_from_zips(base_path: Path):
    """base_path 아래 모든 *.zip을 열고 내부 JSON 데이터를 yield.

    Yields: (zip_path, json_name, parsed_data)
    실패한 ZIP/JSON은 warning(ZIP) / debug(JSON) 처리 후 계속 진행.
    """
    for zip_path in sorted(base_path.rglob("*.zip")):
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                json_names = [n for n in zf.namelist() if n.endswith(".json")]
                for jname in json_names:
                    try:
                        with zf.open(jname) as f:
                            data = json.loads(f.read().decode("utf-8", errors="replace"))
                        yield zip_path, jname, data
                    except Exception as e:
                        logger.debug("JSON 파싱 실패: %s/%s — %s", zip_path.name, jname, e)
        except Exception as e:
            logger.warning("ZIP 열기 실패: %s — %s", zip_path, e)


# ---------------------------------------------------------------------------
# 더미 데이터 생성
# ---------------------------------------------------------------------------

def make_dummy_data(n: int = 700) -> list[dict]:
    templates = [
        ("오늘 정말 기분이 좋아요 설레네요", "긍정", "진술"),
        ("이게 무슨 뜻인가요?", "중립", "질문"),
        ("너무 화가 나서 참을 수가 없어요", "부정", "감탄"),
        ("감사합니다 덕분에 살았어요", "긍정", "감사"),
        ("이 제품을 환불하고 싶습니다", "중립", "요청"),
        ("오늘 날씨가 흐리네요", "중립", "진술"),
        ("정말 슬프고 괴롭습니다", "부정", "진술"),
        ("좋아요 그렇게 하겠습니다", "긍정", "동의"),
        ("아니요 그건 아닌 것 같아요", "중립", "반대"),
        ("네 알겠습니다", "중립", "응답"),
        ("어떻게 해야 할지 너무 불안해요", "부정", "진술"),
        ("드디어 원하던 회사에 합격했어요!", "긍정", "진술"),
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

def load_aihub_dialog(base_dir: Path) -> list[dict]:
    """AI허브 감성 대화 말뭉치 (018.감성대화)

    ZIP 구조:
      감성대화말뭉치(최종데이터)_Training.zip → 감성대화말뭉치(최종데이터)_Training.json
      JSON: list[{profile: {emotion: {type: "E18"}}, talk: {content: {HS01, HS02, HS03}}}]

    사람 발화(HS01, HS02, HS03) 텍스트만 추출.
    E10-E59 → 부정, E60-E69 → 긍정, 중립 없음.
    """
    rows = []

    # ZIP 기반 탐색 (현재 배포 구조)
    zip_count = len(list(base_dir.rglob("*.zip")))
    if zip_count > 0:
        for zip_path, jname, data in _iter_json_data_from_zips(base_dir):
            if not isinstance(data, list):
                continue
            for item in data:
                profile = item.get("profile", {})
                ecode = profile.get("emotion", {}).get("type", "")
                emotion = _ecode_to_emotion(ecode)
                if not emotion:
                    continue
                content = item.get("talk", {}).get("content", {})
                for key in ("HS01", "HS02", "HS03"):
                    text = content.get(key, "").strip()
                    if text:
                        rows.append({
                            "text": text,
                            "emotion": emotion,
                            "dialog_act": "기타",
                            "source": "aihub_dialog",
                        })
    else:
        # 직접 JSON 탐색 (fallback)
        for json_file in base_dir.rglob("*.json"):
            try:
                with json_file.open(encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    continue
                for item in data:
                    profile = item.get("profile", {})
                    ecode = profile.get("emotion", {}).get("type", "")
                    emotion = _ecode_to_emotion(ecode)
                    if not emotion:
                        continue
                    content = item.get("talk", {}).get("content", {})
                    for key in ("HS01", "HS02", "HS03"):
                        text = content.get(key, "").strip()
                        if text:
                            rows.append({
                                "text": text,
                                "emotion": emotion,
                                "dialog_act": "기타",
                                "source": "aihub_dialog",
                            })
            except Exception as e:
                logger.warning("감성대화 JSON 읽기 실패: %s — %s", json_file, e)

    logger.info("AI허브 감성 대화 말뭉치: %d건 로드 (from %s)", len(rows), base_dir)
    return rows


def _load_free_dialog_from_dir(base_dir: Path, source_name: str) -> list[dict]:
    """134-1/134-2 자유대화 공통 로더

    ZIP 구조:
      TL_*.zip → *.json
      JSON: {"Conversation": [{"Text": "...", "VerifyEmotionCategory": "긍정/중립/부정"}]}
    """
    rows = []
    zip_count = len(list(base_dir.rglob("*.zip")))

    if zip_count > 0:
        for zip_path, jname, data in _iter_json_data_from_zips(base_dir):
            if not isinstance(data, dict):
                continue
            for utt in data.get("Conversation", []):
                text = utt.get("Text", "").strip()
                category = utt.get("VerifyEmotionCategory", "").strip()
                if text and category in VALID_FREE_DIALOG_CATEGORIES:
                    rows.append({
                        "text": text,
                        "emotion": category,
                        "dialog_act": "기타",
                        "source": source_name,
                    })
    else:
        json_files = list(base_dir.rglob("*.json"))
        if not json_files:
            logger.warning("%s: JSON 파일 없음 (%s)", source_name, base_dir)
            return rows
        for json_file in json_files:
            try:
                with json_file.open(encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    continue
                for utt in data.get("Conversation", []):
                    text = utt.get("Text", "").strip()
                    category = utt.get("VerifyEmotionCategory", "").strip()
                    if text and category in VALID_FREE_DIALOG_CATEGORIES:
                        rows.append({
                            "text": text,
                            "emotion": category,
                            "dialog_act": "기타",
                            "source": source_name,
                        })
            except Exception as e:
                logger.warning("%s JSON 읽기 실패: %s — %s", source_name, json_file, e)

    logger.info("%s: %d건 로드 (from %s)", source_name, len(rows), base_dir)
    return rows


def load_aihub_free_adult(base_dir: Path) -> list[dict]:
    """AI허브 감정이 태깅된 자유대화(성인) (134-1)"""
    return _load_free_dialog_from_dir(base_dir, "aihub_free_adult")


def load_aihub_free_teen(base_dir: Path) -> list[dict]:
    """AI허브 감정이 태깅된 자유대화(청소년) (134-2)"""
    return _load_free_dialog_from_dir(base_dir, "aihub_free_teen")


def load_empathy_dialog(data_path: Path) -> list[dict]:
    """공감형 대화 데이터셋 로더

    ZIP 구조:
      TL_*.zip → Empathy_*.json
      JSON: {"info": {"speaker_emotion": "분노"}, "utterances": [{"text": "..."}]}

    감정은 대화 단위 info.speaker_emotion 기준 (utterance별 emotion 없음):
      기쁨/설렘/사랑/행복/... → 긍정
      슬픔/분노/혐오/공포/불안/... → 부정
      그 외 → 중립

    내부 학습용 source 컬럼.
    외부 export 시 label_origin/method로 일반화 필요.
    """
    rows = []
    zip_count = len(list(data_path.rglob("*.zip"))) if data_path.is_dir() else 0

    if zip_count > 0:
        for zip_path, jname, data in _iter_json_data_from_zips(data_path):
            if not isinstance(data, dict):
                continue
            info = data.get("info", {})
            emotion_raw = (info.get("speaker_emotion") or "").strip()
            if not emotion_raw:
                continue
            emotion = _empathy_emotion_to_3class(emotion_raw)
            for utt in data.get("utterances", []):
                text = (utt.get("text") or "").strip()
                if text:
                    rows.append({
                        "text": text,
                        "emotion": emotion,
                        "dialog_act": "기타",
                        "source": "empathy_dialog",
                    })
    elif data_path.is_file():
        # 직접 JSON 파일 (fallback)
        try:
            with data_path.open(encoding="utf-8") as f:
                data = json.load(f)
            utterances = data if isinstance(data, list) else data.get("utterances", [])
            for utt in utterances:
                if not isinstance(utt, dict):
                    continue
                text = utt.get("text", "").strip()
                if not text:
                    continue
                emotion_raw = utt.get("emotion", "")
                emotion = _empathy_emotion_to_3class(emotion_raw)
                dialog_act_raw = utt.get("dialog_act", "기타").strip()
                dialog_act = dialog_act_raw if dialog_act_raw in DIALOG_ACT_LABELS else "기타"
                rows.append({
                    "text": text,
                    "emotion": emotion,
                    "dialog_act": dialog_act,
                    "source": "empathy_dialog",
                })
        except Exception as e:
            logger.warning("공감형 대화 JSON 읽기 실패: %s — %s", data_path, e)
    elif data_path.is_dir():
        # 직접 JSON 탐색 (fallback)
        for json_file in data_path.rglob("*.json"):
            try:
                with json_file.open(encoding="utf-8") as f:
                    data = json.load(f)
                utterances = data if isinstance(data, list) else data.get("utterances", [])
                for utt in utterances:
                    if not isinstance(utt, dict):
                        continue
                    text = utt.get("text", "").strip()
                    if not text:
                        continue
                    emotion_raw = utt.get("emotion", "")
                    emotion = _empathy_emotion_to_3class(emotion_raw)
                    dialog_act_raw = utt.get("dialog_act", "기타").strip()
                    dialog_act = dialog_act_raw if dialog_act_raw in DIALOG_ACT_LABELS else "기타"
                    rows.append({
                        "text": text,
                        "emotion": emotion,
                        "dialog_act": dialog_act,
                        "source": "empathy_dialog",
                    })
            except Exception as e:
                logger.warning("공감형 대화 JSON 읽기 실패: %s — %s", json_file, e)
    else:
        logger.warning("공감형 대화: 경로 없음 — %s", data_path)

    logger.info("공감형 대화: %d건 로드 (from %s)", len(rows), data_path)
    return rows


def load_age_speech_dataset(data_path: Path) -> list[dict]:
    """연령별 특징 발화 데이터셋 로더 (말투 연령 헤드 학습용)

    ZIP 구조:
      TL_*.zip → *.json
      JSON: {
        "Category": "20대",
        "Speakers": [{"Speaker": "id", "AgeGroup": "20"}],
        "Dialogs": [{"Speaker": "id", "SpeakerText": "...", "TextConvert": "..."}]
      }

    나이 → age_range:
      10대/20대 → 20s, 30대 → 30s, 40대 → 40s, 50대이상 → 50s+

    내부 학습용 source 컬럼.
    외부 export 시 label_origin/method로 일반화 필요.
    """
    rows = []
    zip_count = len(list(data_path.rglob("*.zip"))) if data_path.is_dir() else 0

    if zip_count > 0:
        for zip_path, jname, data in _iter_json_data_from_zips(data_path):
            if not isinstance(data, dict):
                continue
            # File-level age category (preferred)
            category = (data.get("Category") or "").strip()
            file_age_range = _AGE_CATEGORY_MAP.get(category, None)

            # Build speaker_id → age_range map for utterance-level fallback
            speaker_age: dict[str, str] = {}
            for sp in data.get("Speakers", []):
                sid = str(sp.get("Speaker", "")).strip()
                ag = str(sp.get("AgeGroup", "")).strip()
                if sid and ag:
                    speaker_age[sid] = _age_group_to_range(ag)

            for dialog in data.get("Dialogs", []):
                text = (dialog.get("TextConvert") or dialog.get("SpeakerText") or "").strip()
                if not text or len(text) < 2:
                    continue
                # Utterance-level age: speaker lookup, fallback to file-level
                sid = str(dialog.get("Speaker", "")).strip()
                age_range = speaker_age.get(sid) or file_age_range
                if not age_range or age_range == "unknown":
                    continue
                rows.append({
                    "text": text,
                    "age_range": age_range,
                    "source": "age_speech",
                })
    elif data_path.is_file():
        try:
            with data_path.open(encoding="utf-8") as f:
                data = json.load(f)
            items = data if isinstance(data, list) else data.get("utterances", [])
            for item in items:
                if not isinstance(item, dict):
                    continue
                text = item.get("text", "").strip()
                if not text:
                    continue
                if "age_range" in item:
                    age_range = item["age_range"]
                else:
                    age_range = _age_group_to_range(str(item.get("age", "")))
                rows.append({"text": text, "age_range": age_range, "source": "age_speech"})
        except Exception as e:
            logger.warning("연령별 발화 JSON 읽기 실패: %s — %s", data_path, e)
    elif data_path.is_dir():
        for json_file in data_path.rglob("*.json"):
            try:
                with json_file.open(encoding="utf-8") as f:
                    data = json.load(f)
                items = data if isinstance(data, list) else []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    text = item.get("text", "").strip()
                    if not text:
                        continue
                    age_range = item.get("age_range") or _age_group_to_range(str(item.get("age", "")))
                    rows.append({"text": text, "age_range": age_range, "source": "age_speech"})
            except Exception as e:
                logger.warning("연령별 발화 JSON 읽기 실패: %s — %s", json_file, e)
    else:
        logger.warning("연령별 발화: 경로 없음 — %s", data_path)

    logger.info("연령별 발화: %d건 로드 (from %s)", len(rows), data_path)
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


def write_csv(rows: list[dict], path: Path, fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or ["text", "emotion", "dialog_act", "source"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
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
        "emotion_labels": EMOTION_LABELS,
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
    parser.add_argument("--age-output-dir", default="data/age_speech", help="연령 데이터 출력 디렉토리")
    parser.add_argument("--no-balance", action="store_true", help="클래스 균형 조정 생략")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    age_output_dir = Path(args.age_output_dir)

    age_speech_rows: list[dict] = []

    if args.dummy:
        logger.info("더미 모드: 실제 데이터셋 없이 테스트 데이터 생성")
        all_rows = make_dummy_data(n=700)
    else:
        all_rows = []

        if d := os.environ.get("DATASET_AIHUB_DIALOG_DIR"):
            all_rows += load_aihub_dialog(Path(d))
        else:
            logger.warning("DATASET_AIHUB_DIALOG_DIR 미설정 — 018.감성대화 건너뜀")

        if d := os.environ.get("DATASET_AIHUB_FREE_ADULT_DIR"):
            all_rows += load_aihub_free_adult(Path(d))
        else:
            logger.warning("DATASET_AIHUB_FREE_ADULT_DIR 미설정 — 134-1 자유대화(성인) 건너뜀")

        if d := os.environ.get("DATASET_AIHUB_FREE_TEEN_DIR"):
            all_rows += load_aihub_free_teen(Path(d))
        else:
            logger.info("DATASET_AIHUB_FREE_TEEN_DIR 미설정 — 134-2 자유대화(청소년) 건너뜀 (선택)")

        if d := os.environ.get("EMPATHY_DATASET_PATH"):
            all_rows += load_empathy_dialog(Path(d))
        else:
            logger.info("EMPATHY_DATASET_PATH 미설정 — 공감형 대화 건너뜀 (선택)")

        if d := os.environ.get("AGE_SPEECH_DATASET_PATH"):
            age_speech_rows = load_age_speech_dataset(Path(d))
        else:
            logger.info("AGE_SPEECH_DATASET_PATH 미설정 — 연령별 발화 건너뜀 (선택)")

        if not all_rows:
            logger.error("로드된 데이터 없음. 환경변수를 확인하거나 --dummy 를 사용하세요.")
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

    # 연령별 발화 별도 출력
    if age_speech_rows:
        age_deduped = dedup(age_speech_rows)
        age_train, age_val = split_train_val(age_deduped)
        age_fields = ["text", "age_range", "source"]
        write_csv(age_train, age_output_dir / "train.csv", fieldnames=age_fields)
        write_csv(age_val, age_output_dir / "val.csv", fieldnames=age_fields)
        logger.info("연령별 발화 완료 — train=%d, val=%d", len(age_train), len(age_val))


if __name__ == "__main__":
    main()
