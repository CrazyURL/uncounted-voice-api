"""
감정 데이터셋 병합 스크립트

AI허브 감성대화 + 자유대화(성인/청소년) + 공감형 대화 + 감성발화스타일 +
용도별 목적대화 6개 데이터셋을 읽어 통합 train/val CSV 를 생성한다.

사용법:
  python scripts/prepare_emotion_dataset.py [--dummy] [--output-dir data/emotion]

  --dummy     실제 데이터셋 없이 더미 데이터로 동작 (테스트용)
  --output-dir  출력 디렉토리 (기본: data/emotion)
  --no-balance  클래스 균형 언더샘플링 생략 (중립 비율 보존)

환경 변수 (--dummy 없이 사용 시):
  DATASET_AIHUB_DIALOG_DIR        AI허브 감성 대화 말뭉치 (018.감성대화) 경로
                                  → Training+Validation JSON 파일이 있는 최상위 폴더
  DATASET_AIHUB_FREE_ADULT_DIR    AI허브 자유대화(성인) (134-1) 경로
                                  → extracted/ 폴더를 포함하는 최상위 폴더
  DATASET_AIHUB_FREE_TEEN_DIR     AI허브 자유대화(청소년) (134-2) 경로 (선택)
                                  → 성인과 동일한 구조
  DATASET_AIHUB_EMPATHETIC_DIR    AI허브 공감형 대화 말뭉치 경로 (선택)
                                  → JSON 파일이 있는 최상위 폴더
  DATASET_AIHUB_SPEECH_STYLE_DIR  AI허브 감성 및 발화 스타일별 음성합성 데이터 경로 (선택)
                                  → JSON 메타데이터 파일이 있는 최상위 폴더
  DATASET_AIHUB_PURPOSIVE_DIR     AI허브 용도별 목적대화 데이터 경로 (선택)
                                  → JSON 파일이 있는 최상위 폴더
                                  → dialog_act 15종 보강 및 emotion 레이블 병행 추출

출력:
  data/emotion/train.csv    — text,emotion,dialog_act,source
  data/emotion/val.csv
  data/emotion/dataset_stats.json

감정 라벨 3종:
  긍정 | 중립 | 부정

018.감성대화 E-코드 매핑:
  E10–E59 → 부정  (불안/슬픔/분노/당황/상처 계열 50종)
  E60–E69 → 긍정  (기쁨 계열 10종)
  중립 없음 (해당 데이터셋 특성상 제외)

134-1/134-2 자유대화:
  VerifyEmotionCategory 필드 직접 사용 (긍정/중립/부정)

공감형 대화 감정 매핑:
  기쁨, 감사, 신뢰, 기대 → 긍정
  놀람, 중립             → 중립
  슬픔, 공포, 혐오, 분노 → 부정

감성 발화 스타일 감정 매핑 (TTS 데이터):
  기쁨, 행복, 즐거움, 설렘, 흥분 → 긍정
  슬픔, 우울, 상처               → 부정
  분노, 화남                     → 부정
  공포, 두려움, 불안             → 부정
  혐오, 역겨움                   → 부정
  당황                           → 부정
  놀람, 중립, 평온               → 중립

용도별 목적대화 dialog_act 매핑:
  정보요청, 질문, 의문 → 질문
  정보제공, 설명, 서술 → 진술
  지시, 명령           → 명령
  요청, 부탁           → 요청
  동의, 수락, 허락     → 동의
  거절, 반대, 이의     → 반대
  확인, 재확인         → 확인
  부정, 부인           → 부정
  응답, 대답           → 응답
  제안, 권유           → 제안
  감사                 → 감사
  인사, 작별           → 인사
  사과, 유감           → 사과
  감탄, 놀람           → 감탄
  기타 / 미매핑        → 기타
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

# 공감형 대화 감정 → 3-class 매핑
EMPATHETIC_EMOTION_MAP: dict[str, str] = {
    "기쁨": "긍정", "감사": "긍정", "신뢰": "긍정", "기대": "긍정",
    "놀람": "중립", "중립": "중립",
    "슬픔": "부정", "공포": "부정", "혐오": "부정", "분노": "부정",
}

# 감성 발화 스타일(TTS) 감정 → 3-class 매핑
SPEECH_STYLE_EMOTION_MAP: dict[str, str] = {
    # 긍정
    "기쁨": "긍정", "행복": "긍정", "즐거움": "긍정", "설렘": "긍정",
    "흥분": "긍정", "활기": "긍정", "긍정": "긍정",
    "happy": "긍정", "joy": "긍정", "excited": "긍정", "positive": "긍정",
    # 중립
    "중립": "중립", "평온": "중립", "놀람": "중립",
    "neutral": "중립", "calm": "중립", "surprised": "중립",
    # 부정
    "슬픔": "부정", "우울": "부정", "상처": "부정",
    "분노": "부정", "화남": "부정",
    "공포": "부정", "두려움": "부정", "불안": "부정",
    "혐오": "부정", "역겨움": "부정",
    "당황": "부정",
    "sad": "부정", "angry": "부정", "fearful": "부정", "disgusted": "부정",
    "negative": "부정",
}

# 용도별 목적대화 dialog_act(의도) → 15종 매핑
PURPOSIVE_DIALOG_ACT_MAP: dict[str, str] = {
    # 질문
    "정보요청": "질문", "질문": "질문", "의문": "질문", "묻기": "질문",
    "확인요청": "질문", "information_request": "질문", "question": "질문",
    # 진술
    "정보제공": "진술", "설명": "진술", "서술": "진술", "진술": "진술",
    "information_provide": "진술", "statement": "진술",
    # 명령
    "지시": "명령", "명령": "명령", "directive": "명령", "command": "명령",
    # 요청
    "요청": "요청", "부탁": "요청", "request": "요청",
    # 동의
    "동의": "동의", "수락": "동의", "허락": "동의", "허용": "동의",
    "agreement": "동의", "accept": "동의",
    # 반대
    "거절": "반대", "반대": "반대", "이의": "반대",
    "rejection": "반대", "disagree": "반대",
    # 확인
    "확인": "확인", "재확인": "확인", "confirm": "확인",
    # 부정
    "부정": "부정", "부인": "부정", "denial": "부정", "negation": "부정",
    # 응답
    "응답": "응답", "대답": "응답", "답변": "응답", "response": "응답",
    # 제안
    "제안": "제안", "권유": "제안", "추천": "제안", "suggestion": "제안",
    # 감사
    "감사": "감사", "thanks": "감사", "thank": "감사",
    # 인사
    "인사": "인사", "작별": "인사", "greeting": "인사",
    # 사과
    "사과": "사과", "유감": "사과", "apology": "사과",
    # 감탄
    "감탄": "감탄", "놀람_긍정": "감탄", "exclamation": "감탄",
}

# dialog_act 기존 15종 (감정 데이터에서는 "기타" 기본값)
DIALOG_ACT_LABELS = [
    "진술", "질문", "요청", "감사", "인사", "사과",
    "동의", "반대", "확인", "부정", "응답", "제안",
    "명령", "감탄", "기타",
]


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


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
        # 공감형 대화 스타일
        ("많이 힘드셨겠어요 그렇게 되셔서", "부정", "진술"),
        ("그거 정말 다행이네요 잘 됐어요", "긍정", "진술"),
        ("많이 걱정이 되실 것 같아요", "부정", "진술"),
        ("그 말을 들으니 저도 기쁘네요", "긍정", "진술"),
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

    JSON 구조 (대형 단일 파일, 또는 여러 JSON 파일):
      [
        {
          "profile": { "emotion": { "type": "E18", ... } },
          "talk": { "content": { "HS01": "사람발화", "SS01": "봇응답", ... } }
        },
        ...
      ]

    사람 발화(HS01, HS02, HS03) 텍스트만 추출한다.
    E10-E59 → 부정, E60-E69 → 긍정, 중립 없음.
    """
    rows = []
    # extracted/ 하위 JSON + 직접 JSON 모두 탐색
    for json_file in base_dir.rglob("*.json"):
        try:
            with json_file.open(encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("감성대화 JSON 읽기 실패: %s — %s", json_file, e)
            continue

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

    logger.info("AI허브 감성 대화 말뭉치: %d건 로드 (from %s)", len(rows), base_dir)
    return rows


def _load_free_dialog_from_dir(base_dir: Path, source_name: str) -> list[dict]:
    """134-1/134-2 자유대화 공통 로더

    JSON 구조 (파일 1개 = 대화 1건):
      {
        "Conversation": [
          {
            "Text": "발화 텍스트",
            "VerifyEmotionCategory": "긍정"|"중립"|"부정",
            ...
          }
        ]
      }
    """
    rows = []
    json_files = list(base_dir.rglob("*.json"))

    if not json_files:
        logger.warning("%s: JSON 파일 없음 (%s)", source_name, base_dir)
        return rows

    for json_file in json_files:
        try:
            with json_file.open(encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("%s JSON 읽기 실패: %s — %s", source_name, json_file, e)
            continue

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

    logger.info("%s: %d건 로드 (from %s)", source_name, len(rows), base_dir)
    return rows


def load_aihub_free_adult(base_dir: Path) -> list[dict]:
    """AI허브 감정이 태깅된 자유대화(성인) (134-1)"""
    return _load_free_dialog_from_dir(base_dir, "aihub_free_adult")


def load_aihub_free_teen(base_dir: Path) -> list[dict]:
    """AI허브 감정이 태깅된 자유대화(청소년) (134-2)"""
    return _load_free_dialog_from_dir(base_dir, "aihub_free_teen")


def load_aihub_empathetic(base_dir: Path) -> list[dict]:
    """AI허브 공감형 대화 말뭉치 (5번째 소스)

    공감형 대화는 두 가지 JSON 구조를 지원한다:

    구조 A — 대화 단위 (한 파일 = 대화 N건):
      {
        "data": [
          {
            "utterances": [
              { "text": "발화", "emotion": "기쁨" },
              ...
            ]
          }
        ]
      }

    구조 B — 발화 단위 리스트 (한 파일 = 발화 N건):
      [
        { "utterance": "발화", "emotion": "감사" },
        ...
      ]

    두 구조 모두 시도하고, 감정 레이블을 EMPATHETIC_EMOTION_MAP 으로 3-class 변환한다.
    인식하지 못한 감정 레이블은 스킵하고 경고를 출력한다.
    """
    rows = []
    unknown_emotions: set[str] = set()

    for json_file in base_dir.rglob("*.json"):
        try:
            with json_file.open(encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("공감형 대화 JSON 읽기 실패: %s — %s", json_file, e)
            continue

        utterances_raw: list[dict] = []

        # 구조 A: {"data": [...]} 또는 {"dataset": [...]}
        if isinstance(data, dict):
            container = data.get("data") or data.get("dataset") or []
            for item in container:
                if isinstance(item, dict):
                    for utt in item.get("utterances", []) or item.get("conversations", []):
                        utterances_raw.append(utt)
        # 구조 B: 발화 리스트 직접
        elif isinstance(data, list):
            utterances_raw = data

        for utt in utterances_raw:
            if not isinstance(utt, dict):
                continue
            text = (utt.get("text") or utt.get("utterance") or utt.get("content") or "").strip()
            raw_emotion = (utt.get("emotion") or utt.get("EmotionType") or "").strip()
            if not text or not raw_emotion:
                continue
            emotion = EMPATHETIC_EMOTION_MAP.get(raw_emotion)
            if emotion is None:
                unknown_emotions.add(raw_emotion)
                continue
            rows.append({
                "text": text,
                "emotion": emotion,
                "dialog_act": "기타",
                "source": "aihub_empathetic",
            })

    if unknown_emotions:
        logger.warning("공감형 대화 — 매핑 없는 감정 레이블 (스킵): %s", unknown_emotions)

    logger.info("AI허브 공감형 대화: %d건 로드 (from %s)", len(rows), base_dir)
    return rows


def load_aihub_speech_style(base_dir: Path) -> list[dict]:
    """AI허브 감성 및 발화 스타일별 음성합성 데이터 (6번째 소스)

    TTS 데이터셋이지만 텍스트 + 감정 레이블을 감정 분류 학습에 활용한다.

    지원하는 JSON 구조:
      구조 A — 메타 리스트:
        [{"text": "발화", "emotion": "기쁨", "speaker_id": "...", ...}, ...]

      구조 B — data 래퍼:
        {"data": [{"script": "발화", "EmotionType": "슬픔", ...}, ...]}

      구조 C — 파일명 규칙 + txt 스크립트 (JSON 없는 경우):
        metadata.csv 또는 label.json 에 text, emotion 컬럼

    모든 감정 레이블은 SPEECH_STYLE_EMOTION_MAP 으로 3-class 변환한다.
    """
    rows = []
    unknown_emotions: set[str] = set()

    for json_file in base_dir.rglob("*.json"):
        try:
            with json_file.open(encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("감성발화스타일 JSON 읽기 실패: %s — %s", json_file, e)
            continue

        raw_items: list[dict] = []

        if isinstance(data, list):
            raw_items = data
        elif isinstance(data, dict):
            for key in ("data", "dataset", "items", "scripts"):
                if isinstance(data.get(key), list):
                    raw_items = data[key]
                    break

        for item in raw_items:
            if not isinstance(item, dict):
                continue
            text = (
                item.get("text") or item.get("script") or item.get("sentence")
                or item.get("content") or item.get("Text") or ""
            ).strip()
            raw_emotion = (
                item.get("emotion") or item.get("EmotionType") or item.get("Emotion")
                or item.get("emotion_label") or item.get("style") or ""
            ).strip()
            if not text or not raw_emotion:
                continue

            emotion = SPEECH_STYLE_EMOTION_MAP.get(raw_emotion)
            if emotion is None:
                unknown_emotions.add(raw_emotion)
                continue
            rows.append({
                "text": text,
                "emotion": emotion,
                "dialog_act": "기타",
                "source": "aihub_speech_style",
            })

    if unknown_emotions:
        logger.warning("감성발화스타일 — 매핑 없는 감정 레이블 (스킵): %s", unknown_emotions)

    logger.info("AI허브 감성발화스타일: %d건 로드 (from %s)", len(rows), base_dir)
    return rows


def load_aihub_purposive_dialog(base_dir: Path) -> list[dict]:
    """AI허브 용도별 목적대화 데이터 (dialog_act 15종 보강)

    대화 의도(intent/dialog_act) 레이블을 기반으로 dialog_act 를 채우고,
    감정 레이블이 있는 경우 emotion 도 함께 추출한다.

    지원하는 JSON 구조:
      구조 A — 대화 단위:
        {"dialogues": [{"utterances": [{"text": "...", "intent": "질문", "emotion": "중립"}]}]}

      구조 B — 발화 리스트:
        [{"utterance": "...", "dialog_act": "정보요청", ...}, ...]

      구조 C — AI허브 표준 래퍼:
        {"data": [{"Conversation": [{"Text": "...", "IntentType": "지시", ...}]}]}

    dialog_act 는 PURPOSIVE_DIALOG_ACT_MAP 으로 매핑한다.
    매핑 없는 의도 레이블은 "기타" 로 처리한다.
    """
    rows = []
    unknown_acts: set[str] = set()

    for json_file in base_dir.rglob("*.json"):
        try:
            with json_file.open(encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("목적대화 JSON 읽기 실패: %s — %s", json_file, e)
            continue

        raw_utterances: list[dict] = []

        if isinstance(data, list):
            # 구조 B: 발화 리스트 직접
            raw_utterances = data
        elif isinstance(data, dict):
            # 구조 A: dialogues 래퍼
            for dialogue in data.get("dialogues", []):
                raw_utterances.extend(dialogue.get("utterances", []))
            # 구조 C: data > Conversation 래퍼
            for item in data.get("data", []):
                if isinstance(item, dict):
                    raw_utterances.extend(item.get("Conversation", []))
            # 단순 utterances 키
            if not raw_utterances:
                raw_utterances = data.get("utterances", [])

        for utt in raw_utterances:
            if not isinstance(utt, dict):
                continue
            text = (
                utt.get("text") or utt.get("utterance") or utt.get("Text")
                or utt.get("content") or ""
            ).strip()
            if not text:
                continue

            # dialog_act 추출
            raw_act = (
                utt.get("intent") or utt.get("dialog_act") or utt.get("IntentType")
                or utt.get("DialogAct") or utt.get("act") or ""
            ).strip()
            dialog_act = PURPOSIVE_DIALOG_ACT_MAP.get(raw_act)
            if dialog_act is None and raw_act:
                unknown_acts.add(raw_act)
                dialog_act = "기타"
            elif dialog_act is None:
                dialog_act = "기타"

            # emotion 추출 (있으면 활용, 없으면 중립 기본)
            raw_emotion = (
                utt.get("emotion") or utt.get("EmotionType") or utt.get("Emotion") or ""
            ).strip()
            emotion = (
                EMPATHETIC_EMOTION_MAP.get(raw_emotion)
                or SPEECH_STYLE_EMOTION_MAP.get(raw_emotion)
            )
            if emotion is None:
                emotion = "중립"

            rows.append({
                "text": text,
                "emotion": emotion,
                "dialog_act": dialog_act,
                "source": "aihub_purposive",
            })

    if unknown_acts:
        logger.info("목적대화 — 미매핑 dialog_act (기타 처리): %s", unknown_acts)

    logger.info("AI허브 목적대화: %d건 로드 (from %s)", len(rows), base_dir)
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
    parser.add_argument("--no-balance", action="store_true", help="클래스 균형 조정 생략")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)

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

        if d := os.environ.get("DATASET_AIHUB_EMPATHETIC_DIR"):
            all_rows += load_aihub_empathetic(Path(d))
        else:
            logger.info("DATASET_AIHUB_EMPATHETIC_DIR 미설정 — 공감형 대화 건너뜀 (선택)")

        if d := os.environ.get("DATASET_AIHUB_SPEECH_STYLE_DIR"):
            all_rows += load_aihub_speech_style(Path(d))
        else:
            logger.info("DATASET_AIHUB_SPEECH_STYLE_DIR 미설정 — 감성발화스타일 건너뜀 (선택)")

        if d := os.environ.get("DATASET_AIHUB_PURPOSIVE_DIR"):
            all_rows += load_aihub_purposive_dialog(Path(d))
        else:
            logger.info("DATASET_AIHUB_PURPOSIVE_DIR 미설정 — 목적대화 건너뜀 (선택)")

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


if __name__ == "__main__":
    main()
