"""STAGE 16: 세그먼트 기반 주제 라벨 서비스.

KcELECTRA 임베딩 cosine 유사도 기반 경계 탐지 후 고정 30개 주제 분류.
임베딩 모델이 없으면 단일 세그먼트(주제=null) graceful degradation.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

_COSINE_THRESHOLD = float(os.environ.get("TOPIC_COSINE_THRESHOLD", "0.35"))  # (구) 절대 임계 — depth 경로에선 미사용
_MIN_UTTERANCES_PER_SEGMENT = int(os.environ.get("TOPIC_MIN_UTTERANCES", "3"))
# TextTiling 상대낙폭(depth) 경계탐지 파라미터 (그리드스캔으로 튜닝)
_DEPTH_THRESHOLD_DELTA = float(os.environ.get("TOPIC_DEPTH_DELTA", "0.06"))  # 그리드스캔 최적(F1 0.17→0.36)
_DEPTH_WINDOW = int(os.environ.get("TOPIC_DEPTH_WINDOW", "5"))   # 전후 평균 윈도우 k (스캔 최적)
_SPEAKER_TURN_AMPLIFY = 1.5   # 화자 교대점 depth 증폭(양방향 보강)

# ── 고정 30개 주제 분류 (seed 문구 기반) ─────────────────────────────────────
TOPIC_SEED_PHRASES: dict[str, list[str]] = {
    "건강/의료": ["병원", "진료", "건강", "약", "증상", "치료", "몸이 아파", "수술"],
    "식사/음식": ["밥", "먹었어", "음식", "식당", "요리", "배고파", "맛있어", "식사"],
    "날씨/계절": ["날씨", "더워", "추워", "비", "눈", "바람", "봄", "여름", "가을", "겨울"],
    "직장/업무": ["회사", "일", "업무", "회의", "프로젝트", "출근", "퇴근", "야근"],
    "가족": ["가족", "부모님", "아이", "자녀", "남편", "아내", "형제", "부부"],
    "여행/외출": ["여행", "놀러", "관광", "숙소", "비행기", "기차", "드라이브"],
    "쇼핑/소비": ["쇼핑", "샀어", "구매", "할인", "가격", "비싸", "싸", "마트"],
    "금융/돈": ["돈", "저축", "투자", "대출", "월급", "보험", "세금", "은행"],
    "교육/공부": ["공부", "학교", "수업", "시험", "과외", "학원", "선생님", "성적"],
    "취미/여가": ["취미", "운동", "독서", "영화", "게임", "등산", "낚시", "음악"],
    "연애/관계": ["좋아해", "사귀", "연애", "데이트", "고백", "헤어졌어", "남자친구", "여자친구"],
    "집/생활": ["집", "이사", "청소", "인테리어", "전세", "월세", "아파트"],
    "교통/이동": ["버스", "지하철", "차", "택시", "운전", "길이 막혀", "주차"],
    "뉴스/시사": ["뉴스", "정치", "사회", "사건", "정부", "선거", "법"],
    "스포츠": ["경기", "야구", "축구", "농구", "골프", "운동 경기", "응원"],
    "IT/기술": ["스마트폰", "앱", "컴퓨터", "인터넷", "AI", "프로그래밍", "소프트웨어"],
    "문화/예술": ["영화", "드라마", "음악", "콘서트", "전시", "책", "뮤지컬"],
    "반려동물": ["강아지", "고양이", "반려동물", "동물", "키워", "펫"],
    "종교/신앙": ["교회", "절", "기도", "신앙", "예배", "불교", "성당"],
    "환경/자연": ["환경", "자연", "기후", "재활용", "오염", "식물", "숲"],
    "경조사": ["결혼", "장례", "돌잔치", "생일", "기념일", "축하", "조문"],
    "친목/모임": ["모임", "약속", "술", "파티", "동창", "동호회", "번개"],
    "갈등/고민": ["힘들어", "걱정", "고민", "갈등", "화가 나", "스트레스", "짜증"],
    "감사/칭찬": ["감사해", "고마워", "잘했어", "멋있어", "대단해", "덕분에"],
    "안부/인사": ["잘 지내", "어떻게 지내", "오랜만이야", "반가워", "잘 있어"],
    "계획/약속": ["계획", "예정", "다음에", "언제", "약속", "일정", "스케줄"],
    "추억/과거": ["예전에", "옛날에", "기억나", "그때", "어릴 때", "추억"],
    "자녀양육": ["아기", "육아", "유치원", "초등학교", "학교 준비", "아이 키우"],
    "부동산": ["집값", "아파트 가격", "부동산", "분양", "재건축", "청약"],
    "기타": [],
}

TOPIC_LABELS = list(TOPIC_SEED_PHRASES.keys())


@dataclass
class TopicSegmentResult:
    segment_index: int
    topic: str | None
    start_ms: int
    end_ms: int
    utterance_indices: list[int] = field(default_factory=list)
    topic_confidence: float = 0.0          # 학습모델 분류 신뢰도 (키워드 fallback이면 0.0)
    topic_method: str = "keyword"          # "model"(학습 분류기) | "keyword"(fallback)


# ---------------------------------------------------------------------------
# 주제 분류 (seed phrase 매칭 — 임베딩 없을 때 fallback)
# ---------------------------------------------------------------------------

def _classify_topic_by_keywords(texts: list[str]) -> str | None:
    combined = " ".join(texts)
    best_topic: str | None = None
    best_count = 0
    for topic, seeds in TOPIC_SEED_PHRASES.items():
        if not seeds:
            continue
        count = sum(1 for s in seeds if s in combined)
        if count > best_count:
            best_count = count
            best_topic = topic
    if best_count == 0:
        return "기타"
    return best_topic


# ---------------------------------------------------------------------------
# KcELECTRA 임베딩 기반 세그먼트 경계 탐지
# ---------------------------------------------------------------------------

def _get_kcelectra_embeddings(texts: list[str]):
    """auto_label_service(감정) 인코더로 [CLS] 임베딩을 추출한다. 실패 시 None.

    ※ topic 전용 인코더로 교체 실험했으나 경계 F1 개선 없음(0.357≈0.359, 화자보강
      약화)이라 감정 인코더 유지(2026-06-07 그리드스캔 검증). depth_delta=0.06/win=5는
      이 감정 인코더 기준 최적값.
    """
    try:
        from app.services.auto_label_service import auto_label_service
        if not auto_label_service.is_available():
            return None
        return auto_label_service.encode(texts)
    except (ImportError, AttributeError):
        return None


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 1.0
    return float(np.dot(a, b) / denom)


# 같은 화자 연속 발화 중간의 경계는 문턱을 이 배수로 높여 거짓컷 억제(화자 교대 보강 A).
_SAME_SPEAKER_THRESHOLD_FACTOR = 1.5


def _calculate_depth_scores(sims: list[float], window: int) -> list[float]:
    """TextTiling 상대낙폭: 각 gap의 (좌윈도우평균-sim)+(우윈도우평균-sim). 음수는 0.

    절대 임계가 아닌 *주변 대비 국소 곡곡점(valley)* 을 잡아 과소분할 탈출.
    """
    n = len(sims)
    depths: list[float] = []
    for i in range(n):
        lo = sims[max(0, i - window):i]
        hi = sims[i + 1:i + 1 + window]
        mu_l = (sum(lo) / len(lo)) if lo else sims[i]
        mu_r = (sum(hi) / len(hi)) if hi else sims[i]
        depths.append(max(0.0, (mu_l - sims[i]) + (mu_r - sims[i])))
    return depths


def _detect_boundaries(
    embeddings: np.ndarray,
    threshold: float,                 # (구) 절대임계 — 호환 위해 시그니처 유지, depth 경로 미사용
    min_per_segment: int,
    speaker_ids: list | None = None,
    depth_delta: float | None = None,
    window: int | None = None,
) -> list[int]:
    """TextTiling 상대낙폭(depth) + 양방향 화자보강 경계 탐지.

    - sims[g] = cos(emb[g], emb[g+1]) (gap g, 경계 후보 인덱스 = g+1)
    - depth(g) = (좌평균 - sims[g]) + (우평균 - sims[g]) — 국소 곡곡점
    - 화자보강(양방향): 경계후보가 화자 교대점이면 depth ×1.5 증폭, 같은 화자
      연속이면 ÷1.5 억제. speaker_ids=None 이면 텍스트만.
    - depth >= depth_delta 이고 min_per_segment 간격이면 경계 확정.
    """
    depth_delta = _DEPTH_THRESHOLD_DELTA if depth_delta is None else depth_delta
    window = _DEPTH_WINDOW if window is None else window
    n = len(embeddings)
    if n < 2:
        return []
    sims = [_cosine_sim(embeddings[i], embeddings[i + 1]) for i in range(n - 1)]
    depths = _calculate_depth_scores(sims, window)

    boundaries: list[int] = []
    since_last = 0
    for g in range(n - 1):
        bidx = g + 1
        since_last += 1
        d = depths[g]
        if speaker_ids is not None and bidx < len(speaker_ids):
            if speaker_ids[bidx] is not None and speaker_ids[bidx] == speaker_ids[bidx - 1]:
                d /= _SAME_SPEAKER_THRESHOLD_FACTOR        # 같은 화자 연속 → 억제
            elif speaker_ids[bidx] != speaker_ids[bidx - 1]:
                d *= _SPEAKER_TURN_AMPLIFY                  # 화자 교대 → 증폭
        if d >= depth_delta and since_last >= min_per_segment:
            boundaries.append(bidx)
            since_last = 0
    return boundaries


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

def segment_topics(
    utterances: list[dict],
) -> list[TopicSegmentResult]:
    """utterances를 주제 세그먼트로 분할하고 주제를 분류한다.

    utterances 각 항목은 최소한 아래 키를 포함해야 한다:
      - index: int
      - start_sec: float
      - end_sec: float
      - transcript_text: str

    Returns:
        list[TopicSegmentResult] (비어 있으면 단일 "기타" 세그먼트)
    """
    if not utterances:
        return []

    texts = [u.get("transcript_text", "") for u in utterances]
    # 화자 교대 보강(A) — speaker_id(없으면 speaker) 추출. 둘 다 없으면 None → 기존 동작.
    speaker_ids = [u.get("speaker_id") or u.get("speaker") for u in utterances]
    if not any(s is not None for s in speaker_ids):
        speaker_ids = None

    # KcELECTRA 임베딩 기반 경계 탐지 시도
    boundaries: list[int] = []
    embeddings = _get_kcelectra_embeddings(texts)
    if embeddings is not None and len(embeddings) >= 2:
        try:
            boundaries = _detect_boundaries(
                embeddings,
                threshold=_COSINE_THRESHOLD,
                min_per_segment=_MIN_UTTERANCES_PER_SEGMENT,
                speaker_ids=speaker_ids,
            )
            logger.info("[topic_seg] 임베딩 기반 경계 %d개 탐지", len(boundaries))
        except Exception as exc:
            logger.warning("[topic_seg] 경계 탐지 실패 — fallback: %s", exc)
            boundaries = []

    # 경계 → 세그먼트 구간
    split_points = [0] + boundaries + [len(utterances)]
    results: list[TopicSegmentResult] = []

    # 세그먼트별 텍스트 묶음 — 학습 분류기(세그먼트 단위 0.79)로 일괄 분류, 없으면 키워드 fallback
    seg_text_blobs: list[str] = []
    seg_ranges: list[tuple[int, int]] = []
    for start_i, end_i in zip(split_points[:-1], split_points[1:]):
        seg_utts = utterances[start_i:end_i]
        seg_text_blobs.append(" ".join(u.get("transcript_text", "") for u in seg_utts))
        seg_ranges.append((start_i, end_i))

    model_preds: list[tuple[str | None, float]] = []
    try:
        from app.services.topic_classifier_service import topic_classifier_service
        if topic_classifier_service.is_available():
            model_preds = topic_classifier_service.classify_batch(seg_text_blobs)
    except Exception as exc:
        logger.warning("[topic_seg] 학습 분류기 사용 불가 — 키워드 fallback: %s", exc)

    for seg_idx, (start_i, end_i) in enumerate(seg_ranges):
        seg_utts = utterances[start_i:end_i]
        seg_texts = [u.get("transcript_text", "") for u in seg_utts]

        if seg_idx < len(model_preds) and model_preds[seg_idx][0]:
            topic, topic_conf, method = model_preds[seg_idx][0], model_preds[seg_idx][1], "model"
        else:
            topic, topic_conf, method = _classify_topic_by_keywords(seg_texts), 0.0, "keyword"

        start_ms = int(seg_utts[0]["start_sec"] * 1000) if seg_utts else 0
        end_ms = int(seg_utts[-1]["end_sec"] * 1000) if seg_utts else 0
        utt_indices = [u["index"] for u in seg_utts]

        results.append(TopicSegmentResult(
            segment_index=seg_idx,
            topic=topic,
            start_ms=start_ms,
            end_ms=end_ms,
            utterance_indices=utt_indices,
            topic_confidence=topic_conf,
            topic_method=method,
        ))
        logger.debug("[topic_seg] seg[%d] %s (%s, %d발화)", seg_idx, topic, method, len(seg_utts))

    logger.info("[topic_seg] 주제 세그먼트 %d개 생성", len(results))
    return results
