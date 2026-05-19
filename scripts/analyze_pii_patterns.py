"""AI허브 '숫자가 포함된 패턴 발화 데이터' 분석 스크립트

목적:
    데이터셋 내 발화에서 숫자 포함 패턴을 추출·분류하여
    pii_masker.py 에 추가할 신규 regex 패턴을 제안한다.

사용법:
    export DATASET_AIHUB_NUMBER_DIR=/path/to/aihub_number_dataset
    python scripts/analyze_pii_patterns.py [--top N] [--output FILE]

환경변수:
    DATASET_AIHUB_NUMBER_DIR  — 데이터셋 루트 디렉토리 (*.json 재귀 탐색)

옵션:
    --top N          패턴별 상위 N개 예시 출력 (기본 5)
    --output FILE    분석 결과 JSON 저장 경로 (기본 scripts/pii_pattern_report.json)
    --dummy          더미 데이터 20건으로 동작 검증
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Windows 콘솔 UTF-8 출력 강제
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── 기존 pii_masker.py 패턴 (커버 여부 판단 기준) ─────────────────────────────
EXISTING_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("주민등록번호",   re.compile(r"\d{6}\s*[-]\s*[1-4]\d{6}")),
    ("운전면허번호",   re.compile(r"\d{2}-\d{2}-\d{6}-\d{2}")),
    ("여권번호",      re.compile(r"[A-Z]\d{8}")),
    ("카드번호",      re.compile(r"\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]\d{4}")),
    ("이메일",        re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")),
    ("전화번호_구분",  re.compile(r"0\d{1,2}[\s.-]\d{3,4}[\s.-]\d{4}")),
    ("전화번호_연속",  re.compile(r"01[0-9]\d{3,4}\d{4}")),
    ("계좌번호",      re.compile(r"\b\d{3}\d{8,11}\b")),
    ("IP주소",        re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")),
]

# ── 신규 탐지 후보 패턴 (기존 미포함 가능성 있는 것들) ─────────────────────────
CANDIDATE_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    # label, pattern, description
    ("사업자등록번호", re.compile(r"\b\d{3}-\d{2}-\d{5}\b"), "XXX-XX-XXXXX"),
    ("법인번호",       re.compile(r"\b\d{6}-\d{7}\b"), "XXXXXX-XXXXXXX"),
    ("우편번호",       re.compile(r"\b\d{5}\b"), "5자리 우편번호"),
    ("날짜_슬래시",    re.compile(r"\b\d{2,4}/\d{1,2}/\d{1,2}\b"), "YYYY/MM/DD"),
    ("날짜_점",        re.compile(r"\b\d{2,4}\.\d{1,2}\.\d{1,2}\b"), "YYYY.MM.DD"),
    ("날짜_한국어",    re.compile(r"\d{4}년\s*\d{1,2}월\s*\d{1,2}일"), "N년 N월 N일"),
    ("시각_한국어",    re.compile(r"\d{1,2}시\s*\d{2}분"), "N시 N분"),
    ("은행코드_포함",  re.compile(r"\b\d{3}-\d{6,}-\d{2,}\b"), "은행코드-계좌-체크"),
    ("부동산_면적",    re.compile(r"\d+\.?\d*\s*(?:평|㎡|m²)"), "면적 단위"),
    ("금액_만원",      re.compile(r"\d[\d,]*\s*만\s*원"), "X만 원"),
    ("금액_억원",      re.compile(r"\d[\d,]*\s*억\s*원?"), "X억 원"),
    ("외국번호",       re.compile(r"\+\d{1,3}[\s-]\d{2,4}[\s-]\d{3,4}[\s-]\d{4}"), "+국가코드 번호"),
    ("차량번호_신형",  re.compile(r"\b\d{2,3}[가-힣]\d{4}\b"), "12가1234 형식"),
    ("차량번호_구형",  re.compile(r"\b[가-힣]{2}\d{2}[가-힣]\d{4}\b"), "서울12가1234 형식"),
    ("건물동호수",     re.compile(r"\d+\s*동\s*\d+\s*호"), "N동 N호"),
]

# ── 숫자 포함 범용 패턴 (분류 안 된 숫자열 탐지용) ────────────────────────────
_RAW_NUMBER_PATTERN = re.compile(r"[\d][\d\s가-힣A-Za-z.,/-]{1,30}[\d]")


# ── 더미 데이터 ──────────────────────────────────────────────────────────────
DUMMY_TEXTS = [
    "제 연락처는 010-1234-5678이고 이메일은 hong@example.com입니다",
    "계좌번호는 110-123-456789이고 사업자번호는 123-45-67890입니다",
    "주소는 서울시 강남구 테헤란로 231동 1504호입니다",
    "2025년 3월 15일에 계약했고 금액은 3억 5천만 원입니다",
    "차량번호 12가3456이고 등록일은 2023/01/15입니다",
    "주민번호 900101-1234567이고 여권은 M12345678입니다",
    "카드번호 1234-5678-9012-3456 만료 25/12",
    "법인번호 110111-1234567 대표이사 홍길동",
    "우편번호 06100 서울 강남구",
    "면적은 33.5평이고 보증금은 2억 5천만원입니다",
    "회의는 오후 3시 30분에 시작합니다",
    "해외 연락처 +82 10-1234-5678로 전화주세요",
    "운전면허 12-34-567890-12 번입니다",
    "IP는 192.168.1.100이고 포트는 8080입니다",
    "총 결제금액 1,200,000원입니다",
    "2024.12.31까지 유효한 할인쿠폰",
    "사업자번호 456-78-12345로 세금계산서 발행해드릴게요",
    "거래처 법인등록번호는 200301-0123456입니다",
    "보험번호 1234-5678-90 청구금액 35만원",
    "차량 서울12가5678 주차요금 정산해드립니다",
]


def _load_json_file(path: Path) -> list[str]:
    """JSON 파일에서 발화 텍스트를 추출한다. 여러 구조 허용."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("JSON 파싱 실패 %s: %s", path.name, e)
        return []

    texts: list[str] = []

    def _extract(obj: object) -> None:
        if isinstance(obj, str) and len(obj) > 1:
            texts.append(obj)
            return
        if isinstance(obj, dict):
            for key in ("text", "sentence", "script", "utterance", "content",
                        "transcription", "발화", "원문"):
                if isinstance(obj.get(key), str) and obj[key].strip():
                    texts.append(obj[key].strip())
                    return
            # 키를 못 찾으면 모든 string 값 재귀
            for v in obj.values():
                _extract(v)
        if isinstance(obj, list):
            for item in obj:
                _extract(item)

    _extract(raw)
    return texts


def load_dataset(base_dir: Path) -> list[str]:
    """데이터셋 디렉토리에서 모든 발화 텍스트를 로드한다."""
    all_texts: list[str] = []
    json_files = list(base_dir.rglob("*.json"))
    logger.info("JSON 파일 %d개 발견 (%s)", len(json_files), base_dir)

    for jf in json_files:
        texts = _load_json_file(jf)
        all_texts.extend(texts)

    logger.info("총 발화 %d건 로드", len(all_texts))
    return all_texts


def analyze_existing_coverage(texts: list[str]) -> dict[str, int]:
    """기존 pii_masker.py 패턴별 히트 수 집계."""
    counts: dict[str, int] = {label: 0 for label, _ in EXISTING_PATTERNS}
    for text in texts:
        for label, pat in EXISTING_PATTERNS:
            if pat.search(text):
                counts[label] += 1
    return counts


def analyze_candidates(
    texts: list[str], top_n: int = 5
) -> dict[str, dict]:
    """후보 패턴별 히트 수 + 예시 수집."""
    results: dict[str, dict] = {}
    for label, pat, desc in CANDIDATE_PATTERNS:
        hits = 0
        matched_texts: list[str] = []
        examples: list[str] = []

        for text in texts:
            m = pat.search(text)
            if m:
                hits += 1
                if len(matched_texts) < top_n:
                    matched_texts.append(text[:120])
                    examples.append(m.group(0))

        if hits > 0:
            results[label] = {
                "description": desc,
                "hit_count": hits,
                "hit_rate_pct": round(hits / len(texts) * 100, 2),
                "example_matches": examples[:top_n],
                "example_texts": matched_texts[:top_n],
            }
    return results


def extract_uncovered_number_shapes(
    texts: list[str], top_n: int = 20
) -> list[dict]:
    """기존·후보 패턴 모두 커버하지 못하는 숫자열의 패턴 형태를 추출한다.

    숫자→N, 한글→H, 영문→A, 공백→_ 로 변환하여 shape 집계.
    """
    all_known = [p for _, p in EXISTING_PATTERNS] + [p for _, p, _ in CANDIDATE_PATTERNS]
    shape_counter: Counter = Counter()
    shape_examples: dict[str, list[str]] = defaultdict(list)

    for text in texts:
        # 기존/후보 패턴이 커버하는 위치 마킹
        covered = set()
        for pat in all_known:
            for m in pat.finditer(text):
                covered.update(range(m.start(), m.end()))

        for m in _RAW_NUMBER_PATTERN.finditer(text):
            span_chars = set(range(m.start(), m.end()))
            if span_chars & covered:
                continue  # 이미 알려진 패턴에 포함됨
            raw = m.group(0)
            shape = re.sub(r"[A-Za-z]", "L", raw)
            shape = re.sub(r"[0-9]", "N", shape)
            shape = re.sub(r"[가-힣]", "H", shape)
            shape = re.sub(r"\s+", "_", shape)
            shape_counter[shape] += 1
            if len(shape_examples[shape]) < 3:
                shape_examples[shape].append(raw)

    return [
        {
            "shape": shape,
            "count": cnt,
            "examples": shape_examples[shape],
        }
        for shape, cnt in shape_counter.most_common(top_n)
    ]


def generate_regex_suggestions(candidates: dict[str, dict]) -> list[dict]:
    """히트율 높은 후보 패턴에서 pii_masker.py 추가 제안 생성."""
    # 히트 수 기준 내림차순
    sorted_cands = sorted(
        candidates.items(), key=lambda x: x[1]["hit_count"], reverse=True
    )

    suggestions = []
    regex_templates = {label: pat for label, pat, _ in CANDIDATE_PATTERNS}

    for label, info in sorted_cands:
        if info["hit_count"] < 3:
            continue
        pat = regex_templates.get(label)
        suggestions.append({
            "label": label,
            "description": info["description"],
            "hit_count": info["hit_count"],
            "hit_rate_pct": info["hit_rate_pct"],
            "regex": pat.pattern if pat else "(unknown)",
            "note": (
                "pii_masker.py PII_PATTERNS 리스트에 추가 검토"
                if info["hit_rate_pct"] >= 1.0
                else "빈도 낮음 — 선택적 추가"
            ),
        })
    return suggestions


def print_report(
    texts: list[str],
    coverage: dict[str, int],
    candidates: dict[str, dict],
    uncovered: list[dict],
    suggestions: list[dict],
) -> None:
    total = len(texts)
    print("\n" + "=" * 70)
    print("  PII 패턴 분석 리포트")
    print("=" * 70)
    print(f"\n  분석 발화 수: {total:,}건\n")

    print("── 기존 pii_masker.py 패턴 커버리지 ──────────────────────────────")
    for label, cnt in coverage.items():
        pct = cnt / total * 100 if total else 0
        print(f"  {label:16s}: {cnt:6,}건  ({pct:.2f}%)")

    print("\n── 신규 후보 패턴 (기존 미포함) ──────────────────────────────────")
    if candidates:
        for label, info in sorted(candidates.items(),
                                   key=lambda x: x[1]["hit_count"], reverse=True):
            print(f"  [{label}] {info['description']} - {info['hit_count']:,}건 ({info['hit_rate_pct']}%)")
            for ex in info["example_matches"][:3]:
                print(f"      예: {ex}")
    else:
        print("  후보 패턴 히트 없음")

    print("\n── 미분류 숫자 패턴 형태 (상위 15개) ──────────────────────────────")
    for item in uncovered[:15]:
        exs = " / ".join(item["examples"])
        print(f"  {item['shape']:30s} ({item['count']:4,}건)  예: {exs}")

    print("\n── pii_masker.py 추가 제안 ────────────────────────────────────────")
    if suggestions:
        for s in suggestions:
            note = "★ 추가 권장" if s["hit_rate_pct"] >= 1.0 else "  검토 후 추가"
            print(f"  {note} [{s['label']}] {s['regex']}")
    else:
        print("  신규 제안 없음")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="PII 숫자 패턴 분석")
    parser.add_argument("--top", type=int, default=5, metavar="N", help="패턴별 예시 수 (기본 5)")
    parser.add_argument(
        "--output",
        default="scripts/pii_pattern_report.json",
        help="분석 결과 JSON 저장 경로",
    )
    parser.add_argument("--dummy", action="store_true", help="더미 데이터 20건으로 동작 검증")
    args = parser.parse_args()

    if args.dummy:
        logger.info("더미 모드 — %d건 사용", len(DUMMY_TEXTS))
        texts = DUMMY_TEXTS
    else:
        data_dir = os.environ.get("DATASET_AIHUB_NUMBER_DIR")
        if not data_dir:
            logger.error("DATASET_AIHUB_NUMBER_DIR 환경변수가 설정되지 않았습니다")
            sys.exit(1)
        base = Path(data_dir)
        if not base.is_dir():
            logger.error("디렉토리가 없습니다: %s", base)
            sys.exit(1)
        texts = load_dataset(base)
        if not texts:
            logger.error("발화 데이터를 로드할 수 없습니다. JSON 구조를 확인하세요.")
            sys.exit(1)

    logger.info("기존 패턴 커버리지 분석 중...")
    coverage = analyze_existing_coverage(texts)

    logger.info("신규 후보 패턴 분석 중...")
    candidates = analyze_candidates(texts, top_n=args.top)

    logger.info("미분류 숫자 형태 추출 중...")
    uncovered = extract_uncovered_number_shapes(texts, top_n=30)

    suggestions = generate_regex_suggestions(candidates)

    print_report(texts, coverage, candidates, uncovered, suggestions)

    report = {
        "total_texts": len(texts),
        "existing_coverage": coverage,
        "new_candidates": candidates,
        "uncovered_shapes": uncovered,
        "suggestions": suggestions,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("분석 결과 저장: %s", out_path)


if __name__ == "__main__":
    main()
