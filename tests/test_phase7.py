"""Phase 7 학습 스크립트 테스트.

순수 함수 unit test + dummy 통합 테스트 (subprocess).
- _undersample_kita, _compute_dialog_act_weights
- make_dummy_rows (3개 스크립트)
- compute_wer, compute_cer
- _is_split_path, _normalize_topic
- subprocess: train_X.py --dummy --cpu

NOTE: 이 파일은 scripts/ 디렉토리 옆 tests/ 에서 import 한다.
sys.path에 프로젝트 루트를 추가해 scripts.X 형태로 import.
"""
from __future__ import annotations

import json
import os
import sys
import subprocess
import tempfile
from pathlib import Path

import pytest

# 프로젝트 루트(=uncounted-voice-api)를 sys.path에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

# 스크립트 import (scripts/__init__.py 없을 수도 있어 importlib 사용)
import importlib.util


def _import_script(name: str):
    """scripts/<name>.py 를 모듈로 로드."""
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 모듈 로드 (한 번만)
# ---------------------------------------------------------------------------

train_emotion = _import_script("train_emotion_model")
train_age = _import_script("train_speech_age_model")
train_topic = _import_script("train_topic_model")
train_speech_act = _import_script("train_speech_act_model")
measure_baseline = _import_script("measure_whisperx_baseline")
prepare_topic = _import_script("prepare_topic_dataset")


# ===========================================================================
# train_emotion_model._undersample_kita
# ===========================================================================

class TestUndersampleKita:
    """기타 라벨 언더샘플링 (max_ratio 비율 이하로 유지)."""

    def test_no_change_when_kita_ratio_below_threshold(self):
        """기타 비율이 max_ratio보다 낮으면 변경 없음."""
        rows = [
            {"dialog_act": "진술"} for _ in range(8)
        ] + [{"dialog_act": "기타"} for _ in range(2)]
        result = train_emotion._undersample_kita(rows, max_ratio=0.5)
        # 기타 2건, 비기타 8건 → 비율 0.2 < 0.5 → 변경 없음
        assert len(result) == 10
        assert sum(1 for r in result if r["dialog_act"] == "기타") == 2

    def test_undersamples_when_kita_ratio_exceeds_threshold(self):
        """기타 비율이 max_ratio를 초과하면 max_ratio까지 줄임."""
        # 비기타 4건, 기타 100건 → 기타 비율 100/(100+4)=0.96
        # max_ratio=0.5 → max_kita = 4 * 0.5/0.5 = 4
        rows = [
            {"dialog_act": "진술"} for _ in range(4)
        ] + [{"dialog_act": "기타"} for _ in range(100)]
        result = train_emotion._undersample_kita(rows, max_ratio=0.5)
        kita_count = sum(1 for r in result if r["dialog_act"] == "기타")
        non_kita_count = sum(1 for r in result if r["dialog_act"] != "기타")
        assert non_kita_count == 4
        assert kita_count == 4  # max_kita = non_kita * (0.5/0.5) = 4

    def test_total_equals_non_kita_plus_min_kita_and_max_kita(self):
        """result_total = non_kita + min(kita, max_kita) 불변식."""
        # 비기타 10건, 기타 30건, max_ratio=0.3
        # max_kita = 10 * 0.3/0.7 ≈ 4.28 → 4
        rows = [
            {"dialog_act": "진술"} for _ in range(10)
        ] + [{"dialog_act": "기타"} for _ in range(30)]
        max_ratio = 0.3
        result = train_emotion._undersample_kita(rows, max_ratio=max_ratio)
        max_kita_expected = int(10 * max_ratio / (1 - max_ratio))  # 4
        kita_count = sum(1 for r in result if r["dialog_act"] == "기타")
        non_kita_count = sum(1 for r in result if r["dialog_act"] != "기타")
        assert non_kita_count == 10
        assert kita_count == min(30, max_kita_expected)

    def test_empty_input_returns_empty(self):
        """빈 리스트 입력 → 빈 리스트 반환."""
        result = train_emotion._undersample_kita([], max_ratio=0.5)
        assert result == []

    def test_only_kita_input_keeps_zero_when_no_non_kita(self):
        """비기타가 0이면 max_kita=0 → 모두 제거."""
        rows = [{"dialog_act": "기타"} for _ in range(5)]
        result = train_emotion._undersample_kita(rows, max_ratio=0.5)
        # 비기타 0 → max_kita = 0*1 = 0 → 기타 모두 제거
        kita_count = sum(1 for r in result if r["dialog_act"] == "기타")
        assert kita_count == 0

    def test_missing_dialog_act_key_treated_as_non_kita(self):
        """dialog_act 키 누락 시 빈 문자열로 처리 → 비기타로 분류."""
        rows = [{"text": "x"} for _ in range(3)] + [{"dialog_act": "기타"} for _ in range(100)]
        result = train_emotion._undersample_kita(rows, max_ratio=0.5)
        # 비기타 3건(누락) → max_kita = 3
        kita_count = sum(1 for r in result if r.get("dialog_act", "") == "기타")
        assert kita_count == 3


# ===========================================================================
# train_emotion_model._compute_dialog_act_weights
# ===========================================================================

class TestComputeDialogActWeights:
    """dialog_act 클래스 가중치 계산 (역빈도)."""

    def test_returns_tensor_with_correct_shape(self):
        """torch.Tensor, shape=(15,) 반환."""
        rows = [{"dialog_act": "진술"} for _ in range(10)]
        device = torch.device("cpu")
        weights = train_emotion._compute_dialog_act_weights(rows, device)
        assert isinstance(weights, torch.Tensor)
        assert weights.shape == (15,)

    def test_all_weights_positive(self):
        """모든 가중치 > 0."""
        rows = [
            {"dialog_act": "진술"} for _ in range(10)
        ] + [{"dialog_act": "질문"} for _ in range(5)]
        device = torch.device("cpu")
        weights = train_emotion._compute_dialog_act_weights(rows, device)
        assert torch.all(weights > 0).item()

    def test_low_frequency_class_gets_higher_weight_than_high_frequency(self):
        """빈도 낮은 클래스의 가중치가 빈도 높은 클래스보다 큼."""
        # 진술 100건, 질문 5건 → 질문 가중치 > 진술 가중치
        rows = [
            {"dialog_act": "진술"} for _ in range(100)
        ] + [{"dialog_act": "질문"} for _ in range(5)]
        device = torch.device("cpu")
        weights = train_emotion._compute_dialog_act_weights(rows, device)
        # DIALOG_ACT_LABELS = ["진술", "질문", "요청", ...]
        # 진술=0, 질문=1
        assert weights[1].item() > weights[0].item()

    def test_unseen_class_keeps_default_weight_one(self):
        """관측되지 않은 클래스는 가중치 1.0 유지."""
        # 진술만 등장 → 다른 14개 클래스는 weight=1.0
        rows = [{"dialog_act": "진술"} for _ in range(10)]
        device = torch.device("cpu")
        weights = train_emotion._compute_dialog_act_weights(rows, device)
        # 진술(idx=0)만 관측, 나머지는 1.0
        for i in range(1, 15):
            assert weights[i].item() == pytest.approx(1.0)

    def test_weights_placed_on_specified_device(self):
        """device 인자대로 텐서 배치."""
        rows = [{"dialog_act": "진술"}]
        device = torch.device("cpu")
        weights = train_emotion._compute_dialog_act_weights(rows, device)
        assert weights.device.type == "cpu"


# ===========================================================================
# make_dummy_rows (3개 스크립트)
# ===========================================================================

class TestMakeDummyRows:
    """더미 데이터 생성."""

    def test_emotion_dummy_returns_n_rows(self):
        rows = train_emotion.make_dummy_rows(40)
        assert len(rows) == 40
        for r in rows:
            assert "text" in r
            assert "emotion" in r
            assert "dialog_act" in r

    def test_age_dummy_returns_n_rows_with_age_group(self):
        rows = train_age.make_dummy_rows(40)
        assert len(rows) == 40
        for r in rows:
            assert "text" in r
            assert "age_group" in r
            assert r["age_group"] in train_age.AGE_LABELS

    def test_topic_dummy_returns_n_rows_with_topic(self):
        rows = train_topic.make_dummy_rows(40)
        assert len(rows) == 40
        for r in rows:
            assert "text" in r
            assert "topic" in r
            assert "topic_group" in r
            assert r["topic"] in train_topic.TOPIC_LABELS

    def test_speech_act_dummy_returns_n_rows_with_group(self):
        rows = train_speech_act.make_dummy_rows(40)
        assert len(rows) == 40
        for r in rows:
            assert "text" in r
            assert "speech_act" in r
            assert "speech_act_group" in r
            assert r["speech_act_group"] in train_speech_act.SPEECH_ACT_GROUP_LABELS

    def test_emotion_dummy_n_zero(self):
        """n=0 호출은 빈 리스트 반환."""
        rows = train_emotion.make_dummy_rows(0)
        assert rows == []

    def test_emotion_dummy_n_one(self):
        """n=1 호출은 1건 반환."""
        rows = train_emotion.make_dummy_rows(1)
        assert len(rows) == 1


# ===========================================================================
# measure_whisperx_baseline.compute_wer / compute_cer
# ===========================================================================

class TestComputeWER:
    """단어 오류율."""

    def test_identical_strings_returns_zero(self):
        assert measure_baseline.compute_wer("안녕 하세요", "안녕 하세요") == 0.0

    def test_empty_reference_returns_zero(self):
        """빈 ref → 분모 0 회피로 0.0 반환."""
        assert measure_baseline.compute_wer("", "임의 텍스트") == 0.0

    def test_completely_different_strings_returns_high(self):
        """완전히 다른 문자열은 WER >= 1.0 가능."""
        # ref="A" 1단어, hyp="B C D" 3단어 → 거리=3, ref_len=1 → 3.0
        wer = measure_baseline.compute_wer("안녕", "다른 문장 입니다")
        assert wer > 0.0

    def test_single_word_substitution(self):
        """단어 1개 치환 → 1/3 = 0.333"""
        wer = measure_baseline.compute_wer("나는 학생 입니다", "나는 교사 입니다")
        assert wer == pytest.approx(1 / 3, rel=1e-3)

    def test_handles_special_characters_via_normalize(self):
        """구두점 제거 후 비교 → 동일하게 인식."""
        # _normalize가 [^\w\s가-힣] 제거 → "안녕!" == "안녕"
        wer = measure_baseline.compute_wer("안녕 하세요!", "안녕 하세요")
        assert wer == 0.0


class TestComputeCER:
    """문자 오류율."""

    def test_identical_strings_returns_zero(self):
        assert measure_baseline.compute_cer("가나다", "가나다") == 0.0

    def test_empty_reference_returns_zero(self):
        assert measure_baseline.compute_cer("", "텍스트") == 0.0

    def test_single_char_substitution(self):
        """문자 1개 치환 → 1/3"""
        cer = measure_baseline.compute_cer("가나다", "가니다")
        assert cer == pytest.approx(1 / 3, rel=1e-3)

    def test_completely_different_returns_high(self):
        cer = measure_baseline.compute_cer("가", "가나다라마")
        assert cer > 0.0


# ===========================================================================
# prepare_topic_dataset._is_split_path
# ===========================================================================

class TestIsSplitPath:
    """경로의 train/val 판정 (path.parts 기준, 파일명 의존 X)."""

    def test_training_path_with_tl_zip_is_train(self):
        p = Path("/data/Training/TL_001.zip")
        assert prepare_topic._is_split_path(p, "train") is True
        assert prepare_topic._is_split_path(p, "val") is False

    def test_validation_path_with_vl_zip_is_val(self):
        p = Path("/data/Validation/VL_001.zip")
        assert prepare_topic._is_split_path(p, "val") is True
        assert prepare_topic._is_split_path(p, "train") is False

    def test_validation_path_with_tl_filename_is_not_train(self):
        """파일명이 TL_여도 Validation 경로면 train=False."""
        p = Path("/data/Validation/TL_001.zip")
        assert prepare_topic._is_split_path(p, "train") is False
        assert prepare_topic._is_split_path(p, "val") is True

    def test_unrelated_path_is_neither(self):
        p = Path("/random/path/file.zip")
        assert prepare_topic._is_split_path(p, "train") is False
        assert prepare_topic._is_split_path(p, "val") is False

    def test_case_insensitive(self):
        """대소문자 무관."""
        p = Path("/data/TRAINING/file.zip")
        assert prepare_topic._is_split_path(p, "train") is True


# ===========================================================================
# prepare_topic_dataset._normalize_topic
# ===========================================================================

class TestNormalizeTopic:
    """주제 라벨 정규화."""

    def test_normalizes_known_typo(self):
        assert prepare_topic._normalize_topic("상거래전반") == "상거래 전반"

    def test_unknown_value_returned_as_is(self):
        assert prepare_topic._normalize_topic("식음료") == "식음료"

    def test_strips_whitespace(self):
        assert prepare_topic._normalize_topic("  식음료  ") == "식음료"

    def test_empty_string_returns_empty(self):
        assert prepare_topic._normalize_topic("") == ""


# ===========================================================================
# 통합 테스트 — subprocess로 --dummy --cpu 학습 실행
# ===========================================================================

@pytest.fixture
def tmp_output_dir(tmp_path):
    """각 학습 스크립트의 출력 디렉토리."""
    return tmp_path / "models"


def _run_dummy(script_name: str, tmp_output_dir: Path, extra_args=None) -> Path:
    """scripts/<script_name> --dummy --cpu --output-dir <tmp> 실행.

    반환: 생성된 version 디렉토리 (models/<sub>/v...).
    """
    output_root = tmp_output_dir / script_name
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / f"{script_name}.py"),
        "--dummy",
        "--cpu",
        "--output-dir", str(output_root),
    ]
    if extra_args:
        cmd.extend(extra_args)

    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, (
        f"{script_name} 실패 (rc={proc.returncode})\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    # version 디렉토리 탐색 (v로 시작)
    versions = sorted(p for p in output_root.iterdir() if p.is_dir() and p.name.startswith("v"))
    assert versions, f"version 디렉토리 없음: {output_root}"
    return versions[-1]


@pytest.mark.slow
class TestDummyIntegration:
    """--dummy --cpu 실제 학습 실행. KcELECTRA 캐시 필요. 느림 (~수십초)."""

    def _assert_outputs(self, version_dir: Path, *, expects_dialog_act: bool = False):
        # 공통 산출물
        for fname in ("metrics.json", "label_map.json", "model_card.json"):
            assert (version_dir / fname).exists(), f"missing {fname} in {version_dir}"
        # 내용도 JSON 파싱 가능해야 함
        for fname in ("metrics.json", "label_map.json", "model_card.json"):
            json.loads((version_dir / fname).read_text(encoding="utf-8"))

    def test_train_emotion_dummy_produces_artifacts(self, tmp_output_dir):
        version_dir = _run_dummy("train_emotion_model", tmp_output_dir)
        self._assert_outputs(version_dir)
        label_map = json.loads((version_dir / "label_map.json").read_text(encoding="utf-8"))
        assert label_map["emotion_labels"] == train_emotion.EMOTION_LABELS
        assert label_map["dialog_act_labels"] == train_emotion.DIALOG_ACT_LABELS

    def test_train_speech_age_dummy_produces_artifacts(self, tmp_output_dir):
        version_dir = _run_dummy("train_speech_age_model", tmp_output_dir)
        self._assert_outputs(version_dir)
        label_map = json.loads((version_dir / "label_map.json").read_text(encoding="utf-8"))
        assert label_map["age_labels"] == train_age.AGE_LABELS

    def test_train_topic_dummy_produces_artifacts(self, tmp_output_dir):
        version_dir = _run_dummy("train_topic_model", tmp_output_dir)
        self._assert_outputs(version_dir)
        label_map = json.loads((version_dir / "label_map.json").read_text(encoding="utf-8"))
        assert label_map["topic_labels"] == train_topic.TOPIC_LABELS

    def test_train_speech_act_dummy_produces_artifacts(self, tmp_output_dir):
        version_dir = _run_dummy(
            "train_speech_act_model", tmp_output_dir, extra_args=["--target", "group"]
        )
        self._assert_outputs(version_dir)
        label_map = json.loads((version_dir / "label_map.json").read_text(encoding="utf-8"))
        assert "speech_act_group_labels" in label_map
        assert label_map["speech_act_group_labels"] == train_speech_act.SPEECH_ACT_GROUP_LABELS


# ===========================================================================
# measure_whisperx_baseline._dummy_result 직접 테스트
# ===========================================================================

class TestMeasureBaselineDummy:
    def test_dummy_result_returns_zero_metrics(self):
        import argparse
        args = argparse.Namespace(model_size="large-v2", language="ko")
        result = measure_baseline._dummy_result(args)
        assert result["wer"] == 0.0
        assert result["cer"] == 0.0
        assert result["n_samples"] == 0
        assert result["model_size"] == "large-v2"
        assert result["language"] == "ko"
        assert "timestamp" in result
