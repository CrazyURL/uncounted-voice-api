"""
KcELECTRA 감정/대화행위 멀티태스크 파인튜닝 스크립트

사용법:
  python scripts/train_emotion_model.py [옵션]

주요 옵션:
  --base-model-path   HuggingFace 모델 ID 또는 로컬 경로 (기본: snunlp/KR-ELECTRA-discriminator)
  --previous-model-path  이전 버전 경로 (증분 파인튜닝)
  --data-dir          학습 데이터 디렉토리 (기본: data/emotion)
  --output-base-dir   모델 출력 베이스 디렉토리 (기본: models/emotion)
  --job-id            상태 추적용 UUID (training router 에서 전달)
  --status-file       학습 상태를 기록할 JSON 파일 경로
  --dummy             더미 10-sample CPU 전용 동작 검증 모드
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EMOTION_LABELS = ["기쁨", "놀람", "슬픔", "분노", "불안", "당황", "중립"]

# 세부감정 → 상위 카테고리 파생 (모델 미학습, 추론 시 계산)
EMOTION_TO_CATEGORY: dict[str, str] = {
    "기쁨": "긍정", "놀람": "긍정",
    "슬픔": "부정", "분노": "부정", "불안": "부정", "당황": "부정",
    "중립": "중립",
}

DIALOG_ACT_LABELS = [
    "진술", "질문", "요청", "감사", "인사", "사과",
    "동의", "반대", "확인", "부정", "응답", "제안",
    "명령", "감탄", "기타",
]

EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTION_LABELS)}
DIALOG_ACT_TO_IDX = {d: i for i, d in enumerate(DIALOG_ACT_LABELS)}


# ---------------------------------------------------------------------------
# 상태 파일 유틸
# ---------------------------------------------------------------------------

def write_status(status_file: Optional[Path], data: dict) -> None:
    if status_file is None:
        return
    try:
        status_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("상태 파일 쓰기 실패: %s", e)


# ---------------------------------------------------------------------------
# 더미 데이터셋
# ---------------------------------------------------------------------------

def make_dummy_dataset():
    """torch Dataset 반환 (10건, CPU 전용 학습 검증용)"""
    import torch
    from torch.utils.data import Dataset

    class DummyDataset(Dataset):
        def __init__(self, tokenizer, n=10):
            texts = [f"테스트 발화 {i}" for i in range(n)]
            emotions = [i % len(EMOTION_LABELS) for i in range(n)]
            dialog_acts = [i % len(DIALOG_ACT_LABELS) for i in range(n)]
            self.encodings = tokenizer(
                texts, padding=True, truncation=True, max_length=64, return_tensors="pt"
            )
            self.emotions = torch.tensor(emotions)
            self.dialog_acts = torch.tensor(dialog_acts)

        def __len__(self):
            return len(self.emotions)

        def __getitem__(self, idx):
            return {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "token_type_ids": self.encodings.get("token_type_ids", None),
                "emotion_label": self.emotions[idx],
                "dialog_act_label": self.dialog_acts[idx],
            }

    return DummyDataset


# ---------------------------------------------------------------------------
# CSV 데이터셋
# ---------------------------------------------------------------------------

def load_csv_dataset(tokenizer, csv_path: Path, max_len: int = 256):
    import torch
    from torch.utils.data import Dataset

    rows = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            emotion_idx = EMOTION_TO_IDX.get(row.get("emotion", ""), 1)
            dialog_idx = DIALOG_ACT_TO_IDX.get(row.get("dialog_act", "기타"), 14)
            rows.append((row["text"], emotion_idx, dialog_idx))

    texts = [r[0] for r in rows]
    emotions = torch.tensor([r[1] for r in rows])
    dialog_acts = torch.tensor([r[2] for r in rows])
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")

    class CsvDataset(Dataset):
        def __len__(self):
            return len(emotions)

        def __getitem__(self, idx):
            item = {
                "input_ids": encodings["input_ids"][idx],
                "attention_mask": encodings["attention_mask"][idx],
                "emotion_label": emotions[idx],
                "dialog_act_label": dialog_acts[idx],
            }
            if "token_type_ids" in encodings:
                item["token_type_ids"] = encodings["token_type_ids"][idx]
            return item

    return CsvDataset()


# ---------------------------------------------------------------------------
# 학습 루프
# ---------------------------------------------------------------------------

def train(args, status_file: Optional[Path]) -> None:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() and not args.dummy else "cpu")
    logger.info("디바이스: %s", device)

    # 모델 소스 결정
    model_source = args.previous_model_path or args.base_model_path
    logger.info("모델 소스: %s", model_source)

    write_status(status_file, {
        "status": "running",
        "job_id": args.job_id,
        "current_epoch": 0,
        "total_epochs": args.max_epochs if not args.dummy else 2,
        "progress_pct": 0,
    })

    tokenizer = AutoTokenizer.from_pretrained(model_source)
    backbone = AutoModel.from_pretrained(model_source).to(device)
    hidden_size = backbone.config.hidden_size

    emotion_head = nn.Linear(hidden_size, len(EMOTION_LABELS)).to(device)
    dialog_act_head = nn.Linear(hidden_size, len(DIALOG_ACT_LABELS)).to(device)

    # 이전 버전 head 가중치 이어받기
    if args.previous_model_path:
        prev = Path(args.previous_model_path)
        ep = prev / "emotion_head.pt"
        dp = prev / "dialog_act_head.pt"
        if ep.exists():
            emotion_head.load_state_dict(torch.load(str(ep), map_location=device))
            logger.info("이전 emotion_head 로드")
        if dp.exists():
            dialog_act_head.load_state_dict(torch.load(str(dp), map_location=device))
            logger.info("이전 dialog_act_head 로드")

    # 데이터셋
    if args.dummy:
        DummyDataset = make_dummy_dataset()
        train_ds = DummyDataset(tokenizer, n=10)
        val_ds = DummyDataset(tokenizer, n=4)
        max_epochs = 2
        batch_size = 4
    else:
        data_dir = Path(args.data_dir)
        train_ds = load_csv_dataset(tokenizer, data_dir / "train.csv", args.max_len)
        val_ds = load_csv_dataset(tokenizer, data_dir / "val.csv", args.max_len)
        max_epochs = args.max_epochs
        batch_size = args.batch_size

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(emotion_head.parameters()) + list(dialog_act_head.parameters()),
        lr=args.lr,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    no_improve_count = 0
    patience = 3 if not args.dummy else 999

    for epoch in range(1, max_epochs + 1):
        backbone.train()
        emotion_head.train()
        dialog_act_head.train()

        train_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            e_labels = batch["emotion_label"].to(device)
            d_labels = batch["dialog_act_label"].to(device)

            kwargs = {"input_ids": input_ids, "attention_mask": attn}
            if "token_type_ids" in batch and batch["token_type_ids"] is not None:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(device)

            outputs = backbone(**kwargs)
            cls = outputs.last_hidden_state[:, 0, :]

            e_loss = criterion(emotion_head(cls), e_labels)
            d_loss = criterion(dialog_act_head(cls), d_labels)
            loss = 0.7 * e_loss + 0.3 * d_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        backbone.eval()
        emotion_head.eval()
        dialog_act_head.eval()

        val_loss = 0.0
        correct_e = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                e_labels = batch["emotion_label"].to(device)
                d_labels = batch["dialog_act_label"].to(device)

                kwargs = {"input_ids": input_ids, "attention_mask": attn}
                if "token_type_ids" in batch and batch["token_type_ids"] is not None:
                    kwargs["token_type_ids"] = batch["token_type_ids"].to(device)

                outputs = backbone(**kwargs)
                cls = outputs.last_hidden_state[:, 0, :]

                e_loss = criterion(emotion_head(cls), e_labels)
                d_loss = criterion(dialog_act_head(cls), d_labels)
                val_loss += (0.7 * e_loss + 0.3 * d_loss).item()

                preds = emotion_head(cls).argmax(dim=-1)
                correct_e += (preds == e_labels).sum().item()
                total += e_labels.size(0)

        avg_val_loss = val_loss / max(len(val_loader), 1)
        emotion_acc = correct_e / max(total, 1)
        progress_pct = round(epoch / max_epochs * 100, 1)

        logger.info(
            "Epoch %d/%d — train_loss=%.4f val_loss=%.4f emotion_acc=%.4f",
            epoch, max_epochs, train_loss / max(len(train_loader), 1), avg_val_loss, emotion_acc,
        )

        write_status(status_file, {
            "status": "running",
            "job_id": args.job_id,
            "current_epoch": epoch,
            "total_epochs": max_epochs,
            "val_loss": round(avg_val_loss, 6),
            "val_emotion_acc": round(emotion_acc, 4),
            "progress_pct": progress_pct,
        })

        # 조기 종료
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                logger.info("조기 종료 (patience=%d)", patience)
                break

    # 모델 저장
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(args.output_base_dir)
    save_dir = output_base / f"v{version}"
    save_dir.mkdir(parents=True, exist_ok=True)

    backbone.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    torch.save(emotion_head.state_dict(), str(save_dir / "emotion_head.pt"))
    torch.save(dialog_act_head.state_dict(), str(save_dir / "dialog_act_head.pt"))

    metrics = {
        "best_val_loss": round(best_val_loss, 6),
        "val_emotion_acc": round(emotion_acc, 4),
        "total_epochs_run": epoch,
        "model_version": f"v{version}",
        "emotion_labels": EMOTION_LABELS,
        "dialog_act_labels": DIALOG_ACT_LABELS,
        "emotion_to_category": EMOTION_TO_CATEGORY,
    }
    (save_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # current 심링크 또는 current.txt 갱신
    current_link = output_base / "current"
    current_txt = output_base / "current.txt"
    try:
        if current_link.exists() or current_link.is_symlink():
            current_link.unlink()
        current_link.symlink_to(save_dir.resolve())
        logger.info("symlink 갱신: current → %s", save_dir)
    except (OSError, NotImplementedError):
        # Windows 심링크 실패 시 current.txt fallback
        current_txt.write_text(f"v{version}", encoding="utf-8")
        logger.info("current.txt 갱신: v%s", version)

    logger.info("모델 저장 완료: %s", save_dir)
    write_status(status_file, {
        "status": "completed",
        "job_id": args.job_id,
        "model_version": f"v{version}",
        "val_loss": round(best_val_loss, 6),
        "val_emotion_acc": round(emotion_acc, 4),
        "progress_pct": 100,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="KcELECTRA 감정 모델 파인튜닝")
    parser.add_argument("--base-model-path", default="snunlp/KR-ELECTRA-discriminator")
    parser.add_argument("--previous-model-path", default=None)
    parser.add_argument("--data-dir", default="data/emotion")
    parser.add_argument("--output-base-dir", default=None)
    parser.add_argument("--job-id", default="manual")
    parser.add_argument("--status-file", default=None)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=256)
    args = parser.parse_args()

    if args.output_base_dir is None:
        args.output_base_dir = os.environ.get("EMOTION_MODEL_DIR", "models/emotion")

    status_file = Path(args.status_file) if args.status_file else None

    try:
        train(args, status_file)
    except Exception as exc:
        logger.exception("학습 실패")
        write_status(status_file, {
            "status": "failed",
            "job_id": args.job_id,
            "error": str(exc),
        })
        sys.exit(1)


if __name__ == "__main__":
    main()
