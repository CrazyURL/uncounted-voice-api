"""
KcELECTRA 연령대 분류 모델 학습

사용법:
  python scripts/train_speech_age_model.py [--dummy] [options]

  --dummy               CPU 소규모 테스트 (40샘플, 2에포크, max_len=32)
  --train-csv PATH      학습 CSV (기본: data/age_speech/train.csv)
  --val-csv PATH        검증 CSV (기본: data/age_speech/val.csv)
  --output-dir DIR      모델 저장 루트 (기본: models/speech_age)
  --base-model-path     베이스 모델 (기본: snunlp/KR-ELECTRA-discriminator)
  --max-epochs N        최대 에포크 (기본: 5)
  --batch-size N        배치 크기 (기본: 32)
  --lr FLOAT            학습률 (기본: 2e-5)
  --max-len N           최대 토큰 길이 (기본: 128)
  --seed N              랜덤 시드 (기본: 42)
  --cpu                 GPU 있어도 CPU 강제 사용

출력:
  models/speech_age/v{YYYYMMDD_HHMMSS}/
    encoder/
    tokenizer/
    head.pt
    label_map.json
    model_card.json
    metrics.json
    training_status.json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

AGE_LABELS = ["20s", "30s", "40s", "50s+"]
AGE2ID = {v: i for i, v in enumerate(AGE_LABELS)}

DEFAULT_BASE_MODEL = "snunlp/KR-ELECTRA-discriminator"
EARLY_STOP_PATIENCE = 3


# ---------------------------------------------------------------------------
# 더미 데이터
# ---------------------------------------------------------------------------

def make_dummy_rows(n: int = 40) -> list[dict]:
    templates = [
        ("요즘 유행하는 게임 해봤어", "20s"),
        ("아이들 학교 문제가 걱정이에요", "30s"),
        ("허리가 자꾸 아파서요", "40s"),
        ("요즘 젊은 것들은 이해가 안 돼", "50s+"),
        ("인스타그램 팔로워 늘리는 법", "20s"),
        ("내 집 마련이 목표야", "30s"),
        ("자녀 결혼 준비 중인데요", "40s"),
        ("연금 받을 날만 기다려", "50s+"),
    ]
    rows = []
    for i in range(n):
        tpl = templates[i % len(templates)]
        rows.append({"text": f"{tpl[0]} {i}", "age_group": tpl[1]})
    return rows


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AgeDataset(Dataset):
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        label_id = AGE2ID.get(row.get("age_group", "20s"), 0)
        return {"text": row["text"], "label": label_id}

def _make_collate_fn(tokenizer, max_len: int):
    def collate_fn(batch):
        texts = [b["text"] for b in batch]
        enc = tokenizer(texts, max_length=max_len, padding="max_length",
                        truncation=True, return_tensors="pt")
        n = len(texts)
        tti = enc.get("token_type_ids", torch.zeros(n, max_len, dtype=torch.long))
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "token_type_ids": tti,
            "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        }
    return collate_fn



# ---------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------

class AgeClassifier(nn.Module):
    def __init__(self, base_model_name_or_path: str, num_classes: int = 4,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name_or_path)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids)
        cls = self.dropout(out.last_hidden_state[:, 0])
        return self.head(cls)


# ---------------------------------------------------------------------------
# I/O 헬퍼
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    logger.info("CSV 로드: %s (%d건)", path, len(rows))
    return rows


def _compute_weights(rows: list[dict], device: torch.device) -> torch.Tensor:
    counts = Counter(r.get("age_group", "20s") for r in rows)
    total = sum(counts.values())
    n_classes = len(AGE_LABELS)
    weights = torch.ones(n_classes)
    for label, cnt in counts.items():
        if label in AGE2ID and cnt > 0:
            weights[AGE2ID[label]] = total / (n_classes * cnt)
    logger.info("연령대 클래스 가중치: %s",
                {AGE_LABELS[i]: round(weights[i].item(), 3) for i in range(n_classes)})
    return weights.to(device)


def _save_model(model: AgeClassifier, tokenizer, output_dir: Path,
                version: str, base_model: str) -> None:
    (output_dir / "encoder").mkdir(parents=True, exist_ok=True)
    (output_dir / "tokenizer").mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(str(output_dir / "encoder"))
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))
    torch.save({"head": model.head.state_dict()}, output_dir / "head.pt")
    label_map = {"age_labels": AGE_LABELS}
    (output_dir / "label_map.json").write_text(
        json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    model_card = {
        "version": version,
        "base_model": base_model,
        "heads": ["age_group"],
        "age_labels": AGE_LABELS,
    }
    (output_dir / "model_card.json").write_text(
        json.dumps(model_card, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("모델 저장 완료: %s", output_dir)


def _update_current_symlink(models_root: Path, version_dir: Path) -> None:
    current = models_root / "current"
    try:
        if current.is_symlink() or current.exists():
            current.unlink()
        current.symlink_to(version_dir.resolve())
        logger.info("current 심링크 갱신: %s -> %s", current, version_dir)
    except Exception as e:
        txt = models_root / "current.txt"
        txt.write_text(version_dir.name, encoding="utf-8")
        logger.info("current.txt 갱신: %s (%s)", txt, e)


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------

def evaluate(model: AgeClassifier, loader: DataLoader,
             criterion, device: torch.device) -> dict:
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            all_preds.extend(logits.argmax(-1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return {"val_loss": total_loss / max(len(loader), 1), "age_macro_f1": round(f1, 4)}


# ---------------------------------------------------------------------------
# 학습
# ---------------------------------------------------------------------------

def train(args) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu") if args.cpu else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    logger.info("Device: %s", device)
    # VRAM 85% 상한 — 나머지 15%는 Xwayland/OS 예약
    if not args.cpu and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.85)


    if args.dummy:
        logger.info("더미 모드: 소규모 CPU 테스트")
        all_rows = make_dummy_rows(40)
        train_rows = all_rows[:32]
        val_rows = all_rows[32:]
        max_len = 32
        batch_size = 8
        max_epochs = 2
    else:
        train_rows = load_csv(Path(args.train_csv))
        val_rows = load_csv(Path(args.val_csv))
        if not train_rows:
            logger.error("학습 데이터 없음")
            raise SystemExit(1)
        max_len = args.max_len
        batch_size = args.batch_size
        max_epochs = args.max_epochs

    base_model = args.base_model_path
    version = datetime.now().strftime("v%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / version
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("토크나이저 로드: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    logger.info("모델 로드: %s", base_model)
    model = AgeClassifier(base_model, num_classes=len(AGE_LABELS)).to(device)

    weights = _compute_weights(train_rows, device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    train_ds = AgeDataset(train_rows)
    val_ds = AgeDataset(val_rows)
    _collate = _make_collate_fn(tokenizer, max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=_collate)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * max_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                 num_warmup_steps=total_steps // 10,
                                                 num_training_steps=total_steps)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss = loss.item()
            logger.info("Epoch %d/%d  Step %d/%d  loss=%.4f",
                        epoch, max_epochs, step, len(train_loader), epoch_loss)

        elapsed = round(time.time() - t0, 1)
        logger.info("Epoch %d 완료 -- train_loss=%.4f (%.1fs)", epoch, epoch_loss, elapsed)

        metrics = evaluate(model, val_loader, criterion, device)
        logger.info("Epoch %d 검증 -- val_loss=%.4f  age_f1=%.4f",
                    epoch, metrics["val_loss"], metrics["age_macro_f1"])

        if metrics["val_loss"] < best_val_loss:
            best_val_loss = metrics["val_loss"]
            best_epoch = epoch
            patience_counter = 0
            _save_model(model, tokenizer, output_dir, version, base_model)
            logger.info("Best 모델 저장 (val_loss=%.4f)", best_val_loss)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                logger.info("Early stopping (patience=%d)", EARLY_STOP_PATIENCE)
                break

    final_metrics = {**metrics, "best_epoch": best_epoch, "best_val_loss": round(best_val_loss, 5),
                     "total_epochs_run": epoch, "train_samples": len(train_rows),
                     "val_samples": len(val_rows), "version": version}
    (output_dir / "metrics.json").write_text(
        json.dumps(final_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("최종 메트릭: %s", final_metrics)
    _update_current_symlink(Path(args.output_dir), output_dir)
    logger.info("학습 완료 -- 버전: %s", version)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--train-csv", default="data/age_speech/train.csv")
    parser.add_argument("--val-csv", default="data/age_speech/val.csv")
    parser.add_argument("--output-dir", default="models/speech_age")
    parser.add_argument("--base-model-path", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
