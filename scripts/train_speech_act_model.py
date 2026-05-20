"""
KcELECTRA 화행(speech_act) 분류 모델 학습

1차 학습: speech_act_group (4-class: 단언/지시/표현/언약)
2차 학습: speech_act full (세부 클래스, label_analysis.json 확인 후 결정)

사용법:
  python scripts/train_speech_act_model.py [--dummy] [options]

  --dummy               CPU 소규모 테스트 (40샘플, 2에포크, max_len=32)
  --train-csv PATH      학습 CSV (기본: data/speech_act/train.csv)
  --val-csv PATH        검증 CSV (기본: data/speech_act/val.csv)
  --output-dir DIR      모델 저장 루트 (기본: models/speech_act)
  --target              분류 대상: group (기본) 또는 full
  --base-model-path     베이스 모델 (기본: snunlp/KR-ELECTRA-discriminator)
  --max-epochs N        최대 에포크 (기본: 5)
  --batch-size N        배치 크기 (기본: 32)
  --lr FLOAT            학습률 (기본: 2e-5)
  --max-len N           최대 토큰 길이 (기본: 128)
  --seed N              랜덤 시드 (기본: 42)
  --cpu                 GPU 있어도 CPU 강제 사용

출력:
  models/speech_act/v{YYYYMMDD_HHMMSS}/
    encoder/
    tokenizer/
    head.pt
    label_map.json      speech_act_group_labels (또는 speech_act_labels)
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

SPEECH_ACT_GROUP_LABELS = ["단언", "언약", "지시", "표현"]
SA_GROUP2ID = {v: i for i, v in enumerate(SPEECH_ACT_GROUP_LABELS)}

DEFAULT_BASE_MODEL = "snunlp/KR-ELECTRA-discriminator"
EARLY_STOP_PATIENCE = 3


# ---------------------------------------------------------------------------
# 더미 데이터
# ---------------------------------------------------------------------------

def make_dummy_rows(n: int = 40) -> list[dict]:
    templates = [
        ("오늘 날씨가 맑아요", "(단언) 주장하기", "단언"),
        ("이 일을 해줄 수 있나요", "(지시) 질문하기", "지시"),
        ("정말 기뻐요 감사해요", "(표현) 긍정감정 표현하기", "표현"),
        ("꼭 그렇게 하겠습니다", "(언약) 약속하기(제3자와)/(개인적 수준)", "언약"),
        ("어제 비가 많이 왔네요", "(단언) 주장하기", "단언"),
        ("창문 좀 닫아주세요", "(지시) 질문하기", "지시"),
        ("너무 슬프고 힘들어요", "(표현) 부정감정 표현하기", "표현"),
        ("내일까지 완성할게요", "(언약) 약속하기(제3자와)/(개인적 수준)", "언약"),
    ]
    rows = []
    for i in range(n):
        tpl = templates[i % len(templates)]
        rows.append({
            "text": f"{tpl[0]} {i}",
            "speech_act": tpl[1],
            "speech_act_group": tpl[2],
        })
    return rows


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SpeechActDataset(Dataset):
    def __init__(self, rows: list[dict], label_col: str, label2id: dict) -> None:
        self.rows = rows
        self.label_col = label_col
        self.label2id = label2id
        default_label = next(iter(label2id))
        self.default_id = label2id[default_label]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        label_id = self.label2id.get(row.get(self.label_col, ""), self.default_id)
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

class SpeechActClassifier(nn.Module):
    def __init__(self, base_model_name_or_path: str, num_classes: int,
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


def _save_model(model: SpeechActClassifier, tokenizer, output_dir: Path,
                version: str, base_model: str, target: str,
                labels: list[str]) -> None:
    (output_dir / "encoder").mkdir(parents=True, exist_ok=True)
    (output_dir / "tokenizer").mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(str(output_dir / "encoder"))
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))
    torch.save({"head": model.head.state_dict()}, output_dir / "head.pt")
    label_key = "speech_act_group_labels" if target == "group" else "speech_act_labels"
    label_map = {label_key: labels}
    (output_dir / "label_map.json").write_text(
        json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    model_card = {
        "version": version,
        "base_model": base_model,
        "heads": ["speech_act_group" if target == "group" else "speech_act"],
        label_key: labels,
        "target": target,
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

    target = args.target
    if target == "group":
        label_col = "speech_act_group"
        if args.dummy:
            labels = SPEECH_ACT_GROUP_LABELS
        else:
            seen = sorted({r.get("speech_act_group", "") for r in train_rows
                           if r.get("speech_act_group")})
            ordered = [l for l in SPEECH_ACT_GROUP_LABELS if l in seen]
            extra = [l for l in seen if l not in SPEECH_ACT_GROUP_LABELS]
            labels = ordered + extra
            if extra:
                logger.warning("SPEECH_ACT_GROUP_LABELS에 없는 클래스: %s", extra)
    else:
        label_col = "speech_act"
        seen = sorted({r.get("speech_act", "") for r in train_rows if r.get("speech_act")})
        labels = seen
        logger.info("speech_act full 클래스 수: %d", len(labels))

    label2id = {v: i for i, v in enumerate(labels)}
    n_classes = len(labels)
    logger.info("분류 대상: %s  클래스 수: %d", target, n_classes)

    counts = Counter(r.get(label_col, "") for r in train_rows)
    total = sum(counts.values())
    weights = torch.ones(n_classes)
    for label, cnt in counts.items():
        if label in label2id and cnt > 0:
            weights[label2id[label]] = total / (n_classes * cnt)
    weights = weights.to(device)

    base_model = args.base_model_path
    version = datetime.now().strftime("v%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / version
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("토크나이저 로드: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    logger.info("모델 로드: %s", base_model)
    model = SpeechActClassifier(base_model, num_classes=n_classes).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    train_ds = SpeechActDataset(train_rows, label_col, label2id)
    val_ds = SpeechActDataset(val_rows, label_col, label2id)
    _collate = _make_collate_fn(tokenizer, max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=_collate)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * max_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                 num_warmup_steps=total_steps // 10,
                                                 num_training_steps=total_steps)

    best_val_loss = float("inf")
    best_epoch = 1
    patience_counter = 0
    metrics: dict = {}

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels_t = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels_t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss = loss.item()
            logger.info("Epoch %d/%d  Step %d/%d  loss=%.4f",
                        epoch, max_epochs, step, len(train_loader), epoch_loss)

        elapsed = round(time.time() - t0, 1)
        logger.info("Epoch %d 완료 -- train_loss=%.4f (%.1fs)", epoch, epoch_loss, elapsed)

        model.eval()
        total_loss, all_preds, all_labels_list = 0.0, [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                labels_t = batch["label"].to(device)
                logits = model(input_ids, attention_mask, token_type_ids)
                loss = criterion(logits, labels_t)
                total_loss += loss.item()
                all_preds.extend(logits.argmax(-1).cpu().tolist())
                all_labels_list.extend(labels_t.cpu().tolist())
        val_loss = total_loss / max(len(val_loader), 1)
        f1 = f1_score(all_labels_list, all_preds, average="macro", zero_division=0)
        f1_key = "speech_act_group_macro_f1" if target == "group" else "speech_act_macro_f1"
        metrics = {"val_loss": val_loss, f1_key: round(f1, 4)}
        logger.info("Epoch %d 검증 -- val_loss=%.4f  f1=%.4f",
                    epoch, metrics["val_loss"], f1)

        if metrics["val_loss"] < best_val_loss:
            best_val_loss = metrics["val_loss"]
            best_epoch = epoch
            patience_counter = 0
            _save_model(model, tokenizer, output_dir, version, base_model, target, labels)
            logger.info("Best 모델 저장 (val_loss=%.4f)", best_val_loss)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                logger.info("Early stopping (patience=%d)", EARLY_STOP_PATIENCE)
                break

    final_metrics = {**metrics, "best_epoch": best_epoch, "best_val_loss": round(best_val_loss, 5),
                     "total_epochs_run": epoch, "train_samples": len(train_rows),
                     "val_samples": len(val_rows), "n_classes": n_classes,
                     "target": target, "version": version}
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
    parser.add_argument("--train-csv", default="data/speech_act/train.csv")
    parser.add_argument("--val-csv", default="data/speech_act/val.csv")
    parser.add_argument("--output-dir", default="models/speech_act")
    parser.add_argument("--target", default="group", choices=["group", "full"],
                        help="group=4-class speech_act_group, full=세부 클래스")
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
