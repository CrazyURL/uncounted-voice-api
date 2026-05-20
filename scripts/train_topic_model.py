"""
KcELECTRA 주제 분류 모델 학습

사용법:
  python scripts/train_topic_model.py [--dummy] [options]

  --dummy               CPU 소규모 테스트 (40샘플, 2에포크, max_len=32)
  --train-csv PATH      학습 CSV (기본: data/topic/train.csv)
  --val-csv PATH        검증 CSV (기본: data/topic/val.csv)
  --output-dir DIR      모델 저장 루트 (기본: models/topic)
  --base-model-path     베이스 모델 (기본: snunlp/KR-ELECTRA-discriminator)
  --max-epochs N        최대 에포크 (기본: 5)
  --batch-size N        배치 크기 (기본: 32)
  --lr FLOAT            학습률 (기본: 2e-5)
  --max-len N           최대 토큰 길이 (기본: 128)
  --seed N              랜덤 시드 (기본: 42)
  --cpu                 GPU 있어도 CPU 강제 사용

주의: seed phrase / 룰 기반 로직 없음.
순수 KcELECTRA 분류 — segment_topic 오분류 재현 방지.

출력:
  models/topic/v{YYYYMMDD_HHMMSS}/
    encoder/
    tokenizer/
    head.pt
    label_map.json      topic_labels, topic_to_group
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

# 20 topics
TOPIC_LABELS = [
    "가족", "연애/결혼",
    "건강", "미용", "반려동물", "식음료", "주거와 생활",
    "게임", "방송/연예", "영화/만화", "스포츠/레저", "여행",
    "계절/날씨", "교통", "사회이슈", "타 국가 이슈",
    "교육", "군대",
    "상거래 전반", "회사/아르바이트",
]
TOPIC2ID = {v: i for i, v in enumerate(TOPIC_LABELS)}

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
TOPIC_GROUP_LABELS = sorted(set(TOPIC_TO_GROUP.values()))

DEFAULT_BASE_MODEL = "snunlp/KR-ELECTRA-discriminator"
EARLY_STOP_PATIENCE = 3


# ---------------------------------------------------------------------------
# 더미 데이터
# ---------------------------------------------------------------------------

def make_dummy_rows(n: int = 40) -> list[dict]:
    templates = [
        ("가족이랑 여행 다녀왔어요", "가족"),
        ("오늘 점심 뭐 먹을까요", "식음료"),
        ("이번 주말에 영화 보러 갈 거야", "영화/만화"),
        ("헬스장 다니기 시작했어요", "건강"),
        ("지하철이 너무 복잡하네요", "교통"),
        ("새 게임 나왔다고 해서 설레요", "게임"),
        ("이번 시험 잘 봐야 하는데", "교육"),
        ("회사에서 야근이 많아요", "회사/아르바이트"),
    ]
    rows = []
    for i in range(n):
        tpl = templates[i % len(templates)]
        rows.append({
            "text": f"{tpl[0]} {i}",
            "topic": tpl[1],
            "topic_group": TOPIC_TO_GROUP.get(tpl[1], "기타"),
        })
    return rows


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TopicDataset(Dataset):
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

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

class TopicClassifier(nn.Module):
    def __init__(self, base_model_name_or_path: str, num_classes: int = 20,
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
    counts = Counter(r.get("topic", "식음료") for r in rows)
    total = sum(counts.values())
    n = len(TOPIC_LABELS)
    weights = torch.ones(n)
    for label, cnt in counts.items():
        if label in TOPIC2ID and cnt > 0:
            weights[TOPIC2ID[label]] = total / (n * cnt)
    logger.info("topic 클래스 가중치 (상위 5): %s",
                dict(sorted({TOPIC_LABELS[i]: round(weights[i].item(), 3)
                              for i in range(n)}.items(), key=lambda x: -x[1])[:5]))
    return weights.to(device)


def _save_model(model: TopicClassifier, tokenizer, output_dir: Path,
                version: str, base_model: str, topic_labels: list[str]) -> None:
    (output_dir / "encoder").mkdir(parents=True, exist_ok=True)
    (output_dir / "tokenizer").mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(str(output_dir / "encoder"))
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))
    torch.save({"head": model.head.state_dict()}, output_dir / "head.pt")
    label_map = {
        "topic_labels": topic_labels,
        "topic_to_group": TOPIC_TO_GROUP,
        "topic_group_labels": TOPIC_GROUP_LABELS,
    }
    (output_dir / "label_map.json").write_text(
        json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    model_card = {
        "version": version,
        "base_model": base_model,
        "heads": ["topic"],
        "topic_labels": topic_labels,
        "topic_group_labels": TOPIC_GROUP_LABELS,
        "note": "topic_group is derived from topic via TOPIC_TO_GROUP mapping, no separate head",
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

def evaluate(model: TopicClassifier, loader: DataLoader,
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
    return {"val_loss": total_loss / max(len(loader), 1), "topic_macro_f1": round(f1, 4)}


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

    # 실제 데이터에서 클래스 목록 동적 결정 (더미는 TOPIC_LABELS 그대로 사용)
    if args.dummy:
        topic_labels = TOPIC_LABELS
    else:
        seen_topics = sorted({r.get("topic", "") for r in train_rows if r.get("topic")})
        # TOPIC_LABELS 기준으로 정렬하되, 미등록 클래스는 뒤에 추가
        ordered = [t for t in TOPIC_LABELS if t in seen_topics]
        extra = [t for t in seen_topics if t not in TOPIC_LABELS]
        topic_labels = ordered + extra
        if extra:
            logger.warning("TOPIC_LABELS에 없는 클래스 %d개: %s", len(extra), extra)

    topic2id = {v: i for i, v in enumerate(topic_labels)}
    n_classes = len(topic_labels)
    logger.info("topic 클래스 수: %d", n_classes)

    base_model = args.base_model_path
    version = datetime.now().strftime("v%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / version
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("토크나이저 로드: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    logger.info("모델 로드: %s", base_model)
    model = TopicClassifier(base_model, num_classes=n_classes).to(device)

    # 가중치는 TOPIC_LABELS 기준 (topic2id가 달라도 인덱스 공유)
    counts = Counter(r.get("topic", "") for r in train_rows)
    total = sum(counts.values())
    weights = torch.ones(n_classes)
    for label, cnt in counts.items():
        if label in topic2id and cnt > 0:
            weights[topic2id[label]] = total / (n_classes * cnt)
    weights = weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    class _DynDataset(Dataset):
        def __init__(self, rows, t2id):
            self.rows, self.t2id = rows, t2id
        def __len__(self): return len(self.rows)
        def __getitem__(self, idx):
            row = self.rows[idx]
            return {"text": row["text"], "label": self.t2id.get(row.get("topic", ""), 0)}

    def _dyn_collate(batch):
        texts = [b["text"] for b in batch]
        enc = tokenizer(texts, max_length=max_len, padding="max_length",
                        truncation=True, return_tensors="pt")
        n = len(texts)
        tti = enc.get("token_type_ids", torch.zeros(n, max_len, dtype=torch.long))
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "token_type_ids": tti,
                "label": torch.tensor([b["label"] for b in batch], dtype=torch.long)}

    train_loader = DataLoader(_DynDataset(train_rows, topic2id),
                               batch_size=batch_size, shuffle=True, num_workers=0,
                               collate_fn=_dyn_collate)
    val_loader = DataLoader(_DynDataset(val_rows, topic2id),
                             batch_size=batch_size, shuffle=False, num_workers=0,
                             collate_fn=_dyn_collate)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * max_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                 num_warmup_steps=total_steps // 10,
                                                 num_training_steps=total_steps)

    best_val_loss = float("inf")
    best_epoch = 1
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

        model.eval()
        total_loss, all_preds, all_labels = 0.0, [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                labels = batch["label"].to(device)
                logits = model(input_ids, attention_mask, token_type_ids)
                loss = criterion(logits, labels)
                total_loss += loss.item()
                all_preds.extend(logits.argmax(-1).cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        val_loss = total_loss / max(len(val_loader), 1)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        metrics = {"val_loss": val_loss, "topic_macro_f1": round(f1, 4)}
        logger.info("Epoch %d 검증 -- val_loss=%.4f  topic_f1=%.4f",
                    epoch, metrics["val_loss"], metrics["topic_macro_f1"])

        if metrics["val_loss"] < best_val_loss:
            best_val_loss = metrics["val_loss"]
            best_epoch = epoch
            patience_counter = 0
            _save_model(model, tokenizer, output_dir, version, base_model, topic_labels)
            logger.info("Best 모델 저장 (val_loss=%.4f)", best_val_loss)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                logger.info("Early stopping (patience=%d)", EARLY_STOP_PATIENCE)
                break

    final_metrics = {**metrics, "best_epoch": best_epoch, "best_val_loss": round(best_val_loss, 5),
                     "total_epochs_run": epoch, "train_samples": len(train_rows),
                     "val_samples": len(val_rows), "n_classes": n_classes, "version": version}
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
    parser.add_argument("--train-csv", default="data/topic/train.csv")
    parser.add_argument("--val-csv", default="data/topic/val.csv")
    parser.add_argument("--output-dir", default="models/topic")
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
