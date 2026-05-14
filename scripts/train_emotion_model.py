"""
KcELECTRA 감정 + 대화행위 멀티태스크 파인튜닝

사용법:
  python scripts/train_emotion_model.py [--dummy] [options]

  --dummy               CPU 소규모 테스트 (40샘플, 2에포크, max_len=32)
  --train-csv PATH      학습 CSV (기본: data/emotion/train.csv)
  --val-csv PATH        검증 CSV (기본: data/emotion/val.csv)
  --output-dir DIR      모델 저장 루트 (기본: models/emotion)
  --base-model-path     베이스 모델 (기본: snunlp/KR-ELECTRA-discriminator)
  --previous-model-path 이전 모델 경로 (증분 파인튜닝 시)
  --max-epochs N        최대 에포크 (기본: 5)
  --batch-size N        배치 크기 (기본: 32)
  --lr FLOAT            학습률 (기본: 2e-5)
  --max-len N           최대 토큰 길이 (기본: 256)
  --seed N              랜덤 시드 (기본: 42)
  --cpu                 GPU 있어도 CPU 강제 사용

출력:
  models/emotion/v{YYYYMMDD_HHMMSS}/
    encoder/            AutoModel.save_pretrained
    tokenizer/          AutoTokenizer.save_pretrained
    heads.pt            emotion_head + dialog_act_head state_dicts
    label_map.json      EMOTION_LABELS, DIALOG_ACT_LABELS
    metrics.json        학습 완료 후 최종 지표
    training_status.json 실시간 진행 상황
  models/emotion/current  symlink (Linux) 또는 current.txt (Windows)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

EMOTION_LABELS = ["긍정", "중립", "부정"]
EMOTION2ID = {v: i for i, v in enumerate(EMOTION_LABELS)}

DIALOG_ACT_LABELS = [
    "진술", "질문", "요청", "감사", "인사", "사과",
    "동의", "반대", "확인", "부정", "응답", "제안",
    "명령", "감탄", "기타",
]
DIALOG_ACT2ID = {v: i for i, v in enumerate(DIALOG_ACT_LABELS)}

DEFAULT_BASE_MODEL = "snunlp/KR-ELECTRA-discriminator"
ALPHA_EMOTION = 0.7
ALPHA_DIALOG_ACT = 0.3
EARLY_STOP_PATIENCE = 3


# ---------------------------------------------------------------------------
# 더미 데이터
# ---------------------------------------------------------------------------

def make_dummy_rows(n: int = 40) -> list[dict]:
    templates = [
        ("오늘 정말 기분이 좋아요", "긍정", "진술"),
        ("이게 무슨 뜻인가요", "중립", "질문"),
        ("너무 화가 나서 참을 수가 없어요", "부정", "감탄"),
        ("감사합니다 덕분에 살았어요", "긍정", "감사"),
        ("이 제품을 환불하고 싶습니다", "중립", "요청"),
        ("오늘 날씨가 흐리네요", "중립", "진술"),
        ("정말 슬프고 괴롭습니다", "부정", "진술"),
        ("좋아요 그렇게 하겠습니다", "긍정", "동의"),
        ("아니요 그건 아닌 것 같아요", "중립", "반대"),
        ("드디어 원하던 회사에 합격했어요", "긍정", "진술"),
    ]
    rows = []
    for i in range(n):
        tpl = templates[i % len(templates)]
        rows.append({"text": f"{tpl[0]} {i}", "emotion": tpl[1], "dialog_act": tpl[2]})
    return rows


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EmotionDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_len: int = 256) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        enc = self.tokenizer(
            row["text"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        emotion_id = EMOTION2ID.get(row.get("emotion", "중립"), 1)
        dialog_act_id = DIALOG_ACT2ID.get(row.get("dialog_act", "기타"), 14)
        token_type_ids = enc.get(
            "token_type_ids", torch.zeros(self.max_len, dtype=torch.long)
        ).squeeze(0)
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": token_type_ids,
            "emotion_label": torch.tensor(emotion_id, dtype=torch.long),
            "dialog_act_label": torch.tensor(dialog_act_id, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------

class EmotionClassifier(nn.Module):
    def __init__(
        self,
        base_model_name_or_path: str,
        num_emotions: int = 3,
        num_dialog_acts: int = 15,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name_or_path)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.emotion_head = nn.Linear(hidden, num_emotions)
        self.dialog_act_head = nn.Linear(hidden, num_dialog_acts)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls = self.dropout(out.last_hidden_state[:, 0])
        return self.emotion_head(cls), self.dialog_act_head(cls)


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


def _save_model(model: EmotionClassifier, tokenizer, output_dir: Path) -> None:
    (output_dir / "encoder").mkdir(parents=True, exist_ok=True)
    (output_dir / "tokenizer").mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(str(output_dir / "encoder"))
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))
    torch.save(
        {
            "emotion_head": model.emotion_head.state_dict(),
            "dialog_act_head": model.dialog_act_head.state_dict(),
        },
        output_dir / "heads.pt",
    )
    label_map = {"emotion_labels": EMOTION_LABELS, "dialog_act_labels": DIALOG_ACT_LABELS}
    (output_dir / "label_map.json").write_text(
        json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("모델 저장 완료: %s", output_dir)


def _write_status(output_dir: Path, status: dict) -> None:
    (output_dir / "training_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _update_current_symlink(models_root: Path, version_dir: Path) -> None:
    current = models_root / "current"
    try:
        if current.is_symlink() or current.exists():
            current.unlink()
        current.symlink_to(version_dir.resolve())
        logger.info("current 심링크 갱신: %s -> %s", current, version_dir)
    except (OSError, NotImplementedError):
        (models_root / "current.txt").write_text(
            str(version_dir.resolve()), encoding="utf-8"
        )
        logger.info("current.txt 갱신: %s", version_dir)


# ---------------------------------------------------------------------------
# 평가
# ---------------------------------------------------------------------------

def _evaluate(model: EmotionClassifier, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    e_preds: list[int] = []
    e_labels: list[int] = []
    d_preds: list[int] = []
    d_labels: list[int] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            e_lbl = batch["emotion_label"].to(device)
            d_lbl = batch["dialog_act_label"].to(device)

            e_logit, d_logit = model(input_ids, attention_mask, token_type_ids)
            loss = (
                ALPHA_EMOTION * criterion(e_logit, e_lbl)
                + ALPHA_DIALOG_ACT * criterion(d_logit, d_lbl)
            )
            total_loss += loss.item()
            e_preds.extend(e_logit.argmax(dim=-1).cpu().tolist())
            e_labels.extend(e_lbl.cpu().tolist())
            d_preds.extend(d_logit.argmax(dim=-1).cpu().tolist())
            d_labels.extend(d_lbl.cpu().tolist())

    try:
        from sklearn.metrics import f1_score
        e_f1 = float(f1_score(e_labels, e_preds, average="macro", zero_division=0))
        d_f1 = float(f1_score(d_labels, d_preds, average="macro", zero_division=0))
    except ImportError:
        n = max(len(e_labels), 1)
        e_f1 = sum(p == lb for p, lb in zip(e_preds, e_labels)) / n
        d_f1 = sum(p == lb for p, lb in zip(d_preds, d_labels)) / n

    return {
        "val_loss": total_loss / max(len(loader), 1),
        "emotion_macro_f1": round(e_f1, 4),
        "dialog_act_macro_f1": round(d_f1, 4),
    }


# ---------------------------------------------------------------------------
# 학습
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    version = datetime.now().strftime("v%Y%m%d_%H%M%S")
    models_root = Path(args.output_dir)
    output_dir = models_root / version
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("출력 디렉토리: %s", output_dir)

    device = (
        torch.device("cpu")
        if args.cpu
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("장치: %s", device)

    if args.dummy:
        logger.info("더미 모드: 소규모 CPU 테스트")
        all_rows = make_dummy_rows(40)
        split = int(len(all_rows) * 0.8)
        train_rows, val_rows = all_rows[:split], all_rows[split:]
        max_len = 32
        batch_size = 4
        max_epochs = 2
    else:
        train_rows = load_csv(Path(args.train_csv))
        val_rows = load_csv(Path(args.val_csv))
        max_len = args.max_len
        batch_size = args.batch_size
        max_epochs = args.max_epochs

    if args.previous_model_path:
        encoder_path = str(Path(args.previous_model_path) / "encoder")
        tokenizer_path = str(Path(args.previous_model_path) / "tokenizer")
        logger.info("증분 파인튜닝: %s", args.previous_model_path)
    else:
        encoder_path = args.base_model_path
        tokenizer_path = args.base_model_path

    logger.info("토크나이저 로드: %s", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    logger.info("모델 로드: %s", encoder_path)
    model = EmotionClassifier(encoder_path).to(device)

    if args.previous_model_path:
        heads_path = Path(args.previous_model_path) / "heads.pt"
        if heads_path.exists():
            ckpt = torch.load(str(heads_path), map_location="cpu")
            model.emotion_head.load_state_dict(ckpt["emotion_head"])
            model.dialog_act_head.load_state_dict(ckpt["dialog_act_head"])
            logger.info("이전 head weights 로드 완료")

    train_loader = DataLoader(
        EmotionDataset(train_rows, tokenizer, max_len),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        EmotionDataset(val_rows, tokenizer, max_len),
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * max_epochs
    warmup_steps = max(1, total_steps // 10)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    no_improve_count = 0
    best_epoch = 0
    epoch = 0
    start_time = time.time()

    status: dict = {
        "version": version,
        "status": "running",
        "current_epoch": 0,
        "max_epochs": max_epochs,
        "best_val_loss": None,
        "best_epoch": None,
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "device": str(device),
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
    }
    _write_status(output_dir, status)

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            e_lbl = batch["emotion_label"].to(device)
            d_lbl = batch["dialog_act_label"].to(device)

            optimizer.zero_grad()
            e_logit, d_logit = model(input_ids, attention_mask, token_type_ids)
            loss = (
                ALPHA_EMOTION * criterion(e_logit, e_lbl)
                + ALPHA_DIALOG_ACT * criterion(d_logit, d_lbl)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

            if step % max(1, len(train_loader) // 5) == 0:
                logger.info(
                    "Epoch %d/%d  Step %d/%d  loss=%.4f",
                    epoch, max_epochs, step, len(train_loader), epoch_loss / step,
                )

        avg_train_loss = epoch_loss / max(len(train_loader), 1)
        logger.info(
            "Epoch %d 완료 -- train_loss=%.4f (%.1fs)",
            epoch, avg_train_loss, time.time() - t0,
        )

        val_metrics = _evaluate(model, val_loader, device)
        val_loss = val_metrics["val_loss"]
        logger.info(
            "Epoch %d 검증 -- val_loss=%.4f  emotion_f1=%.4f  dialog_act_f1=%.4f",
            epoch, val_loss, val_metrics["emotion_macro_f1"], val_metrics["dialog_act_macro_f1"],
        )

        status["current_epoch"] = epoch
        status["last_val_loss"] = round(val_loss, 6)
        status["last_emotion_f1"] = val_metrics["emotion_macro_f1"]
        _write_status(output_dir, status)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve_count = 0
            _save_model(model, tokenizer, output_dir)
            logger.info("Best 모델 저장 (val_loss=%.4f)", val_loss)
        else:
            no_improve_count += 1
            logger.info("개선 없음 %d/%d", no_improve_count, EARLY_STOP_PATIENCE)
            if no_improve_count >= EARLY_STOP_PATIENCE:
                logger.info("조기 종료 (epoch %d)", epoch)
                break

    final_metrics = _evaluate(model, val_loader, device)
    final_metrics.update(
        {
            "best_epoch": best_epoch,
            "best_val_loss": round(best_val_loss, 6),
            "total_epochs_run": epoch,
            "train_samples": len(train_rows),
            "val_samples": len(val_rows),
            "version": version,
            "total_time_sec": round(time.time() - start_time, 1),
        }
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(final_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("최종 메트릭: %s", final_metrics)

    status["status"] = "completed"
    status["best_val_loss"] = round(best_val_loss, 6)
    status["best_epoch"] = best_epoch
    status["completed_at"] = datetime.now().isoformat()
    _write_status(output_dir, status)

    _update_current_symlink(models_root, output_dir)
    logger.info("학습 완료 -- 버전: %s", version)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="KcELECTRA 감정+대화행위 멀티태스크 파인튜닝")
    parser.add_argument("--dummy", action="store_true", help="CPU 소규모 테스트 모드")
    parser.add_argument("--train-csv", default="data/emotion/train.csv")
    parser.add_argument("--val-csv", default="data/emotion/val.csv")
    parser.add_argument("--output-dir", default="models/emotion")
    parser.add_argument("--base-model-path", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--previous-model-path", default=None)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if not args.dummy and not Path(args.train_csv).exists():
        logger.error("학습 CSV 없음: %s -- --dummy 또는 prepare_emotion_dataset.py 먼저 실행", args.train_csv)
        raise SystemExit(1)

    train(args)


if __name__ == "__main__":
    main()
