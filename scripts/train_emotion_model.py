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
    label_map.json      EMOTION_LABELS, DIALOG_ACT_LABELS, DIALOG_ACT_GROUP_LABELS
    model_card.json     heads, base_model, version 정보
    metrics.json        학습 완료 후 최종 지표
    training_status.json 실시간 진행 상황
  models/emotion/current  symlink (Linux) 또는 current.txt (Windows)

추론 시 dialog_act_group:
  dialog_act 예측 → DIALOG_ACT_TO_GROUP 매핑으로 6-class group 자동 파생
  별도 head 없음 (label_map.json에 매핑 포함)
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
from torch.amp import GradScaler, autocast
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

# 15-class dialog_act → 6-class group (추론 시 자동 파생, 별도 head 없음)
DIALOG_ACT_TO_GROUP: dict[str, str] = {
    "진술": "정보",
    "질문": "질문/확인",
    "확인": "질문/확인",
    "요청": "요청/제안",
    "제안": "요청/제안",
    "감사": "감사/사과",
    "사과": "감사/사과",
    "인사": "사회적",
    "동의": "응답",
    "반대": "응답",
    "부정": "응답",
    "응답": "응답",
    "명령": "지시",
    "감탄": "감정 표현",
    "기타": "기타",
}
DIALOG_ACT_GROUP_LABELS = sorted(set(DIALOG_ACT_TO_GROUP.values()))

DEFAULT_BASE_MODEL = "snunlp/KR-ELECTRA-discriminator"
ALPHA_EMOTION = 0.7
ALPHA_DIALOG_ACT = 0.3
EARLY_STOP_PATIENCE = 3


# ---------------------------------------------------------------------------
# 데이터 전처리 유틸
# ---------------------------------------------------------------------------

def _undersample_kita(rows: list[dict], max_ratio: float = 0.5) -> list[dict]:
    """dialog_act '기타' 비율을 max_ratio(기본 50%) 이하로 언더샘플.
    '기타'가 편향되어 다른 15개 클래스 학습을 방해하는 것을 방지.
    non_kita=0 이면 전체가 '기타' 뿐인 데이터셋 — 언더샘플 불필요, 원본 반환.
    """
    kita = [r for r in rows if r.get("dialog_act", "") == "기타"]
    non_kita = [r for r in rows if r.get("dialog_act", "") != "기타"]
    if not non_kita:
        logger.info("dialog_act non_kita=0 — 전체 '기타' 데이터셋, 언더샘플 스킵 (%d행)", len(rows))
        return rows
    max_kita = int(len(non_kita) * max_ratio / max(1 - max_ratio, 1e-9))
    if len(kita) > max_kita:
        random.shuffle(kita)
        kita = kita[:max_kita]
        logger.info("dialog_act '기타' 언더샘플: %d → %d (non_kita=%d)",
                    len([r for r in rows if r.get("dialog_act", "") == "기타"]),
                    len(kita), len(non_kita))
    result = non_kita + kita
    random.shuffle(result)
    return result


def _compute_dialog_act_weights(rows: list[dict], device: torch.device) -> torch.Tensor:
    """dialog_act 클래스별 역빈도 가중치 계산.
    희소 클래스에 높은 가중치를 부여해 불균형 보정.
    """
    counts = Counter(r.get("dialog_act", "기타") for r in rows)
    total = sum(counts.values())
    n_classes = len(DIALOG_ACT_LABELS)
    weights = torch.ones(n_classes)
    for label, cnt in counts.items():
        if label in DIALOG_ACT2ID and cnt > 0:
            weights[DIALOG_ACT2ID[label]] = total / (n_classes * cnt)
    logger.info("dialog_act 클래스 가중치: %s",
                {DIALOG_ACT_LABELS[i]: round(weights[i].item(), 3) for i in range(n_classes)})
    return weights.to(device)


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
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        return {
            "text": row["text"],
            "emotion_label": EMOTION2ID.get(row.get("emotion", "중립"), 1),
            "dialog_act_label": DIALOG_ACT2ID.get(row.get("dialog_act", "기타"), 14),
        }


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
            "emotion_label": torch.tensor([b["emotion_label"] for b in batch], dtype=torch.long),
            "dialog_act_label": torch.tensor([b["dialog_act_label"] for b in batch], dtype=torch.long),
        }
    return collate_fn


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


def _save_model(model: EmotionClassifier, tokenizer, output_dir: Path, version: str,
                base_model: str) -> None:
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
    label_map = {
        "emotion_labels": EMOTION_LABELS,
        "dialog_act_labels": DIALOG_ACT_LABELS,
        "dialog_act_group_labels": DIALOG_ACT_GROUP_LABELS,
        "dialog_act_to_group": DIALOG_ACT_TO_GROUP,
    }
    (output_dir / "label_map.json").write_text(
        json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    model_card = {
        "version": version,
        "base_model": base_model,
        "heads": ["emotion", "dialog_act", "dialog_act_group"],
        "emotion_labels": EMOTION_LABELS,
        "dialog_act_labels": DIALOG_ACT_LABELS,
        "dialog_act_group_labels": DIALOG_ACT_GROUP_LABELS,
        "note": "dialog_act_group is derived from dialog_act via DIALOG_ACT_TO_GROUP mapping, no separate head",
    }
    (output_dir / "model_card.json").write_text(
        json.dumps(model_card, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("모델 저장 완료: %s", output_dir)


_external_status_file: Path | None = None


def _write_status(output_dir: Path, status: dict) -> None:
    (output_dir / "training_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if _external_status_file is not None:
        try:
            _external_status_file.write_text(
                json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass


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

            with autocast("cuda", enabled=device.type == "cuda"):
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
    # VRAM 85% 상한 — 나머지 15%는 Xwayland/OS 예약
    if not args.cpu and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.85)


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

    # dialog_act '기타' 언더샘플 (학습 데이터만)
    if not args.dummy:
        train_rows = _undersample_kita(train_rows, max_ratio=0.5)

    # --resume: ckpt 디렉토리의 encoder/tokenizer 를 우선 사용
    resume_ckpt_dir: Path | None = None
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_file() and resume_path.name == "ckpt.pt":
            resume_ckpt_dir = resume_path.parent
        elif resume_path.is_dir():
            resume_ckpt_dir = resume_path
        else:
            logger.error("--resume 경로 잘못됨 (디렉토리 또는 ckpt.pt 파일): %s", resume_path)
            raise SystemExit(1)
        encoder_path = str(resume_ckpt_dir / "encoder")
        tokenizer_path = str(resume_ckpt_dir / "tokenizer")
        if not Path(encoder_path).is_dir():
            logger.error("resume encoder 디렉토리 없음: %s", encoder_path)
            raise SystemExit(1)
        logger.info("RESUME 모드: %s 에서 재개", resume_ckpt_dir)
    elif args.previous_model_path:
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

    _collate = _make_collate_fn(tokenizer, max_len)
    train_loader = DataLoader(
        EmotionDataset(train_rows),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        EmotionDataset(val_rows),
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * max_epochs
    warmup_steps = max(1, total_steps // 10)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # dialog_act는 불균형 클래스에 역빈도 가중치 적용; emotion은 prepare에서 이미 균형화
    emotion_criterion = nn.CrossEntropyLoss()
    if args.dummy:
        dialog_act_criterion = nn.CrossEntropyLoss()
    else:
        da_weights = _compute_dialog_act_weights(train_rows, device)
        dialog_act_criterion = nn.CrossEntropyLoss(weight=da_weights)

    use_amp = (device.type == "cuda")
    scaler = GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    no_improve_count = 0
    best_epoch = 0
    epoch = 0
    start_epoch = 1
    start_time = time.time()

    # --resume: head/optimizer/scheduler 복원, 마지막 epoch+1 부터 재개
    if resume_ckpt_dir is not None:
        ckpt_pt = resume_ckpt_dir / "ckpt.pt"
        if not ckpt_pt.exists():
            logger.error("ckpt.pt 없음: %s", ckpt_pt)
            raise SystemExit(1)
        ck = torch.load(str(ckpt_pt), map_location=device, weights_only=False)
        model.emotion_head.load_state_dict(ck["emotion_head"])
        model.dialog_act_head.load_state_dict(ck["dialog_act_head"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        last_epoch = int(ck["epoch"])
        last_step = int(ck["step"])
        start_epoch = last_epoch + 1
        if start_epoch > max_epochs:
            logger.warning(
                "resume: 이미 모든 epoch 완료 (last_epoch=%d, max_epochs=%d). "
                "검증/저장만 진행",
                last_epoch, max_epochs,
            )
        else:
            logger.info(
                "resume: epoch %d (step %d 까지) → epoch %d 부터 재개",
                last_epoch, last_step, start_epoch,
            )

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

    for epoch in range(start_epoch, max_epochs + 1):
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
            with autocast("cuda", enabled=use_amp):
                e_logit, d_logit = model(input_ids, attention_mask, token_type_ids)
                loss = (
                    ALPHA_EMOTION * emotion_criterion(e_logit, e_lbl)
                    + ALPHA_DIALOG_ACT * dialog_act_criterion(d_logit, d_lbl)
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item()

            if step % 500 == 0:
                elapsed = time.time() - t0
                steps_per_sec = step / max(elapsed, 1e-6)
                logger.info(
                    "Epoch %d/%d  Step %d/%d  loss=%.4f  %.2f steps/s",
                    epoch, max_epochs, step, len(train_loader), epoch_loss / step, steps_per_sec,
                )

            # 중간 체크포인트: 재부팅 보험 (최신 1개만 유지, 검증 없이 weights만 저장)
            if args.save_steps > 0 and step % args.save_steps == 0:
                ckpt_dir = output_dir / "checkpoint_latest"
                ckpt_dir.mkdir(exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "step": step,
                        "emotion_head": model.emotion_head.state_dict(),
                        "dialog_act_head": model.dialog_act_head.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch_loss_so_far": epoch_loss / step,
                    },
                    ckpt_dir / "ckpt.pt",
                )
                model.encoder.save_pretrained(str(ckpt_dir / "encoder"))
                tokenizer.save_pretrained(str(ckpt_dir / "tokenizer"))
                logger.info("중간 체크포인트 저장 (epoch=%d step=%d)", epoch, step)

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
            _save_model(model, tokenizer, output_dir, version, encoder_path)
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
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save-steps", type=int, default=2000,
                        help="N 스텝마다 중간 체크포인트 저장 (0=비활성화)")
    parser.add_argument("--resume", default=None,
                        help="checkpoint_latest 디렉토리 또는 ckpt.pt 경로. "
                             "지정 시 encoder/tokenizer/heads/optimizer/scheduler 복원 후 "
                             "마지막 epoch+1 부터 재개")
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--status-file", default=None)
    args = parser.parse_args()

    global _external_status_file
    if args.status_file:
        _external_status_file = Path(args.status_file)

    if not args.dummy and not Path(args.train_csv).exists():
        logger.error("학습 CSV 없음: %s", args.train_csv)
        raise SystemExit(1)

    train(args)


if __name__ == "__main__":
    main()
