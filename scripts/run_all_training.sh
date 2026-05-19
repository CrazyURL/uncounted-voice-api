#!/bin/bash
# 전체 모델 순차 학습 체인 스크립트
# 각 모델 완료 + 저장 검증 후 다음 모델 진행
# 실행: bash scripts/run_all_training.sh [--start-from emotion|speech_age|topic|speech_act]
# 로그: logs/chain_training_YYYYMMDD.log (각 모델 로그는 별도)

set -euo pipefail

PROJ="/home/gdash/project/Uncounted-root/uncounted-voice-api"
PYTHON="$PROJ/venv/bin/python"
# CPU 양보(nice 15) + OOM 우선 종료 대상(score 500) 설정
RUN_PYTHON() { nice -n 15 "$PYTHON" "$@"; }
set_oom_score() { echo 500 > /proc/$1/oom_score_adj 2>/dev/null || true; }
LOG_DATE=$(date +%Y%m%d)
CHAIN_LOG="$PROJ/logs/chain_training_${LOG_DATE}.log"
START_FROM="${1:-emotion}"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$CHAIN_LOG"; }
fail() { log "❌ FAIL: $*"; exit 1; }

# 저장 검증 함수: 필수 파일 존재 + non-empty 확인
verify_save() {
    local model_dir="$1"
    local label="$2"
    local latest
    latest=$(ls -td "$PROJ/models/$model_dir/v"* 2>/dev/null | head -1)
    if [ -z "$latest" ]; then
        fail "$label: models/$model_dir/v* 없음"
    fi
    for f in metrics.json label_map.json model_card.json; do
        if [ ! -s "$latest/$f" ]; then
            fail "$label: $latest/$f 없음 또는 비어있음"
        fi
    done
    local f1
    f1=$(python3 -c "
import json, sys
m = json.load(open('$latest/metrics.json'))
keys = [k for k in m if 'f1' in k.lower()]
for k in keys: print(f'{k}={m[k]:.4f}')
" 2>/dev/null || echo "f1=N/A")
    log "✅ $label 저장 확인: $latest  ($f1)"
}

cd "$PROJ"

log "=== 체인 학습 시작 (start_from=$START_FROM) ==="

# ─── emotion ──────────────────────────────────────────────
if [[ "$START_FROM" == "emotion" ]]; then
    log "--- [1/4] emotion 학습 시작 ---"
    RUN_PYTHON scripts/train_emotion_model.py \
        --max-epochs 5 --batch-size 32 --save-steps 10000 \
        2>&1 | tee "logs/train_emotion_${LOG_DATE}.log"
    verify_save "emotion" "emotion"
    log "--- emotion 완료 ---"
fi

# ─── speech_age ───────────────────────────────────────────
if [[ "$START_FROM" == "emotion" || "$START_FROM" == "speech_age" ]]; then
    log "--- [2/4] speech_age 학습 시작 ---"
    RUN_PYTHON scripts/train_speech_age_model.py \
        --max-epochs 5 --batch-size 32 \
        2>&1 | tee "logs/train_speech_age_${LOG_DATE}.log"
    verify_save "speech_age" "speech_age"
    log "--- speech_age 완료 ---"
fi

# ─── topic ────────────────────────────────────────────────
if [[ "$START_FROM" == "emotion" || "$START_FROM" == "speech_age" || "$START_FROM" == "topic" ]]; then
    log "--- [3/4] topic 학습 시작 ---"
    RUN_PYTHON scripts/train_topic_model.py \
        --max-epochs 5 --batch-size 32 \
        2>&1 | tee "logs/train_topic_${LOG_DATE}.log"
    verify_save "topic" "topic"
    log "--- topic 완료 ---"
fi

# ─── speech_act ───────────────────────────────────────────
log "--- [4/4] speech_act 학습 시작 ---"
RUN_PYTHON scripts/train_speech_act_model.py \
    --max-epochs 5 --batch-size 32 --target group \
    2>&1 | tee "logs/train_speech_act_${LOG_DATE}.log"
verify_save "speech_act" "speech_act"
log "--- speech_act 완료 ---"

# ─── 최종 요약 ────────────────────────────────────────────
log ""
log "=== 체인 학습 전체 완료 ==="
for model in emotion speech_age topic speech_act; do
    latest=$(ls -td "$PROJ/models/$model/v"* 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        f1=$(python3 -c "
import json
m = json.load(open('$latest/metrics.json'))
keys = [k for k in m if 'f1' in k.lower()]
vals = ' '.join(f'{k}={m[k]:.4f}' for k in keys[:2])
print(vals or 'f1=N/A')
" 2>/dev/null || echo "?")
        log "  $model: $(basename $latest)  $f1"
    else
        log "  $model: 저장 없음"
    fi
done
log "체인 로그: $CHAIN_LOG"
