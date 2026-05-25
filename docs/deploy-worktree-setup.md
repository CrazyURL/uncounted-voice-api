# 배포 Worktree 설정 런북

## 개요

`uncounted-voice-api`는 git worktree를 활용해 배포 환경을 분리한다.
메인 개발은 `uncounted-voice-api/`에서, 운영 배포는 `uncounted-voice-api-deploy-s1/`에서 실행한다.

## models/ 심볼릭 링크 필수

배포 worktree는 `.gitignore`에 의해 `models/`를 추적하지 않는다.
운영 서버에서 `auto_label_service`가 emotion/speech_age 모델을 찾으려면
다음 심볼릭 링크가 worktree 루트에 반드시 있어야 한다.

```bash
# 배포 worktree 루트에서 1회 실행
cd /home/gdash/project/Uncounted-root/uncounted-voice-api-deploy-s1
ln -sfn /home/gdash/project/Uncounted-root/uncounted-voice-api/models models
```

확인:
```bash
ls -la models/emotion/current
# → lrwxrwxrwx ... current -> .../models/emotion/v{VERSION}
```

## uvicorn 기동 방법

배포 worktree의 `run.sh`는 메인 repo의 `.env.dev`를 명시적으로 지정해야 한다.
(worktree 자체에는 `.env.dev`가 없으므로)

```bash
cd /home/gdash/project/Uncounted-root/uncounted-voice-api-deploy-s1

# tmux 세션으로 기동
tmux new-session -d -s voice-api-dev -x 220 -y 50 \
  'cd /home/gdash/project/Uncounted-root/uncounted-voice-api-deploy-s1 && \
   source /home/gdash/project/Uncounted-root/uncounted-voice-api/.env.dev && \
   export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
          NUMEXPR_MAX_THREADS=4 TOKENIZERS_PARALLELISM=false \
          CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
   /home/gdash/project/Uncounted-root/uncounted-voice-api/venv/bin/uvicorn \
     app.main:app --host 0.0.0.0 --port 8001 --workers 1 --log-level debug \
     2>&1 | tee /tmp/uvicorn.log'
```

## venv

배포 worktree는 별도 venv 없이 메인 repo의 venv를 사용한다.

```
메인 repo: /home/gdash/project/Uncounted-root/uncounted-voice-api/venv/
```

## emotion 모델 승격 절차

1. 새 모델 학습 완료 확인 (`models/emotion/v{VERSION}/metrics.json`)
2. gold eval set으로 성능 검증
3. symlink 변경: `ln -sfn $(pwd)/models/emotion/v{VERSION} models/emotion/current`
4. uvicorn graceful restart (tmux 세션 재기동)
5. `auto_label_service` 로드 확인 (로그에서 `emotion-only 모델 — dialog_act_head 없음` 확인)

## emotion-only 모델 호환성

`v20260524_095713` 이후 모델은 `emotion_head`만 포함하며 `dialog_act_head`가 없다.
`auto_label_service.py`는 이를 자동으로 감지해 `dialog_act_head=None`으로 처리한다.
STT 결과의 `dialog_act` 필드는 `None`으로 반환된다.
