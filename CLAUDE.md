# uncounted-voice-api — WhisperX STT Server

WhisperX 기반 음성 처리 API. STT + 화자분리 + 발화분리 + 텍스트/음성 PII 마스킹.

## 기술 스택

- Python 3.12 / FastAPI / Uvicorn
- WhisperX 3.8.5 (large-v3, CUDA)
- pyannote (화자분리, HF_TOKEN 필요)
- numpy + soundfile (오디오 처리)
- pytest (테스트)

## 필수 명령어

```bash
./run.sh dev              # dev 서버 (port 8001)
./run.sh live             # live 서버 (port 8000)
python -m pytest -q       # 테스트 (178개)
sudo systemctl restart voice-api@dev   # 서비스 재시작
sudo systemctl restart voice-api@live
```

## 디렉토리 구조

```
app/
├── main.py              # FastAPI 진입점
├── config.py            # 환경변수 설정
├── stt_processor.py     # STT 파이프라인 (핵심)
├── pii_masker.py        # PII 마스킹
├── routers/             # health, transcribe
├── services/            # audio_preprocessor, audio_splitter, pii_service, utterance_segmenter, whisperx_service
├── models/schemas.py    # Pydantic 스키마
└── core/                # job_store (TTL 1h, max 100), exceptions
tests/                   # pytest (178개)
scripts/                 # 디버그 스크립트 (VRAM 확인, 파이프라인 검사 등)
```

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| GET | /api/v1/health | 서버 상태 |
| POST | /api/v1/transcribe | 음성 업로드 → STT |
| GET | /api/v1/jobs/{task_id} | 결과 조회 |
| GET | /api/v1/jobs/{task_id}/audio/{filename} | WAV 다운로드 |

## 상세 참조

- `docs/api-reference.md` — API 상세 + Swagger 필드 설명
- `.claude/rules/python/stt-pipeline.md` — STT 파이프라인 아키텍처 + 비동기 작업 패턴
- `.claude/rules/python/performance.md` — 성능 최적화 + DeepFilterNet + 청크 모드
- `.claude/rules/python/config-and-pii.md` — 환경변수 테이블 + PII 마스킹 상세
- Swagger UI: `http://{host}:{port}/docs`

## ⚠️ 운영 사실 — 창 간 공유 (정본: Windows `uncounted-root/CLAUDE.md` §1~15)

> 본 절은 **GPU/voice-api 작업에 필요한 사실만 발췌·미러**한 것이다. 권위 원본은 Windows 워크스테이션의 `uncounted-root/CLAUDE.md`(§1~15) + `scripts/analysis/INDEX.md`(창 간 작업 위키). 상충 시 루트 CLAUDE.md 가 우선. GPU 접속정보·자격증명은 보안상 본 파일에 두지 않는다.
> ⚠️ 이력·사실 기록일 뿐, 성능·기능 개선의 제약이 아니다.

### 워커/서비스 단일화
- **GPU voice-api(:8001 dev / :8000 live) + `voice-worker@dev` 단독 운영.** uncounted-api 의 TS gpu-worker 는 비활성(`GPU_WORKER_ENABLED=false`). 새 기능은 Python(`app/worker.py`) 측에 반영.
- voice-api 리포 = `uncounted-voice-api`(hyphen) 단일. underscore 리포는 archived(read-only).

### 자동라벨·품질 모델 (자주 헷갈림)
- **모델 symlink 확인 경로 함정**: live 모델은 **worker cwd 기준 `models/*/current`** 만 유효(`readlink /proc/<voice-worker MainPID>/cwd`). 같은 서버 부모레벨 `…/Uncounted-root/models/emotion/current`(→`v20260514_174430`)는 **stray·미배선** — `find /home/gdash/*/*/models` 류 glob 으로 잡으면 emotion current 가 바뀐 것으로 오인.
- **emotion 버전 성격**: `v20260519_232848`(32샘플 토이), `v20260519_232923`(1.87M 정식이나 **dialog_act 100% 기타 붕괴**), `v20260524_092815`(dialog_act head 제거=emotion-only, 토이런). dialog_act 학습라벨 부재가 근본원인 → **emotion current 가 dialog_act 기타 100% 를 내면 데이터버그 재발 신호.** `macro_f1=1.0` 은 단일클래스 degenerate artifact(누수 아님).
- **dialog_act**: 학습데이터 라벨 부재로 어느 버전도 미신뢰 → 납품 제외.
- **speech_act / topic**: standalone `models/speech_act`·학습 `models/topic` head 는 **추론 미배선(orphan)**. topic 문자열은 cosine+seed-phrase(30종)로 별도 생성.
- **speech_age 로더버그(현존)**: `auto_label_service._try_load_speech_age()` 는 `heads.pt`+key `speech_age_head` 를 찾으나 산출물은 `head.pt`+key `head`(`train_speech_age_model.py`) → 가중치 미로드 = **랜덤**. 말투연령(`speaker_speech_age_range`) 값은 채워져도 무의미.
- **로드 캐시 함정**: `auto_label_service` 는 `models/*/current` 를 **lazy-load 후 메모리 캐시** → symlink 교체는 프로세스 재시작(또는 첫 추론 전 교체) 전까지 미반영. 적용여부는 `/proc/<pid>/maps` 또는 신규 처리분 `model_version` 으로 확인.
- 자동라벨 emit 은 `source:"automatic"`(사람 검수 정답 아님). `honorific_level`/`confidence_tier` 등은 emit 시작 전 null 가능.

### 품질 (B-60 버그, 수정됨)
- 옛 `worker.py` 가 `ffprobe -af`(미지원) 사용 → astats/silencedetect 실패 → fallback rms=peak=-60 → **snr=0·grade B·speech_ratio=1.0 고착**. fix `d40a8fd`(ffmpeg -v info + stderr 파싱). **기존 DB 의 snr=0/grade B 값은 실측 아님** — 재측정/backfill 전까지 품질 근거로 쓰지 말 것(신규 처리분은 정상).
- `worker.py` 는 `numeric_patterns`/`utterance_form`/`clipping_ratio`/`snr_db`/`speech_ratio` 를 audio·transcript 에서 **직접 계산**(Voice API 응답 의존 X).

### 화자 성별/목소리연령
- `speaker_analysis_service._detect_gender_and_voice_age()` = librosa F0(pyin) 중앙값(male<165, ≥165 female, 겹침 None). 함수내 **lazy `import librosa`** 실패 시 `(None,None)`=「미상」. **venv 재생성 시 librosa 누락 재발 주의** → `venv/bin/pip install 'librosa>=0.10.0'`(restart 불필요, 다음 신규 세션부터 적용). 「말투연령은 나오는데 목소리연령/성별만 미상」=librosa 누락 신호.

### STT
- dev `.env.dev` `MODEL_SIZE=large-v3-turbo`(large-v3 아님). GPU 8GB → large-v3 동시 로딩 OOM 위험. 운영 권장: **turbo + 중립 prompt + 지명 HOTWORDS**. 모델/prompt/hotword A/B 는 **CPU(device=cpu, float32) 무중단 비교** 가능.

### PII
- transcript_text 는 voice-api 가 **이미 마스킹된 상태로 emit**(`UtteranceResult` 엔 utterance-level `pii_intervals` 없음, `pii_summary` 는 job-level). 표시-시점 실명 denylist 마스킹(`[이름]`)은 **API 측 admin 화면·admin-downloads** 경로에서 수행(voice-api 무관). export-v2 패키지는 denylist 이름 발견 시 safety-checks **fail-closed 차단**.
