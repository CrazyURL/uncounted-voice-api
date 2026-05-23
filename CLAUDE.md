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

## 품질 지표 B-60 고착 버그 수정 — 2026-05-23

**커밋**: `d40a8fd` (feat/window-b2-worker-fields)

**원인**:
`app/worker.py` `_get_audio_stats_sync()` 에서 `ffprobe -af`를 사용했으나,
`ffprobe`는 오디오 필터 옵션 `-af`를 지원하지 않는다.
`astats`/`silencedetect` 필터가 묵시적으로 실패하고 fallback 기본값이 반환되는 문제.

```
fallback: rms_db=-60, peak_db=-60, silence_ratio=0
-> snr_db=0, speech_ratio=1.0, quality_score=60, quality_grade=B (모든 파일 동일)
```

**수정** (`app/worker.py`):
- `ffprobe -af astats=...` -> `ffmpeg -v info -i <path> -af astats=... -f null -`
- `ffprobe -af silencedetect=...` -> `ffmpeg -v info -i <path> -af silencedetect=... -f null -`
- `returncode != 0` 시 `log.warning()` 추가
- `_compute_quality()` 공식은 변경하지 않음
- stderr 파싱 기존 로직 유지

**검증**:
| 파일 | score | grade |
|------|-------|-------|
| 무음 4초 | 20 | C |
| 440Hz tone | 64 | B |
| 백색잡음 | 66 | B |
| 실제 음성 | 84 | A |

**주의**: `app/worker.py` 수정 시 `ffprobe`를 audio filter 용도로 사용하지 말 것.
audio filter는 반드시 `ffmpeg` 명령어로 실행하고 결과는 `proc.stderr`에서 파싱.
