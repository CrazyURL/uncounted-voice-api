# raw_direct activation preflight — baseline (read-only)

작성: 2026-05-31 21:34 UTC
목적: raw_direct(SPEAKER_MAPPING_MODE) activation 직전 최종 안전성 기준선. **activation/restart/canary 미실행 — read-only.**

## 1. 운영 baseline snapshot
| 항목 | 값 |
|------|-----|
| timestamp_utc | 2026-05-31T21:34:16Z |
| sessions | done 583 / running 0 / failed 0 / pending 734 / done_empty 110 |
| orphan | 199 rows / 133 sessions |
| api NRestarts | 0 (active) |
| worker NRestarts | 1 (active) |
| VRAM idle | 4,625 MiB / 8,188 MiB |
| queue | depth 0, gpu_busy False (idle) |
| model | large-v3 |
| SHA | **b3c338e** |
| SPEAKER_MAPPING_MODE | unset → **whisperx** (default, raw_direct 비활성) |

## 2. PR #21 / #22 / config 확인
- origin/main HEAD = **b3c338e** "feat(stt): raw pyannote direct speaker mapping (Phase 3 draft) (#21)"
- git 이력 순서: b3c338e(#21) → fa5ea74(#22 None hardening) → 915bf23(#20 large-v3) → 4f80fd1 → 33e535d
- **PR #22(None hardening) HEAD 이력 포함 확인**: `git merge-base --is-ancestor fa5ea74 HEAD` = YES ✓
- `app/speaker_mapping.py` 존재 ✓ (assign_speakers, whisperx 호환 schema: result dict + segments list)
- `app/config.py:161` `SPEAKER_MAPPING_MODE = os.environ.get("SPEAKER_MAPPING_MODE", "whisperx")` — **default whisperx 확인** ✓

## 3. raw_direct activation impact (read-only 분석)
- **env gate 미설정 시 분기 = whisperx**: stt_processor.py:161-167
  ```
  speaker_mapping_mode = getattr(config,"SPEAKER_MAPPING_MODE","whisperx")
  raw = (speaker_mapping_mode or "whisperx").strip().lower()
  if raw == "raw_direct":  result = app.speaker_mapping.assign_speakers(...)
  else:                    result = whisperx.assign_word_speakers(...)
  ```
  → env 없으면 항상 whisperx. **코드 머지(b3c338e)만으로 raw_direct 자동 활성화 안 됨** ✓
- **restart 발생해도 자동 활성화 안 됨**: env 미설정 상태 restart → 여전히 default whisperx ✓
- **PR#22 None coerce 반영**: speaker=None → "SPEAKER_UNKNOWN" (assign_speakers + worker persist). config.py:165-166 명시. raw_direct 활성 시에도 None corruption 차단됨 ✓
- **rollback path 유효**:
  - drop-in `voice-api@dev.service.d/*.conf`에서 SPEAKER_MAPPING_MODE 제거 또는 =whisperx → daemon-reload + restart → 즉시 legacy 복귀
  - 예상 rollback 시간: **~15초** (api restart + 모델로딩 ~10-15s, 코드 변경 없어 daemon-reload 즉시)

## 4. activation 절차 (참고 — 본 문서는 실행 안 함)
- activation: `voice-api@dev.service.d/speaker-mapping.conf`에 `Environment=SPEAKER_MAPPING_MODE=raw_direct` + daemon-reload + voice-api restart
- worker는 화자매핑 비주체(voice-api가 assign) → api restart만 필수, worker는 코드 일관성용 선택

## activation readiness 판정
| 게이트 | 상태 |
|--------|------|
| 코드 머지(#21) | ✅ b3c338e |
| None hardening(#22) 선반영 | ✅ 이력 포함 |
| default 안전(whisperx) | ✅ 자동활성화 차단 |
| rollback 유효 | ✅ drop-in 제거+restart (~15s) |
| 운영 baseline 안정 | ✅ failed0·NRestarts안정·queue idle·VRAM 4.6GB |
| blocker | **없음** |

→ **activation 가능 상태** (단 사용자 승인 + restart 필요. 현재는 preflight만).
