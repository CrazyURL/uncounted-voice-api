# CONTEXT — voice-api 진행 맥락 (개발PC 간 공유용)

> **이 문서의 목적**: GPU 서버에서 진행 중인 voice-api 개선 작업의 맥락을, 다른 개발 PC에서도
> `git pull`로 공유받기 위한 스냅샷. 코드/git 히스토리에서 자명하게 드러나지 않는 **결정 사항·
> 진행 상태·미완결 트랙·함정**만 기록한다.
>
> **스냅샷 기준일**: 2026-06-02 · **출처**: GPU 서버 작업 세션 누적 맥락
> **주의**: 이 문서는 시점 스냅샷이다. 파일·함수·플래그·PR 번호를 인용하므로, 권고/적용 전
> `git fetch` 후 origin/main과 실제 코드를 다시 확인할 것(아래 [협업 수칙](#협업-수칙) 참조).

## 상태 범례

- ✅ 완료/머지 · 🟡 DRAFT/승인대기 · 🔬 PoC/조사 · 📋 스펙확정·미구현 · ⏸ HOLD · ⚠️ 함정/주의

---

## 1. 오디오 전처리 파이프라인 (gain / LUFS)

**현재 코드 동작** (`app/services/audio_preprocessor.py`, `app/config.py`):
- LUFS/EBU R128 정규화는 **사용하지 않음** (`loudnorm`/`pyloudnorm`/`ebur128` 코드 0건).
- 대신 **RMS 기반 게인**: `normalize_gain()` = `TARGET_GAIN_RMS(0.1) / rms`, 최대 `MAX_GAIN_X=10x`.
- `local_normalize_gain()` = 500ms 슬라이딩 윈도우 + 100ms hop, 선형보간 게인커브, **부스트 전용**,
  `LOCAL_MAX_GAIN_X=30x`.
- ⚠️ **기본 config 상태**: `PREPROCESS_GAIN_ENABLED=true`, **denoise/dedup/silence = 전부 false(OFF)**.
  즉 기본 파이프라인은 게인만 돈다. (Round 1~4 점진 재활성화 진행 중.)
- ⚠️ **측정된 cascade 문제**(코드 주석): DeepFilterNet denoise가 voice RMS를 **median 23배 감쇠** →
  regain 필요, `MAX_GAIN_X=10`으로는 완전 복원 불가. denoise 후 silence 임계값 동적 하향
  (`SILENCE_RMS_THRESHOLD_DENOISE=0.0005`)으로 보완.
- DeepFilterNet은 **이미 구현돼 있음**(별도 subprocess, 파일기반 통신, CPU 격리)이나 기본 OFF.

**🟡 PR #29 DRAFT — LUFS 정규화 + noise breathing 게이트** (2026-06-02):
- 문제: denoise OFF인데 `local 30x` 부스트 → 조용한 윈도우의 미세 소음 증폭(noise breathing) 시한폭탄.
- 변경: `LOCAL_MAX_GAIN_X 30→10` + **통화 단위 균일 LUFS**(클립별 적분 LUFS의 짧은-클립 불안정 회피) +
  noise floor 게이트. 통화 단위 정규화는 `snr_db` 불변 → **품질 등급 중립**(실측 확인).
- 전부 **env-gate OFF**, canary 후 활성. **테스트 590 passed**.

**발화 클립 추출 지점** (대안2 LUFS 삽입점): `app/services/audio_splitter.py`의
`extract_utterance_audio()`는 순수 슬라이스(정규화 없음) → `stt_processor.py`에서 `utterance_NNN.wav`로 저장.

---

## 2. 화자분리 / 발화분리

**✅ 하이브리드 발화분리 PR #27** (머지·가동·e2e 검증완료, 2026-06-02, `e5dabc1`):
- 도입부 30초를 **NeMo MSDD로 재분리** + 코사인 ID 매핑 + 하드 오버라이트.
- ★ 세션 `93c28f57` 재처리에서 GT1 **ABAB**(본인-상대-본인-상대) 복원 성공.
- 결합 VRAM 3987MiB 안전. **NeMo 서비스 :8009** (background, **systemd 아님 → 재부팅 시 수동 기동**).
- env gate ON(`.env.dev`). 후속: systemd 유닛화 / chunked / 일반화.

**🔬 도입부 화자분리 PoC** (2026-06-02): pyannote가 짧은 turn(0.3~0.45s)을 단일화자로 오분리 →
NeMo MSDD가 100% 정확 분리·ID 매핑 명확(전략2 viable). NeMo 격리 venv = `/home/gdash/_poc/nemo_venv`.

**🟡 GPU 프로세스 락 PR #28 DRAFT** (2026-06-02): voice-api STT ↔ NeMo 동시추론 VRAM peak(7.5GB)
충돌 방지 filelock. 데드락 회피 위해 하이브리드를 락 밖으로. 검증 데드락0·VRAM 7610<임계.
⚠️ 단 하이브리드 GT1 하류 편차 → 게이트 OFF 보류.

**📋 Task5 화자중첩** (스펙확정·미구현, 2026-06-02): `is_overlapping` boolean 플래그로 확정.
**음원분리 금지**(정합성 붕괴). 탐지 소스만 pyannote segmentation으로 upgrade 권장.

---

## 3. STT 세그먼테이션

**✅ Segmenter v2 PR #9** (MERGED, 기본 OFF wiring): STT 과분할 1차 원인 = postprocess.
bidirectional 권장(-3.3%/-11%). dev 무회귀 검증완료(검증만).
⚠️ dev=live 동일 Supabase → worker 토폴로지 정리 후 카나리.

**STT v2 트랙 종료** (2026-05-26): 알고리즘·문장종결 가드·replay·정성·rollout 설계 완료,
**운영 적용만 승인대기**. 산출물: `scripts/stt_v2_canary_replay.py`, `scripts/stt_v2_qualitative_review.py`.

**⚠️ Worker 토폴로지**: 단일 worker가 **유일한 Supabase writer**, 세그먼테이션은 API(8001)측.
라이브(deploy-s2a)엔 v2 코드 없음 → `MERGE_V2`만 켜선 안 됨. 안전 카나리 = 오프라인 리플레이(8002, DB 무접촉).

Phase 8 STT: cond5c 결정 + hotword 설정.

---

## 4. PII 마스킹

- **✅ PR-S1(regex) · S2A(tier)** 둘 다 MERGED (`8ec434b` / `d14012b`). ⚠️ main에 PR-S1 단독 병합 금지.
  C안(2026-05-26): IP는 최종 auto_confirm 복귀 가능하나 현재 live는 needs_human 유지.
- **🟡 PII graded confidence PR #11** (정본; #10은 CLOSED): S2A선 `d14012b` 위 cherry-pick, 충돌0,
  diff 5파일, 158+54+6 통과, IP demotion 보존. **merge/backfill 승인대기**.
- **✅ PII-1A detect-batch** live 검증 통과(2026-05-24). backfill은 개발PC에서.
- ⚠️ **PII 학습 export**(2026-05-26): 53/72/1, 전부 이름·피처 degenerate, 텍스트 없이 학습 불가.

---

## 5. 품질 등급 & 재측정

- **✅ 품질등급 단가기준 확정**(2026-06-02): "**전처리 후 제공 파일 품질**"(원본 아님).
  재측정 배치는 **utterance WAV(전처리본)** 기준.
- **✅ B-60 재측정 dry-run 전량완료**(2026-06-02): 오염 878 재측정 → A133/B644/C101
  (상향133 ≈ 하향101 균형). 단일세션 순감은 비대표 착시였음(정정). 산출물:
  `scripts/analysis/quality_remeasure_dryrun.py`.
- ⏸ **quality_tier write-orphan**: `session_quality_tier`/`topic_summary` 100% null = 버그 아닌
  **미완 기능**(write-orphan). null-safe. 활성 populator = `worker.py`(TS gpu-worker는 dead).
  HOLD — B-60 Phase2 이후 재평가. 지금 채우면 오염tier·export자격역전·납품물변경 위험.

---

## 6. 관계 / 화자 identity

- **✅ relationship/peer 모델 v2 확정**(2026-05-27): cross-kind 자동merge 제거(admin 수동만) ·
  override lock(`override_locked`+`locked_scope`+append history) · propagation 강키 조건부 AND ·
  약키 자동전파 영구금지 · owner≠SPEAKER_00.
- **Layer A/B migration plan 입력 명세**(ground truth, 2026-05-27): `session_labels`=runtime-dead ·
  `session_speakers`(066)=active · `peers`=greenfield. §4 질문 7개 종료.
  상태=**DDL 입력명세 완료/실행대기**(reset 이후 착수). durable doc=
  `scripts/analysis/relationship_migration_plan_inputs_20260527.md`(미커밋).
- **🔬 peer 정적자산 캐싱**(미구현, 2026-06-02): relation+화자프로필(성별/연령)을 longest-call에서
  1회 계산 후 캐싱. `voice_profiles`는 user(본인)용만 존재 → peer용 신규 필요.
  v2 propagation gate + 공용번호 필터 정합 필요. (PR #29 LUFS 통화단위 1회측정과 결 동일.)

---

## 7. 동의 / consent

⚠️ **consent locked 적체**(2026-05-27): `locked`=동의 前(`consented_at` NULL), `both_agreed`=동의완료.
265건 적체 = 전이(write) 유실 추정. **동의 write 코드는 이 리포에 없음**(모바일/Supabase).
결함3 = `pick_next`에 consent 필터 추가(미실행). 보고서=
`scripts/analysis/consent_locked_investigation_20260527.md`.

---

## 8. 운영 사고 이력 (재발 방지)

- ⚠️ **utt=0 사고**(2026-05-27): 런처 `set -a` 누락 → uvicorn에 `HF_TOKEN` 미export → 화자분리 skip
  → 발화 0. 조치=deploy-s2a cwd + 메인 venv 재기동 + `HF_TOKEN` 추가(검증완료, 수정 후 done-empty=0).
  **uvicorn=deploy-s2a / worker=메인, 공용 venv**.
- ⚠️ **빈 done 사고**: 근본원인 = `HF_TOKEN` 누락 → 화자분리 스킵. systemd 토폴로지 / 2026-05-27 freeze.
- **세션 93c28f57 파이프라인 검수 ZIP**(2026-06-02): PR-α/β/γ'/δ/ε 5개 머지. 하드코딩0·전부 자동산출.
  relation 배우자→직장동료(Ollama), speaker_source 329/329, PII 마스킹(시간교차),
  safety detector ID 오탐 수정. 운영영향 0.

---

## 9. 협업 수칙 (이 리포 작업 시)

- **구현 착수 전**: `git fetch` + origin/main stale 여부 + **이미 머지된 PR 존재** 확인 선행.
  (로컬 main stale / stale checkout 착시가 반복 발생 — "blocked"가 실제론 stale인 경우 많았음.)
- **검증 후에만 결론**: canary/검증/diff 결론은 실측(결과파일 Read·코드 확인) 후에만 기록.
  추정 성공수치 금지.
- **공유 멀티윈도우 운영**: 프로세스 kill 전 "누가 띄웠나" 확인 필수, freeze 우선.

---

## 부록 — 이 맥락의 원본 위치

이 문서의 상세 원본은 GPU 서버 로컬 Claude 메모리에만 있으며(`~/.claude/projects/.../memory/`)
**다른 PC로 동기화되지 않는다.** 그래서 공유 가능한 기술 맥락만 이 문서로 추출했다.
크로스레포(운영/배포/admin·api) 맥락은 `Uncounted-root/docs/CONTEXT_uncounted_ecosystem.md` 참조.
