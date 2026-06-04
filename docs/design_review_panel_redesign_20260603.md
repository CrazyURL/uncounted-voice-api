# 검수자 패널 재설계 & STT 평가 정규화 정책

> **Status:** §0 정규화 정책 **LOCKED (2026-06-03)** / §1 환각·오역 처리 결정 확정 / §2~ 검수 UI 설계 진행
> **Date:** 2026-06-03
> **Scope:** uncounted-voice-api (평가·측정), uncounted-admin (검수자 UI)
> **정본 코드:** `scripts/analysis/stt_score_policy.py` (규칙① 정식 구현, self-check 포함)

---

## §0 정규화 정책 (LOCKED — 데이터 공장 통제 변수)

> **선언**: 한국어 STT 점수의 단일 기준점. 이 전처리를 거치지 않은 모든 통계·단가 계산은 무효로 간주한다.
> 문자단위와 단어단위 점수의 괴리(아래 §2)는 **Whisper의 지능 실패가 아니라 한국어 띄어쓰기의 문법적 가변성에서 오는 착시 노이즈**다. 기준을 통일하지 않으면 모든 지표·납품 단가가 모래성처럼 흔들린다.

### 규칙① — 평가/바이어 리포트 정식 지표 = 공백제거 Character 정확도
- 모델 평가·바이어 리포트 점수는 텍스트 내 **모든 공백을 제거**하고 한글·영숫자만 남긴 순수 글자 시퀀스를 **Levenshtein 거리(jiwer.cer)**로 비교한다.
- **정식 지표 = `char_accuracy = 1 − CER`** (`stt_score_policy.char_accuracy`).
- 효과: `어떤 거`↔`어떤거`, `되는 거죠`↔`되는거죠`, `IT 처리 요청서`↔`IT처리요청서` 같은 **껍데기 노이즈를 제거**하고, 본질적 오역(`수석님`→`선생님`)만 정밀 감점한다.
- **단어단위(strict token match)는 단독 품질지표로 쓰지 않는다.** 보조 참고용(`word_accuracy`)으로만 병기.

### 규칙② — 형태소 기반 완화 토큰 매칭 (Kiwi/KoNLPy)
- 띄어쓰기가 달라도 **품사 단위 정합성이 맞으면 정답 인정**하는 완화 매칭 모듈을 평가 엔진에 배선.
- 상태: `kiwipiepy` **미설치** → 설치 후 `stt_score_policy`에 `morph_accuracy()` 추가 예정. 규칙①이 1차 기준이므로 차순위.

### 규칙③ — 검수자 패널 자동 띄어쓰기 정정 (PyKoSpacing)
- 검수자가 띄어쓰기를 한 땀 한 땀 교정하는 건 리소스 낭비. **검수 완료 시 백엔드가 국문 띄어쓰기 라이브러리(PyKoSpacing)로 1차 표준화**한 뒤 `utterance_gt`에 저장.
- 상태: `pykospacing` **미설치** → P1 검수 UI 백엔드 훅에 배선 예정.

---

## §1 환각·오역 처리 결정 (확정)

7종 신호(RMS/SNR·wav2vec2 align score·Whisper word.prob·CTC 텍스트·토큰밀도·params·음소거리) 전수 실측 결과, **부연 오전사는 ①확신(high prob) + ②문법정상 + ③정상음성 + ④부분 음소일치가 동시라 cheap signal로 정답과 구분 불가**(= 모델 천장). 답안지(GT)로 환각/오역을 두 형으로 분리한다.

### A형 — 삽입환각: **자동마킹 즉시 기각 (REJECT)**
- 정의: 정답에 **없는** 단어를 Whisper가 발명(예: `여러개 브라우저로` → `여러 개의 제품하고 이제 뭐 브라우저라`).
- 기각 근거(실측): 단어단위 difflib `insert` 파싱이 **주변 조사/공백 노이즈 때문에 `replace` 블록에 뭉개져** 수율이 안 나옴(sess3에서 ① 미검출, 오탐 `그` 1건). 빈도도 rare(n=1). → 검수자가 QA 중 마킹.

### B형 — 오역: **검수자 화면 소프트 플래그 (P1 UI)**
- 정의: **들리는** 진짜 발화를 음향적으로 잘못 들음(`우리쪽 이슈` → `조이 쇼`). 음질 정상 → 마킹 대상 아님, **정정 대상**.
- 처리: 답안지 diff의 `replace` 목록이 실제 오역을 정밀 적출 → 검수자 화면에 **하이라이팅(소프트 플래그)**. 자동치환 금지.
- sess3 적출 예: `선생님`←`수석님`, `결제`←`결재`, `품이`←`품의`, `경증`←`[이름]`(PII 구간).
- (후속 가설) **통화 내 self-consistency**: `우리쪽 이슈`가 같은 통화에 4회 정상 전사 vs 1회 `조이 쇼` 깨짐 → 텍스트 빈도 기반 오역 후보 탐지. 음향·확률과 독립이라 검증 가치 있음.

---

## §2 측정 방법론 & 실측 증거

### 정식 측정 코드 (정본 = `scripts/analysis/stt_score_policy.py`)
```python
from scripts.analysis.stt_score_policy import char_accuracy, word_accuracy
acc = char_accuracy(reference_gt, hypothesis_stt)   # 규칙① 정식 지표
```

### 실측 (sess3 = 01dd38b9, IT보안 통화, B프롬프트, 답안지 gt_01dd38b9.json)
| 지표 | 값 | 비고 |
|---|---|---|
| **규칙① 공백제거 Character 정확도** | **79.5%** (CER 20.5%) | ← **정식 락 숫자** |
| 단어단위(strict) | 38.7% (WER 61.3%) | 띄어쓰기/조사 노이즈 포함, 단독사용 금지 |
| (폐기) difflib SequenceMatcher.ratio | ~87% | **측정 오류** — ratio는 CER이 아님. 사용 중단 |

> **주의**: 과거 보고된 "STT 87%"는 difflib 유사도(비대칭·과대평가)였으며 **폐기**한다. 정식 지표는 규칙①(jiwer.cer) 기준 **79.5%**다.

### 검증 (self-check, 모듈 내장)
- 띄어쓰기만 다른 문장: char=100% / word=0% → 규칙①이 노이즈 제거 확인.
- 진짜 오역(`수석님`→`선생님`): char=33% → 본질 오역은 정상 감점 확인.

---

## §3 STT 품질 천장 (참고)

- 코드 천장(8GB·추가모델 없음) = **truncation 수정(배포완료) + B 발음페어링(약어) + D 문맥교정(혼동쌍)**.
- 환각/불명료 자동탐지·자동마킹 = **불가**(7종 실측). 답안지 diff + 사람 검수가 정답.
- 진짜 천장 돌파 = 도메인 fine-tuning (별도 트랙).

---

## §4 B형 오역 소프트플래그 — 검수 가속 (설계, 승인 대기, 미구현)

> **결정(2026-06-03)**: 신호 = **신뢰도 기반**(GT 불요, 프로덕션 검수 가속) / 범위 = 3리포 / **설계먼저 락 후 승인**.
> **핵심**: "답안지 diff"는 검수 전 GT가 없어 **프로덕션 검수엔 불가**(닭-달걀). 신뢰도 신호로 **저신뢰 단어를 하이라이트** → 검수자 글랜스. 자동수정 금지.

### 검증된 아키텍처
- 데이터플로우: `uncounted-admin`(프론트) → `uncounted-api`(:3001) → Supabase.
- 기존재: 검수 큐·`review_status`·transcript 편집(`transcripts.ts`/`transcriptStore.ts`)·`UtteranceQualityReviewControls.tsx`. `TranscriptWord`에 `probability` **타입 존재**.

### ⚠️ 결정적 갭 (실측)
- voice-api word 직렬화 = `{word, start, end, speaker}` (`app/models/schemas.py:383`) — **신뢰도 필드 없음, 적재 0**. `TranscriptWord.probability`는 타입만, 미충전.
- → **백엔드 플러밍 필수** (프론트 단독 불가).

### 신호 선택지
| | 신호 | 비용 | 한계 |
|---|---|---|---|
| **A (저렴)** | wav2vec2 align `score` (align이 이미 산출, 직렬화만) | 낮음 | 확신 오역에 high(놓침), 짧은 실단어 low(오탐) |
| **B (권장)** | Whisper `word.probability` (faster-whisper word_timestamps 캡처→align 단어에 시간매칭 머지) | 중간 | align보다 우수(제품하고 0.566), 그래도 확신오역 부분적 |
| C (조잡) | 세그먼트 `avg_logprob` (이미 있음, 세그먼트 단위 하이라이트) | 낮음 | 단어 정밀도 없음 |

→ **권장 B**: align score는 확신오역에 high라 소프트플래그 목적에 거의 무용. word.probability가 옳은 신호.

### 정직한 한계 (반드시 명시)
신뢰도 소프트플래그는 **불명료/garbled 구간은 잡지만, 확신에 찬 오역(수석님→선생님, 우리쪽이슈→조이쇼)은 못 잡음**(7종 검증과 동일 이유). **"불명료 우선검토" 도구지 만능 적출기 아님.** 마퀴 오역은 여전히 검수자 청취가 답.

### UX
- `word.probability < τ`(기본 보수값, env/설정) 단어에 배경 하이라이트. 검수자 글랜스 → 무시 or 수정.
- 오탐 비용 = 글랜스 1회(저렴). recall 지향. **자동치환 절대 금지.**

### 규칙③ PyKoSpacing 저장 훅
- 검수완료 시 백엔드(uncounted-api)가 PyKoSpacing으로 띄어쓰기 1차 표준화 후 `utterance_gt` 저장. `pykospacing` **미설치** → 설치 후 배선.

### 위상 (Phase)
- **P1-c (uncounted-admin) — ✅ 완료(2026-06-03, 프론트 단독 PoC 우선)**: 임계 기반 하이라이트 렌더 + dismiss UX + 시각 캘리브레이션(임계/최소길이 슬라이더). Mock 데이터(sess3 실측 probability).
  - `src/lib/confidenceFlag.ts`(순수로직)+`.test.ts`(**vitest 11/11**) · `src/lib/confidenceFlagMock.ts` · `src/components/domain/ConfidenceFlaggedTranscript.tsx` · `src/pages/admin/AdminConfidenceFlagPocPage.tsx` · 라우트 `/admin/confidence-flag-poc`. tsc 0 에러.
  - **API 계약 선제정의**: `TranscriptWord.probability`(이미 타입 존재)가 단일 입력. 백엔드는 이 필드만 채우면 됨.
- **P1-a (voice-api)**: word.probability 캡처 + 직렬화 추가 (신호 적재). 유닛테스트. **GPU 안전윈도우 대기.**
- **P1-b (uncounted-api)**: transcript 엔드포인트에 probability 통과 + (선택) Supabase 컬럼.
- **P2**: 규칙③ PyKoSpacing 저장 훅.

### 별도 트랙 (혼동 방지)
- **답안지 diff 오역 적출**(replace/insert) = GT 있는 **검증세션 QA 전용**. 프로토타입 `_test_gtmark` 존재. 프로덕션 검수와 분리.
- 규칙②(Kiwi) 형태소 완화 매칭 = 평가엔진, 독립.
- 채널 프로브(ffprobe) 검증 → speaker-scope 락 (별도, P1 선행).

---

## §5 도메인 오역 핫워드 엔진 (B+D) — 설계 (승인 대기, 미구현)

> **목표**: 도메인 약어 오역(DLP→DAP)·문맥 혼동(수석님→선생님)을 **가드레일 갖춰** 교정. **family-safe**(가족통화 등 비도메인 통화 무영향)가 1차 안전속성.
> **상태 (2026-06-03)**: **코어 구현 완료 + 유닛테스트 20/20 green (CPU)**. env 기본 OFF → byte-identical (전체 596 collection 무파손 검증). **e2e eval만 GPU 안전윈도우 대기.**
> **구현물**: `app/hotword_engine/{profiles,guard,engine,__init__}.py` · `tests/test_hotword_engine.py` · config.py(env 3종) · stt_processor.py(B 프롬프트 + D 후처리 게이트 배선).

### 검증된 메커니즘 (실측 근거)
- **B — 발음 페어링 프롬프트**: `initial_prompt`에 `"보안 IT 용어: DLP(디엘피), NAC(엔에이씨), EPP(이피피), DRM(디알엠), OA망, 공동인증서, 예외정책, 팝업창."` 부착 → DLP→DAP 오역 교정(+약 5pp, 환각 미증가).
- **D — 문맥게이트 큐레이트 혼동쌍 후처리**: 일반 자모거리 아님. **큐레이트된** 혼동쌍(`{선생님→수석님}`)만, **문맥 prior(보안/IT 키워드 ≥2)** + **세션사전에 정답어 존재** 동시 충족 시에만 치환. 가족통화(키워드 0) → 무발동.

### 아키텍처: `app/hotword_engine/` (신규 패키지)
| 파일 | 책임 |
|---|---|
| `profiles.py` | `DomainProfile`(frozen dataclass): `phonetic_pairs`/`confusion_pairs`/`context_keywords`/`min_kw`. IT보안 프로파일 1종(데이터, immutable) |
| `guard.py` | **Token Guard** — 교정 안전규칙(아래) |
| `engine.py` | `build_domain_prompt()`[B] · `detect_domain()`[문맥게이트] · `correct_confusions()`[D] |
| `__init__.py` | 공개 API |
| `tests/test_hotword_engine.py` | 유닛테스트(CPU, GPU불요) |

### Token Guard (안전규칙 — 오교정 차단)
1. **최소길이**: 교정 대상 단어 ≥ 2글자 (`그`·`네` 등 단일/초단 금지).
2. **공통어 블록리스트**: `네/응/그/거/저/음...` 등 고빈도어는 교정 절대 금지.
3. **호칭 단독 금지**: 문맥(`detect_domain`) 미충족 시 호칭 혼동쌍 미발동.
4. **약어 allowlist**: B 발음페어링은 화이트리스트 약어에만.
5. **경계 가드(수정판)**: 좌경계 `(?<![가-힣])`만 — 우경계 `(?![가-힣])`는 **제거**(조사 `선생님이` 차단 버그 회피). `선생님이`→`수석님이` 정상 교정.
6. **세션사전 게이트**: 정답어가 세션 승인사전에 존재할 때만 치환(검수 승인 경로).

### 통합지점 (env-gate, 기본 OFF → byte-identical)
- **B**: `stt_processor.py:463-469` `asr_options["initial_prompt"]` 빌드 시 `build_domain_prompt(base, profile)`로 도메인 페어 부착. 모델 로드가 전역 1회라 **프롬프트는 워커 전역**(운영자가 통화믹스 보고 env로 결정, eval로 비도메인 무회귀 검증).
- **D**: `stt_processor.py:864` PII 마스킹/관계스냅샷 **직전**에 `correct_confusions(segments, profile, session_dict)` 패스. 관계탐지가 교정된 호칭을 보게 됨(이득).

### env (config.py, 전부 기본 OFF/보수)
- `HOTWORD_ENGINE_ENABLED`="false" (D 후처리 게이트)
- `HOTWORD_ENGINE_PROMPT_DOMAIN`="" (비면 B 프롬프트 OFF; 예 "it_security")
- 기본값에서 프롬프트 변경 0·후처리 0 → 현행과 동일.

### Eval 계획 (GPU 안전윈도우 — 분리)
- 4개 답안지 세션 전체를 `stt_score_policy.char_accuracy`로 before/after.
- **합격기준**: 도메인 통화 char_accuracy ↑, **비도메인(가족) 통화 무회귀**, D 치환은 문맥 내에서만 발동, 환각 미증가.

### 미구현/명시적 보류
- 세션 승인사전(`session_dict`) **인프라 미존재** → v1은 프로파일을 전역 데이터로, `session_dict`는 인터페이스만(후속 DB 배선, 코어 로직 무수정 플러그인).
- **chunked 모드(>1h, `_transcribe_chunked`)**: D 후처리 미배선(normal 모드만). 후속 — chunked는 청크별 세그먼트 emit 경로라 별도 훅 필요.
- 규칙②(Kiwi) 형태소 매칭과 독립.

### 회귀 비상탈출구 (승인 2026-06-03 — 비용 $0)
eval에서 **비도메인(가족) 통화 회귀율이 임계 초과** 시 → **2-Pass 재전사로 고치지 말 것**(디코딩 2배·레이턴시 폭등, ROI 최악). 대신 **워커 큐 격리**: `worker-it`(IT 프롬프트 ON, 도메인 전용) / `worker-general`(프롬프트 OFF, 일반 전용)로 **기동 env만 분리 배포** → VRAM·연산 증가 0, 완벽 격리. (백엔드 라우팅 단계 조치, 코어 코드 무수정)
