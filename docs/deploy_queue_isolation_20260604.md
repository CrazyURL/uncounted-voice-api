# B 프롬프트 도메인 격리 (Queue Isolation) — 배포 정본

> **Status:** v0(env 격리) 즉시 적용 가능 / v1(세션 태깅) 설계
> **Date:** 2026-06-04
> **Scope:** uncounted-voice-api (worker + voice-api)
> **목적:** B(IT 약어 발음페어링 프롬프트)의 비IT 통화 WER 회귀 리스크를 **아키텍처적으로 0%** 로 만든다.

---

## 0. ⚠️ 아키텍처 실측 (Celery 아님)

원안(Celery `-Q queue_it`, `session.domain`)은 우리 스택과 불일치. 실측 확정:
- 워커 = **Supabase 폴링**(`gpu_upload_status=pending` → claim → voice-api 호출). Celery/RabbitMQ **없음**.
- sessions에 **pre-STT 도메인 필드 없음**(`session_topic_summary`는 post-STT, null).
- 프롬프트는 `load_models()`에서 **전역 1회 베이크**(`stt_processor.py:591`), transcribe 콜별 주입 아님.
- 도메인은 transcript에서 탐지(post-STT) → **프롬프트(pre-STT)를 탐지결과로 게이트 불가**(닭-달걀).

→ **세션별 B 격리 = "STT 전 도메인 신호" 필요(현재 부재).** 따라서 단계 분리.

## 1. D(혼동쌍 교정)는 이미 family-safe — 격리 불요
`correct_confusions`는 `detect_domain`(transcript 키워드 ≥2, post-STT)로 게이트.
- 비IT 통화(병원/HDMI/가족) → detect_domain=False → **D 무발동**(실측: ba059bf0/d0212e24 False).
- 병원 "의사 선생님" → 수석님 오염 **구조적 차단**.
→ `HOTWORD_ENGINE_ENABLED=true` **항상 ON 무방.**

## 2. B(프롬프트) 격리

### v0 — env 워커-런 격리 (0 인프라, 지금)
B는 pre-STT 전역이라 **세션별 불가**, **워커-런 단위**로만 분리:

| 워커 용도 | env | B |
|---|---|---|
| 혼합/일반/라이브 | `HOTWORD_ENGINE_PROMPT_DOMAIN=""` | **OFF** → 비IT WER 회귀 0% |
| IT 전용 배치 재처리 | `HOTWORD_ENGINE_PROMPT_DOMAIN=it_security` | ON → DLP 교정 |

운영 규칙: **혼합 트래픽 워커는 B OFF 고정.** 알려진 IT 배치를 재처리할 때만 env ON으로 띄워 처리 후 OFF 복귀. (단일 워커라 세션별 아님 = 한계.)

적용: `.env.dev`의 `HOTWORD_ENGINE_PROMPT_DOMAIN` 값만 토글 + `sudo systemctl restart voice-api@dev.service`. 코드 변경 0(기존 env 게이트 재사용).

### v1 — 진짜 세션격리 (도메인 태깅 선행)
대표님이 그린 라우팅 분리. 선행조건 = **sessions.domain 태그**(STT 전):
1. **도메인 신호 소스 결정**: (a) 클라이언트/번호 매핑 (b) 수동 태그 (c) 경량 분류기(통화 메타/첫 발화).
2. **2-uvicorn**: `voice-api-it`(B ON, :8001) / `voice-api-general`(B OFF, :8002) systemd 분리 기동.
3. **워커 라우팅**: `submit_to_voice_api`가 `session.domain`으로 endpoint 분기(`VOICE_API_URL` 선택).
4. 결과: 비IT 세션은 B 없는 워커로만 → **회귀확률 0% 보장 + 라이브 IT도 B 혜택**.

→ v1은 도메인 태깅 인프라(크로스 컴포넌트)가 핵심. 그 신호가 생기면 코드는 endpoint 분기 몇 줄.

## 3. 무회귀 검증 근거 (2026-06-04 CPU 실측)
- D: 비IT detect_domain=False, 교정 0 ✅
- 과잉마스킹(NER): 비IT 0 + DB전역 FP 0 ✅
- 토큰 오버헤드: B +~37토큰/청크 1회성 ✅
- ΔWER(B): 비IT 오디오 미보유로 미측정 → **v0(B OFF) 채택 시 측정 불요**(회귀 0 보장).

## 4. 권장 집행
1. **혼합/라이브 워커: B OFF**(`HOTWORD_ENGINE_PROMPT_DOMAIN=""`), D ON 유지 → 비IT WER 0% 즉시 달성.
2. IT 배치는 env ON 임시 기동으로 처리.
3. 라이브 IT 자동 B 혜택이 필요해지면 → v1(도메인 태깅) 트랙 착수.
