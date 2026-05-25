# PR-S2 설계 — 구조 PII auto_confirmed tier 정책 재검토

> 상태: **설계 문서**. 작성일 2026-05-25. **2026-05-26 C안으로 일부 정정됨(아래 갱신 참조).**
> 선행: PR-S1(정규식 정밀화 + overlap/substring dedup) **live 반영 완료**(merge `8ec434b`, new uvicorn live).
> 본 문서는 정책 판단만 다룬다. 코드 수정·재시작·DB write·`pii_candidates` 변경·prod 반영 없음.

---

## 갱신 (2026-05-26, C안) — IP tier 방향 정정

> 이 갱신이 본 문서 §0·§3 의 **"IP 영구 needs_human 강등"** 표현보다 우선한다.
> 해당 표현들은 **stale** 로 간주한다(아래 인라인 표기 참조).

**C안 결정 (2026-05-26):**
- **IP주소는 "최종적으로 auto_confirm 복귀 가능성"이 있는 타입이다.** 본 문서 작성 시점(2026-05-25)의
  "IP 영구 강등" 판단은 stale 이며, IP 를 계좌번호와 동일한 "영구 loose" 로 묶지 않는다.
- **단, 현재 live S2A 에서는 IP 를 `needs_human_decision` 으로 유지한다.** 지금 당장 IP 를
  auto_confirm 으로 되돌리지 않는다.
- **복귀 조건:** PR-S1 regex hardening 검증 통과 **+ 별도 S2 후속 검증/승인** 이후에만 IP 의
  auto_confirm 복귀가 가능하다. 그 전까지 IP auto_confirm 복귀 금지(가드).
- **계좌번호는 변동 없음** — 계속 `needs_human_decision` 유지(가장 loose, 영구 검수).
- **이번 세션 tier 코드 변경 없음.** 본 갱신은 문서 정정만이며 `pii_confidence.py` 등 tier 코드는
  건드리지 않았다. PR-S1 은 regex/masking 개선만 유지한다.
- **병합 가드:** `main` 에는 **PR-S1 단독 병합 금지.** S2A 라인(`fix/pii-pr-s2a`/`d14012b`)
  기준으로만 취급한다. PR-S1 만 단독으로 main 에 들어가면 S2A 의 IP→needs_human demotion 없이
  IP 가 auto_confirm 으로 되살아나므로 가드 위반이다.

**in-range IP/버전번호 애매성의 책임 분리 (PR-S1 vs tier):**
- `10.45.49.12` 같은 in-range 4-octet 문자열은 regex 만으로 IP/버전번호 구분 불가(동일 문자열).
  PR-S1(regex 층)은 이를 IP 로 **탐지/마스킹**한다 — 이것은 한계가 아니라 책임 분리다.
- in-range 오탐의 흡수는 **tier(needs_human) 트랙**이 담당한다. 회귀 계약은
  `tests/test_pii_pr_s1.py::TestIpVersionAmbiguityCriteria`(regex 탐지 경계) +
  `tests/test_pii_s2_tier_policy.py::test_review_required_types_needs_human`(IP→needs_human)
  로 고정돼 있다.

---

## 0. 한 줄 결론과 핵심 구분 (먼저 읽을 것)

**결론:** 느슨한 구조 PII 타입(**IP주소·계좌번호**)을 `auto_confirmed`에서 `needs_human_decision`으로 강등하여 사람 검수 큐를 거치게 한다. 단단한 타입(**주민/운전면허/여권/카드/이메일**)은 유지한다. 전화번호는 형식별 차등을 검토한다.

> ⚠️ **[2026-05-26 C안 정정]** 위 결론 중 **IP 의 "강등"은 영구 정책이 아니다(stale).** IP 는
> 현재 live 에서만 `needs_human` 으로 유지하며, regex hardening + S2 후속 승인 후 auto_confirm
> 복귀가 가능하다. **계좌번호 강등만 영구 유지.** 상단 「갱신 (2026-05-26, C안)」 절이 우선한다.

**가장 중요한 구분 (오해 방지):**

| 경로 | 코드 | tier 영향 | 담당 PR |
|------|------|-----------|---------|
| **마스킹 경로** (납품 transcript) | `mask_pii()` → `detect_pii_spans()` → 직접 치환 | **tier 안 봄.** 탐지된 PII는 전부 마스킹 | **PR-S1** (탐지 정밀화로 오탐↓ → 오마스킹 방지) |
| **후보 큐 경로** (admin 검수) | `detect-batch` → `score_candidates()` → `classify_tier()` | tier가 `auto_confirmed`면 사람 검수 **우회** | **PR-S2** (본 문서) |

> 코드 확인: `mask_pii`는 `pii_confidence`(tier 산정)를 **호출하지 않는다**. 따라서 **tier 정책을 바꿔도 납품 마스킹 동작은 변하지 않는다.** PR-S2의 가치는 "마스킹 품질"이 아니라 **검수 커버리지 / 데이터 거버넌스** — *어떤 PII가 사람 확인 없이 자동 확정되어 후보 데이터로 굳는가*의 문제다.
>
> 즉 PR-S1은 "오탐을 [IP주소]로 바꾸지 않게" 했고(완료), PR-S2는 "오탐 가능성이 있는 후보가 사람 눈을 거치지 않고 auto_confirmed로 굳지 않게" 한다.

---

## 1. 현재 tier 정책 (코드 사실관계)

출처: `app/pii_confidence.py`, `app/routers/pii.py`, `app/pii_masker.py` (PR-S1 반영본 `8ec434b`).

### 1.1 분류 함수 `classify_tier()`
```
auto_confirmed         : confidence ≥ 0.90 AND high_precision_pattern
auto_rejected          : confidence < 0.50 AND NOT high_precision (애매유형 제외)
needs_human_decision   : 그 외 전부 (애매유형은 항상 여기)
```

### 1.2 타입 분류와 기본 confidence
- `HIGH_PRECISION_TYPES` = { 주민등록번호, 운전면허번호, 여권번호, 카드번호, 이메일, **전화번호**, **계좌번호**, **IP주소** }
- `AMBIGUOUS_TYPES` = { 이름 }
- 기본 confidence: high_precision → **0.95**, ambiguous → 0.70, weak → 0.40
- 임계값: `_AUTO_CONFIRMED_MIN = 0.90`, `_NEEDS_HUMAN_MIN = 0.50`

### 1.3 귀결
- 위 8개 구조 타입은 전부 `high_precision`=True, confidence=0.95 → **모두 `auto_confirmed`** → admin 검수 큐 **우회**.
- 이름만 `needs_human_decision`.
- per-span hint(예: 음성 전사형 전화 `detect_spoken_phone_spans`)는 confidence/high_precision을 개별 지정 가능하나, 현재는 0.95/True로 동일하게 auto_confirmed.

### 1.4 위험 지점
`high_precision_pattern`이라는 이름은 "정규식이 구체적"이라는 뜻일 뿐, **오탐률이 낮다는 보장이 아니다.** 특히 IP·계좌는 정규식이 구조적이어도 컨텍스트 의존 오탐이 남는다(아래 §2). 그럼에도 일괄 auto_confirmed → 사람 확인 없이 후보 확정.

---

## 2. 타입별 위험도 평가

| 타입 | 패턴 강도 | 컨텍스트 오탐 위험 | 근거 | 위험 등급 |
|------|-----------|---------------------|------|-----------|
| 주민등록번호 | 매우 강함 (`6digit-[1-4]6digit`) | 매우 낮음 | 형식·구분자·성별자리 강제 | **낮음 (rigid)** |
| 이메일 | 매우 강함 (`local@domain.tld`) | 매우 낮음 | `@`+도메인 구조 | **낮음 (rigid)** |
| 운전면허번호 | 강함 (`12-34-567890-12`) | 낮음 | 4블록 구분자 | **낮음 (rigid)** |
| 카드번호 | 강함 (`4-4-4-4` 구분자) | 중간 | 구분자 강하나 **Luhn 미검증** → 임의 16자리 오탐 여지 | **중간** |
| 전화번호 | 형식 의존 | 형식별 상이 | 하이픈형(`010-1234-5678`) 명확 / 붙여쓰기 `01X+10~11자리`는 긴 숫자열 substring 위험(PR-S1 dedup으로 완화) | **중간** |
| 계좌번호 | **약함** (`\d{11,14}` 연속) | **높음** | 은행별 자리수 다양, 주문번호·송장·일반 긴 숫자열과 구분 불가. 실측: 010-시작 14자리가 계좌로 확정됨 | **높음 (loose)** |
| IP주소 | 중간 (octet 0~255, PR-S1) | **높음** | octet 검증 통과해도 버전/빌드번호 in-range 오탐. 실측: 기존 5건 전부 `in_range` | **높음 (loose)** |

핵심: **IP·계좌는 "정규식이 구체적"이지만 "의미가 모호"**하다. auto_confirmed로 사람 검수를 우회시키기엔 오탐 비용이 크다.

---

## 3. 권장 tier 정책

### 3.1 분류 재정의 (권장안)

| 그룹 | 타입 | tier |
|------|------|------|
| **Rigid (유지)** | 주민등록번호, 운전면허번호, 여권번호, 카드번호, 이메일 | `auto_confirmed` 유지 |
| **Loose (강등)** | **IP주소, 계좌번호** | **`needs_human_decision`로 강등** |
| **Conditional (검토)** | 전화번호 | 하이픈/표준형식 → `auto_confirmed`, 그 외(붙여쓰기·긴숫자열 인접) → `needs_human` 검토 |

### 3.2 010-시작 14자리 숫자열 (실측 핵심 케이스)
- PR-S1 dedup 결과: 전화(앞 11자리 substring) 탈락, **계좌(14자리)로 단일화**.
- §3.1에서 **계좌를 강등**하면 이 케이스는 자동으로 `needs_human_decision` → admin 검수 큐로 유입. **별도 특수 규칙 불필요** (계좌 강등으로 흡수됨).
- 권장: "010-prefix 14자리를 계좌로 auto_confirm" 금지는 **계좌 타입 전체 강등**으로 달성. (타입별 세분 규칙은 복잡도만 키움 → 지양.)

### 3.3 구현 방식 옵션 (코드는 PR-S2A에서)
- **옵션 A (권장):** `HIGH_PRECISION_TYPES`를 rigid 집합으로 축소하고, loose 타입은 별도 `REVIEW_REQUIRED_TYPES`로 분리해 `classify_tier` 입력 시 `high_precision=False`로 전달 → 자연히 `needs_human_decision`. 임계값/함수 시그니처 변경 없음, per-type 매핑만 추가.
- 옵션 B: 타입별 confidence 상수 차등(loose=0.70). 가능하나 "확신도"와 "검수 필요"를 뒤섞어 의미가 흐려짐 → 비권장.
- 어느 옵션이든 **`classify_tier`의 total-function 성격, 임계값(0.90/0.50)은 불변** 권장.

---

## 4. 기존 auto_confirmed 7건 — shape 기준 영향 분석

원문/matched_text/snippet 미사용. dev PC 추출 shape(offset/길이/type)만.

### 4.1 PR-S1 이후 현황 (7 → 5)
| 후보 | 세션 | shape | PR-S1 처리 | PR-S1 후 |
|------|------|-------|-----------|----------|
| IP a | e61debb9 | 5–19, octets=4, in_range | shift쌍 중 채택 | 잔존 |
| IP b | e61debb9 | 6–20 (1글자 시프트 중복) | **dedup 제거** | 제거 |
| IP c | (타발화) | octets=4, in_range | 비중첩 | 잔존 |
| IP d | (타발화) | octets=4, in_range | 비중첩 | 잔존 |
| IP e | (타발화) | octets=4, in_range | 비중첩 | 잔존 |
| 전화 | 18b98cca | 5–16, 11자리, 010 | **계좌에 흡수(substring 탈락)** | 제거 |
| 계좌 | 18b98cca | 5–19, 14자리, 010 | 단일 채택 | 잔존 |

→ PR-S1 후 잔존 **5건 = IP 4 + 계좌 1**.

### 4.2 PR-S2 권장 정책 적용 시 tier 재분류
| 잔존 후보 | 현재 tier | PR-S2 권장 tier |
|-----------|-----------|------------------|
| IP a, c, d, e (4건) | auto_confirmed | **needs_human_decision** |
| 계좌 1건 (010-14자리) | auto_confirmed | **needs_human_decision** |

→ **잔존 5건 전부 `needs_human_decision`로 강등** (auto_confirmed 0건). 즉 PR-S2 적용 시 기존 7건 중 자동 확정으로 남는 건 없고, 5건은 사람 검수 큐로 이동.

> 주의: 이는 **후보 큐**의 재분류일 뿐, 이 5건이 들어간 통화의 **마스킹 결과는 PR-S1 시점과 동일**하다(마스킹은 tier 무관).

---

## 5. API / Admin 영향 (voice-api 밖 — 확인 필요 항목)

본 repo(voice-api)에서 확정 불가. **PR-S2 착수 전 확인 필요:**
- [ ] `detect-batch` 응답 스키마는 이미 `confidence_tier`를 반환(변경 불필요). tier 분포만 바뀜.
- [ ] uncounted-api 후보 적재 로직이 `needs_human_decision` 구조 PII를 정상 큐잉하는가? (현재 이름만 큐에 오는 전제일 수 있음)
- [ ] admin 검수 UI가 `predicted_type`=이름 외(IP/계좌/전화 등) 구조 타입도 **표시/배지**할 수 있는가?
- [ ] 구조 PII 후보의 검수 문구("맞음/아님/보류")가 이름용 문구와 동일해도 적절한가? (offset만 있고 원문 미표시인 점 고려)
- [ ] 강등으로 admin 큐 유입량이 급증할 수 있음 → 검수 부담/우선순위 정책 필요.

---

## 6. 구현 PR 분리안

| PR | 범위 | 위치 | 비고 |
|----|------|------|------|
| **PR-S2A** | tier 정책 변경(§3.3 옵션 A): rigid/loose 분리, IP·계좌 강등 | voice-api `app/pii_confidence.py` (+테스트) | mask_pii 무영향 회귀 포함 |
| **PR-S2B** | 후보 적재/큐잉이 구조 타입 needs_human을 수용하는지 | uncounted-api (dev PC) | 확인 우선, 필요 시 수정 |
| **PR-S2C** | admin UI 구조 PII 문구/배지 보강 | admin (별도 repo) | S2B 결과 의존 |
| **PR-S2D** | 기존 auto_confirmed 후보 재평가/정리 | DB (dev PC, read→판단) | DB write는 별도 승인 |

- 의존: S2A(정책) → S2B(수용) → S2C(표시) → S2D(소급 정리).
- S2A는 voice-api 단독으로 안전 착수 가능(마스킹 무영향). S2B~D는 voice-api 밖.

---

## 7. 검증 계획 (PR-S2A 구현 시)

1. **합성 케이스** (detect-batch tier 단언):
   - 주민/카드/이메일 → `auto_confirmed` 유지
   - IP(`192.168.0.1`) → `needs_human_decision`
   - 계좌(14자리, 010-시작 포함) → `needs_human_decision`
   - 전화(하이픈형) → 정책 결정대로
2. **기존 7건 shape 기준**: 잔존 5건 전부 `needs_human_decision` 되는지(§4.2), count만.
3. **mask_pii 회귀 (불변 확인 — 최重要)**: 동일 입력에 대해 PR-S1과 `masked_text`/`pii_detected`/`total_masked` **완전 동일**. (tier 변경이 마스킹에 새지 않음을 증명.)
4. **detect-batch tier 분포 변화**: auto_confirmed↓ / needs_human↑ 방향 확인.
5. **admin 큐 유입 수 변화**(S2B/C 이후): 구조 PII 후보가 큐에 표시되는지.

---

## 8. 범위 / 금지 (PR-S2 설계 단계)

- 본 문서는 **설계만**. 코드 미수정.
- 금지: voice-api 코드 수정, tier 즉시 변경, uvicorn 재시작, DB write, `pii_candidates` 수정/삭제, prod 반영, 원문/matched_text/snippet 출력.
- 구현 착수(PR-S2A~) 여부는 본 설계 승인 후 별도 판단.

---

## 부록 A. 결정 포인트 (승인 필요 사항)

1. IP주소 강등에 동의하는가? (정상 IP도 검수 큐로 → 검수 부담 ↔ 오탐 자동확정 방지)
2. 계좌번호 강등에 동의하는가? (loose 타입, 010-14자리 문제 흡수)
3. 전화번호는 (a) 전부 유지 / (b) 하이픈형만 유지 / (c) 전부 강등 중 무엇인가?
4. 카드번호 Luhn 체크 추가를 PR-S2 범위에 넣을 것인가, 별도로 둘 것인가?
5. 기존 7건 소급 재평가(S2D)를 정책 반영 후 진행할 것인가?
