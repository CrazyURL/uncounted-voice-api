"""worker.py 모듈-로드 시점 환경변수 해석 helper.

worker.py 가 supabase env 를 import-시점에 요구하므로(L46 `os.environ["SUPABASE_URL"]`),
dev PC 처럼 그 env 가 없는 환경에서도 단위 테스트가 가능하도록 env 해석 로직만 별도 모듈로 둔다.
``sanitize_json.py`` 와 동일한 분리 전략(supabase/aiohttp/boto3 무의존).
"""

from __future__ import annotations

import os
from typing import Mapping, Optional


def resolve_voice_api_max_wait_sec(env: Optional[Mapping[str, str]] = None) -> int:
    """``VOICE_API_MAX_WAIT_SEC`` 환경변수를 읽어 worker 의 poll timeout 상수를 반환.

    의도:
      장통화(>5분) 처리 시 worker 가 voice-api 응답을 충분히 기다릴 수 있도록 timeout 을
      운영자가 env 로 조정할 수 있게 한다(코드 변경·재배포 없이 systemd EnvironmentFile
      교체만으로 적용 가능).

    의미론:
      - **미설정**: 기본 300(현 하드코딩과 동일, 회귀 없음).
      - **유효 정수 문자열**(예 ``"7200"``): ``int()`` 변환 결과 그대로 사용.
      - **invalid**(비-정수 문자열 / 빈 문자열 / 소수점 등): ``ValueError`` 가 import
        시점에 그대로 propagate → **fail-loud 시작 실패**(운영자가 즉시 알아차림).
        ``worker.py`` L42 ``WORKER_CONCURRENCY = int(os.getenv("WORKER_CONCURRENCY", "2"))``
        의 기존 컨벤션과 동일.
      - **0 / 음수**: 검증하지 않고 그대로 수용(operator 의 명시 선택; ``poll_job`` 의
        ``loop.time() < deadline`` 검사는 의미 그대로 동작 — 즉시 timeout).

    매개변수:
      ``env``: 테스트에서 ``os.environ`` 대신 dict 를 주입할 수 있도록 한 hook. None 이면
      프로세스 환경(``os.environ``) 을 사용.

    반환: int. 호출자(worker.py)는 모듈 로드 시 단 한 번 호출해 상수에 보관한다(live
      reload 의도 없음).
    """
    e = env if env is not None else os.environ
    return int(e.get("VOICE_API_MAX_WAIT_SEC", "300"))


# ─────────────────────────────────────────────────────────────────────────────
# Orphan cleanup env helpers (PR-1, default OFF, dry-run by default)
#
# 설계 정본: uncounted-root/scripts/analysis/orphan_cleanup_strategy_20260528.md
# 운영 게이트: WORKER_ORPHAN_CLEANUP_ENABLED 가 명시적으로 truthy 일 때만 cleanup 호출
# 진입(persist_results), 그리고 그 안에서도 WORKER_ORPHAN_CLEANUP_DRY_RUN 가 truthy
# 이면 DELETE 없이 로그만 emit. 두 env 모두 기본 OFF/DRY → 머지 자체는 운영 영향 0.
# ─────────────────────────────────────────────────────────────────────────────

_TRUE_STRS = frozenset({"true", "1", "yes", "on"})
_FALSE_STRS = frozenset({"false", "0", "no", "off"})


def _parse_bool_env(raw: Optional[str], default: bool, env_name: str) -> bool:
    """공통 bool env 파서.

    의미론(`WORKER_CONCURRENCY = int(os.getenv(...))` 의 fail-loud 컨벤션 동형):
      - 미설정 / 빈 문자열 → ``default``.
      - ``"true"`` / ``"1"`` / ``"yes"`` / ``"on"`` (case-insensitive) → True.
      - ``"false"`` / ``"0"`` / ``"no"`` / ``"off"`` → False.
      - 그 외 → ``ValueError`` 가 import 시점에 propagate(fail-loud).
    """
    if raw is None or raw == "":
        return default
    v = raw.strip().lower()
    if v in _TRUE_STRS:
        return True
    if v in _FALSE_STRS:
        return False
    raise ValueError(
        f"invalid bool env {env_name}={raw!r} "
        f"(허용: {sorted(_TRUE_STRS)} / {sorted(_FALSE_STRS)})"
    )


def resolve_orphan_cleanup_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """``WORKER_ORPHAN_CLEANUP_ENABLED`` — orphan cleanup 코드 경로 활성화 여부.

    의미론:
      - **미설정 / false 류**: ``False`` (기본). cleanup 함수 호출 자체가 일어나지 않음
        → 머지 직후 운영 영향 0.
      - **true 류**: ``True``. cleanup 호출 진입(단 실제 DELETE 여부는 dry_run flag 가 결정).
      - **invalid**: ``ValueError`` import 시점 fail-loud.

    모듈 로드 시점 1회 호출(live reload 없음). 변경 적용은 worker restart 필요.
    """
    e = env if env is not None else os.environ
    return _parse_bool_env(
        e.get("WORKER_ORPHAN_CLEANUP_ENABLED"),
        default=False,
        env_name="WORKER_ORPHAN_CLEANUP_ENABLED",
    )


def resolve_orphan_cleanup_dry_run(env: Optional[Mapping[str, str]] = None) -> bool:
    """``WORKER_ORPHAN_CLEANUP_DRY_RUN`` — 실 DELETE 여부 게이트.

    의미론:
      - **미설정 / true 류**: ``True`` (기본). DELETE 하지 않고 로그만 emit. 운영자가
        분포·후보 검토 후 명시적으로 false 로 전환해야 실 DELETE.
      - **false 류**: ``False``. 안전조건 통과 행 실제 DELETE.
      - **invalid**: ``ValueError`` import 시점 fail-loud.

    ``WORKER_ORPHAN_CLEANUP_ENABLED=false`` 이면 본 flag 는 아예 평가되지 않음(cleanup
    함수 미진입). 두 flag 모두 켜야 실 DELETE.
    """
    e = env if env is not None else os.environ
    return _parse_bool_env(
        e.get("WORKER_ORPHAN_CLEANUP_DRY_RUN"),
        default=True,
        env_name="WORKER_ORPHAN_CLEANUP_DRY_RUN",
    )


def resolve_orphan_cleanup_min_ratio(env: Optional[Mapping[str, str]] = None) -> float:
    """``WORKER_ORPHAN_CLEANUP_MIN_RATIO`` — 신규/기존 ratio sanity guard.

    의미론:
      - **미설정**: ``0.5`` (기본). N(신규) / N'(이전 = 신규 + 후보) < 0.5 면 cleanup skip.
      - **유효 float**: 그 값으로 비교.
      - **invalid**(비-숫자): ``ValueError`` import 시점 fail-loud.
      - 0.0 / 음수: 가드 disable 효과(모든 ratio 통과). operator 명시 선택, 위험 자기책임.
      - >1.0: 어떤 ratio 도 skip(cleanup 사실상 비활성), operator 명시 선택.

    의도: 갑작스러운 voice-api 빈 응답(N << N') 시 stale-기존 전부 삭제하는 사고 차단.
    """
    e = env if env is not None else os.environ
    return float(e.get("WORKER_ORPHAN_CLEANUP_MIN_RATIO", "0.5"))
