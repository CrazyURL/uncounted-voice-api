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
