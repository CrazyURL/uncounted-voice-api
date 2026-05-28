"""NaN/Inf sanitize at the persist boundary.

worker.py 가 supabase 로 row dict 를 보내기 직전, NaN/+Inf/-Inf 가 페이로드에 섞이지 않도록
재귀적으로 None 으로 치환한다. Python ``json.dumps`` 의 기본 ``allow_nan=True`` 는 NaN/Infinity
를 비표준 JSON 리터럴로 출력하지만, PostgreSQL JSONB 입력은 이를 거부하므로 supabase REST →
PostgREST → JSONB 경로의 어느 지점에서도 NaN 행은 reject 된다.

순수 함수 — 외부 의존(supabase env / aiohttp / boto3) 없음. ``tests/test_worker_nan_sanitize.py``
에서 직접 import 한다(worker.py 는 모듈 로드 시 supabase env 를 요구하므로 함께 import 불가).

알려진 한계:
    ``np.float32`` 는 ``float`` 의 subclass 가 아니므로 본 helper 가 감지하지 못한다. WhisperX
    pipeline 의 post-processing 이 Python ``float`` 로 좁혀 emit 하는 것이 현재 계약이며, 모델
    직출 numpy 스칼라가 누설되는 케이스는 별도 추적(검출 시 voice-api 발신지 가드).
"""

from __future__ import annotations

import math
from typing import Any


def sanitize_json_safe(obj: Any) -> Any:
    """Return a copy of ``obj`` with NaN/+Inf/-Inf scalars replaced by ``None``.

    - dict: 재귀 walk(키 보존, 값만 sanitize).
    - list / tuple: 재귀 walk(시퀀스 타입은 list 로 정규화).
    - float: ``math.isfinite`` 가 False (NaN/+Inf/-Inf) 면 ``None``, 정상 float 은 그대로.
    - bool: float 의 subclass 가 아니므로 그대로 통과(int subclass 가드는 불필요).
    - int / str / None / 기타: 그대로 반환.

    원본은 mutate 하지 않는다(call-site 의 row 변수가 후속 로깅·반환에서 재사용될 수 있음).
    """
    if isinstance(obj, dict):
        return {k: sanitize_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_json_safe(item) for item in obj]
    # bool 는 int 의 subclass — float 검사 전에 분기 우선순위 무관.
    if isinstance(obj, float):
        # math.isfinite: NaN/+Inf/-Inf 모두 False → 한 줄로 세 케이스 커버.
        return obj if math.isfinite(obj) else None
    return obj
