import numpy as np
from app.pii_masker import detect_pii_spans

def find_pii_word_ranges(
    segments: list[dict], enable_name_masking: bool = False, pad_sec: float = 0.15
) -> list[tuple[float, float, str]]:
    """STT 세그먼트의 텍스트 PII를 오디오 시간 범위로 매핑한다.
    
    Returns:
        list of (start_sec, end_sec, pii_type)
    """
    pii_time_ranges = []

    for seg in segments:
        text = seg.get("text", "")
        words = seg.get("words", [])
        if not text or not words:
            continue

        # 텍스트에서 PII 스팬 감지
        spans = detect_pii_spans(text, enable_name_masking)
        if not spans:
            continue

        # 각 단어의 텍스트 내 시작/끝 위치(offset) 계산
        current_offset = 0
        word_offsets = []
        text_lower = text.lower()
        for w in words:
            w_text = w.get("word", "")
            if not w_text:
                word_offsets.append({"start": -1, "end": -1, "word": w})
                continue
            start_idx = text_lower.find(w_text.lower(), current_offset)
            if start_idx == -1:
                # 못 찾으면 offset 오염 방지 — 이 word는 매핑 제외, current_offset 유지
                word_offsets.append({"start": -1, "end": -1, "word": w})
                continue
            end_idx = start_idx + len(w_text)
            word_offsets.append({
                "start": start_idx,
                "end": end_idx,
                "word": w
            })
            current_offset = end_idx

        for span in spans:
            s_start = span["char_start"]
            s_end = span["char_end"]

            # span에 걸치는 word 찾기 (매핑 실패한 word는 제외)
            matching_words = [
                wo["word"] for wo in word_offsets
                if wo["start"] >= 0 and not (wo["end"] <= s_start or wo["start"] >= s_end)
            ]
            
            # word들 중 start/end 시간이 있는 것들만 필터링
            timed_words = [w for w in matching_words if "start" in w and "end" in w]
            
            if timed_words:
                # 걸치는 word들의 min start, max end
                t_start = min(w["start"] for w in timed_words)
                t_end = max(w["end"] for w in timed_words)

                # 패딩 적용
                t_start = max(0.0, t_start - pad_sec)
                t_end = t_end + pad_sec

                pii_time_ranges.append((t_start, t_end, span["type"]))

    # PR-B2: 같은 (start_sec, end_sec, pii_type) 튜플 중복 제거.
    # 기존 PII_PATTERNS 와 PR-B extended 룰이 동일 char 영역을 같은 type 으로 두 번
    # emit 하거나, 같은 word 가 여러 span 에 걸쳐 중복 매핑되는 경우 idempotent 보장.
    # 다른 type 의 시간 overlap (예: 이름 ∩ korean_name_like_candidate) 은 정보적
    # 중복 아님 — type 별로 별도 emit 유지 (build_pii_intervals 가 piiType 별 행 emit).
    # 안정 순서 유지: 첫 등장 순서대로.
    seen: set[tuple[float, float, str]] = set()
    deduped: list[tuple[float, float, str]] = []
    for r in pii_time_ranges:
        key = (round(r[0], 4), round(r[1], 4), r[2])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped

def mask_audio_ranges(
    audio: np.ndarray, ranges: list[tuple[float, float, str]], sr: int, method: str = "beep"
) -> np.ndarray:
    """오디오의 지정된 시간 범위를 마스킹(비프음 등) 처리한다."""
    if not ranges:
        return audio
        
    masked_audio = audio.copy()
    fade_len = int(0.01 * sr) # 10ms fade
    
    for start_t, end_t, _ in ranges:
        start_idx = int(start_t * sr)
        end_idx = int(end_t * sr)
        
        # 경계 제한
        start_idx = max(0, min(start_idx, len(audio) - 1))
        end_idx = max(0, min(end_idx, len(audio)))
        
        if start_idx >= end_idx:
            continue
            
        if method == "beep":
            num_samples = end_idx - start_idx
            # 1kHz sine wave (진폭 0.1)
            t = np.arange(num_samples) / sr
            beep = 0.1 * np.sin(2 * np.pi * 1000 * t)
            
            # Fade in/out 적용
            if num_samples > 2 * fade_len:
                fade_in = np.linspace(0, 1, fade_len)
                fade_out = np.linspace(1, 0, fade_len)
                beep[:fade_len] *= fade_in
                beep[-fade_len:] *= fade_out
            elif num_samples > 0:
                # 너무 짧으면 전체적으로 fade in-out (삼각형 모양)
                mid = num_samples // 2
                if mid > 0:
                    beep[:mid] *= np.linspace(0, 1, mid)
                    beep[mid:] *= np.linspace(1, 0, num_samples - mid)
                else:
                    beep[:] = 0 # 너무 짧으면 그냥 0

            masked_audio[start_idx:end_idx] = beep
            
    return masked_audio
