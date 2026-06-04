# -*- coding: utf-8 -*-
"""B+D 핫워드 엔진 유닛테스트 (CPU, GPU 불요).

설계: docs/design_review_panel_redesign_20260603.md §5.
핵심 회귀잠금: family-safe(문맥 미충족 무변경), Token Guard, 조사 경계, immutability.
"""
from app.hotword_engine import (
    build_domain_prompt,
    correct_confusions,
    detect_domain,
    get_profile,
)
from app.hotword_engine.guard import is_correctable
from app.hotword_engine.profiles import IT_SECURITY, DomainProfile

IT = IT_SECURITY


# ── B: 발음페어링 프롬프트 ──
class TestBuildDomainPrompt:
    def test_profile_none_returns_base_unchanged(self):
        assert build_domain_prompt("기본프롬프트", None) == "기본프롬프트"
        assert build_domain_prompt(None, None) is None

    def test_appends_phonetic_pairs(self):
        out = build_domain_prompt("기본.", IT)
        assert "DLP(디엘피)" in out
        assert "NAC(엔에이씨)" in out
        assert out.startswith("기본.")

    def test_empty_pronunciation_renders_term_only(self):
        out = build_domain_prompt("", IT)
        assert "공동인증서" in out
        assert "공동인증서()" not in out  # 빈 발음은 괄호 없이

    def test_idempotent_no_double_append(self):
        once = build_domain_prompt("기본.", IT)
        twice = build_domain_prompt(once, IT)
        assert once == twice


# ── 문맥게이트 ──
class TestDetectDomain:
    def test_business_call_detected(self):
        text = "DLP 팝업창 뜨는데 공동인증서 정책 때문에"  # 키워드 다수
        assert detect_domain(text, IT) is True

    def test_family_call_not_detected(self):
        text = "엄마 오늘 저녁에 뭐 먹어 날씨도 추운데 선생님이 그랬어"
        assert detect_domain(text, IT) is False

    def test_profile_none_false(self):
        assert detect_domain("DLP 보안 정책", None) is False

    def test_below_threshold_false(self):
        # 키워드 1개(보안)만 → min_keywords=2 미달
        assert detect_domain("보안 얘기 잠깐 했어", IT) is False


# ── D: 혼동쌍 교정 (family-safe 핵심) ──
class TestCorrectConfusions:
    def _segs(self, *texts):
        return [{"text": t, "speaker": "SPEAKER_00"} for t in texts]

    def test_business_context_corrects(self):
        segs = self._segs("DLP 정책 관련해서 선생님 한가지 여쭤볼게요 공동인증서")
        out, n = correct_confusions(segs, IT)
        assert n == 1
        assert "수석님" in out[0]["text"]
        assert "선생님" not in out[0]["text"]

    def test_family_context_does_not_correct(self):
        # 비도메인: 선생님(학교) 그대로 보존 — family-safe
        segs = self._segs("우리 애 선생님이 오늘 전화 왔어 저녁 뭐먹지")
        out, n = correct_confusions(segs, IT)
        assert n == 0
        assert out[0]["text"] == segs[0]["text"]

    def test_particle_boundary_preserved(self):
        # 조사 붙은 '선생님이' → '수석님이' (우경계 제거 회귀잠금)
        segs = self._segs("DLP 보안 정책 선생님이 공동인증서 결재 말씀하셨어요")
        out, n = correct_confusions(segs, IT)
        assert n == 1
        assert "수석님이" in out[0]["text"]

    def test_corrects_word_level_for_utterance_rebuild(self):
        # utterance transcript_text 는 seg["words"] 에서 재구성되므로 워드도 교정돼야 한다
        # (회귀잠금: 텍스트만 고치면 발화 persist 에 반영 안 됨 — 01dd38b9 실증)
        segs = [{
            "text": "DLP 보안 공동인증서 정책 아 선생님",
            "words": [{"word": "아", "start": 0, "end": 1}, {"word": "선생님", "start": 1, "end": 2}],
        }]
        out, n = correct_confusions(segs, IT)
        assert n >= 1
        joined = " ".join(w["word"] for w in out[0]["words"])
        assert "수석님" in joined
        assert "선생님" not in joined

    def test_session_dict_gate_blocks_when_absent(self):
        segs = self._segs("DLP 보안 공동인증서 정책 선생님 결재")
        # 세션사전에 정답어(수석님) 없음 → 미발동
        out, n = correct_confusions(segs, IT, session_dict=frozenset({"DLP"}))
        assert n == 0
        assert "선생님" in out[0]["text"]

    def test_session_dict_gate_allows_when_present(self):
        segs = self._segs("DLP 보안 공동인증서 정책 선생님 결재")
        out, n = correct_confusions(segs, IT, session_dict=frozenset({"수석님"}))
        assert n == 1
        assert "수석님" in out[0]["text"]

    def test_immutability_input_unchanged(self):
        segs = self._segs("DLP 보안 공동인증서 정책 선생님 결재")
        original = segs[0]["text"]
        correct_confusions(segs, IT)
        assert segs[0]["text"] == original  # 입력 미변형

    def test_profile_none_noop(self):
        segs = self._segs("아무 텍스트")
        out, n = correct_confusions(segs, None)
        assert n == 0
        assert out[0]["text"] == "아무 텍스트"


# ── Token Guard ──
class TestTokenGuard:
    def test_rejects_short(self):
        assert is_correctable("그") is False
        assert is_correctable("거") is False

    def test_rejects_common_words(self):
        assert is_correctable("네") is False
        assert is_correctable("그래서") is False

    def test_allows_safe_honorific(self):
        assert is_correctable("선생님") is True
        assert is_correctable("공동인증서") is True

    def test_guard_blocks_unsafe_pair_in_engine(self):
        # 혼동쌍에 위험한(단일/공통) wrong이 있으면 엔진이 컷
        unsafe = DomainProfile(
            name="x",
            phonetic_pairs=(),
            confusion_pairs=(("그", "수석님"),),  # 단일어 → 차단돼야
            context_keywords=("보안", "DLP", "정책"),
            min_keywords=2,
        )
        segs = [{"text": "보안 DLP 정책 그 어쩌고"}]
        out, n = correct_confusions(segs, unsafe)
        assert n == 0


def test_get_profile():
    assert get_profile("it_security") is IT_SECURITY
    assert get_profile("") is None
    assert get_profile(None) is None
    assert get_profile("nonexistent") is None
