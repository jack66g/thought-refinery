# -*- coding: utf-8 -*-
from rsi.core.cleaner import (health_check, language_mix_ratio, looks_truncated,
                              mask_pii, ngram_repeat_ratio, strip_think_tags)

CFG = {"ngram_n": 10, "max_repeat_ratio": 0.3, "minor_lang_ratio": 0.35,
       "max_total_chars": 120000}


def test_strip_think_tags():
    assert strip_think_tags("<think>abc</think>") == "abc"
    assert strip_think_tags("no tags") == "no tags"


def test_repeat_ratio():
    text = "这是一段完全相同的句子。这是一段完全相同的句子。这是一段完全相同的句子。"
    assert ngram_repeat_ratio(text, 10) > 0.3
    unique = "先设未知数 x 等于三。再代入第二个方程。解得 y 等于七。所以答案是十七。"
    assert ngram_repeat_ratio(unique, 10) < 0.1


def test_lang_mix():
    assert language_mix_ratio("纯中文文本没有英文") == 0.0
    assert language_mix_ratio("pure english text only") == 0.0
    mixed = "这段中文 text 是混排 with english words 很多的 words 场景"
    assert 0 < language_mix_ratio(mixed) < 1


def test_truncated():
    assert looks_truncated("推导到一半，最终的结果是") is True  # 悬空在"是"
    assert looks_truncated("答案是 42") is False
    assert looks_truncated("括号没闭合（差一半") is True
    assert looks_truncated("完整句子。") is False


def test_pii_mask():
    masked = mask_pii("联系 test@example.com 或 13812345678")
    assert "test@example.com" not in masked
    assert "13812345678" not in masked
    assert "已脱敏" in masked


def test_health_check_drop_short_thinking():
    rep = health_check("任务", "太短", "", CFG)
    assert not rep.ok
    assert any("思考过短" in r for r in rep.drop_reasons)


def test_health_check_ok():
    thinking = ("先设未知数 x，根据题意列出方程 2x+3=17。"
                "移项得到 2x=14，两边除以二。"
                "解得 x=7，代回原方程验证：左边等于十七，右边也是十七，成立。"
                "所以答案是 17。")
    rep = health_check("解方程 2x+3=17", thinking, "x=7", CFG)
    assert rep.ok, rep.drop_reasons
