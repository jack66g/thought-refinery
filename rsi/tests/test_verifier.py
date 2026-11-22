# -*- coding: utf-8 -*-
import asyncio

from rsi.core.verifier import extract_answer, hard_verify, math_equal


def test_extract_answer():
    assert extract_answer("一步步推导。答案是 3.14。") == "3.14"
    assert extract_answer("最终答案：x=7") == "x=7"
    assert extract_answer("没有答案标记") == ""


def test_math_equal():
    assert math_equal("42", "42")
    assert math_equal("42。", "$42$")
    assert math_equal("2*21", "84/2")
    assert math_equal("x=7", "7") is False
    assert math_equal("1/3", "0.3333333333")
    assert math_equal("5", "6") is False


def test_hard_verify_math():
    ok = asyncio.run(hard_verify("答案是 50", "50", "math_expression"))
    assert ok is True
    ok2 = asyncio.run(hard_verify("答案是 51", "50", "math_expression"))
    assert ok2 is False


def test_hard_verify_short_text():
    assert asyncio.run(hard_verify("答案是 秋天", "秋天", "short_text"))
    assert not asyncio.run(hard_verify("答案是 冬天", "秋天", "short_text"))
