# -*- coding: utf-8 -*-
import pytest

from rsi.connectors.adapters import normalize

ZCODE = {
    "messages": [
        {"role": "user", "content": "帮我推导 2x+3=17"},
        {"role": "assistant", "reasoning_content": "移项得2x=14，除以2得x=7，验证成立。",
         "content": "x=7"},
        {"role": "user", "content": "验证一下"},
        {"role": "assistant", "reasoning_content": "代回2*7+3=17，成立。",
         "content": "验证通过，x=7"},
    ]
}

DEEPSEEK = {
    "task": "优化脚本启动速度",
    "reasoning": "先测量启动耗时，发现导入慢，改成懒加载后从3秒降到0.5秒。",
    "result": "懒加载方案有效",
}

OPENCODE = {
    "messages": [
        {"role": "user", "content": "这段代码为什么报空指针"},
        {"role": "assistant",
         "content": "<think>变量在分支内未初始化，另一分支使用时为空，需先判空。</think>原因是分支未初始化。"},
    ]
}


def test_zcode_messages_shape():
    ev = normalize("zcode", ZCODE)
    assert ev["task"] == "帮我推导 2x+3=17"
    assert "移项" in ev["thinking"] and "代回" in ev["thinking"]
    assert ev["conclusion"] == "验证通过，x=7"
    assert ev["event_kind"] == "host_chat"
    assert ev["context"]["host"] == "zcode"
    # V3：transcript 透传（role+content，think 块剥离）
    assert len(ev["transcript"]) == 4
    assert ev["transcript"][0]["content"] == "帮我推导 2x+3=17"


def test_deepseek_task_shape():
    ev = normalize("deepseek-harness", DEEPSEEK)
    assert ev["task"] == "优化脚本启动速度"
    assert "懒加载" in ev["thinking"]
    assert ev["conclusion"] == "懒加载方案有效"


def test_opencode_think_block():
    ev = normalize("opencode", OPENCODE)
    assert "未初始化" in ev["thinking"]
    assert ev["conclusion"] == "原因是分支未初始化。"


def test_generic_passthrough():
    ev = normalize("generic", {"task": "任务", "thinking": "思考" * 30, "conclusion": "结论"})
    assert ev["source"] == "generic"


def test_missing_thinking_rejected():
    # V3：纯聊天（有对话轮、无思考）放行——CoT 线丢弃，预训练/风格/画像线消费
    ev = normalize("zcode", {"messages": [{"role": "user", "content": "hi"},
                                          {"role": "assistant", "content": "ok"}]})
    assert ev["thinking"] == "" and len(ev["transcript"]) == 2
    # 思考与对话轮全空 → 拒收
    with pytest.raises(ValueError):
        normalize("generic", {"task": "只有任务"})
    with pytest.raises(ValueError):
        normalize("zcode", {"messages": []})
