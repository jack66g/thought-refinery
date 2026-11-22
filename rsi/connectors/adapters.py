"""宿主适配器：zcode / deepseek-harness / opencode / generic → NormalizedEvent。

各宿主轨迹格式不同，这里统一吃多种键名拼写；解析不了就报错（400 给宿主），
绝不静默丢字段。
"""
from __future__ import annotations

import re
from typing import Any

from ..core.pipeline import new_event

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def _first(values: list[Any], *keys: str) -> Any:
    for item in values:
        if isinstance(item, dict):
            for key in keys:
                if item.get(key):
                    return item[key]
    return None


def _from_messages(payload: dict[str, Any]) -> tuple[str, str, str]:
    messages = payload.get("messages") or payload.get("session") or payload.get("turns") or []
    if not isinstance(messages, list):
        raise ValueError("messages/session/turns 必须是数组")
    users = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"]
    assistants = [m for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]
    task = payload.get("task") or (_first(users, "content") if users else None)
    if not task:
        raise ValueError("找不到任务文本（需要首个 user 消息或 task 字段）")
    thinking_parts: list[str] = []
    for a in assistants:
        for key in ("reasoning_content", "reasoning", "thinking", "think"):
            if a.get(key):
                thinking_parts.append(str(a[key]))
                break
        else:
            content = str(a.get("content", ""))
            think_blocks = _THINK_RE.findall(content)
            if think_blocks:
                thinking_parts.extend(t.strip() for t in think_blocks)
    thinking = payload.get("thinking") or "\n".join(p for p in thinking_parts if p.strip())
    conclusion = payload.get("conclusion") or payload.get("result") or ""
    if not conclusion and assistants:
        conclusion = str(assistants[-1].get("content", ""))
    conclusion = _THINK_RE.sub("", str(conclusion)).strip()
    return str(task), str(thinking), conclusion


def _from_task_shape(payload: dict[str, Any]) -> tuple[str, str, str]:
    task = payload.get("task") or payload.get("prompt") or payload.get("question")
    thinking = (payload.get("thinking") or payload.get("reasoning")
                or payload.get("reasoning_trace") or payload.get("cot") or "")
    conclusion = (payload.get("conclusion") or payload.get("answer")
                  or payload.get("result") or payload.get("final") or "")
    if not task:
        raise ValueError("缺少 task/prompt/question 字段")
    return str(task), str(thinking), str(conclusion)


def normalize(source: str, payload: dict[str, Any]) -> dict[str, Any]:
    """source ∈ {zcode, deepseek-harness, opencode, generic}；返回 NormalizedEvent。

    V3 起纯聊天事件（无 AI 思考轨迹）也放行：CoT 线会自然体检丢弃，
    但预训练/风格/画像三条线依然有价值。思考与对话轮都为空才拒收。
    """
    if not isinstance(payload, dict):
        raise ValueError("payload 必须是对象")
    has_conv = any(k in payload for k in ("messages", "session", "turns"))
    task, thinking, conclusion = _from_messages(payload) if has_conv else _from_task_shape(payload)
    transcript: list[dict[str, str]] = []
    if has_conv:
        raw = payload.get("messages") or payload.get("session") or payload.get("turns") or []
        transcript = [{"role": m.get("role", "user"),
                       "content": _THINK_RE.sub("", str(m.get("content", ""))).strip()}
                      for m in raw if isinstance(m, dict) and m.get("content")]
    if not str(thinking).strip() and not transcript:
        raise ValueError("找不到思考轨迹且无对话轮（reasoning/thinking/<think>/messages 均为空）")
    kind = "host_chat" if source in ("zcode",) and has_conv else "host_task"
    return new_event(source, task, thinking, conclusion, event_kind=kind,
                     transcript=transcript, context={"host": source})
