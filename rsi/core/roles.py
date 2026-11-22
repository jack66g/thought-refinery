"""六个 LLM 角色的薄封装：加载种子提示词 → 调用 → 解析。"""
from __future__ import annotations

from typing import Any

from ..llm.client import LLMClient
from . import prompts

CATEGORIES = ("math", "chinese", "code", "reasoning", "misc")


async def classify(llm: LLMClient, task: str, thinking: str) -> dict[str, Any]:
    system, user = prompts.render("classifier", task=task[:300],
                                  thinking=thinking[:600])
    data = await llm.chat_json(system, user)
    cat = data.get("category", "misc")
    if cat not in CATEGORIES:
        cat = "misc"
    data["category"] = cat
    data.setdefault("tags", [])
    data.setdefault("confidence", 0.5)
    return data


async def value_gate(llm: LLMClient, task: str, thinking: str,
                     conclusion: str) -> dict[str, Any]:
    system, user = prompts.render("value_gate", task=task[:400],
                                  thinking=thinking[:2500], conclusion=conclusion[:400])
    data = await llm.chat_json(system, user)
    data.setdefault("total", 0.0)
    data.setdefault("verdict", "drop")
    return data


async def audit(llm: LLMClient, task: str, conclusion: str,
                segments: list[str]) -> dict[str, Any]:
    from .segmenter import format_segments
    system, user = prompts.render("auditor", task=task[:400],
                                  conclusion=conclusion[:300],
                                  segments=format_segments(segments))
    data = await llm.chat_json(system, user)
    data.setdefault("verdict", "clean")
    data.setdefault("first_bad_step", None)
    data.setdefault("error_type", None)
    return data


async def repair(llm: LLMClient, task: str, prev_context: str, bad_segment: str,
                 next_context: str, conclusion: str) -> str:
    system, user = prompts.render("repairer", task=task[:400],
                                  prev_context=prev_context[-300:],
                                  bad_segment=bad_segment,
                                  next_context=next_context[:300],
                                  conclusion=conclusion[:200])
    out = await llm.chat(system, user)
    return out.strip()


async def purify(llm: LLMClient, task: str, thinking: str) -> str:
    system, user = prompts.render("purifier", task=task[:400], thinking=thinking)
    out = await llm.chat(system, user, max_tokens=4096)
    return out.strip()


async def propose(llm: LLMClient, category_hint: str, recent_tasks: str,
                  difficulty_anchor: str = "（暂无，出中等难度题）") -> dict[str, Any]:
    system, user = prompts.render("proposer", category_hint=category_hint,
                                  recent_tasks=recent_tasks or "（无）",
                                  difficulty_anchor=difficulty_anchor)
    return await llm.chat_json(system, user)


async def observe(llm: LLMClient, content: str) -> dict[str, Any]:
    system, user = prompts.render("biodata_observer", content=content[:3000])
    return await llm.chat_json(system, user)


async def narrate(llm: LLMClient, facts: list[str]) -> dict[str, Any]:
    system, user = prompts.render("fact_narrator",
                                  facts="\n".join(f"- {f}" for f in facts))
    return await llm.chat_json(system, user)


async def explore_mission(llm: LLMClient, persona: str, anchors: str) -> dict[str, Any]:
    system, user = prompts.render("explore_mission", persona=persona, anchors=anchors)
    return await llm.chat_json(system, user)


async def style_rewrite(llm: LLMClient, content: str) -> str:
    system, user = prompts.render("style_rewriter", content=content[:4000])
    out = await llm.chat_json(system, user)
    return str(out.get("rewrite", "")).strip()


async def solve(llm: LLMClient, task: str) -> str:
    system, user = prompts.render("solver", task=task)
    return await llm.chat(system, user, max_tokens=4096)


async def meta_learn(llm: LLMClient, stats: str, current_lessons: str) -> dict[str, Any]:
    system, user = prompts.render("meta_learner", stats=stats,
                                  current_lessons=current_lessons or "（无）")
    return await llm.chat_json(system, user)
