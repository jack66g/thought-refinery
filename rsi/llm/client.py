"""LLM 客户端：openai_compatible（真跑）与 fake（确定性测试）双模式。

provider 优先级：daemon.json llm.provider < 环境变量 RSI_LLM_PROVIDER。
密钥只从环境变量读（api_key_env 指定变量名），绝不落盘。
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import httpx

from ..core.cleaner import mask_pii
from .budget import DailyBudget


class LLMNotConfigured(RuntimeError):
    """LLM 未配置（无密钥）——调用方决定降级行为，绝不静默假装成功。"""


class BaseLLM:
    async def chat(self, system: str, user: str, *, temperature: float | None = None,
                   max_tokens: int | None = None) -> str:
        raise NotImplementedError


class OpenAICompatLLM(BaseLLM):
    def __init__(self, cfg: dict[str, Any], budget: DailyBudget | None = None):
        self.budget = budget
        self.base_url: str = cfg["base_url"].rstrip("/")
        self.model: str = cfg["model"]
        self.api_key_env: str = cfg.get("api_key_env", "RSI_LLM_API_KEY")
        self.api_key: str = os.environ.get(self.api_key_env, "")
        self.temperature: float = cfg.get("temperature", 0.6)
        self.max_tokens: int = cfg.get("max_tokens", 4096)
        self.timeout_s: float = cfg.get("timeout_s", 120)
        self.reasoning_effort: str | None = cfg.get("reasoning_effort") or None
        if not self.api_key:
            raise LLMNotConfigured(
                f"环境变量 {self.api_key_env} 未设置（base_url={self.base_url}）")
        self._client = httpx.AsyncClient(timeout=self.timeout_s)

    async def chat(self, system: str, user: str, *, temperature: float | None = None,
                   max_tokens: int | None = None) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else min(self.max_tokens, max_tokens),
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        reservation = None
        if self.budget is not None:
            upper_input = len(system.encode("utf-8")) + len(user.encode("utf-8")) + 256
            reservation = self.budget.reserve(upper_input, body["max_tokens"])
        resp = await self._client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()
        if reservation is not None:
            self.budget.settle(reservation, data.get("usage"))
        return data["choices"][0]["message"]["content"] or ""

    async def aclose(self) -> None:
        await self._client.aclose()


def _extract_json_block(text: str) -> dict[str, Any]:
    """从模型输出里稳健提取第一个 JSON 对象。"""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"输出中无 JSON 对象: {text[:120]!r}")
    return json.loads(match.group(0))


class FakeLLM(BaseLLM):
    """确定性假模型：按 system 首行 [ROLE:x] 分发脚本化响应，供全链路测试。"""

    def __init__(self, _cfg: dict[str, Any] | None = None):
        pass

    async def chat(self, system: str, user: str, *, temperature: float | None = None,
                   max_tokens: int | None = None) -> str:
        role = system.split("]")[0].replace("[ROLE:", "").strip()
        handler = getattr(self, f"_role_{role}", None)
        if handler is None:
            raise ValueError(f"FakeLLM 未知角色: {role}")
        return handler(system, user)

    # ---- 各角色脚本 ----

    def _role_classifier(self, system: str, user: str) -> str:
        text = user.split("任务：", 1)[-1]  # 只看载荷区，模板文字里的"数学"等不算
        if "[cat:" in text:  # 测试注入通道
            cat = re.search(r"\[cat:(\w+)\]", text).group(1)
            return json.dumps({"category": cat, "tags": ["测试"], "confidence": 0.95},
                              ensure_ascii=False)
        if any(k in text for k in ("方程", "计算", "求值", "数学", "+", "求解")):
            cat = "math"
        elif any(k in text for k in ("代码", "函数", "debug", "bug", "脚本")):
            cat = "code"
        elif any(k in text for k in ("摘要", "翻译", "写作", "改写")):
            cat = "chinese"
        else:
            cat = "reasoning"
        return json.dumps({"category": cat, "tags": [], "confidence": 0.9},
                          ensure_ascii=False)

    def _role_value_gate(self, system: str, user: str) -> str:
        payload = user.split("任务：", 1)[-1]  # 只看载荷区，模板文字不算
        if "寒暄" in payload or "今天天气" in payload:
            return json.dumps({"total": 2.0, "verdict": "drop", "reason": "无实质推理"},
                              ensure_ascii=False)
        return json.dumps({
            "scores": {"verifiable": 8, "density": 8, "novelty": 7,
                       "non_template": 7, "completeness": 8},
            "total": 7.7, "verdict": "keep", "reason": "有完整推理链"}, ensure_ascii=False)

    def _role_auditor(self, system: str, user: str) -> str:
        bad = re.search(r"(?m)^\[(\d+)\][^\n]*【坏段】", user)
        if bad:
            return json.dumps({"verdict": "suspect", "first_bad_step": int(bad.group(1)),
                               "error_type": "跳步", "evidence": "标记坏段"},
                              ensure_ascii=False)
        return json.dumps({"verdict": "clean", "first_bad_step": None,
                           "error_type": None, "evidence": ""}, ensure_ascii=False)

    def _role_repairer(self, system: str, user: str) -> str:
        if "不可修复" in user:
            return "UNREPAIRABLE"
        segment_text = user.split("待修复段落：", 1)[-1].split("\n后文", 1)[0]
        return re.sub(r"【坏段】[^。]*(?:。|$)", "", segment_text).strip()

    def _role_purifier(self, system: str, user: str) -> str:
        return user.split("已审计的思考：", 1)[-1].split("\n直接输出", 1)[0].strip()

    def _role_proposer(self, system: str, user: str) -> str:
        return json.dumps({
            "task": "计算 7*6+8 的值",
            "answer": "50",
            "answer_type": "math_expression",
            "category_hint_used": "math"}, ensure_ascii=False)

    def _role_solver(self, system: str, user: str) -> str:
        from ..core.verifier import deterministic_answer
        task = user.split("任务：", 1)[-1].splitlines()[0].strip()
        answer = deterministic_answer(task)
        if answer is not None:
            return f"先算括号和乘除，再计算加减，对原式逐项代入并检查运算优先级。答案是 {answer}。"
        return "INSUFFICIENT"

    def _role_meta_learner(self, system: str, user: str) -> str:
        return json.dumps({
            "lessons": ["审计员：数学类样本注意换元后的跳步，逐段核对中间量"],
            "supersede": []}, ensure_ascii=False)

    def _role_answer_probe(self, system: str, user: str) -> str:
        from ..core.verifier import deterministic_answer
        task = user.removeprefix("任务：")
        return deterministic_answer(task.splitlines()[0]) or "INSUFFICIENT"

    def _role_choice_scene(self, system: str, user: str) -> str:
        inputs = json.loads(user)
        label = inputs["context"]
        return json.dumps({
            "title": "周末工作坊",
            "background": f"在一段虚构的{label}故事中，你和朋友要准备一场小型工作坊。"
                          "剩下一个下午，可以选一种推进方式；三种做法都来得及完成。",
            "question": "接下来你会怎么做？",
            "a": f"优先采用{inputs['side_a']}的方式推进，留出相同的复盘时间。",
            "b": f"优先采用{inputs['side_b']}的方式推进，留出相同的复盘时间。",
            "mixed": "把下午分为两段，先并行准备两种方案，再根据现场状况切换。",
            "memory_refs": [1],
        }, ensure_ascii=False)

    def _role_choice_reasons(self, system: str, user: str) -> str:
        return json.dumps({
            "question": "哪一个因素最影响你刚才的决定？",
            "r1": "这能更快让我掌握现场反馈，方便随后调整。",
            "r2": "我希望先明确每个人的时间和责任，便于协调。",
            "r3": "我更看重回撤空间，遇到变化也能恢复原计划。",
        }, ensure_ascii=False)

    def _role_biodata_observer(self, system: str, user: str) -> str:
        text = user.split("用户内容：", 1)[-1]
        mood = {"p": 0.0, "a": 0.0, "d": 0.0}
        emo = {"label": None, "intensity": 0}
        pulse = {"energy": 0, "stress": 0, "curiosity": 0, "social": 0,
                 "autonomy": 0, "competence": 0}
        if "累" in text or "熬夜" in text:
            pulse["energy"] = -1
            mood["p"] = -0.2
        if any(k in text for k in ("太好了", "开心", "高兴", "成了")):
            mood["p"] = 0.5
            emo = {"label": "喜", "intensity": 60}
        if any(k in text for k in ("烦", "气人", "愤怒")):
            mood["p"] = -0.5
            emo = {"label": "怒", "intensity": 70}
        if "好奇" in text or "想了解" in text:
            pulse["curiosity"] = 1
        facts = [s.strip() for s in re.split(r"[。！？\n]", text)
                 if s.strip().startswith("我") and len(s.strip()) > 8][:3]
        return json.dumps({"mood_delta": mood, "emotion": emo, "need_pulse": pulse,
                           "trait_hint": {}, "facts": facts}, ensure_ascii=False)

    def _role_fact_narrator(self, system: str, user: str) -> str:
        section = user.split("既定事实列表：", 1)[-1]
        facts = [ln.lstrip("- ").strip() for ln in section.splitlines()
                 if ln.strip().startswith("-")]
        return json.dumps({"narratives": facts},
                          ensure_ascii=False)

    def _role_explore_mission(self, system: str, user: str) -> str:
        return json.dumps({
            "mission": "用 Python 实现一个命令行待办工具，支持增删查与本地 JSON 持久化，附单元测试",
            "type": "curiosity", "acceptance": "pytest 全绿；add/list/del 三个子命令可用",
            "motivation": "好奇心缺口：探索 CLI 工具构建"}, ensure_ascii=False)

    def _role_style_rewriter(self, system: str, user: str) -> str:
        content = user.split("原始发言：", 1)[-1].split("\n只输出")[0]
        cut = max(20, int(len(content) * 0.8))
        return json.dumps({"rewrite": "（改写）" + content[:cut]}, ensure_ascii=False)


class LLMClient:
    """模型调用边界。"""

    def __init__(self, cfg: dict[str, Any], budget: dict[str, Any] | None = None,
                 budget_path: Path | None = None):
        provider = os.environ.get("RSI_LLM_PROVIDER") or cfg.get("provider", "fake")
        self.provider = provider
        self.budget = DailyBudget(budget, budget_path) if provider != "fake" and budget is not None else None
        if provider == "fake":
            self._impl: BaseLLM = FakeLLM(cfg)
        elif provider == "openai_compatible":
            self._impl = OpenAICompatLLM(cfg, self.budget)
        else:
            raise ValueError(f"未知 LLM provider: {provider}")
        self.total_chars_in = 0
        self.total_chars_out = 0
        self.calls = 0

    async def chat(self, system: str, user: str, **kw: Any) -> str:
        system, user = mask_pii(system), mask_pii(user)
        out = mask_pii(await self._impl.chat(system, user, **kw))
        self.calls += 1
        self.total_chars_in += len(system) + len(user)
        self.total_chars_out += len(out)
        return out

    async def chat_json(self, system: str, user: str, **kw: Any) -> dict[str, Any]:
        out = await self.chat(system, user, **kw)
        return _extract_json_block(out)

    def estimated_cost(self, rate_per_1k: float = 0.001) -> float:
        return self.budget.spent() if self.budget is not None else 0.0

    async def aclose(self) -> None:
        if isinstance(self._impl, OpenAICompatLLM):
            await self._impl.aclose()
