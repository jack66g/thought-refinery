"""LLM 客户端：openai_compatible（真跑）与 fake（确定性测试）双模式。

provider 优先级：daemon.json llm.provider < 环境变量 RSI_LLM_PROVIDER。
密钥只从环境变量读（api_key_env 指定变量名），绝不落盘。
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx


class LLMNotConfigured(RuntimeError):
    """LLM 未配置（无密钥）——调用方决定降级行为，绝不静默假装成功。"""


class BaseLLM:
    async def chat(self, system: str, user: str, *, temperature: float | None = None,
                   max_tokens: int | None = None) -> str:
        raise NotImplementedError


class OpenAICompatLLM(BaseLLM):
    def __init__(self, cfg: dict[str, Any]):
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
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        resp = await self._client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()
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
        if "【坏段】" in user:
            return json.dumps({"verdict": "suspect", "first_bad_step": 2,
                               "error_type": "跳步", "evidence": "标记坏段"},
                              ensure_ascii=False)
        return json.dumps({"verdict": "clean", "first_bad_step": None,
                           "error_type": None, "evidence": ""}, ensure_ascii=False)

    def _role_repairer(self, system: str, user: str) -> str:
        if "不可修复" in user:
            return "UNREPAIRABLE"
        return "（已修复）补全省略的推导：设中间量并代入原式，逐步计算得证。"

    def _role_purifier(self, system: str, user: str) -> str:
        # 确定性变换：保留前 60% + 提纯标记；若答案句被截掉则补回（模拟真实提纯保结论）
        thinking = user.split("已审计的思考：", 1)[-1].split("\n直接输出", 1)[0]
        cut = max(20, int(len(thinking) * 0.6))
        out = thinking[:cut]
        if "答案是" in thinking and "答案是" not in out:
            out += "。" + thinking[thinking.rfind("答案是"):]
        return out + "\n（已提纯：去除冗余回溯，保留关键推理路径）"

    def _role_proposer(self, system: str, user: str) -> str:
        return json.dumps({
            "task": "计算 7*6+8 的值",
            "answer": "50",
            "answer_type": "math_expression",
            "category_hint_used": "math"}, ensure_ascii=False)

    def _role_solver(self, system: str, user: str) -> str:
        # 从题面中提取纯算术表达式并真算，保证硬验证可确定性通过
        expr = re.search(r"[-+*/().\d\s]{3,}[\d)]", user)
        if expr:
            try:
                value = eval(expr.group(0).strip(), {"__builtins__": {}}, {})  # noqa: S307 仅测试假模型
                return f"先算乘除后算加减：对 {expr.group(0).strip()} 求值。答案是 {value}"
            except Exception:
                pass
        return "思考：按题意逐步推导。答案是 0"

    def _role_meta_learner(self, system: str, user: str) -> str:
        return json.dumps({
            "lessons": ["审计员：数学类样本注意换元后的跳步，逐段核对中间量"],
            "supersede": []}, ensure_ascii=False)

    def _role_answer_probe(self, system: str, user: str) -> str:
        m = re.search(r"已知参照结论[：:](.+?)）", user)
        ans = m.group(1).strip() if m else "0"
        return f"答案是 {ans}"

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
        return json.dumps({"narratives": [f"（叙事）{f}" for f in facts]},
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
    """门面：按配置构造真/Fake 客户端；chat() 返回纯文本。"""

    def __init__(self, cfg: dict[str, Any]):
        provider = os.environ.get("RSI_LLM_PROVIDER") or cfg.get("provider", "fake")
        self.provider = provider
        if provider == "fake":
            self._impl: BaseLLM = FakeLLM(cfg)
        elif provider == "openai_compatible":
            self._impl = OpenAICompatLLM(cfg)
        else:
            raise ValueError(f"未知 LLM provider: {provider}")
        self.total_chars_in = 0
        self.total_chars_out = 0
        self.calls = 0

    async def chat(self, system: str, user: str, **kw: Any) -> str:
        self.calls += 1
        self.total_chars_in += len(system) + len(user)
        out = await self._impl.chat(system, user, **kw)
        self.total_chars_out += len(out)
        return out

    async def chat_json(self, system: str, user: str, **kw: Any) -> dict[str, Any]:
        out = await self.chat(system, user, **kw)
        return _extract_json_block(out)

    def estimated_cost(self, rate_per_1k: float = 0.001) -> float:
        tokens = (self.total_chars_in + self.total_chars_out) / 1.6
        return tokens / 1000 * rate_per_1k

    async def aclose(self) -> None:
        if isinstance(self._impl, OpenAICompatLLM):
            await self._impl.aclose()
