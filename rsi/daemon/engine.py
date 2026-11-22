"""RSI 守护进程引擎：队列消费 + 预算 + 自驱出题(T1) + 元学习循环。

每时每刻运行的保证：宿主没人投喂时，T1 自驱出题让队列永不停转；
预算封顶时暂停自驱（显式记入状态，不静默）。
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from pathlib import Path
from typing import Any

from ..biodata.state import load_biodata
from ..core import roles
from ..core.pipeline import Pipeline, new_event
from ..core.splitter import Splitter
from ..explore import Explorer
from ..llm.client import LLMClient, LLMNotConfigured
from ..sme_wrap.client import SMEClient

RSI_BASE = Path(__file__).resolve().parent.parent


class RSIDaemon:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.llm = LLMClient(cfg.get("llm", {}))
        self.sme = SMEClient(cfg.get("sme", {}).get("base_url", "http://127.0.0.1:8000"),
                             backlog_path=RSI_BASE / "logs" / "sme_backlog.jsonl")
        self.pipeline = Pipeline(self.llm, self.sme, cfg, RSI_BASE)
        # V3：仿生画像 + 三线分流 + 自探索（biodata 默认开；自探索默认关）
        self.biodata = load_biodata(RSI_BASE / "biodata" / "state.json")
        self.splitter = (Splitter(self.llm, self.biodata, self.sme, RSI_BASE)
                         if cfg.get("biodata", {}).get("enabled", True) else None)
        self.explorer = Explorer(self.llm, self.biodata, self.sme, cfg, RSI_BASE)
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=int(cfg.get("queue", {}).get("max_len", 2000)))
        self.workers = int(cfg.get("queue", {}).get("workers", 2))
        self._tasks: list[asyncio.Task] = []
        self._last_activity = time.time()
        self._today = time.strftime("%Y-%m-%d")
        self._day_events = 0
        self._recent_self_tasks: deque[str] = deque(maxlen=10)
        self._lessons_cache: list[str] = []
        self.running = False
        self.last_results: deque[dict[str, Any]] = deque(maxlen=50)

    # ---------- 生命周期 ----------

    async def start(self) -> None:
        self.running = True
        await self.sme.health()
        for i in range(self.workers):
            self._tasks.append(asyncio.create_task(self._worker(i), name=f"worker-{i}"))
        self._tasks.append(asyncio.create_task(self._self_propose_loop(), name="self-propose"))
        self._tasks.append(asyncio.create_task(self._meta_loop(), name="meta"))
        self._tasks.append(asyncio.create_task(self._sme_watchdog(), name="sme-watchdog"))

    async def stop(self) -> None:
        self.running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.llm.aclose()
        await self.sme.aclose()

    # ---------- 入队 ----------

    async def enqueue(self, event: dict[str, Any]) -> bool:
        if self._today != time.strftime("%Y-%m-%d"):
            self._today = time.strftime("%Y-%m-%d")
            self._day_events = 0
        self._day_events += 1
        self._last_activity = time.time()
        try:
            self.queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            return False

    # ---------- 消费 ----------

    async def _worker(self, wid: int) -> None:
        while self.running:
            try:
                event = await asyncio.wait_for(self.queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            try:
                result = await self.pipeline.process(event)
                if self.splitter is not None:
                    try:
                        result["split"] = await self.splitter.process(event)
                    except LLMNotConfigured:
                        raise
                    except Exception as exc:  # noqa: BLE001 分流失败不拖垮 CoT 线
                        (RSI_BASE / "logs" / "errors.jsonl").open("a", encoding="utf-8").write(
                            json.dumps({"ts": time.strftime("%F %T"),
                                        "err": f"splitter:{exc!r}"}, ensure_ascii=False) + "\n")
                self.last_results.append(result)
            except LLMNotConfigured as exc:
                # LLM 未配置：事件退回队列尾部等配置，台账记录（不静默丢）
                (RSI_BASE / "logs" / "errors.jsonl").open("a", encoding="utf-8").write(
                    json.dumps({"ts": time.strftime("%F %T"), "err": str(exc),
                                "event_id": event.get("id")}, ensure_ascii=False) + "\n")
                await asyncio.sleep(30)
                if self.queue.qsize() < self.queue.maxsize - 1:
                    self.queue.put_nowait(event)
            except Exception as exc:  # noqa: BLE001 单事件失败不拖垮常驻进程
                (RSI_BASE / "logs" / "errors.jsonl").open("a", encoding="utf-8").write(
                    json.dumps({"ts": time.strftime("%F %T"), "err": repr(exc),
                                "event_id": event.get("id")}, ensure_ascii=False) + "\n")

    # ---------- T1 自驱出题 ----------

    def _budget_ok(self) -> bool:
        cap = float(self.cfg.get("budget", {}).get("daily_cost_cap", 5.0))
        max_events = int(self.cfg.get("self_propose", {}).get("max_events_per_day", 200))
        return (self.llm.estimated_cost() < cap
                and self._day_events < max_events * 3)

    async def _self_propose_loop(self) -> None:
        idle_after = float(self.cfg.get("self_propose", {}).get("idle_after_s", 120))
        while self.running:
            await asyncio.sleep(10)
            sp = self.cfg.get("self_propose", {})
            if not sp.get("enabled", True):
                continue
            if self.queue.qsize() > 0 or time.time() - self._last_activity < idle_after:
                continue
            if not self._budget_ok():
                continue
            try:
                stats = self.pipeline.stats
                anchor = (f"已收编 {stats.get('accepted_sft', 0)} 条；"
                          f"平均思考长度 {stats.get('avg_think_chars', '暂无')} 字符")
                proposal = await roles.propose(
                    self.llm, "auto", "\n".join(self._recent_self_tasks),
                    difficulty_anchor=anchor)
                task = proposal.get("task", "").strip()
                expected = str(proposal.get("answer", "")).strip()
                if not task or not expected:
                    continue
                solution = await roles.solve(self.llm, task)
                self._recent_self_tasks.append(task[:80])
                event = new_event("self", task, solution,
                                  conclusion="", event_kind="self_proposed",
                                  expected_answer=expected,
                                  answer_type=proposal.get("answer_type", "math_expression"))
                await self.enqueue(event)
            except LLMNotConfigured:
                await asyncio.sleep(60)  # 无密钥：自驱暂停，等配置（status 可见）
            except Exception as exc:  # noqa: BLE001
                (RSI_BASE / "logs" / "errors.jsonl").open("a", encoding="utf-8").write(
                    json.dumps({"ts": time.strftime("%F %T"),
                                "err": f"self_propose:{exc!r}"}, ensure_ascii=False) + "\n")
                await asyncio.sleep(30)

    # ---------- 元学习循环 ----------

    async def _meta_loop(self) -> None:
        every = 20  # 每 20 个事件复盘一次
        while self.running:
            await asyncio.sleep(15)
            total = self.pipeline.stats.get("events_total", 0)
            if total == 0 or total % every != 0 or total == getattr(self, "_last_meta_total", -1):
                continue
            self._last_meta_total = total
            try:
                hits = await self.sme.search("提纯教训", ns="rsi.lessons", top_k=5)
                current = [h.get("memory", h).get("text", "") if isinstance(h, dict) else str(h)
                           for h in hits]
                stats_str = json.dumps(self.pipeline.stats, ensure_ascii=False)
                out = await roles.meta_learn(self.llm, stats_str, "\n".join(current))
                for lesson in out.get("lessons", [])[:5]:
                    await self.sme.add(str(lesson), ns="rsi.lessons",
                                       tags=["lesson"], importance=0.8)
                    self._lessons_cache.append(str(lesson))
                for old in out.get("supersede", []):
                    await self.sme.add(f"更正：{old} —— 该教训已被取代。",
                                       ns="rsi.lessons", tags=["lesson", "superseded"])
                with (RSI_BASE / "logs" / "lessons.jsonl").open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"ts": time.strftime("%F %T"), **out},
                                        ensure_ascii=False) + "\n")
            except LLMNotConfigured:
                continue
            except Exception as exc:  # noqa: BLE001
                (RSI_BASE / "logs" / "errors.jsonl").open("a", encoding="utf-8").write(
                    json.dumps({"ts": time.strftime("%F %T"),
                                "err": f"meta:{exc!r}"}, ensure_ascii=False) + "\n")

    # ---------- SME 看门狗（backlog 补写） ----------

    async def _sme_watchdog(self) -> None:
        while self.running:
            await asyncio.sleep(60)
            try:
                if not self.sme.online:
                    await self.sme.health()
                if self.sme.online:
                    await self.sme.flush_backlog()
            except Exception:  # noqa: BLE001
                pass

    # ---------- 状态 ----------

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "llm_provider": self.llm.provider,
            "llm_calls": self.llm.calls,
            "estimated_cost": round(self.llm.estimated_cost(), 4),
            "budget_cap": self.cfg.get("budget", {}).get("daily_cost_cap", 5.0),
            "day_events": self._day_events,
            "queue_len": self.queue.qsize(),
            "sme_online": self.sme.online,
            "sme_last_error": self.sme.last_error,
            "stats": self.pipeline.stats,
            "self_propose_enabled": self.cfg.get("self_propose", {}).get("enabled", True),
            "biodata": self.biodata.summary() if self.splitter else None,
            "self_explore_enabled": self.cfg.get("self_explore", {}).get("enabled", False),
            "recent_results": list(self.last_results)[-5:],
        }
