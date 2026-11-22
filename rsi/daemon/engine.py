"""RSI 守护进程引擎：队列消费 + 预算 + 自驱出题(T1) + 元学习循环。

每时每刻运行的保证：宿主没人投喂时，T1 自驱出题让队列永不停转；
预算封顶时暂停自驱（显式记入状态，不静默）。
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from datetime import date, datetime, time as wall_time, timedelta
from pathlib import Path
from typing import Any

from ..biodata.state import load_biodata
from ..core import roles
from ..core.cleaner import mask_pii
from ..core.pipeline import Pipeline, new_event
from ..core.splitter import Splitter
from ..explore import Explorer
from ..llm.budget import LLMBudgetExceeded
from ..llm.client import LLMClient
from ..sme_wrap.client import SMEClient
from .journal import EventJournal

RSI_BASE = Path(__file__).resolve().parent.parent


class RSIDaemon:
    def __init__(self, cfg: dict[str, Any], base: Path = RSI_BASE):
        self.cfg = cfg
        self.base = base
        self.llm = LLMClient(cfg.get("llm", {}), cfg.setdefault("budget", {}), base / "logs" / "budget.json")
        self.sme = SMEClient(cfg.get("sme", {}).get("base_url", "http://127.0.0.1:8000"),
                             backlog_path=base / "logs" / "sme_backlog.jsonl")
        self.pipeline = Pipeline(self.llm, self.sme, cfg, base)
        # V3：仿生画像 + 三线分流 + 自探索（biodata 默认开；自探索默认关）
        self.biodata = load_biodata(base / "biodata" / "state.json")
        self.splitter = (Splitter(self.llm, self.biodata, self.sme, base)
                         if cfg.get("biodata", {}).get("enabled", True) else None)
        self.explorer = Explorer(self.llm, self.biodata, self.sme, cfg, base)
        self.journal = EventJournal(base / "logs" / "events.sqlite3",
                                    int(cfg.get("queue", {}).get("max_len", 2000)))
        self._ready = asyncio.Event()
        self._last_budget_error = ""
        self.workers = int(cfg.get("queue", {}).get("workers", 2))
        self._tasks: list[asyncio.Task] = []
        self._last_activity = time.time()
        self._recent_self_tasks: deque[str] = deque(maxlen=10)
        self._lessons_cache: list[str] = []
        self.running = False

    # ---------- 生命周期 ----------

    async def start(self) -> None:
        self.journal.recover()
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
        self.journal.close()

    # ---------- 入队 ----------

    async def enqueue(self, event: dict[str, Any]) -> bool:
        accepted = self.journal.enqueue(event)
        if accepted:
            self._last_activity = time.time()
            self._ready.set()
        return accepted

    def _record_error(self, event_id: str, exc: Exception) -> str:
        message = mask_pii(str(exc) if isinstance(exc, LLMBudgetExceeded) else type(exc).__name__)
        path = self.base / "logs" / "errors.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"ts": time.time(), "event_id": event_id,
                                     "err": message}, ensure_ascii=False) + "\n")
        return message

    async def _worker(self, wid: int) -> None:
        while self.running:
            job = self.journal.claim()
            if job is None:
                self._ready.clear()
                try:
                    await asyncio.wait_for(self._ready.wait(), timeout=1)
                except asyncio.TimeoutError:
                    pass
                continue
            event, result, attempts = job
            event_id = event["id"]
            try:
                if result is None:
                    result = await self.pipeline.process(event)
                    self.journal.save_cot(event_id, result)
                if self.splitter is not None:
                    result["split"] = await self.splitter.process(event)
                self.journal.finish(event_id, result)
                self._last_budget_error = ""
            except LLMBudgetExceeded as exc:
                self._last_budget_error = self._record_error(event_id, exc)
                tomorrow = datetime.combine(date.today() + timedelta(days=1), wall_time.min)
                self.journal.retry(event_id, tomorrow.timestamp(), count_attempt=False)
            except Exception as exc:
                error = self._record_error(event_id, exc)
                if attempts < 3:
                    self.journal.retry(event_id, time.time() + 2 ** attempts)
                else:
                    failed = result or {"event_id": event_id, "outcome": "failed"}
                    failed["processing_error"] = error
                    self.journal.finish(event_id, failed, failed=True)

    # ---------- T1 自驱出题 ----------

    def _budget_ok(self) -> bool:
        cap = float(self.cfg.get("budget", {}).get("daily_cost_cap", 5.0))
        max_events = int(self.cfg.get("self_propose", {}).get("max_events_per_day", 200))
        return (self.llm.estimated_cost() < cap
                and self.journal.count_today("self_proposed") < max_events)

    async def _self_propose_loop(self) -> None:
        idle_after = float(self.cfg.get("self_propose", {}).get("idle_after_s", 120))
        while self.running:
            await asyncio.sleep(10)
            sp = self.cfg.get("self_propose", {})
            if not sp.get("enabled", False):
                continue
            if self.journal.count_pending() > 0 or time.time() - self._last_activity < idle_after:
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
            except Exception as exc:
                error = self._record_error("self_propose", exc)
                if isinstance(exc, LLMBudgetExceeded):
                    self._last_budget_error = error
                await asyncio.sleep(30)

    # ---------- 元学习循环 ----------

    async def _meta_loop(self) -> None:
        every = 20  # 每 20 个事件复盘一次
        while self.running:
            await asyncio.sleep(15)
            total = self.pipeline.stats.get("events_total", 0)
            batch = total // every
            if batch <= getattr(self, "_last_meta_batch", 0):
                continue
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
                with (self.base / "logs" / "lessons.jsonl").open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"ts": time.strftime("%F %T"), **out},
                                        ensure_ascii=False) + "\n")
                self._last_meta_batch = batch
            except Exception as exc:
                error = self._record_error("meta", exc)
                if isinstance(exc, LLMBudgetExceeded):
                    self._last_budget_error = error

    # ---------- SME 看门狗（backlog 补写） ----------

    async def _sme_watchdog(self) -> None:
        while self.running:
            await asyncio.sleep(60)
            try:
                if not self.sme.online:
                    await self.sme.health()
                if self.sme.online:
                    await self.sme.flush_backlog()
            except Exception as exc:
                self._record_error("sme_watchdog", exc)

    # ---------- 状态 ----------

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "llm_provider": self.llm.provider,
            "llm_calls": self.llm.calls,
            "estimated_cost": round(self.llm.estimated_cost(), 4),
            "budget_cap": self.cfg.get("budget", {}).get("daily_cost_cap", 5.0),
            "day_events": self.journal.count_today(),
            "queue_len": self.journal.count_pending(),
            "budget_last_error": self._last_budget_error,
            "sme_online": self.sme.online,
            "sme_last_error": self.sme.last_error,
            "stats": self.pipeline.stats,
            "self_propose_enabled": self.cfg.get("self_propose", {}).get("enabled", False),
            "biodata": self.biodata.summary() if self.splitter else None,
            "self_explore_enabled": self.cfg.get("self_explore", {}).get("enabled", False),
            "recent_results": self.journal.recent(),
        }
