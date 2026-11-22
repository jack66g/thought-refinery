"""RSI daemon HTTP 面：/ingest（宿主投喂）/distill/status/query/lessons/set_budget。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..connectors.adapters import normalize
from .engine import RSIDaemon


class IngestBody(BaseModel):
    source: str = "generic"
    payload: dict[str, Any]


class DistillBody(BaseModel):
    task: str
    thinking: str
    conclusion: str = ""
    source: str = "mcp"


class SetBudgetBody(BaseModel):
    daily_cost_cap: float | None = None
    self_propose_enabled: bool | None = None
    self_explore_enabled: bool | None = None
    biodata_enabled: bool | None = None


def create_app(daemon: RSIDaemon) -> FastAPI:
    app = FastAPI(title="RSI daemon", version="0.1.0")

    @app.on_event("startup")
    async def _startup() -> None:
        await daemon.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await daemon.stop()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True}

    @app.post("/ingest")
    async def ingest(body: IngestBody) -> dict[str, Any]:
        """T0/T2 通道：宿主连接器把轨迹 POST 进来。source 决定适配器。"""
        try:
            event = normalize(body.source, body.payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        ok = await daemon.enqueue(event)
        if not ok:
            raise HTTPException(status_code=429, detail="队列已满，事件被拒（低价值洪峰保护）")
        return {"queued": True, "event_id": event["id"]}

    @app.post("/distill")
    async def distill(body: DistillBody) -> dict[str, Any]:
        """T3/MCP 通道：直接投喂一段思考。"""
        if not body.task.strip() or not body.thinking.strip():
            raise HTTPException(status_code=400, detail="task/thinking 不能为空")
        from ..core.pipeline import new_event
        event = new_event(body.source, body.task, body.thinking, body.conclusion,
                          event_kind="host_task")
        ok = await daemon.enqueue(event)
        if not ok:
            raise HTTPException(status_code=429, detail="队列已满")
        return {"queued": True, "event_id": event["id"]}

    @app.get("/status")
    async def status() -> dict[str, Any]:
        return daemon.status()

    @app.get("/query")
    async def query(category: str = "", k: int = 3) -> dict[str, Any]:
        base = Path(__file__).resolve().parent.parent / "dataset"
        cats = [category] if category else ["math", "chinese", "code", "reasoning", "misc"]
        out: dict[str, Any] = {}
        for cat in cats:
            path = base / cat / "sft.jsonl"
            if not path.exists():
                out[cat] = []
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
            out[cat] = [json.loads(x) for x in lines[-k:]]
        return out

    @app.get("/lessons")
    async def lessons(k: int = 5) -> dict[str, Any]:
        hits = await daemon.sme.search("提纯教训", ns="rsi.lessons", top_k=k)
        items = []
        for h in hits:
            mem = h.get("memory", h) if isinstance(h, dict) else {}
            items.append(mem.get("text", str(h)) if isinstance(mem, dict) else str(h))
        return {"lessons": items, "local_cache": daemon._lessons_cache[-k:]}

    @app.post("/set_budget")
    async def set_budget(body: SetBudgetBody) -> dict[str, Any]:
        if body.daily_cost_cap is not None:
            daemon.cfg.setdefault("budget", {})["daily_cost_cap"] = body.daily_cost_cap
        if body.self_propose_enabled is not None:
            daemon.cfg.setdefault("self_propose", {})["enabled"] = body.self_propose_enabled
        if body.self_explore_enabled is not None:
            daemon.cfg.setdefault("self_explore", {})["enabled"] = body.self_explore_enabled
        if body.biodata_enabled is not None:
            daemon.cfg.setdefault("biodata", {})["enabled"] = body.biodata_enabled
            if body.biodata_enabled and daemon.splitter is None:
                from ..core.splitter import Splitter
                from .engine import RSI_BASE
                daemon.splitter = Splitter(daemon.llm, daemon.biodata, daemon.sme, RSI_BASE)
            elif not body.biodata_enabled:
                daemon.splitter = None
        return {"ok": True, "cfg": {"budget": daemon.cfg.get("budget"),
                                    "self_propose": daemon.cfg.get("self_propose"),
                                    "self_explore": daemon.cfg.get("self_explore"),
                                    "biodata": daemon.cfg.get("biodata")}}

    @app.get("/next_mission")
    async def next_mission() -> dict[str, Any]:
        """自探索：生成下一个宿主任务（默认关；cap/开关状态在返回里）。"""
        return await daemon.explorer.next_mission()

    return app
