"""仅在本机提供选择模式页面与地址。"""
from __future__ import annotations

import ipaddress
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..llm.budget import LLMBudgetExceeded
from . import ChoiceError, ChoiceMode

_UI = Path(__file__).resolve().parent / "ui"
_UI_HEADERS = {
    "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": "default-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'none'",
}


class Settings(BaseModel):
    enabled: bool
    open_window: bool = True


class Answer(BaseModel):
    round_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    step: Literal["decision", "reason"]
    option: Literal["0", "1", "2", "skip"]


def _loopback(request: Request, response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"
    try:
        local = request.client is not None and ipaddress.ip_address(request.client.host).is_loopback
    except ValueError:
        local = False
    if not local or request.url.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise HTTPException(status_code=403, detail="选择模式只支持本机访问。")
    origin = request.headers.get("origin")
    if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
        raise HTTPException(status_code=403, detail="请从选择模式页面提交操作。")


async def _call(action: Awaitable[dict[str, Any]]) -> dict[str, Any]:
    try:
        return await action
    except ChoiceError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.message) from exc
    except LLMBudgetExceeded as exc:
        raise HTTPException(status_code=429, detail="今日模型预算不足，题目进度已保留。") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="暂时无法生成题目，题目进度已保留。") from exc
    except OSError as exc:
        raise HTTPException(status_code=503, detail="暂时无法写入训练文件，请重试。") from exc


def choice_router(mode: ChoiceMode, open_window: Callable[[], None] | None) -> APIRouter:
    router = APIRouter(prefix="/choice", dependencies=[Depends(_loopback)], tags=["choice"])

    @router.get("", include_in_schema=False)
    async def page():
        return FileResponse(_UI / "index.html", media_type="text/html", headers=_UI_HEADERS)

    @router.get("/ui/{asset}", include_in_schema=False)
    async def ui_asset(asset: Literal["choice.css", "choice.js"]):
        kind = "text/css" if asset.endswith(".css") else "text/javascript"
        return FileResponse(_UI / asset, media_type=kind, headers=_UI_HEADERS)

    @router.get("/api/state")
    async def state():
        return await _call(mode.status())

    @router.post("/api/settings")
    async def settings(body: Settings, background: BackgroundTasks):
        was_enabled = mode.enabled
        await mode.set_enabled(body.enabled)
        if body.enabled and not was_enabled and body.open_window and open_window:
            background.add_task(open_window)
        return await _call(mode.status())

    @router.post("/api/start")
    async def start():
        return await _call(mode.begin())

    @router.post("/api/answer")
    async def answer(body: Answer):
        return await _call(mode.submit(body.round_id, body.step, body.option))

    return router
