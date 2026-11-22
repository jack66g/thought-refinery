"""SME 客户端：纯 REST，不 import sme 包（零耦合）。

降级契约（显式，不静默）：SME 不可达时 add 记入本地 backlog 文件，
health 恢复后由调用方 flush_backlog 补写。status 永远暴露 sme 在线状态。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx


class SMEClient:
    def __init__(self, base_url: str, backlog_path: Path):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=15)
        self._backlog_path = backlog_path
        self.online = False
        self._last_error: str = ""

    async def health(self) -> bool:
        try:
            resp = await self._client.get(f"{self.base_url}/health")
            self.online = resp.status_code == 200
            self._last_error = "" if self.online else f"HTTP {resp.status_code}"
        except Exception as exc:  # noqa: BLE001 需要把任何网络错误都显式暴露
            self.online = False
            self._last_error = repr(exc)
        return self.online

    @property
    def last_error(self) -> str:
        return self._last_error

    async def add(self, text: str, *, ns: str, tags: list[str] | None = None,
                  metadata: dict[str, Any] | None = None,
                  importance: float = 0.5, source: str = "rsi") -> str | None:
        """写入一条记忆，返回 memory_id；SME 不在线则进 backlog 返回 None。"""
        body: dict[str, Any] = {"text": text, "ns": ns, "tags": tags or [],
                                "metadata": metadata or {}, "importance": importance,
                                "source": source}
        try:
            resp = await self._client.post(f"{self.base_url}/memories", json=body)
            resp.raise_for_status()
            self.online = True
            return str(resp.json().get("id", ""))
        except Exception as exc:  # noqa: BLE001
            self.online = False
            self._last_error = repr(exc)
            self._append_backlog({"op": "add", "body": body, "ts": time.time()})
            return None

    async def search(self, text: str, *, ns: str | None = None, top_k: int = 5
                     ) -> list[dict[str, Any]]:
        body: dict[str, Any] = {"text": text, "top_k": top_k}
        if ns:
            # SME extraction 管线把 ns 收进 metadata.ns（实测），双保险过滤
            body["ns"] = ns
            body["metadata_filters"] = {"ns": ns}
        try:
            resp = await self._client.post(f"{self.base_url}/memories/search", json=body)
            resp.raise_for_status()
            self.online = True
            data = resp.json()
            return data.get("results", data if isinstance(data, list) else [])
        except Exception as exc:  # noqa: BLE001
            self.online = False
            self._last_error = repr(exc)
            return []

    async def reinforce(self, memory_id: str) -> bool:
        try:
            resp = await self._client.post(f"{self.base_url}/memories/{memory_id}/hit")
            self.online = resp.status_code == 200
            return self.online
        except Exception as exc:  # noqa: BLE001
            self.online = False
            self._last_error = repr(exc)
            return False

    # ---- backlog（显式降级） ----

    def _append_backlog(self, record: dict[str, Any]) -> None:
        self._backlog_path.parent.mkdir(parents=True, exist_ok=True)
        with self._backlog_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def flush_backlog(self, limit: int = 50) -> int:
        """SME 恢复在线后补写积压。返回成功补写条数。"""
        if not self._backlog_path.exists():
            return 0
        if not await self.health():
            return 0
        lines = self._backlog_path.read_text(encoding="utf-8").splitlines()
        remaining: list[str] = []
        flushed = 0
        for line in lines:
            if flushed >= limit:
                remaining.append(line)
                continue
            try:
                record = json.loads(line)
                resp = await self._client.post(
                    f"{self.base_url}/memories", json=record["body"])
                resp.raise_for_status()
                flushed += 1
            except Exception:  # noqa: BLE001
                remaining.append(line)
        self._backlog_path.write_text(
            "\n".join(remaining) + ("\n" if remaining else ""), encoding="utf-8")
        return flushed

    async def aclose(self) -> None:
        await self._client.aclose()
