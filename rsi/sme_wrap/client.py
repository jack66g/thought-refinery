"""SME 客户端：纯 REST，不 import sme 包（零耦合）。

降级契约（显式，不静默）：SME 不可达时 add 记入本地 backlog 文件，
health 恢复后由调用方 flush_backlog 补写。status 永远暴露 sme 在线状态。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from ..core.cleaner import mask_pii, redact


class SMEClient:
    def __init__(self, base_url: str, backlog_path: Path):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=15)
        self._backlog_path = backlog_path
        self._receipt_path = backlog_path.with_suffix(".receipts.jsonl")
        self._lock = asyncio.Lock()
        self._receipts = {}
        if self._receipt_path.exists():
            for line in self._receipt_path.read_text(encoding="utf-8").splitlines():
                try:
                    receipt = json.loads(line)
                    self._receipts[receipt["key"]] = receipt["memory_id"]
                except (ValueError, KeyError, TypeError):
                    continue
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
        """先写出站记录，再投递到 SME。"""
        body = redact({"text": text, "ns": ns, "tags": tags or [],
                       "metadata": metadata or {}, "importance": importance, "source": source})
        event_id = body["metadata"].get("event_id")
        if event_id:
            identity = json.dumps([event_id, ns, body["text"]], ensure_ascii=False).encode()
            key = hashlib.sha256(identity).hexdigest()
        else:
            key = uuid.uuid4().hex
        body["metadata"]["rsi_delivery_id"] = key
        async with self._lock:
            if key in self._receipts:
                return self._receipts[key]
            pending = self._pending()
            if any(self._key(line) == key for line in pending):
                return None
            line = json.dumps({"op": "add", "body": body, "ts": time.time()}, ensure_ascii=False)
            self._backlog_path.parent.mkdir(parents=True, exist_ok=True)
            with self._backlog_path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
            try:
                memory_id = await self._send(body)
                self._write_pending(pending)
                return memory_id
            except (httpx.HTTPError, ValueError, KeyError, TypeError, OSError) as exc:
                self.online = False
                self._last_error = type(exc).__name__
                return None

    async def search(self, text: str, *, ns: str | None = None, top_k: int = 5
                     ) -> list[dict[str, Any]]:
        body: dict[str, Any] = {"text": mask_pii(text), "top_k": top_k}
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

    def _pending(self) -> list[str]:
        if self._backlog_path.exists():
            return self._backlog_path.read_text(encoding="utf-8").splitlines()
        return []

    @staticmethod
    def _key(line: str) -> str | None:
        try:
            return json.loads(line)["body"]["metadata"].get("rsi_delivery_id")
        except (ValueError, KeyError, TypeError, AttributeError):
            return None

    def _write_pending(self, lines: list[str]) -> None:
        temp = self._backlog_path.with_suffix(".tmp")
        temp.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        temp.replace(self._backlog_path)

    async def _send(self, body: dict[str, Any], *, replay: bool = False) -> str:
        delivery_id = body.get("metadata", {}).get("rsi_delivery_id")
        memory_id = self._receipts.get(delivery_id)
        if not memory_id and replay and delivery_id:
            response = await self._client.post(f"{self.base_url}/memories/search", json={
                "text": body["text"], "top_k": 1,
                "metadata_filters": {"ns": body["ns"], "rsi_delivery_id": delivery_id}})
            response.raise_for_status()
            for hit in response.json().get("results", []):
                memory = hit.get("memory", hit)
                if memory.get("metadata", {}).get("rsi_delivery_id") == delivery_id:
                    memory_id = memory.get("id")
                    break
        if not memory_id:
            response = await self._client.post(f"{self.base_url}/memories", json=body)
            response.raise_for_status()
            memory_id = response.json()["id"]
            if not isinstance(memory_id, str) or not memory_id:
                raise ValueError("SME 写入缺少记录 ID")
        if delivery_id and delivery_id not in self._receipts:
            with self._receipt_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"key": delivery_id, "memory_id": memory_id}) + "\n")
            self._receipts[delivery_id] = memory_id
        self.online, self._last_error = True, ""
        return memory_id

    async def flush_backlog(self, limit: int = 50) -> int:
        """重试未确认的写入，保留旧版积压格式。"""
        async with self._lock:
            lines = self._pending()
            if not lines or not await self.health():
                return 0
            remaining = []
            flushed = 0
            for i, line in enumerate(lines):
                if i >= limit:
                    remaining.extend(lines[i:])
                    break
                try:
                    await self._send(redact(json.loads(line)["body"]), replay=True)
                    flushed += 1
                except (httpx.HTTPError, ValueError, KeyError, TypeError, OSError):
                    remaining.append(line)
            self._write_pending(remaining)
            return flushed

    async def aclose(self) -> None:
        await self._client.aclose()
