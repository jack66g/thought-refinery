# -*- coding: utf-8 -*-
"""MCP 插件独立冒烟：不挂任何真实宿主，用官方 SDK 客户端 + 桩 daemon 驱动 stdio server。"""
import asyncio
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

MCP_SERVER = str(Path(__file__).resolve().parent.parent / "mcp_server.py")


class _StubDaemon(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, body):
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/status"):
            self._send(200, {"running": True, "queue_len": 0, "sme_online": True})
        elif self.path.startswith("/query"):
            self._send(200, {"math": [], "misc": []})
        elif self.path.startswith("/lessons"):
            self._send(200, {"lessons": ["审计员：示例教训"]})
        elif self.path.startswith("/next_mission"):
            self._send(200, {"enabled": False, "message": "自探索默认关闭"})
        else:
            self._send(404, {})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        if self.path in ("/distill", "/set_budget"):
            self._send(200, {"queued": True, "event_id": "evt-stub"})
        else:
            self._send(404, {})


def test_mcp_stdio_tools():
    server = HTTPServer(("127.0.0.1", 0), _StubDaemon)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        result = asyncio.run(_drive(port))
    finally:
        server.shutdown()
    assert "rsi_distill" in result["tools"]
    assert result["status_ok"]
    assert result["distill_queued"]


async def _drive(port: int) -> dict:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = {**os.environ, "RSI_DAEMON_URL": f"http://127.0.0.1:{port}",
           "PYTHONIOENCODING": "utf-8"}
    params = StdioServerParameters(command=sys.executable, args=[MCP_SERVER], env=env)
    async with asyncio.timeout(60):
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                names = {t.name for t in tools.tools}
                assert {"rsi_distill", "rsi_status", "rsi_query",
                        "rsi_lessons", "rsi_set_budget", "rsi_next_mission"} <= names

                mission_res = await session.call_tool("rsi_next_mission", {})
                assert "自探索默认关闭" in mission_res.content[0].text

                status_res = await session.call_tool("rsi_status", {})
                status_text = status_res.content[0].text
                status_ok = '"running": true' in status_text.lower()

                distill_res = await session.call_tool(
                    "rsi_distill",
                    {"task": "冒烟任务", "thinking": "一段足够长的冒烟思考" * 5,
                     "conclusion": "冒烟结论", "source": "test"})
                distill_text = distill_res.content[0].text
                queued = "queued" in distill_text and "错误" not in distill_text
                return {"tools": names, "status_ok": status_ok,
                        "distill_queued": queued}


if __name__ == "__main__":
    test_mcp_stdio_tools()
    print("MCP 冒烟通过")
