"""RSI MCP 插件本体（stdio）：挂载进 ZCode / OpenCode 等任意 MCP 宿主。

工具是 daemon HTTP 的薄转发；daemon 不在则自动拉起失败并给出明确错误。
宿主挂载示例（与挂 SME 同款，一行 JSON）：
  {"mcpServers": {"rsi": {"command": "<venv-python>", "args": ["<RSI>/rsi/mcp_server.py"]}}}

注意：本文件不做任何宿主配置修改——挂载由宿主侧自行决定。
"""
from __future__ import annotations

import os

import httpx
from mcp.server.mcpserver import MCPServer

DAEMON_URL = os.environ.get("RSI_DAEMON_URL", "http://127.0.0.1:8100")

mcp = MCPServer("rsi")


def _post(path: str, json_body: dict | None = None) -> str:
    try:
        resp = httpx.post(f"{DAEMON_URL}{path}", json=json_body, timeout=30)
        resp.raise_for_status()
        return resp.text
    except httpx.HTTPError as exc:
        return f"错误：daemon 不可达或报错（{exc}）。请确认 daemon 已启动：python -m rsi.daemon.main"


def _get(path: str) -> str:
    try:
        resp = httpx.get(f"{DAEMON_URL}{path}", timeout=30)
        resp.raise_for_status()
        return resp.text
    except httpx.HTTPError as exc:
        return f"错误：daemon 不可达（{exc}）"


@mcp.tool()
def rsi_distill(task: str, thinking: str, conclusion: str = "",
                source: str = "mcp") -> str:
    """投喂一段思考进提纯队列（主入口）。task=任务/问题，thinking=思考轨迹，
    conclusion=最终结论。异步提纯，立即返回回执。"""
    return _post("/distill", {"task": task, "thinking": thinking,
                              "conclusion": conclusion, "source": source})


@mcp.tool()
def rsi_status() -> str:
    """查看循环状态：队列深度、各分类累计、接受率、API 成本、SME 在线状态。"""
    return _get("/status")


@mcp.tool()
def rsi_query(category: str = "", k: int = 3) -> str:
    """按类抽样查看已沉淀的训练数据（category ∈ math/chinese/code/reasoning/misc）。"""
    return _get(f"/query?category={category}&k={k}")


@mcp.tool()
def rsi_lessons(k: int = 5) -> str:
    """读当前生效的提纯教训（improver 学到了什么）。"""
    return _get(f"/lessons?k={k}")


@mcp.tool()
def rsi_set_budget(daily_cost_cap: float | None = None,
                   self_propose_enabled: bool | None = None,
                   self_explore_enabled: bool | None = None,
                   biodata_enabled: bool | None = None) -> str:
    """调预算/暂停/恢复自驱出题、自探索、仿生画像。"""
    body: dict = {}
    if daily_cost_cap is not None:
        body["daily_cost_cap"] = daily_cost_cap
    if self_propose_enabled is not None:
        body["self_propose_enabled"] = self_propose_enabled
    if self_explore_enabled is not None:
        body["self_explore_enabled"] = self_explore_enabled
    if biodata_enabled is not None:
        body["biodata_enabled"] = biodata_enabled
    return _post("/set_budget", body)


@mcp.tool()
def rsi_next_mission() -> str:
    """自探索：领取一个由你的画像需求驱动的小任务（默认关，先 rsi_set_budget 打开）。
    完成任务后把思考轨迹用 rsi_distill 投喂回来即闭环。"""
    return _get("/next_mission")


if __name__ == "__main__":
    mcp.run(transport="stdio")
