"""自探索任务生成器（V3-T9）——默认关、独立开关、日封顶。

需求层缺口 → 任务类型 → 生成宿主可执行的工程任务（宿主经 rsi_next_mission
MCP 工具领取，或外部调度器打 /next_mission）。任务完成后的思考链经
rsi_distill 自然回流 CoT 线——闭环。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..biodata.state import BioData
from ..core import roles
from ..llm.client import LLMClient, LLMNotConfigured
from ..sme_wrap.client import SMEClient


class Explorer:
    def __init__(self, llm: LLMClient, biodata: BioData, sme: SMEClient,
                 cfg: dict[str, Any], rsi_base: Path):
        self.llm = llm
        self.biodata = biodata
        self.sme = sme
        self.cfg = cfg.get("self_explore", {})
        self.log_path = rsi_base / "logs" / "explore.jsonl"
        self.stats_path = rsi_base / "logs" / "stats.json"

    def _today_count(self) -> int:
        today = time.strftime("%Y-%m-%d")
        if not self.log_path.exists():
            return 0
        n = 0
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            try:
                if json.loads(line).get("date") == today:
                    n += 1
            except json.JSONDecodeError:
                continue
        return n

    def _anchors(self) -> str:
        anchors = ["（暂无已完成任务记录）"]
        if self.stats_path.exists():
            try:
                stats = json.loads(self.stats_path.read_text(encoding="utf-8"))
                anchors = [f"已收编 CoT 样本 {stats.get('accepted_sft', 0)} 条；"
                           f"平均思考长度 {stats.get('avg_think_chars', '暂无')} 字符"]
            except json.JSONDecodeError:
                pass
        return "；".join(anchors)

    async def next_mission(self) -> dict[str, Any]:
        if not self.cfg.get("enabled", False):
            return {"enabled": False, "message": "自探索默认关闭（self_explore.enabled）"}
        cap = int(self.cfg.get("max_missions_per_day", 5))
        if self._today_count() >= cap:
            return {"enabled": True, "capped": True,
                    "message": f"今日任务数已达上限 {cap}"}
        self.biodata.tick()
        try:
            mission = await roles.explore_mission(
                self.llm, self.biodata.persona_card(), self._anchors())
        except LLMNotConfigured as exc:
            return {"enabled": True, "error": str(exc)}
        mission["date"] = time.strftime("%Y-%m-%d")
        mission["ts"] = time.strftime("%F %T")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(mission, ensure_ascii=False) + "\n")
        await self.sme.add(
            f"自探索任务({mission.get('type')})：{str(mission.get('mission'))[:80]}",
            ns="rsi.biodata", tags=["mission"],
            metadata={"date": mission["date"]})
        return {"enabled": True, "mission": mission}
