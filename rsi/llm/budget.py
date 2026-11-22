"""本地日预算。模型请求发出前持久记录保守预留。"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any


class LLMBudgetExceeded(RuntimeError):
    """剩余额度不足以覆盖下一次调用的预留。"""


class DailyBudget:
    def __init__(self, config: dict[str, Any], path: Path | None):
        self.config = config
        self.path = path
        self.input_rate = float(config.get("input_usd_per_million", 0.30))
        self.output_rate = float(config.get("output_usd_per_million", 1.20))
        if not all(math.isfinite(rate) and rate > 0 for rate in (self.input_rate, self.output_rate)):
            raise ValueError("模型 token 单价必须是有效正数")
        self.day = time.strftime("%Y-%m-%d")
        self.amount = 0.0
        if path is not None and path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved["day"] == self.day:
                self.amount = float(saved["amount"])
                if not math.isfinite(self.amount) or self.amount < 0:
                    raise ValueError("预算台账金额无效")

    def _write(self, amount: float) -> None:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix(".tmp")
            temp.write_text(json.dumps({"day": self.day, "amount": amount}), encoding="utf-8")
            temp.replace(self.path)
        self.amount = amount

    def spent(self) -> float:
        today = time.strftime("%Y-%m-%d")
        if today != self.day:
            self.day, self.amount = today, 0.0
        return self.amount

    def _price(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in * self.input_rate + tokens_out * self.output_rate) / 1_000_000

    def reserve(self, tokens_in: int, tokens_out: int) -> tuple[str, float]:
        reserved = self._price(tokens_in, tokens_out)
        total = self.spent() + reserved
        cap = float(self.config.get("daily_cost_cap", 5.0))
        if not math.isfinite(cap) or total > cap:
            raise LLMBudgetExceeded("剩余日预算不足以预留本次模型调用")
        self._write(total)
        return self.day, reserved

    def settle(self, reservation: tuple[str, float], usage: Any) -> None:
        day, reserved = reservation
        total = self.spent()
        if self.day != day or not isinstance(usage, dict):
            return
        tokens_in, tokens_out = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if all(type(value) is int and value >= 0 for value in (tokens_in, tokens_out)):
            self._write(max(0.0, total - reserved + self._price(tokens_in, tokens_out)))
