"""选择模式的行为维度与保守的个人内估计。"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

AXES = {
    "risk": ("风险余地", "控制风险", "接受可承受的风险"),
    "pace": ("行动节奏", "先小步试验", "先补全信息"),
    "autonomy": ("协作方式", "独立推进", "尽早协商"),
    "novelty": ("探索偏好", "尝试新路径", "沿用熟悉路径"),
    "planning": ("计划弹性", "先定清晰计划", "随情况灵活调整"),
}
CONTEXTS = ("项目与工作", "日常安排", "学习与尝试", "协作与沟通")


def estimates(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(rows)
    results = []
    for key, (label, side_a, side_b) in AXES.items():
        evidence = [row for row in rows if row["axis"] == key]
        contexts = len({row["context"] for row in evidence})
        direction = sum({"a": 1.0, "b": 0.0, "mixed": 0.5}[row["decision"]] for row in evidence)
        score = None
        if len(evidence) >= 3 and contexts >= 2:
            score = round(100 * (direction + 2) / (len(evidence) + 4))
        results.append({"key": key, "label": label, "a": side_a, "b": side_b,
                        "observations": len(evidence), "contexts": contexts, "a_share": score})
    return results
