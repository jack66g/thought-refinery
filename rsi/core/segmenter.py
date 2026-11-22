"""思考分段：按句群聚段（约 200 字/段），供审计员逐段检查与 FCS 截断。"""
from __future__ import annotations

import re

_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;\n])")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    return parts


def segment(thinking: str, max_chars: int = 200) -> list[str]:
    """句群聚段：句子顺序累加，段长达到 max_chars 即封段。"""
    sentences = split_sentences(thinking)
    segments: list[str] = []
    buf: list[str] = []
    size = 0
    for sent in sentences:
        buf.append(sent)
        size += len(sent)
        if size >= max_chars:
            segments.append("".join(buf))
            buf, size = [], 0
    if buf:
        segments.append("".join(buf))
    return segments


def format_segments(segments: list[str]) -> str:
    return "\n".join(f"[{i}] {seg}" for i, seg in enumerate(segments, start=1))


def find_conclusion_segment(segments: list[str]) -> int | None:
    """定位首个'给出最终答案'的段（FCS 锚点）。返回段号（1 起）。"""
    for i, seg in enumerate(segments, start=1):
        if re.search(r"答案是|最终答案|答案为|结论[是为：:]", seg):
            return i
    return None


def fcs_truncate(segments: list[str], keep_reflection: bool = True) -> list[str]:
    """FCS / FCS+Reflection：首正确段后截断，可选再保留一个后续段。"""
    anchor = find_conclusion_segment(segments)
    if anchor is None:
        return segments
    end = min(anchor + (1 if keep_reflection else 0), len(segments))
    return segments[:end]
