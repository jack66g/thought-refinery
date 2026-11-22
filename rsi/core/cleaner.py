"""廉价体检与清洗：纯规则、零 LLM 成本（阈值来源：调研规格书 v1.0，工程经验值已标注）。

输出 HealthReport（结构化），任何 drop 都带原因——弃单进台账，不静默。
PII 只对"进入数据集的文本"脱敏，原始轨迹留在 logs（BigScience/FineWeb 口径：
检测→替换，不删文档）。
"""
from __future__ import annotations

import re
import zlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone_cn": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    "id_card": re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
    "ipv4": re.compile(
        r"(?<!\d)(25[0-5]|2[0-4]\d|1?\d?\d)(\.(25[0-5]|2[0-4]\d|1?\d?\d)){3}(?!\d)"),
    "api_key": re.compile(r"(sk|ghp|gho|AKIA|AIza)[A-Za-z0-9_\-]{16,}"),
}

_LOOP_MARKER = re.compile(r"等等|让我再|换个思路|重新想|不对，")


def normalize_width(text: str) -> str:
    out = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:
            out.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:
            out.append(chr(code - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


def strip_think_tags(text: str) -> str:
    """去掉 <think></think> 包裹，保留内部文本。"""
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    return text.strip()


def mask_pii(text: str) -> str:
    text = _PATTERNS["api_key"].sub("[密钥已脱敏]", text)
    text = _PATTERNS["id_card"].sub("[身份证已脱敏]", text)
    text = _PATTERNS["phone_cn"].sub("[手机号已脱敏]", text)
    text = _PATTERNS["email"].sub("[邮箱已脱敏]", text)
    text = _PATTERNS["ipv4"].sub("[IP已脱敏]", text)
    return text


@dataclass
class HealthReport:
    ok: bool = True
    drop_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


def ngram_repeat_ratio(text: str, n: int = 10) -> float:
    """文档内部重复率：重复出现的 n-gram 占比（char 级，中文友好）。"""
    if len(text) <= n:
        return 0.0
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    counts = Counter(grams)
    repeated = sum(c for c in counts.values() if c > 1)
    return repeated / len(grams)


def compression_ratio(text: str) -> float:
    """zlib 压缩比作信息密度代理（规格书 R4：正常中文散文 0.35-0.60）。"""
    raw = text.encode("utf-8")
    if not raw:
        return 1.0
    return len(zlib.compress(raw, 6)) / len(raw)


def language_mix_ratio(text: str) -> float:
    """次要语言占比：中文与英文词都在场时，少的一方占比；0=纯单语。"""
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_words = len(re.findall(r"[A-Za-z]+", text))
    total = cjk + latin_words
    if total == 0 or cjk == 0 or latin_words == 0:
        return 0.0
    return min(cjk, latin_words) / total


def looks_truncated(text: str) -> bool:
    """截断启发式：悬空标点/未闭合括号。"""
    t = text.rstrip()
    if not t:
        return True
    if t[-1] in "，、的是和与及或（(":
        return True
    for open_c, close_c in (("（", "）"), ("(", ")"), ("【", "】"), ("[", "]")):
        if t.count(open_c) != t.count(close_c):
            return True
    return False


def health_check(task: str, thinking: str, conclusion: str,
                 cfg: dict[str, Any]) -> HealthReport:
    """廉价体检。任何一项不过 → ok=False 并记录原因。"""
    rep = HealthReport()
    n = int(cfg.get("ngram_n", 10))
    max_repeat = float(cfg.get("max_repeat_ratio", 0.3))
    minor_limit = float(cfg.get("minor_lang_ratio", 0.35))
    max_chars = int(cfg.get("max_total_chars", 120000))
    min_compress = float(cfg.get("min_compress_ratio", 0.30))
    max_loops = int(cfg.get("max_loop_markers", 8))

    think_clean = strip_think_tags(thinking)
    rep.metrics["think_chars"] = len(think_clean)
    rep.metrics["repeat_ratio"] = round(ngram_repeat_ratio(think_clean, n), 4)
    rep.metrics["lang_mix"] = round(language_mix_ratio(think_clean), 4)
    rep.metrics["compress_ratio"] = round(compression_ratio(think_clean), 4)
    rep.metrics["loop_markers"] = len(_LOOP_MARKER.findall(think_clean))

    if not task.strip():
        rep.drop_reasons.append("空任务")
    if len(think_clean) < 50:
        rep.drop_reasons.append(f"思考过短({len(think_clean)}字符)")
    if len(task) + len(thinking) + len(conclusion) > max_chars:
        rep.drop_reasons.append("总长超限")
    if not conclusion.strip() and "答案是" not in think_clean:
        rep.drop_reasons.append("无结论")
    if rep.metrics["repeat_ratio"] > max_repeat:
        rep.drop_reasons.append(f"内部重复率{rep.metrics['repeat_ratio']:.2f}>{max_repeat}")
    if rep.metrics["compress_ratio"] < min_compress:
        rep.drop_reasons.append(f"压缩比{rep.metrics['compress_ratio']:.2f}<{min_compress}(疑似复读)")
    if rep.metrics["lang_mix"] > minor_limit:
        rep.drop_reasons.append(f"语言混杂{rep.metrics['lang_mix']:.2f}>{minor_limit}")
    if rep.metrics["loop_markers"] > max_loops:
        rep.drop_reasons.append(f"思考循环标记{rep.metrics['loop_markers']}>{max_loops}")
    if looks_truncated(think_clean) and looks_truncated(conclusion or task):
        rep.drop_reasons.append("疑似截断")

    rep.ok = not rep.drop_reasons
    return rep


def clean_for_dataset(task: str, thinking: str, conclusion: str) -> tuple[str, str, str]:
    """进入数据集前的文本处理：宽度归一 + PII 脱敏。"""
    return (mask_pii(normalize_width(task)).strip(),
            mask_pii(normalize_width(strip_think_tags(thinking))).strip(),
            mask_pii(normalize_width(conclusion)).strip())
