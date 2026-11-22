"""验证器：硬验证（sympy 表达式等价）与软验证（自一致性投票）。"""
from __future__ import annotations

import re
from typing import Any

from ..llm.client import LLMClient
from . import prompts
from .cleaner import normalize_width

try:  # sympy 是硬验证依赖，缺失时显式报错而不是假装通过
    import sympy  # noqa: F401
    _HAS_SYMPY = True
except ImportError:  # pragma: no cover
    _HAS_SYMPY = False


def _norm_answer(ans: str) -> str:
    ans = normalize_width(str(ans)).strip()
    ans = re.sub(r"[\s$，,。．；;！!？?：:、（）()\[\]【】]+", "", ans)
    ans = ans.replace("×", "*").replace("÷", "/").replace("^", "**")
    ans = re.sub(r"答案是为?", "", ans)
    return ans


def math_equal(a: str, b: str) -> bool:
    """sympy 等价：先串等，再符号等价，再数值等价。"""
    na, nb = _norm_answer(a), _norm_answer(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if not _HAS_SYMPY:
        raise RuntimeError("sympy 未安装：无法硬验证（pip install sympy）")
    try:
        expr_a = sympy.sympify(na, evaluate=True)
        expr_b = sympy.sympify(nb, evaluate=True)
        if sympy.simplify(expr_a - expr_b) == 0:
            return True
    except (sympy.SympifyError, TypeError, ValueError):
        pass
    try:  # 数值兜底
        fa, fb = float(sympy.N(na)), float(sympy.N(nb))
        return abs(fa - fb) <= 1e-9 * max(1.0, abs(fa), abs(fb))
    except (ValueError, TypeError, sympy.SympifyError):
        return False


def extract_answer(text: str) -> str:
    """从思考/结论文本提取"答案是 X"的 X。"""
    m = re.search(r"答案是[：:\s]*(.+)", text)
    if m:
        return m.group(1).strip().strip("。，.；;！!？?")
    m = re.search(r"(?:最终答案|答案为|答案|结论)[：:\s=]+(.+)", text)
    return m.group(1).strip().strip("。，.；;！!？?") if m else ""


async def hard_verify(conclusion: str, expected: str, answer_type: str) -> bool:
    """硬验证：self_proposed 事件有出题者的标准答案。"""
    got = extract_answer(conclusion) or conclusion
    if answer_type == "short_text":
        return _norm_answer(got) == _norm_answer(expected)
    # 数值型：从"50个/共50"这类表述中取首个数字再比（防单位尾巴误杀）
    if re.fullmatch(r"[-+]?[\d.]+", _norm_answer(expected)):
        m = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", _norm_answer(got))
        if m:
            return math_equal(m.group(0).replace(",", ""), expected)
    return math_equal(got, expected)


def _fuzzy_same(a: str, b: str) -> bool:
    """模糊一致：相等 / 包含 / 序列相似度≥0.7（自由文本结论措辞会漂移，精确相等太脆）。"""
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio() >= 0.7


async def soft_verify(llm: LLMClient, task: str, thinking: str, conclusion: str,
                      cfg: dict[str, Any]) -> tuple[bool, float]:
    """软验证：自一致性——独立采样 N 次结论，模糊一致比例达标即过。"""
    n = int(cfg.get("self_consistency_n", 3))
    min_agree = int(cfg.get("self_consistency_min_agree", 2))
    system = ("[ROLE:answer_probe] 给定任务与一段思考，从中提取最终结论并**逐字引用**"
              "（数值/表达式/结论原句），不要改写措辞，不要任何解释。")
    answers = []
    for _ in range(n):
        out = await llm.chat(system, f"任务：{task}\n思考：{thinking}\n（已知参照结论：{conclusion}）")
        answers.append(_norm_answer(extract_answer(out) or out))
    target = _norm_answer(extract_answer(conclusion) or conclusion)
    agree = sum(1 for a in answers if _fuzzy_same(a, target))
    return agree >= min_agree, agree / max(n, 1)
