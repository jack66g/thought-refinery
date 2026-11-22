"""验证结论，保留算式的运算顺序。"""
from __future__ import annotations

import ast
import json
import operator
import re
from typing import Any

import sympy
from sympy.ntheory.modular import solve_congruence

from ..llm.client import LLMClient
from .cleaner import normalize_width

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv}
_NUMBER = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d{1,2})?")
_TASK = re.compile(
    r"(?:请)?(?:计算|求值|求)\s*[:：]?\s*([\d+\-*/^×÷().\s]+?)\s*(?:的值)?[。.!！?？]?")
_INTEGERS = r"[0-9]{1,9}(?:\s*[,、]\s*[0-9]{1,9}){1,8}"
_CONGRUENCES = re.compile(
    rf"(?:求|计算)(?:同时满足)?\s*[a-zA-Z]?\s*除以\s*({_INTEGERS})\s*分别余\s*({_INTEGERS})"
    r"\s*的最小正整数\s*[a-zA-Z]?\s*(?:,\s*并证明不存在更小的正整数解)?[。.?？]?")


def _norm_answer(value: str) -> str:
    value = normalize_width(str(value)).strip().strip("$").strip()
    value = re.sub(r"^答案是为?", "", value).strip().rstrip("。，,；;！!？?")
    return re.sub(r"\s+", "", value).replace("×", "*").replace("÷", "/").replace("^", "**")


def _expression(text: str) -> sympy.Expr | None:
    """在有界表达式语法内构造数值或符号表达式。"""
    if not text or len(text) > 128:
        return None

    def evaluate(node: ast.AST) -> sympy.Expr:
        if isinstance(node, ast.Constant):
            literal = ast.get_source_segment(text, node) or ""
            if _NUMBER.fullmatch(literal) and len(literal) <= 30:
                return sympy.Rational(literal)
        if isinstance(node, ast.Name):
            if node.id in ("pi", "e"):
                return {"pi": sympy.pi, "e": sympy.E}[node.id]
            if re.fullmatch(r"[A-Za-z]", node.id):
                return sympy.Symbol(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            if type(node.op) in _OPS:
                return _OPS[type(node.op)](left, right)
            if isinstance(node.op, ast.Pow) and right.is_Integer and abs(right) <= 12:
                if len(str(left)) <= 96:
                    return left ** right
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "sqrt" and len(node.args) == 1 and not node.keywords):
            return sympy.sqrt(evaluate(node.args[0]))
        raise ValueError("unsupported expression")

    try:
        root = ast.parse(text, mode="eval")
        if sum(1 for _ in ast.walk(root)) > 48:
            return None
        value = evaluate(root.body)
        return value if not value.has(sympy.nan, sympy.zoo, sympy.oo, -sympy.oo) else None
    except (SyntaxError, ValueError, TypeError, ArithmeticError, RecursionError):
        return None


def math_equal(a: str, b: str) -> bool:
    ea, eb = _expression(_norm_answer(a)), _expression(_norm_answer(b))
    if ea is None or eb is None:
        return False
    diff = sympy.simplify(ea - eb)
    if diff == 0:
        return True
    if not (ea.is_number and eb.is_number) or (ea.is_Integer and eb.is_Integer):
        return False
    try:
        return abs(float(diff)) <= 1e-9
    except (ValueError, TypeError, OverflowError):
        return False


def deterministic_answer(task: str) -> str | None:
    task = normalize_width(task).strip()
    match = _TASK.fullmatch(task)
    if match:
        value = _expression(_norm_answer(match.group(1)))
        if value is not None and value.is_number:
            return str(value)
    congruences = _CONGRUENCES.fullmatch(task)
    if congruences:
        moduli, residues = [list(map(int, re.split(r"[,、]", group))) for group in congruences.groups()]
        if len(moduli) == len(residues) and all(0 <= r < m for m, r in zip(moduli, residues)):
            result = solve_congruence(*zip(residues, moduli))
            return str(result[0] or result[1]) if result is not None else "无解"
    return None


def extract_answer(text: str) -> str:
    matches = re.findall(r"(?:答案是[：:\s]*|(?:最终答案|答案为|答案|结论)[：:\s=]+)([^\n]+)", text)
    return matches[-1].strip().strip("。，.；;！!？?") if matches else ""


def answers_agree(a: str, b: str) -> bool:
    a = _norm_answer(extract_answer(a) or a)
    b = _norm_answer(extract_answer(b) or b)
    if not a or not b:
        return False
    if _expression(a) is not None or _expression(b) is not None:
        return math_equal(a, b)
    return a.rstrip(".") == b.rstrip(".")


async def hard_verify(conclusion: str, expected: str, answer_type: str) -> bool:
    got = extract_answer(conclusion) or conclusion
    if answer_type == "short_text":
        return bool(_norm_answer(got)) and _norm_answer(got) == _norm_answer(expected)
    return math_equal(got, expected)


async def soft_verify(llm: LLMClient, task: str, thinking: str, conclusion: str,
                      cfg: dict[str, Any]) -> tuple[bool, float]:
    """独立求解只接收题面，参考结论在本地比较。"""
    independent = deterministic_answer(task)
    if independent is not None:
        passed = answers_agree(independent, conclusion)
        return passed, float(passed)
    count = max(1, int(cfg.get("self_consistency_n", 3)))
    minimum = max(1, int(cfg.get("self_consistency_min_agree", 2)))
    system = ('[ROLE:answer_probe] 独立解题并核算。最后一行输出 JSON 对象 {"answer": "..."}。'
              'answer 只放最终结果。单个数值只放数值，不加变量名、单位或公式包装。'
              '条件不足时 answer 填 INSUFFICIENT。')
    request = f"任务：{task}\n请独立核算，最后一行返回结果 JSON。"
    agree = 0
    for _ in range(count):
        answer = (await llm.chat(system, request, max_tokens=1536)).strip()
        candidate = answer
        if match := re.search(r'\{\s*"answer"\s*:[^{}]*\}', answer):
            try:
                value = json.loads(match.group())["answer"]
                candidate = str(value) if type(value) in (str, int, float) else ""
            except (ValueError, KeyError):
                candidate = ""
        if candidate != "INSUFFICIENT" and answers_agree(candidate, conclusion):
            agree += 1
    return agree >= minimum, agree / count
