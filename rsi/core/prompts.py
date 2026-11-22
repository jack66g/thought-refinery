"""种子提示词加载器：rsi/prompts/v0/*.md，{var} 占位符 format 填充。"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts" / "v0"


@lru_cache(maxsize=32)
def _load_raw(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"种子提示词缺失: {path}")
    return path.read_text(encoding="utf-8")


def render(name: str, **vars_: str) -> tuple[str, str]:
    """返回 (system, user)。system = 首行 [ROLE:x] 起到第一个空行；其余为 user 模板。"""
    raw = _load_raw(name)
    raw = raw.replace("{", "{{").replace("}", "}}")
    # 还原我们自己的占位符（支持 list/dict 传入，str() 化）
    for key in vars_:
        raw = raw.replace("{{" + key + "}}", "{" + key + "}")
    body = raw.format(**{k: (v if isinstance(v, str) else str(v)) for k, v in vars_.items()})
    lines = body.splitlines()
    system = lines[0].strip()
    user = "\n".join(lines[1:]).strip()
    return system, user
