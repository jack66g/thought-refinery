"""冻结 RSI 三线训练输入。运行方式：python -m rsi.core.export --help。"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .cleaner import redact
from .dedup import normalize_text
from .formatter import PRETRAIN_KINDS, _balanced, _rows
from .roles import CATEGORIES

_COT = re.compile(r"\s*<think>\s*\S[\s\S]*?</think>\s*\S[\s\S]*")


def _target(messages: Any) -> int | None:
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    for message in messages:
        if (not isinstance(message, dict) or message.get("role") not in ("user", "assistant", "system")
                or not isinstance(message.get("content"), str) or not message["content"].strip()):
            return None
    return len(messages) - 1 if messages[-1]["role"] == "assistant" else None


def _eligible(product: str, row: dict[str, Any]) -> bool:
    meta = row.get("meta", {})
    if meta.get("quality_version") != 2 or not meta.get("event_id"):
        return False
    if product == "pretrain":
        return (isinstance(row.get("text"), str) and bool(row["text"].strip())
                and (not meta.get("subjective_fact") or bool(meta.get("evidence"))))
    if product == "dpo":
        return (isinstance(row.get("prompt"), str) and bool(row["prompt"].strip())
                and meta.get("paired") is True and meta.get("chosen_quality_version") == 2
                and isinstance(row.get("chosen"), str) and bool(_COT.fullmatch(row["chosen"]))
                and isinstance(row.get("rejected"), str) and row["chosen"] != row["rejected"])
    target = _target(row.get("messages"))
    if product == "personality_choice":
        return (target == 3 and row["messages"][0]["role"] == "user"
                and row["messages"][1]["role"] == "assistant" and row["messages"][2]["role"] == "user"
                and meta.get("synthetic_scenario") is True
                and meta.get("target_source") == "confirmed_model_option")
    return target is not None and (
        product == "style" or bool(_COT.fullmatch(row["messages"][target]["content"])))


def _groups(rows: list[tuple[str, dict[str, Any]]]) -> dict[str, str]:
    parents: dict[str, str] = {}

    def find(key: str) -> str:
        parents.setdefault(key, key)
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    docs = {row.get("id"): row["meta"]["event_id"] for _, row in rows}
    for product, row in rows:
        meta = row["meta"]
        event = find("event:" + meta["event_id"])
        related = []
        if meta.get("session_id"):
            related.append("session:" + meta["session_id"])
        if meta.get("rewrite_of") in docs and meta.get("rewrite_of"):
            related.append("event:" + docs[meta["rewrite_of"]])
        if product in ("cot", "lema", "dpo"):
            task = row["prompt"] if product == "dpo" else row["messages"][0]["content"]
            related.append("task:" + normalize_text(task))
        for key in related:
            previous = find(key)
            winner, child = sorted((event, previous))
            parents[child] = winner
            event = winner
    return {row["meta"]["event_id"]: find("event:" + row["meta"]["event_id"]) for _, row in rows}


def freeze(source: Path, destination: Path, validation_percent: int = 20,
           per_bucket_cap: int = 100) -> dict[str, Any]:
    if not 0 <= validation_percent < 100:
        raise ValueError("validation_percent 必须在 0 到 99 之间")
    if destination.exists():
        raise FileExistsError(destination)
    paths = [("style", source / "style" / "sft.jsonl"),
             ("personality_choice", source / "style" / "choices.jsonl")]
    paths.extend((product, source / category / f"{name}.jsonl")
                 for category in CATEGORIES for product, name in (("cot", "sft"), ("lema", "lema"), ("dpo", "dpo")))
    # 先从通过质量版本门的 CPT 记录分层取样，旧版记录不占用配额。
    cpt = [row for kind in PRETRAIN_KINDS for row in _rows(source / "pretrain" / f"{kind}.jsonl")]
    proposed = [(product, row) for product, path in paths for row in _rows(path)]
    cpt_choice_ids = {row["meta"]["event_id"] for row in cpt
                      if _eligible("pretrain", row) and row["meta"].get("source") == "choice_mode"}
    sft_choice_ids = {row["meta"]["event_id"] for product, row in proposed
                      if product == "personality_choice" and _eligible(product, row)}
    complete_choices = cpt_choice_ids & sft_choice_ids
    unique_cpt = {}
    for row in cpt:
        if (_eligible("pretrain", row) and (row["meta"].get("source") != "choice_mode"
                                           or row["meta"]["event_id"] in complete_choices)):
            unique_cpt.setdefault(normalize_text(row["text"]), row)
    selected = _balanced(list(unique_cpt.values()), per_bucket_cap)
    rows = [(product, redact(row)) for product, row in proposed if _eligible(product, row)
            and (product != "personality_choice" or row["meta"]["event_id"] in complete_choices)]
    rows.extend(("pretrain", redact(row)) for row in selected)
    groups = _groups(rows)
    files: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for product, row in rows:
        group = groups[row["meta"]["event_id"]]
        bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
        split = "validation" if bucket < validation_percent else "train"
        row["meta"]["split_group"] = hashlib.sha256(group.encode()).hexdigest()[:16]
        if product in ("cot", "lema", "style"):
            row["meta"]["train_message_indices"] = [_target(row["messages"])]
        elif product == "personality_choice":
            row["meta"]["train_message_indices"] = [1, 3]
        files.setdefault((split, product), []).append(row)
    manifest = {
        "schema_version": 2,
        "counts": {f"{split}/{product}": len(content) for (split, product), content in sorted(files.items())},
        "rejected_schema_or_gate": len(proposed) + len(cpt)
        - sum(_eligible(product, row) for product, row in proposed)
        - sum(_eligible("pretrain", row) for row in cpt),
        "selected_pretrain_moods": dict(Counter(row["meta"].get("mood", "neutral") for row in selected)),
        "unpaired_choice_rounds": len(cpt_choice_ids ^ sft_choice_ids),
        "validation_percent": validation_percent,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as temp:
        output = Path(temp) / "snapshot"
        output.mkdir()
        for (split, product), content in files.items():
            directory = output / split
            directory.mkdir(exist_ok=True)
            with (directory / f"{product}.jsonl").open("w", encoding="utf-8") as stream:
                for record in content:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
        output.rename(destination)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-percent", type=int, default=20)
    parser.add_argument("--per-bucket-cap", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(freeze(args.dataset, args.output, args.validation_percent, args.per_bucket_cap),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
