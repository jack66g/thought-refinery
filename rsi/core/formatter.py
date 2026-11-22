"""沉淀器：写三条产品线 jsonl（追加、线程安全）+ 质检元数据。"""
from __future__ import annotations

import hashlib
import json
import threading
from collections import defaultdict, deque
from datetime import datetime
import time
from pathlib import Path
from typing import Any

from .cleaner import redact

_lock = threading.Lock()


def _append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(redact(record), ensure_ascii=False) + "\n")


class Formatter:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir
        self._written: dict[Path, set[str]] = {}

    def _path(self, category: str, line: str) -> Path:
        return self.dataset_dir / category / f"{line}.jsonl"

    @staticmethod
    def _id(prefix: str, content: Any, meta: dict[str, Any]) -> str:
        event_id = meta.get("event_id")
        if not event_id:
            return f"{prefix}-{time.time_ns()}"
        material = json.dumps([event_id, content], ensure_ascii=False).encode()
        return f"{prefix}-{hashlib.sha256(material).hexdigest()[:24]}"

    def _persist(self, path: Path, row: dict[str, Any]) -> None:
        if path not in self._written:
            self._written[path] = {entry["id"] for entry in _rows(path) if "id" in entry}
        if row["id"] not in self._written[path]:
            _append(path, row)
            self._written[path].add(row["id"])

    def find_sft(self, category: str, event_id: str) -> dict[str, Any] | None:
        for row in _rows(self._path(category, "sft")):
            if row.get("meta", {}).get("event_id") == event_id:
                return row
        return None

    def write_sft(self, category: str, task: str, thinking: str, conclusion: str,
                  meta: dict[str, Any]) -> str:
        doc_id = self._id(category, "sft", meta)
        assistant = f"<think>\n{thinking}\n</think>\n\n{conclusion}"
        record = {
            "id": doc_id,
            "messages": [
                {"role": "user", "content": task},
                {"role": "assistant", "content": assistant},
            ],
            "meta": meta,
        }
        self._persist(self._path(category, "sft"), record)
        return doc_id

    def write_lema(self, category: str, task: str, flawed_thinking: str,
                   error_step_desc: str, corrected_thinking: str,
                   conclusion: str, meta: dict[str, Any]) -> str:
        doc_id = self._id(f"{category}-lema", "lema", meta)
        record = {
            "id": doc_id,
            "messages": [
                {"role": "user", "content": task},
                {"role": "assistant",
                 "content": f"<think>\n{flawed_thinking}\n</think>"},
                {"role": "user",
                 "content": f"上面的推理存在问题：{error_step_desc}。请纠正并完成解答。"},
                {"role": "assistant",
                 "content": f"<think>\n{corrected_thinking}\n</think>\n\n{conclusion}"},
            ],
            "meta": meta,
        }
        self._persist(self._path(category, "lema"), record)
        return doc_id

    def write_dpo_negative(self, category: str, task: str, rejected_thinking: str,
                           reject_reason: str, meta: dict[str, Any]) -> str:
        """负样本池：chosen 由同题通过样本事后配对（无配对则不用于训练，仅档案）。"""
        doc_id = self._id(f"{category}-dpo", "dpo", meta)
        record = {
            "id": doc_id,
            "prompt": task,
            "chosen": None,
            "rejected": f"<think>\n{rejected_thinking}\n</think>",
            "meta": {**meta, "reject_reason": reject_reason, "paired": False},
        }
        self._persist(self._path(category, "dpo"), record)
        return doc_id

    def try_pair_dpo(self, category: str, task: str, chosen_text: str) -> int:
        """给同题的未配对负样本回填 chosen。返回配对数。"""
        path = self._path(category, "dpo")
        if not path.exists():
            return 0
        with _lock:
            lines = path.read_text(encoding="utf-8").splitlines()
            changed = 0
            out_lines = []
            for line in lines:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    out_lines.append(line)
                    continue
                if (not rec.get("meta", {}).get("paired") and rec.get("chosen") is None
                        and rec["prompt"] == task):
                    rec["chosen"] = chosen_text
                    rec["meta"]["paired"] = True
                    rec["meta"]["chosen_quality_version"] = 2
                    changed += 1
                out_lines.append(json.dumps(rec, ensure_ascii=False))
            if changed:
                tmp = path.with_suffix(".tmp")
                tmp.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
                tmp.replace(path)
        return changed

    def write_meta_report(self, stats: dict[str, Any]) -> None:
        path = self.dataset_dir / "_meta" / "stats_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    # ---------- V3：预训练线 / 风格SFT线 ----------

    def write_pretrain(self, kind: str, text: str, meta: dict[str, Any]) -> str:
        """kind ∈ transcripts | narratives | rewrites。预训练语料行：纯文本 + meta。"""
        doc_id = self._id(f"pt-{kind}", text, meta)
        self._persist(self.dataset_dir / "pretrain" / f"{kind}.jsonl",
                {"id": doc_id, "text": text, "meta": meta})
        return doc_id

    def write_style(self, messages: list[dict[str, str]], meta: dict[str, Any]) -> str:
        """风格SFT：assistant=用户原话（模型学做用户）。"""
        doc_id = self._id("style", messages, meta)
        self._persist(self.dataset_dir / "style" / "sft.jsonl",
                {"id": doc_id, "messages": messages, "meta": meta})
        return doc_id


PRETRAIN_RECIPE = {
    "混合配方": "个人语料 : 通用语料回放 = 1:4 ~ 1:9（调研 B2：回放 5-15% 即显著防遗忘，保守取高回放）",
    "epoch上限": 4,
    "epoch说明": "arXiv:2305.16264：重复 ≤4 epoch 价值≈新数据，4 后骤减",
    "学习率建议": "5e-6 ~ 5e-5（基座预训峰值 1/10 以下），无 warmup，cosine 衰到底（arXiv:2407.07263）",
    "适配器建议": "底色CPT 用全参或 rank≥128/ReLoRA（LoRA 学得少，2405.09673）；风格SFT 用 rank16-64 LoRA, lr 1e-4~5e-5, epoch 1-3（WeClone 实践）",
    "清洗口径": "保守清洗：仅 PII 脱敏 + 宽松去重(jaccard≥0.9)；句长/停用词/困惑度过滤一概不做（保底色，C4 审计 2104.08758）",
    "记忆化警示": "逐字记忆化在早期 epoch 暴涨且早于验证 loss 恶化（2406.11507）——训练后必须做隐私抽取测试",
}


PRETRAIN_KINDS = ("transcripts", "narratives", "rewrites")
MOODS = ("positive", "negative", "neutral")


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _mood(row: dict[str, Any]) -> str:
    value = row.get("meta", {}).get("mood")
    return value if value in MOODS else "neutral"


def _week(row: dict[str, Any]) -> str:
    meta = row.get("meta", {})
    raw = meta.get("captured_at") or meta.get("ts")
    try:
        year, week, _ = datetime.fromisoformat(raw).isocalendar()
        return f"{year:04}-W{week:02}"
    except (ValueError, TypeError):
        return "unknown"


def _balanced(rows: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, deque]] = {mood: defaultdict(deque) for mood in MOODS}
    for row in sorted(rows, key=lambda r: str(r.get("id", ""))):
        groups[_mood(row)][_week(row)].append(row)
    limits = {mood: min(max(0, cap), sum(map(len, weeks.values()))) for mood, weeks in groups.items()}
    total = sum(limits.values())
    if sum(value > 0 for value in limits.values()) >= 2:
        limits = {mood: min(count, total - count) for mood, count in limits.items()}
    chosen = []
    for mood in MOODS:
        weeks = deque(groups[mood][week] for week in sorted(groups[mood]))
        for _ in range(limits[mood]):
            batch = weeks.popleft()
            chosen.append(batch.popleft())
            if batch:
                weeks.append(batch)
    return chosen


def build_pretrain_manifest(dataset_dir: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = {"lines": {}, "mood_buckets": {}, "capture_weeks": {}, "recipe": PRETRAIN_RECIPE}
    all_rows = []
    for kind in PRETRAIN_KINDS:
        rows = _rows(dataset_dir / "pretrain" / f"{kind}.jsonl")
        manifest["lines"][kind] = {
            "count": len(rows), "chars": sum(len(row.get("text", "")) for row in rows),
            "subjective_fact": sum(bool(row.get("meta", {}).get("subjective_fact")) for row in rows)}
        all_rows.extend(rows)
    for row in all_rows:
        for field, key in (("mood_buckets", _mood(row)), ("capture_weeks", _week(row))):
            counts = manifest[field]
            counts[key] = counts.get(key, 0) + 1
    manifest["lines"]["style_sft"] = {"count": len(_rows(dataset_dir / "style" / "sft.jsonl"))}
    manifest["total_pretrain"] = len(all_rows)
    sampled = _balanced(all_rows, 100)
    buckets = {mood: sum(_mood(row) == mood for row in sampled) for mood in MOODS}
    manifest["balanced_preview"] = {
        "per_bucket_cap": 100, "mood_buckets": buckets,
        "mood_cap_met": bool(sampled) and max(buckets.values()) * 2 <= len(sampled),
        "missing_buckets": [mood for mood, count in buckets.items() if not count],
        "capture_weeks": sorted({_week(row) for row in sampled}),
    }
    path = dataset_dir / "_meta" / "pretrain_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)
    return manifest


def sample_balanced(dataset_dir: Path, per_bucket_cap: int = 100) -> list[dict[str, Any]]:
    """确定性跨周采样。两桶以上可用时，每桶至多占一半。"""
    rows = [row for kind in PRETRAIN_KINDS
            for row in _rows(dataset_dir / "pretrain" / f"{kind}.jsonl")]
    return _balanced(rows, per_bucket_cap)
