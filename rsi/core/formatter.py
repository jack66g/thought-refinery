"""沉淀器：写三条产品线 jsonl（追加、线程安全）+ 质检元数据。"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

_lock = threading.Lock()


def _append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


class Formatter:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir

    def _path(self, category: str, line: str) -> Path:
        return self.dataset_dir / category / f"{line}.jsonl"

    def write_sft(self, category: str, task: str, thinking: str, conclusion: str,
                  meta: dict[str, Any]) -> str:
        doc_id = f"{category}-{int(time.time()*1000)}-{meta.get('event_id', 'x')[:8]}"
        assistant = f"<think>\n{thinking}\n</think>\n\n{conclusion}"
        record = {
            "id": doc_id,
            "messages": [
                {"role": "user", "content": task},
                {"role": "assistant", "content": assistant},
            ],
            "meta": meta,
        }
        _append(self._path(category, "sft"), record)
        return doc_id

    def write_lema(self, category: str, task: str, flawed_thinking: str,
                   error_step_desc: str, corrected_thinking: str,
                   conclusion: str, meta: dict[str, Any]) -> str:
        doc_id = f"{category}-lema-{int(time.time()*1000)}"
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
        _append(self._path(category, "lema"), record)
        return doc_id

    def write_dpo_negative(self, category: str, task: str, rejected_thinking: str,
                           reject_reason: str, meta: dict[str, Any]) -> str:
        """负样本池：chosen 由同题通过样本事后配对（无配对则不用于训练，仅档案）。"""
        doc_id = f"{category}-dpo-{int(time.time()*1000)}"
        record = {
            "id": doc_id,
            "prompt": task,
            "chosen": None,
            "rejected": f"<think>\n{rejected_thinking}\n</think>",
            "meta": {**meta, "reject_reason": reject_reason, "paired": False},
        }
        _append(self._path(category, "dpo"), record)
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
                if (not rec.get("paired") and rec.get("chosen") is None
                        and rec["prompt"] == task):
                    rec["chosen"] = chosen_text
                    rec["meta"]["paired"] = True
                    changed += 1
                out_lines.append(json.dumps(rec, ensure_ascii=False))
            if changed:
                path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        return changed

    def write_meta_report(self, stats: dict[str, Any]) -> None:
        path = self.dataset_dir / "_meta" / "stats_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    # ---------- V3：预训练线 / 风格SFT线 ----------

    def write_pretrain(self, kind: str, text: str, meta: dict[str, Any]) -> str:
        """kind ∈ transcripts | narratives | rewrites。预训练语料行：纯文本 + meta。"""
        doc_id = f"pt-{kind}-{int(time.time()*1000)}-{meta.get('event_id','x')[:6]}"
        _append(self.dataset_dir / "pretrain" / f"{kind}.jsonl",
                {"id": doc_id, "text": text, "meta": meta})
        return doc_id

    def write_style(self, messages: list[dict[str, str]], meta: dict[str, Any]) -> str:
        """风格SFT：assistant=用户原话（模型学做用户）。"""
        doc_id = f"style-{int(time.time()*1000)}-{meta.get('event_id','x')[:6]}"
        _append(self.dataset_dir / "style" / "sft.jsonl",
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


def build_pretrain_manifest(dataset_dir: Path) -> dict[str, Any]:
    """汇总预训练三线清单 + 情绪分桶分布 + 训练配方（拿走即训的说明书）。"""
    manifest: dict[str, Any] = {"lines": {}, "mood_buckets": {}, "recipe": PRETRAIN_RECIPE}
    total = 0
    for kind in ("transcripts", "narratives", "rewrites"):
        path = dataset_dir / "pretrain" / f"{kind}.jsonl"
        rows = []
        if path.exists():
            rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        manifest["lines"][kind] = {"count": len(rows),
                                   "chars": sum(len(r.get("text", "")) for r in rows),
                                   "subjective_fact": sum(1 for r in rows
                                                          if r.get("meta", {}).get("subjective_fact"))}
        total += len(rows)
        for r in rows:
            bucket = r.get("meta", {}).get("mood", "neutral")
            manifest["mood_buckets"][bucket] = manifest["mood_buckets"].get(bucket, 0) + 1
    style_path = dataset_dir / "style" / "sft.jsonl"
    style_rows = []
    if style_path.exists():
        style_rows = [json.loads(x) for x in style_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    manifest["lines"]["style_sft"] = {"count": len(style_rows)}
    manifest["total_pretrain"] = total
    manifest["balance_note"] = ("情绪时间平衡：训练集组装时按 mood_buckets 配额抽样，"
                                "任一桶占比不应超过 50%（防低落期底色污染）")
    path = dataset_dir / "_meta" / "pretrain_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def sample_balanced(dataset_dir: Path, per_bucket_cap: int = 100) -> list[dict[str, Any]]:
    """按情绪桶配额抽样，返回平衡后的预训练行（正负中三桶各 ≤ cap）。"""
    rows: list[dict[str, Any]] = []
    for kind in ("transcripts", "narratives", "rewrites"):
        path = dataset_dir / "pretrain" / f"{kind}.jsonl"
        if path.exists():
            rows.extend(json.loads(x) for x in
                        path.read_text(encoding="utf-8").splitlines() if x.strip())
    by_bucket: dict[str, list[dict[str, Any]]] = {"positive": [], "negative": [], "neutral": []}
    for r in rows:
        by_bucket[r.get("meta", {}).get("mood", "neutral")].append(r)
    out: list[dict[str, Any]] = []
    for bucket, items in by_bucket.items():
        out.extend(items[:per_bucket_cap])
    return out
