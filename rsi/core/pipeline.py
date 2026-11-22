"""提纯管线：单个事件的全生命周期（判类→查重→价值门→审计→修复→FCS→净化→验证→沉淀）。

所有 drop 都带原因写入 logs/funnel.jsonl（门控漏斗），不静默。
"""
from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from ..llm.client import LLMClient
from ..sme_wrap.client import SMEClient
from . import roles
from .cleaner import clean_for_dataset, health_check, looks_truncated, mask_pii, redact, strip_think_tags
from .dedup import DedupConfig, DedupIndex
from .formatter import Formatter
from .segmenter import fcs_truncate, segment
from .verifier import answers_agree, deterministic_answer, extract_answer, hard_verify, soft_verify

# 审计错误类型 → 可修性（设计书 §3.5）
_REPAIRABLE = {"跳步", "计算错误", "内部矛盾", "表述不严谨"}

_DEAD_WAIT = re.compile(
    r"^(等等|嗯+|wait|不对|唉)[，,。！!]?\s*(让我|我)?(再)?(想|看看|检查|算|试试)?",
    re.IGNORECASE)


def remove_dead_waits(text: str) -> str:
    """删空转：删掉无信息量的回头句（句内无数字/无新计算的自我怀疑）。"""
    from .segmenter import split_sentences
    kept = []
    for sent in split_sentences(text):
        if _DEAD_WAIT.match(sent) and not re.search(r"\d", sent):
            continue
        kept.append(sent)
    return "".join(kept)


class Pipeline:
    def __init__(self, llm: LLMClient, sme: SMEClient, cfg: dict[str, Any],
                 rsi_base: Path):
        self.llm = llm
        self.sme = sme
        self.cfg = cfg
        self.dataset_dir = rsi_base / "dataset"
        self.logs_dir = rsi_base / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.formatter = Formatter(self.dataset_dir)
        self.dedup = DedupIndex(
            base_dir=self.dataset_dir,
            cfg=DedupConfig(**{k: v for k, v in cfg.get("dedup", {}).items()
                               if k in DedupConfig.__dataclass_fields__}),
        )
        self.stats: dict[str, Any] = self._load_stats()

    # ---------- 统计与漏斗 ----------

    def _load_stats(self) -> dict[str, Any]:
        path = self.logs_dir / "stats.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {"events_total": 0, "by_category": {}, "funnel": {}, "drops": {},
                "accepted_sft": 0, "lema": 0, "dpo_negatives": 0,
                "updated_at": ""}

    def _save_stats(self) -> None:
        self.stats["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        (self.logs_dir / "stats.json").write_text(
            json.dumps(self.stats, ensure_ascii=False, indent=2), encoding="utf-8")
        self.formatter.write_meta_report(self.stats)

    def _funnel(self, event: dict[str, Any], stage: str, outcome: str,
                detail: str = "") -> None:
        rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "event_id": event["id"],
               "source": event.get("source", ""), "stage": stage,
               "outcome": outcome, "detail": mask_pii(detail)}
        with (self.logs_dir / "funnel.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if outcome == "drop":
            key = f"{stage}:{detail}" if detail else stage
            self.stats["drops"][key] = self.stats["drops"].get(key, 0) + 1
        self.stats["funnel"][stage] = self.stats["funnel"].get(stage, 0) + 1

    # ---------- 主流程 ----------

    async def process(self, event: dict[str, Any]) -> dict[str, Any]:
        """event: NormalizedEvent（connectors.adapters 产出或 self_propose 构造）。"""
        event = redact(event)
        if event.get("replaying"):
            for category in roles.CATEGORIES:
                stored = self.formatter.find_sft(category, event["id"])
                if stored is not None:
                    return await self._recover_sft(event, category, stored)
        self.stats["events_total"] += 1
        result: dict[str, Any] = {"event_id": event["id"], "outcome": "dropped",
                                  "category": None, "doc_ids": [], "stages": []}

        # 0) 廉价体检
        health = health_check(event["task"], event["thinking"],
                              event.get("conclusion", ""),
                              self.cfg.get("cleaner", {}))
        if not health.ok:
            self._funnel(event, "health", "drop", ";".join(health.drop_reasons))
            self._save_stats()
            result["drop_reason"] = ";".join(health.drop_reasons)
            return result
        result["stages"].append("health=ok")

        # 0.5) 结论一致性（规格书 T3：思考尾部答案 vs 结论字段，不一致即丢）
        concl_field = (event.get("conclusion") or "").strip()
        think_answer = extract_answer(strip_think_tags(event["thinking"]))
        if concl_field and think_answer and not answers_agree(concl_field, think_answer):
            self._funnel(event, "consistency", "drop", "answer_mismatch")
            self._save_stats()
            result["drop_reason"] = "结论不一致"
            return result
        result["stages"].append("consistency=ok")

        # 1) 判类
        cls = await roles.classify(self.llm, event["task"], event["thinking"])
        category = cls["category"]
        result["category"] = category
        self.stats["by_category"].setdefault(
            category, {"enqueued": 0, "accepted": 0, "dropped": 0})
        self.stats["by_category"][category]["enqueued"] += 1
        result["stages"].append(f"classify={category}({cls['confidence']})")

        # 2) 预查重（原文级，早杀）
        raw_text = event["task"] + "\n" + strip_think_tags(event["thinking"])
        verdict = self.dedup.check(raw_text, category, event["id"])
        if verdict.verdict != "unique":
            self._funnel(event, "dedup_pre", "drop",
                         f"{verdict.verdict}@{verdict.level} sim={verdict.similarity}")
            self.stats["by_category"][category]["dropped"] += 1
            self._save_stats()
            result["drop_reason"] = verdict.verdict
            return result
        result["stages"].append("dedup_pre=unique")

        # 3) 价值门（可验证的 self_proposed 豁免分数，只走双门）
        is_self = event.get("event_kind") == "self_proposed"
        gate_cfg = self.cfg.get("value_gate", {})
        if not (is_self and gate_cfg.get("verifiable_exempt", True)):
            vg = await roles.value_gate(self.llm, event["task"],
                                        event["thinking"],
                                        event.get("conclusion", ""))
            if vg["verdict"] == "drop" or float(vg["total"]) < gate_cfg.get("min_score", 6.0):
                self._funnel(event, "value_gate", "drop",
                             f"score={vg['total']}")
                self.stats["by_category"][category]["dropped"] += 1
                self._save_stats()
                result["drop_reason"] = f"value_gate:{vg['total']}"
                return result
            result["stages"].append(f"value_gate={vg['total']}")

        # 4) 分段审计和修复后复审
        task = event["task"]
        thinking = strip_think_tags(event["thinking"])
        conclusion = concl_field or think_answer
        segments = segment(thinking)
        audit = await roles.audit(self.llm, task, conclusion, segments)
        lineage = "seed"
        flawed = None
        if audit.get("verdict") != "clean":
            position = audit.get("first_bad_step")
            reason = audit.get("error_type")
            can_repair = (audit.get("verdict") == "suspect" and isinstance(reason, str)
                          and reason in _REPAIRABLE and type(position) is int
                          and 1 <= position <= len(segments))
            if not can_repair:
                detail = reason if isinstance(reason, str) else "unreviewed"
                return self._reject(event, result, thinking, "audit", f"unrepairable:{detail}")
            index = position - 1
            fixed = strip_think_tags(await roles.repair(
                self.llm, task, segments[index - 1] if index else "（开头）",
                segments[index], segments[position] if position < len(segments) else "（结尾）",
                conclusion))
            if not fixed or fixed == "UNREPAIRABLE":
                return self._reject(event, result, thinking, "repair", f"unrepairable:{reason}")
            repaired = "\n".join([*segments[:index], fixed, *segments[position:]])
            if (await roles.audit(self.llm, task, conclusion, segment(repaired))).get("verdict") != "clean":
                return self._reject(event, result, thinking, "repair", f"reaudit_failed:{reason}")
            flawed = thinking
            thinking = repaired
            lineage = "repaired"
            result["stages"].append(f"repair=ok({reason})")
        else:
            reason = None
            result["stages"].append("audit=clean")

        # 5–6) 截断和净化产生候选，只有通过复审的版本才能替换已有审计结果。
        trimmed = remove_dead_waits("\n".join(fcs_truncate(segment(thinking), keep_reflection=True)))
        candidate = strip_think_tags(await roles.purify(self.llm, task, trimmed))
        leaked = any(marker in candidate for marker in (
            "已审计的思考", "直接输出重写后的完整思考文本", "只输出 JSON", "[ROLE:"))
        candidate_answer = extract_answer(candidate)
        purified = thinking
        if (len(candidate) >= 20 and not leaked and not looks_truncated(candidate)
                and (not candidate_answer or answers_agree(candidate_answer, conclusion))):
            if candidate == thinking or (await roles.audit(
                    self.llm, task, conclusion, segment(candidate))).get("verdict") == "clean":
                purified = candidate
        result["stages"].append("purify=ok" if purified == candidate else "purify=fallback_audited")

        # 7) 答案门。可独立计算的自驱题需要同时通过题面推导和标准答案。
        if is_self and event.get("expected_answer"):
            standard = event["expected_answer"]
            answer_type = event.get("answer_type", "math_expression")
            verified = await hard_verify(conclusion, standard, answer_type)
            computed = deterministic_answer(task)
            if computed is not None:
                verified = verified and answers_agree(conclusion, computed)
            verify_kind = "hard" if computed is not None else "hard+independent"
            if computed is None and verified:
                verified, _ = await soft_verify(self.llm, task, purified, conclusion, self.cfg.get("verify", {}))
        else:
            verified, agree_ratio = await soft_verify(
                self.llm, task, purified, conclusion, self.cfg.get("verify", {}))
            verify_kind = f"soft({agree_ratio:.0%})"
        if not verified:
            return self._reject(event, result, purified, "verify", f"verify_fail:{verify_kind}")
        result["stages"].append(f"verify={verify_kind}")

        # 8) 终查重（净化后文本为权威文本）
        final_text = event["task"] + "\n" + purified
        verdict2 = self.dedup.check(final_text, category, event["id"])
        if verdict2.verdict != "unique":
            self._funnel(event, "dedup_final", "drop", verdict2.verdict)
            self.stats["by_category"][category]["dropped"] += 1
            self._save_stats()
            result["drop_reason"] = f"dedup_final:{verdict2.verdict}"
            return result

        # 9) 沉淀
        task_c, think_c, concl_c = clean_for_dataset(event["task"], purified, conclusion)
        meta = {"event_id": event["id"], "source": event.get("source", ""),
                "lineage": lineage, "category_tags": cls.get("tags", []),
                "quality_version": 2, "captured_at": event.get("occurred_at"),
                "repair_reason": reason, "repair_position": position if flawed is not None else None,
                "session_id": event.get("session_id"),
                "verify": verify_kind, "deterministic_verifier": deterministic_answer(task) is not None,
                "think_chars": len(purified),
                "stages": result["stages"]}
        doc_id = self.formatter.write_sft(category, task_c, think_c, concl_c, meta)
        result["doc_ids"].append(doc_id)
        self.dedup.add(final_text, category, doc_id)
        # 同步登记原文指纹：相同事件重灌时预查重 L1 直接秒杀（省全部 LLM 调用）
        self.dedup.add(raw_text, category, doc_id + "-raw", count_skeleton=False)
        self.stats["accepted_sft"] += 1
        # 难度锚（V3-T8）：已收编思考长度的 EMA，供自适应出题
        prev_avg = self.stats.get("avg_think_chars")
        self.stats["avg_think_chars"] = round(
            len(purified) if prev_avg is None else 0.9 * prev_avg + 0.1 * len(purified), 1)
        self.stats["by_category"][category]["accepted"] += 1
        paired = self.formatter.try_pair_dpo(category, task_c,
                                             f"<think>\n{think_c}\n</think>\n\n{concl_c}")
        if paired:
            result["stages"].append(f"dpo_paired={paired}")

        if flawed is not None:
            lmeta = {**meta, "lineage": "lema"}
            _, flawed_clean, _ = clean_for_dataset("", flawed, "")
            ldoc = self.formatter.write_lema(
                category, task_c, flawed_clean, f"第{position}段 {reason}", think_c, concl_c, lmeta)
            result["doc_ids"].append(ldoc)
            self.stats["lema"] += 1

        result["outcome"] = "accepted"
        self._funnel(event, "accept", "accept", lineage)
        self._save_stats()
        await self._publish_index(category, doc_id, task_c, meta)
        return result

    async def _recover_sft(self, event: dict[str, Any], category: str, stored: dict[str, Any]) -> dict[str, Any]:
        meta = stored["meta"]
        task = stored["messages"][0]["content"]
        answer = stored["messages"][-1]["content"]
        thinking, _, conclusion = answer.removeprefix("<think>").partition("</think>")
        self.dedup.add(task + "\n" + thinking.strip(), category, stored["id"])
        self.dedup.add(event["task"] + "\n" + strip_think_tags(event["thinking"]), category,
                       stored["id"] + "-raw", count_skeleton=False)
        self.formatter.try_pair_dpo(category, task, answer)
        documents = [stored["id"]]
        if meta.get("repair_reason"):
            _, flawed, _ = clean_for_dataset("", event["thinking"], "")
            documents.append(self.formatter.write_lema(
                category, task, flawed, f"第{meta['repair_position']}段 {meta['repair_reason']}",
                thinking.strip(), conclusion.strip(), {**meta, "lineage": "lema"}))
        await self._publish_index(category, stored["id"], task, meta)
        return {"event_id": event["id"], "outcome": "accepted", "category": category,
                "doc_ids": documents, "stages": meta["stages"]}

    async def _publish_index(self, category: str, doc_id: str, task: str, meta: dict[str, Any]) -> None:
        lineage, verify = meta["lineage"], meta["verify"]
        await self.sme.add(
            f"{category}类已收录：{task[:60]}（{lineage}，{verify}验证通过）",
            ns=f"cls.{category}", tags=[f"cat.{category}", *meta["category_tags"][:3]],
            metadata={"doc_id": doc_id, "event_id": meta["event_id"], "lineage": lineage,
                      "path": str(self._path_of(category, "sft"))})
        if reason := meta.get("repair_reason"):
            await self.sme.add(f"{category}类错误模式：{reason}（{task[:50]}）", ns="rsi.trace_index",
                               tags=["error_pattern", f"cat.{category}"], metadata={"event_id": meta["event_id"]})

    def _path_of(self, category: str, line: str) -> Path:
        return self.dataset_dir / category / f"{line}.jsonl"

    def _reject(self, event: dict[str, Any], result: dict[str, Any], thinking: str,
                stage: str, reason: str) -> dict[str, Any]:
        category = result["category"]
        reason = mask_pii(reason)
        self._funnel(event, stage, "drop", reason)
        result["doc_ids"].append(self._write_dpo(event, category, thinking, reason))
        result["drop_reason"] = reason
        self.stats["dpo_negatives"] += 1
        self.stats["by_category"][category]["dropped"] += 1
        self._save_stats()
        return result

    def _write_dpo(self, event: dict[str, Any], category: str, thinking: str,
                   reason: str) -> str:
        task_c, think_c, _ = clean_for_dataset(event["task"], thinking, "")
        return self.formatter.write_dpo_negative(category, task_c, think_c, reason,
                                                 {"event_id": event["id"], "quality_version": 2,
                                                  "session_id": event.get("session_id"),
                                                  "source": event.get("source", "")})


def new_event(source: str, task: str, thinking: str, conclusion: str = "",
              event_kind: str = "host_task", **extra: Any) -> dict[str, Any]:
    return {"id": uuid.uuid4().hex[:12], "source": source, "event_kind": event_kind,
            "task": task, "thinking": thinking, "conclusion": conclusion,
            "occurred_at": datetime.now().astimezone().isoformat(), **extra}
