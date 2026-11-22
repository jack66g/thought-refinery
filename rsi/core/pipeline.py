"""提纯管线：单个事件的全生命周期（判类→查重→价值门→审计→修复→FCS→净化→验证→沉淀）。

所有 drop 都带原因写入 logs/funnel.jsonl（门控漏斗），不静默。
"""
from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

from ..llm.client import LLMClient
from ..sme_wrap.client import SMEClient
from . import roles
from .cleaner import clean_for_dataset, health_check, strip_think_tags
from .dedup import DedupConfig, DedupIndex
from .formatter import Formatter
from .segmenter import fcs_truncate, find_conclusion_segment, segment
from .verifier import extract_answer, hard_verify, math_equal, soft_verify

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
               "outcome": outcome, "detail": detail}
        with (self.logs_dir / "funnel.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if outcome == "drop":
            key = f"{stage}:{detail}" if detail else stage
            self.stats["drops"][key] = self.stats["drops"].get(key, 0) + 1
        self.stats["funnel"][stage] = self.stats["funnel"].get(stage, 0) + 1

    # ---------- 主流程 ----------

    async def process(self, event: dict[str, Any]) -> dict[str, Any]:
        """event: NormalizedEvent（connectors.adapters 产出或 self_propose 构造）。"""
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
        if concl_field and think_answer:
            both_num = re.fullmatch(r"[-+*/().\d\s]+", concl_field) and \
                re.fullmatch(r"[-+*/().\d\s]+", think_answer)
            if both_num and not math_equal(concl_field, think_answer):
                self._funnel(event, "consistency", "drop",
                             f"{concl_field}!={think_answer}")
                self._save_stats()
                result["drop_reason"] = f"结论不一致:{think_answer}vs{concl_field}"
                return result
            if not both_num and concl_field.strip("。，.") != think_answer.strip("。，."):
                self._funnel(event, "consistency", "drop", "text_mismatch")
                self._save_stats()
                result["drop_reason"] = "结论不一致(文本)"
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

        # 4) 分段审计
        thinking = strip_think_tags(event["thinking"])
        segments = segment(thinking)
        audit_res = await roles.audit(self.llm, event["task"],
                                      event.get("conclusion", ""), segments)
        lineage = "seed"
        lema_material = None
        if audit_res["verdict"] == "suspect":
            bad = int(audit_res.get("first_bad_step") or 0)
            error_type = audit_res.get("error_type") or "未知"
            if bad < 1 or bad > len(segments):
                bad = 1
            if error_type in _REPAIRABLE:
                prev_ctx = segments[bad - 2] if bad >= 2 else "（开头）"
                next_ctx = segments[bad] if bad < len(segments) else "（结尾）"
                fixed_seg = await roles.repair(
                    self.llm, event["task"], prev_ctx, segments[bad - 1],
                    next_ctx, event.get("conclusion", ""))
                if fixed_seg == "UNREPAIRABLE":
                    lema_material = None
                    self._funnel(event, "repair", "drop", f"unrepairable:{error_type}")
                    doc = self._write_dpo(event, category, thinking, f"unrepairable:{error_type}")
                    result["doc_ids"].append(doc)
                    self.stats["dpo_negatives"] += 1
                    self.stats["by_category"][category]["dropped"] += 1
                    self._save_stats()
                    result["drop_reason"] = f"unrepairable:{error_type}"
                    return result
                repaired_segments = (segments[:bad - 1] + [fixed_seg]
                                     + segments[bad:])
                repaired_thinking = "\n".join(repaired_segments)
                re_audit = await roles.audit(self.llm, event["task"],
                                             event.get("conclusion", ""),
                                             segment(repaired_thinking))
                if re_audit["verdict"] == "suspect":
                    # 修复失败 → LEMA 纠错对（错误+纠正+终对仍可用）
                    lema_material = {"flawed": "\n".join(segments[:bad]),
                                     "error_desc": f"第{bad}段 {error_type}",
                                     "corrected": repaired_thinking}
                    thinking = repaired_thinking
                    lineage = "lema"
                    result["stages"].append(f"repair=failed->lema({error_type})")
                else:
                    thinking = repaired_thinking
                    lineage = "repaired"
                    result["stages"].append(f"repair=ok({error_type})")
            else:
                # 不可修类 → 负样本
                self._funnel(event, "audit", "drop", f"unrepairable:{error_type}")
                doc = self._write_dpo(event, category, thinking, error_type)
                result["doc_ids"].append(doc)
                self.stats["dpo_negatives"] += 1
                self.stats["by_category"][category]["dropped"] += 1
                self._save_stats()
                result["drop_reason"] = error_type
                return result
        else:
            result["stages"].append("audit=clean")

        # 5) FCS 截断 + 删空转
        segs2 = fcs_truncate(segment(thinking), keep_reflection=True)
        final_thinking = remove_dead_waits("\n".join(segs2))
        result["stages"].append(f"fcs={len(segments)}→{len(segs2)}段")

        # 6) 重述净化
        purified = await roles.purify(self.llm, event["task"], final_thinking)
        # 模板回显守卫：真模型也可能把提示词指令抄进输出——发现即退回截断版
        _LEAK_MARKERS = ("已审计的思考", "直接输出重写后的完整思考文本",
                         "只输出 JSON", "[ROLE:")
        if (len(purified) < 20
                or any(marker in purified for marker in _LEAK_MARKERS)):
            purified = final_thinking
            result["stages"].append("purify=fallback_raw")
        else:
            result["stages"].append("purify=ok")

        # 7) 终验
        conclusion = event.get("conclusion", "") or extract_answer(purified) \
            or extract_answer(thinking)
        if is_self and event.get("expected_answer"):
            verified = await hard_verify(conclusion, event["expected_answer"],
                                         event.get("answer_type", "math_expression"))
            verify_kind = "hard"
        else:
            verified, agree_ratio = await soft_verify(
                self.llm, event["task"], purified, conclusion,
                self.cfg.get("verify", {}))
            verify_kind = f"soft({agree_ratio:.0%})"
        if not verified:
            self._funnel(event, "verify", "drop", verify_kind)
            doc = self._write_dpo(event, category, purified, f"verify_fail:{verify_kind}")
            result["doc_ids"].append(doc)
            self.stats["dpo_negatives"] += 1
            self.stats["by_category"][category]["dropped"] += 1
            self._save_stats()
            result["drop_reason"] = f"verify_fail:{verify_kind}"
            return result
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
                "verify": verify_kind, "think_chars": len(purified),
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

        if lema_material:
            lmeta = {**meta, "lineage": "lema"}
            ldoc = self.formatter.write_lema(
                category, task_c, lema_material["flawed"],
                lema_material["error_desc"], lema_material["corrected"],
                concl_c, lmeta)
            result["doc_ids"].append(ldoc)
            self.stats["lema"] += 1

        # 10) SME 索引卡 + 错误模式
        await self.sme.add(
            f"{category}类已收录：{task_c[:60]}（{lineage}，{verify_kind}验证通过）",
            ns=f"cls.{category}",
            tags=[f"cat.{category}"] + cls.get("tags", [])[:3],
            metadata={"doc_id": doc_id, "lineage": lineage,
                      "path": str(self._path_of(category, "sft"))})
        if audit_res.get("error_type"):
            await self.sme.add(
                f"{category}类错误模式：{audit_res['error_type']}（{event['task'][:50]}）",
                ns="rsi.trace_index",
                tags=["error_pattern", f"cat.{category}"],
                metadata={"event_id": event["id"]})

        result["outcome"] = "accepted"
        self._funnel(event, "accept", "accept", lineage)
        self._save_stats()
        return result

    def _path_of(self, category: str, line: str) -> Path:
        return self.dataset_dir / category / f"{line}.jsonl"

    def _write_dpo(self, event: dict[str, Any], category: str, thinking: str,
                   reason: str) -> str:
        task_c, think_c, _ = clean_for_dataset(event["task"], thinking, "")
        return self.formatter.write_dpo_negative(category, task_c, think_c, reason,
                                                 {"event_id": event["id"],
                                                  "source": event.get("source", "")})


def new_event(source: str, task: str, thinking: str, conclusion: str = "",
              event_kind: str = "host_task", **extra: Any) -> dict[str, Any]:
    return {"id": uuid.uuid4().hex[:12], "source": source, "event_kind": event_kind,
            "task": task, "thinking": thinking, "conclusion": conclusion,
            "occurred_at": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
