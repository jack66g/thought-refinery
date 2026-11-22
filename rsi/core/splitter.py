"""三线分流器（V3-T4）：一个事件 → biodata 更新 + 预训练线 + 风格SFT线。

与 CoT 线（pipeline）相互独立：CoT 线被门拦下的事件，其用户内容仍可进
预训练/风格线（三线各判各的，这是刻意的——聊天没通过思考门不代表没底色价值）。

分工依据（调研 B5）：主观既定事实=第一人称叙事进 CPT + 结构化进 SME(rsi.facts)
供 RAG；对话转录=用户原话进 CPT；用户表达=风格 SFT（assistant=用户）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..biodata.state import BioData
from ..llm.client import LLMClient
from ..sme_wrap.client import SMEClient
from . import roles
from .cleaner import mask_pii, normalize_width
from .dedup import DedupConfig, DedupIndex
from .formatter import Formatter, build_pretrain_manifest


def _render_user_transcript(transcript: list[dict[str, str]]) -> str:
    """预训练转录只收用户原话（保底色纯度：AI 回复不混入）。"""
    lines = []
    for turn in transcript:
        if turn.get("role") == "user":
            text = str(turn.get("content", "")).strip()
            if text:
                lines.append(text)
    return "\n".join(lines)


def _swap_role(role: str) -> str:
    return "assistant" if role == "user" else "user"


def build_style_samples(transcript: list[dict[str, str]],
                        max_samples: int = 3) -> list[list[dict[str, str]]]:
    """风格SFT样本：把对话里的角色对调——用户的话变成 assistant 目标，
    模型由此学会'作为用户那样说话'（WeClone 式）。"""
    samples = []
    for i, turn in enumerate(transcript):
        if turn.get("role") != "user" or i == 0:
            continue
        target = str(turn.get("content", "")).strip()
        if len(target) < 4:
            continue
        context = [{"role": _swap_role(t["role"]), "content": str(t.get("content", ""))}
                   for t in transcript[:i]][-6:]
        if not context:
            continue
        samples.append(context + [{"role": "assistant", "content": target}])
        if len(samples) >= max_samples:
            break
    return samples


class Splitter:
    def __init__(self, llm: LLMClient, biodata: BioData, sme: SMEClient,
                 rsi_base: Path):
        self.llm = llm
        self.biodata = biodata
        self.sme = sme
        self.base = rsi_base
        self.dataset_dir = rsi_base / "dataset"
        self.formatter = Formatter(self.dataset_dir)
        # 预训练线宽松去重（jaccard 0.9，防误杀口头禅——调研 B3）
        self.dedup_pt = DedupIndex(
            base_dir=self.dataset_dir / "pretrain",
            cfg=DedupConfig(jaccard_threshold=0.9, ngram_overlap_threshold=0.8))
        self.dedup_style = DedupIndex(
            base_dir=self.dataset_dir / "style",
            cfg=DedupConfig(jaccard_threshold=0.85))
        self.state_path = rsi_base / "biodata" / "state.json"
        self.hist_path = rsi_base / "biodata" / "history.jsonl"

    async def process(self, event: dict[str, Any]) -> dict[str, Any]:
        if event.get("event_kind") in ("self_proposed",):
            return {"skipped": "self_event"}
        out: dict[str, Any] = {"event_id": event["id"], "narratives": 0,
                               "transcripts": 0, "style": 0, "facts": 0}
        content = event["task"]
        transcript = event.get("transcript") or []
        if transcript:
            content += "\n" + "\n".join(f"{t.get('role')}: {t.get('content', '')}"
                                        for t in transcript[:20])
        elif event.get("thinking"):
            content += "\n" + event["thinking"]

        # 1) 观察者：状态 Δ + 事实抽取（一次调用两用）
        obs = await roles.observe(self.llm, content)
        self.biodata.apply(obs)
        self.biodata.save(self.state_path)
        self.biodata.append_history(self.hist_path, obs)
        bucket = self.biodata.mood_bucket()
        out["mood_bucket"] = bucket

        # 2) 事实 → 双落点：SME 结构化（RAG 用）+ 第一人称叙事（CPT 用）
        facts = [str(f).strip() for f in (obs.get("facts") or []) if str(f).strip()]
        for fact in facts:
            await self.sme.add(mask_pii(fact), ns="rsi.facts",
                               tags=["subjective_fact"], importance=0.7,
                               metadata={"event_id": event["id"],
                                         "mood": bucket})
        out["facts"] = len(facts)
        if facts:
            narr = await roles.narrate(self.llm, facts)
            for para in narr.get("narratives", []):
                text = mask_pii(normalize_width(str(para))).strip()
                if len(text) < 30:
                    continue
                if self.dedup_pt.check(text, "narratives", event["id"]).verdict != "unique":
                    continue
                doc = self.formatter.write_pretrain(
                    "narratives", text,
                    {"event_id": event["id"], "subjective_fact": True,
                     "mood": bucket, "ts": time.strftime("%F %T")})
                self.dedup_pt.add(text, "narratives", doc)
                out["narratives"] += 1

        # 3) 对话转录 → 预训练（用户原话）+ 风格保持改写（配方 20% 形态）
        user_text = _render_user_transcript(transcript)
        if len(user_text) >= 40:
            text = mask_pii(normalize_width(user_text))
            if self.dedup_pt.check(text, "transcripts", event["id"]).verdict == "unique":
                doc = self.formatter.write_pretrain(
                    "transcripts", text,
                    {"event_id": event["id"], "subjective_fact": False,
                     "mood": bucket, "ts": time.strftime("%F %T")})
                self.dedup_pt.add(text, "transcripts", doc)
                out["transcripts"] += 1
                try:
                    rewrite = await roles.style_rewrite(self.llm, text)
                    if len(rewrite) >= 30:
                        rewrite = mask_pii(normalize_width(rewrite))
                        v = self.dedup_pt.check(rewrite, "rewrites", event["id"])
                        if v.verdict == "unique":
                            rdoc = self.formatter.write_pretrain(
                                "rewrites", rewrite,
                                {"event_id": event["id"], "subjective_fact": False,
                                 "mood": bucket, "rewrite_of": doc,
                                 "ts": time.strftime("%F %T")})
                            self.dedup_pt.add(rewrite, "rewrites", rdoc)
                            out["rewrites"] = out.get("rewrites", 0) + 1
                except Exception as exc:  # noqa: BLE001 改写失败不拖垮其余线，但必须留痕
                    err_path = self.base / "logs" / "errors.jsonl"
                    err_path.parent.mkdir(parents=True, exist_ok=True)
                    with err_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"ts": time.strftime("%F %T"),
                                             "err": f"rewrite:{exc!r}",
                                             "event_id": event["id"]},
                                            ensure_ascii=False) + "\n")

        # 4) 风格 SFT 样本（assistant=用户原话，不修正风格）
        for messages in build_style_samples(transcript):
            fingerprint = messages[-1]["content"] + "|" + messages[-2]["content"]
            if self.dedup_style.check(fingerprint, "style",
                                      event["id"]).verdict != "unique":
                continue
            doc = self.formatter.write_style(
                messages, {"event_id": event["id"], "mood": bucket,
                           "ts": time.strftime("%F %T")})
            self.dedup_style.add(fingerprint, "style", doc)
            out["style"] += 1

        # 5) biodata 索引卡节流入 SME（每 10 次更新一条）
        if self.biodata.updates % 10 == 0:
            await self.sme.add(f"画像快照：{self.biodata.persona_card()}",
                               ns="rsi.biodata", tags=["snapshot"],
                               metadata={"updates": self.biodata.updates})

        build_pretrain_manifest(self.dataset_dir)
        return out
