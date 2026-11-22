# -*- coding: utf-8 -*-
import asyncio
import json

import pytest

from rsi.biodata.state import BioData
from rsi.core import roles
from rsi.core.splitter import Splitter, build_style_samples
from rsi.llm.client import LLMClient
from rsi.sme_wrap.client import SMEClient

TRANSCRIPT = [
    {"role": "user", "content": "我上周和合伙人把公司注册下来了，折腾了三个月终于成了"},
    {"role": "assistant", "content": "恭喜！"},
    {"role": "user", "content": "我觉得这次能成主要是我们俩分工清楚"},
    {"role": "assistant", "content": "分工确实关键。"},
    {"role": "user", "content": "累死了，先这样"},
]


def _event():
    return {"id": "ev-test-01", "source": "zcode", "event_kind": "host_chat",
            "task": TRANSCRIPT[0]["content"], "thinking": "思考" * 40,
            "conclusion": "", "transcript": TRANSCRIPT,
            "occurred_at": "2026-11-22 16:00:00"}


def _make_splitter(tmp_path):
    llm = LLMClient({"provider": "fake"})
    sme = SMEClient("http://127.0.0.1:59999",
                    backlog_path=tmp_path / "logs" / "backlog.jsonl")
    bd = BioData()
    return Splitter(llm, bd, sme, tmp_path), tmp_path


def test_style_samples_role_swap():
    samples = build_style_samples(TRANSCRIPT)
    assert samples, "应产出至少一个风格样本"
    last = samples[0][-1]
    assert last["role"] == "assistant"
    assert last["content"] == TRANSCRIPT[2]["content"]  # 用户的话成为学习目标
    assert samples[0][0]["role"] == "assistant"         # 用户的原话轮被换成 assistant
    assert samples[0][0]["content"] == TRANSCRIPT[0]["content"]


def test_splitter_three_lines(tmp_path):
    s, base = _make_splitter(tmp_path)
    out = asyncio.run(s.process(_event()))
    # 观察者(FakeLLM)从用户话里抽到"我"开头事实 → 叙事线
    assert out["facts"] >= 1
    assert out["narratives"] >= 1
    assert out["transcripts"] == 1
    assert out.get("rewrites") == 1          # 风格保持改写线（补课项）
    assert out["style"] >= 1
    # 落盘检查
    narr = [json.loads(x) for x in
            (base / "dataset" / "pretrain" / "narratives.jsonl").read_text(encoding="utf-8").splitlines()]
    assert narr[0]["meta"]["subjective_fact"] is True
    assert "mood" in narr[0]["meta"]
    tr = [json.loads(x) for x in
          (base / "dataset" / "pretrain" / "transcripts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert "合伙人" in tr[0]["text"] and "恭喜" not in tr[0]["text"]  # 只收用户原话
    rw = [json.loads(x) for x in
          (base / "dataset" / "pretrain" / "rewrites.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rw[0]["meta"]["rewrite_of"]  # 改写与原文有血缘关联
    style = [json.loads(x) for x in
             (base / "dataset" / "style" / "sft.jsonl").read_text(encoding="utf-8").splitlines()]
    assert style[0]["messages"][-1]["role"] == "assistant"
    manifest = json.loads((base / "dataset" / "_meta" / "pretrain_manifest.json")
                          .read_text(encoding="utf-8"))
    assert manifest["lines"]["narratives"]["count"] >= 1
    assert manifest["lines"]["rewrites"]["count"] == 1
    assert manifest["recipe"]["epoch上限"] == 4
    # biodata 状态持久化 + SME 离线降级进 backlog
    assert (base / "biodata" / "state.json").exists()
    backlog = (base / "logs" / "backlog.jsonl").read_text(encoding="utf-8")
    assert "rsi.facts" in backlog


def test_splitter_dedup_second_event(tmp_path):
    s, base = _make_splitter(tmp_path)
    asyncio.run(s.process(_event()))
    out2 = asyncio.run(s.process(_event()))
    assert out2["narratives"] == 0 and out2["transcripts"] == 0


def test_supported_narrative_and_identical_transcript_both_retained(tmp_path, monkeypatch):
    splitter, base = _make_splitter(tmp_path)
    quotation = "我上周整理好一套虚构模型展板和说明卡片，周日把它们收到蓝色纸盒里，方便下次接着用"

    async def observe(*args):
        return {"facts": [quotation, "我实际说过的是不存在的第三方话语"]}

    async def narrate(*args):
        return {"narratives": [quotation, quotation + "。我还给朋友付了一万元。"]}

    monkeypatch.setattr(roles, "observe", observe)
    monkeypatch.setattr(roles, "narrate", narrate)
    event = _event()
    event["transcript"] = [{"role": "user", "content": quotation}]
    output = asyncio.run(splitter.process(event))
    assert output["facts"] == 1 and output["narratives"] == 1 and output["transcripts"] == 1
    assert "一万元" not in (base / "dataset/pretrain/narratives.jsonl").read_text(encoding="utf-8")


def test_self_event_skipped(tmp_path):
    s, base = _make_splitter(tmp_path)
    ev = _event()
    ev["event_kind"] = "self_proposed"
    out = asyncio.run(s.process(ev))
    assert out == {"skipped": "self_event"}
