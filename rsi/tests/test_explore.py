# -*- coding: utf-8 -*-
import asyncio
import json

import pytest

from rsi.biodata.state import BioData
from rsi.explore import Explorer
from rsi.llm.client import LLMClient
from rsi.sme_wrap.client import SMEClient


@pytest.fixture()
def explorer(tmp_path):
    llm = LLMClient({"provider": "fake"})
    sme = SMEClient("http://127.0.0.1:59999",
                    backlog_path=tmp_path / "logs" / "backlog.jsonl")
    return Explorer(llm, BioData(), sme,
                    {"self_explore": {"enabled": True, "max_missions_per_day": 1}},
                    tmp_path), tmp_path


def test_disabled_returns_flag(tmp_path):
    llm = LLMClient({"provider": "fake"})
    sme = SMEClient("http://127.0.0.1:59999", backlog_path=tmp_path / "b.jsonl")
    ex = Explorer(llm, BioData(), sme, {}, tmp_path)
    out = asyncio.run(ex.next_mission())
    assert out == {"enabled": False, "message": "自探索默认关闭（self_explore.enabled）"}


def test_mission_generated_and_capped(explorer):
    ex, base = explorer
    out = asyncio.run(ex.next_mission())
    assert out["enabled"] is True
    m = out["mission"]
    assert m["mission"] and m["acceptance"] and m["motivation"]
    assert (base / "logs" / "explore.jsonl").exists()
    out2 = asyncio.run(ex.next_mission())  # 上限1 → capped
    assert out2.get("capped") is True


def test_mission_json_serializable(explorer):
    ex, base = explorer
    asyncio.run(ex.next_mission())
    line = (base / "logs" / "explore.jsonl").read_text(encoding="utf-8").splitlines()[0]
    assert json.loads(line)["type"] == "curiosity"
