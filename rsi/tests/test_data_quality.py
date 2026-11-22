"""数据发布、接收恢复和模型费用的回归。"""
import asyncio
import json

import httpx
import pytest

from rsi.biodata.state import BioData
from rsi.core.export import freeze
from rsi.core.formatter import Formatter, build_pretrain_manifest, sample_balanced
from rsi.core.pipeline import new_event
from rsi.core.splitter import build_style_samples
from rsi.core.verifier import deterministic_answer, hard_verify, math_equal, soft_verify
from rsi.daemon.journal import EventJournal
from rsi.llm.budget import DailyBudget, LLMBudgetExceeded
from rsi.llm.client import LLMClient
from rsi.sme_wrap.client import SMEClient


def test_arithmetic_respects_grouping_and_independent_answer():
    assert math_equal("(2+3)*4", "20")
    assert not math_equal("(2+3)*4", "2+3*4")
    assert not asyncio.run(hard_verify("(2+3)*4", "2+3*4", "math_expression"))
    fake = LLMClient({"provider": "fake"})
    assert asyncio.run(soft_verify(fake, "计算 7+5", "答案是 99", "99", {})) == (False, 0.0)


def test_independent_congruence_solver():
    assert deterministic_answer(
        "求同时满足 n 除以 7、11、13 分别余 3、5、8 的最小正整数 n，并证明不存在更小的正整数解。"
    ) == "346"
    assert deterministic_answer("求 n 除以 4、6 分别余 0、0 的最小正整数 n。") == "12"
    assert deterministic_answer("求 n 除以 4、6 分别余 1、2 的最小正整数 n。") == "无解"
    assert deterministic_answer("求 n 除以 4、6 分别余 0、0 的最小正整数 n，且 n 大于 99。") is None


def test_independent_answer_never_receives_candidate():
    class BlindSolver:
        async def chat(self, system, user, **kwargs):
            assert "99" not in system + user
            assert "待核验轨迹" not in system + user
            return "12"
    assert asyncio.run(soft_verify(BlindSolver(), "七加五是多少？", "待核验轨迹", "99", {})) == (False, 0.0)
    assert asyncio.run(soft_verify(BlindSolver(), "七加五是多少？", "", "12", {})) == (True, 1.0)


def test_budget_reserves_before_billing_and_redacts_model_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("RSI_BUDGET_TEST_TOKEN", "test-token")
    cap = {"daily_cost_cap": 0.0005}
    path = tmp_path / "budget.json"
    sent = []

    async def verify():
        client = LLMClient({"provider": "openai_compatible", "base_url": "https://example.test",
                            "model": "mock", "api_key_env": "RSI_BUDGET_TEST_TOKEN", "max_tokens": 256},
                           cap, path)
        entered = asyncio.Event()
        release = asyncio.Event()

        async def backend(request):
            sent.append(json.loads(request.content))
            entered.set()
            await release.wait()
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "out@example.test"}}],
                "usage": {"prompt_tokens": 40, "completion_tokens": 12}})

        await client._impl._client.aclose()
        client._impl._client = httpx.AsyncClient(transport=httpx.MockTransport(backend))
        first = asyncio.create_task(client.chat("system", "我使用 user@example.test"))
        await entered.wait()
        assert DailyBudget(cap, path).spent() > 0
        with pytest.raises(LLMBudgetExceeded):
            await client.chat("second", "another input")
        release.set()
        assert await first == "[邮箱已脱敏]"
        assert client.estimated_cost() == pytest.approx(0.0000264)
        await client.aclose()

    asyncio.run(verify())
    assert len(sent) == 1
    assert "user@example.test" not in json.dumps(sent)


def test_journal_recovers_checkpoint_deduplicates_and_scrubs(tmp_path):
    path = tmp_path / "events.sqlite3"
    journal = EventJournal(path, capacity=1)
    event = new_event("zcode", "联系 me@example.test", "proof", source_event_id="line-1", session_id="run")
    assert journal.enqueue(event)
    duplicate = new_event("zcode", "联系 me@example.test", "proof", source_event_id="line-1", session_id="run")
    assert journal.enqueue(duplicate) and duplicate["id"] == event["id"]
    assert not journal.enqueue(new_event("mcp", "second", "trace"))
    claimed, prior, attempts = journal.claim()
    assert "[邮箱已脱敏]" in claimed["task"] and prior is None and attempts == 1
    journal.save_cot(event["id"], {"event_id": event["id"], "outcome": "accepted"})
    journal.close()
    restored = EventJournal(path, capacity=1)
    restored.recover()
    claimed, prior, attempts = restored.claim()
    assert claimed["id"] == event["id"] and attempts == 2
    assert prior["outcome"] == "accepted"
    restored.finish(event["id"], prior)
    assert restored.recent() == [prior]
    assert restored.count_pending() == 0
    assert restored.enqueue(duplicate) and restored.count_pending() == 0
    restored.close()


def test_sme_replay_reconciles_delivery_without_double_post(tmp_path):
    stored = {}
    sends = []

    async def verify():
        client = SMEClient("http://sme.test", tmp_path / "backlog.jsonl")

        async def backend(request):
            if request.url.path == "/health":
                return httpx.Response(200)
            body = json.loads(request.content)
            if request.url.path == "/memories/search":
                key = body["metadata_filters"]["rsi_delivery_id"]
                return httpx.Response(200, json={"results": [{"memory": stored[key]}]})
            key = body["metadata"]["rsi_delivery_id"]
            sends.append(key)
            stored[key] = {"id": "record-1", "metadata": body["metadata"]}
            assert "[邮箱已脱敏]" in body["text"]
            raise httpx.ReadTimeout("response lost after write", request=request)

        await client._client.aclose()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(backend))
        assert await client.add("email me@example.test", ns="rsi.facts", metadata={"event_id": "1"}) is None
        assert await client.flush_backlog() == 1
        assert await client.add("email me@example.test", ns="rsi.facts", metadata={"event_id": "1"}) == "record-1"
        await client.aclose()

    asyncio.run(verify())
    assert len(sends) == 1


def test_style_redacts_all_turns_without_losing_punctuation():
    result = build_style_samples([
        {"role": "system", "content": "不进入训练上下文"},
        {"role": "assistant", "content": "前情：assist@example.test！"},
        {"role": "user", "content": "我的邮箱是 ｕｓｅｒ＠ｅｘａｍｐｌｅ．ｔｅｓｔ，记好！"},
    ])
    assert result == [[{"role": "user", "content": "前情：[邮箱已脱敏]！"},
                       {"role": "assistant", "content": "我的邮箱是 [邮箱已脱敏]，记好！"}]]


def test_balancing_covers_weeks_and_caps_dominant_bucket(tmp_path):
    fmt = Formatter(tmp_path)
    for i in range(9):
        mood = "neutral" if i < 7 else "positive"
        fmt.write_pretrain("transcripts", f"文章-{i}", {
            "event_id": str(i), "mood": mood,
            "captured_at": f"2026-11-{1 + i * 3:02d}T00:00:00+08:00"})
    sample = sample_balanced(tmp_path)
    assert len(sample) == 4
    neutral = [row for row in sample if row["meta"]["mood"] == "neutral"]
    assert len({row["meta"]["captured_at"][:10] for row in neutral}) == 2
    manifest = build_pretrain_manifest(tmp_path)
    assert manifest["balanced_preview"]["mood_cap_met"] is True
    assert manifest["balanced_preview"]["missing_buckets"] == ["negative"]


def test_freeze_keeps_same_session_and_derived_text_together(tmp_path):
    dataset, output = tmp_path / "dataset", tmp_path / "snapshot"
    fmt = Formatter(dataset)
    first = {"event_id": "1", "session_id": "one", "quality_version": 2, "mood": "neutral"}
    second = {**first, "event_id": "2"}
    transcript = fmt.write_pretrain("transcripts", "我保存好原始记录。", first)
    fmt.write_pretrain("rewrites", "原始记录我已保存。", {**second, "rewrite_of": transcript})
    fmt.write_sft("math", "计算 2+2", "二加二等于四", "4", first)
    fmt.write_style([{"role": "user", "content": "然后呢？"},
                     {"role": "assistant", "content": "我再试试。"}], second)
    fmt.write_dpo_negative("math", "计算 2+2", "wrong", "suspect", second)
    fmt.write_pretrain("transcripts", "旧版未经审核的条目", {"event_id": "old"})
    manifest = freeze(dataset, output, 50)
    assert sum(manifest["counts"].values()) == 4 and manifest["rejected_schema_or_gate"] == 2
    assert len({name.split("/")[0] for name in manifest["counts"]}) == 1
    styles = list(output.rglob("style.jsonl"))
    assert json.loads(styles[0].read_text(encoding="utf-8"))["meta"]["train_message_indices"] == [1]


def test_24h_logical_time_and_trait_step_bound():
    state = BioData()
    for _ in range(150):
        state.apply({}, now=1000)
    prior = state.traits["openness"]
    state.apply({"trait_hint": {"openness": 1}, "mood_delta": {"p": 1}}, now=1000)
    assert 0 < state.traits["openness"] - prior <= 0.1
    state.tick(now=999)
    state.tick(now=1000 + 24 * 3600)
    assert state.mood["p"] == pytest.approx(0.1)
