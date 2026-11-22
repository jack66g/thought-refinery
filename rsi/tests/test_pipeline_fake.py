# -*- coding: utf-8 -*-
"""管线全链路集成测试：FakeLLM + SME 不可达（降级→backlog→恢复补写）。"""
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from rsi.core import roles
from rsi.core.pipeline import Pipeline, new_event
from rsi.llm.client import LLMClient
from rsi.sme_wrap.client import SMEClient

CFG = {
    "cleaner": {"ngram_n": 10, "max_repeat_ratio": 0.3,
                "minor_lang_ratio": 0.35, "max_total_chars": 120000,
                "min_compress_ratio": 0.30, "max_loop_markers": 8},
    "dedup": {"minhash_perms": 64, "lsh_bands": 8, "jaccard_threshold": 0.8,
              "shingle_n": 5, "ngram_n": 10, "ngram_overlap_threshold": 0.7,
              "skeleton_keep_k": 2},
    "value_gate": {"min_score": 6.0, "verifiable_exempt": True},
    "verify": {"self_consistency_n": 3, "self_consistency_min_agree": 2},
}

GOOD_THINKING = ("先算括号内的加法，三加五等于八。得到中间结果八。"
                 "再把八乘以二得到十六。验证一下：十六除以二等于八，"
                 "与括号内结果一致，说明计算无误。所以答案是 16。")


class _StubSME(BaseHTTPRequestHandler):
    def log_message(self, *args):  # 静默桩
        pass

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True})
        else:
            self._send(404, {})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        if self.path in ("/memories", "/memories/search"):
            self._send(200, {"id": "m-stub", "results": []})
        else:
            self._send(404, {})

    def _send(self, code, body):
        data = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def pipeline(tmp_path):
    llm = LLMClient({"provider": "fake"})
    sme = SMEClient("http://127.0.0.1:59999",  # 不可达：触发显式降级
                    backlog_path=tmp_path / "logs" / "sme_backlog.jsonl")
    return Pipeline(llm, sme, CFG, tmp_path), tmp_path


def _run(pipe, event):
    return asyncio.run(pipe.process(event))


def test_good_math_event_accepted(pipeline):
    pipe, tmp = pipeline
    ev = new_event("zcode", "计算 (3+5)*2 的值", GOOD_THINKING, conclusion="16")
    result = _run(pipe, ev)
    assert result["outcome"] == "accepted", result
    assert result["category"] == "math"
    sft = (tmp / "dataset" / "math" / "sft.jsonl").read_text(encoding="utf-8")
    rec = json.loads(sft.splitlines()[-1])
    assert "<think>" in rec["messages"][1]["content"]
    assert rec["meta"]["verify"].startswith("soft")
    # SME 离线 → 索引卡进 backlog（显式降级，不静默）
    backlog = (tmp / "logs" / "sme_backlog.jsonl").read_text(encoding="utf-8")
    assert "cls.math" in backlog


def test_exact_duplicate_dropped(pipeline):
    pipe, tmp = pipeline
    ev1 = new_event("zcode", "计算 (3+5)*2 的值", GOOD_THINKING, conclusion="16")
    r1 = _run(pipe, ev1)
    assert r1["outcome"] == "accepted"
    ev2 = new_event("zcode", "计算 (3+5)*2 的值", GOOD_THINKING, conclusion="16")
    r2 = _run(pipe, ev2)
    assert r2["outcome"] == "dropped"
    # 预查重(原文级)或终查重(净化后)任一层抓住都算对
    assert r2["drop_reason"] in ("exact_dup", "dedup_final:exact_dup")


def test_near_duplicate_dropped(pipeline):
    pipe, tmp = pipeline
    ev1 = new_event("zcode", "计算 (3+5)*2 的值", GOOD_THINKING, conclusion="16")
    assert _run(pipe, ev1)["outcome"] == "accepted"
    variant = GOOD_THINKING.replace("先算括号内的加法", "首先计算括号里加法")
    ev2 = new_event("zcode", "计算 (3+5)*2 的值", variant, conclusion="16")
    r2 = _run(pipe, ev2)
    assert r2["outcome"] == "dropped"
    assert "dup" in r2["drop_reason"] or "polluted" in r2["drop_reason"]


SELF_THINKING_OK = ("先算乘法：七乘六等于四十二，这一步是基础乘法运算。"
                    "再把四十二加上八：四十二加八等于五十。"
                    "验算一下：五十减八等于四十二，四十二除以七等于六，无误。"
                    "所以答案是 50")


def test_self_proposed_hard_verify(pipeline):
    pipe, tmp = pipeline
    ev = new_event("self", "计算 7*6+8 的值", SELF_THINKING_OK,
                   event_kind="self_proposed", expected_answer="50",
                   answer_type="math_expression")
    result = _run(pipe, ev)
    assert result["outcome"] == "accepted", result
    assert any("hard" in s for s in result["stages"])


def test_self_proposed_wrong_answer_dropped(pipeline):
    pipe, tmp = pipeline
    bad = SELF_THINKING_OK.replace("答案是 50", "答案是 51")
    ev = new_event("self", "计算 7*6+8 的值", bad,
                   event_kind="self_proposed", expected_answer="50",
                   answer_type="math_expression")
    result = _run(pipe, ev)
    assert result["outcome"] == "dropped", result
    assert "verify_fail" in result["drop_reason"]
    dpo = (tmp / "dataset" / "math" / "dpo.jsonl").read_text(encoding="utf-8")
    assert "verify_fail" in dpo


def test_unrepairable_goes_dpo_negative(pipeline):
    pipe, tmp = pipeline
    thinking = ("第一步设定参数范围并列举全部约束条件，确认问题规模有限。"
                "第二步开始枚举验证各候选值是否满足约束集合。"
                "【坏段】此处不可修复：直接跳到结论没有过程。"
                "所以答案是 99。")
    ev = new_event("zcode", "求解约束问题", thinking, conclusion="99")
    result = _run(pipe, ev)
    assert result["outcome"] == "dropped"
    assert "unrepairable" in result["drop_reason"]
    dpo = (tmp / "dataset" / result["category"] / "dpo.jsonl").read_text(encoding="utf-8")
    assert "unrepairable" in dpo


def test_repairable_then_accepted(pipeline):
    pipe, tmp = pipeline
    thinking = ("先计算括号中的加法，两个二相加得到四，括号外的乘数是二。"
                "【坏段】这句没有提供有效的运算。"
                "再用括号内得到的四乘以二，结果是八；用八除以二可验回得到四。答案是 8。")
    ev = new_event("zcode", "计算 (2+2)*2 的值", thinking, conclusion="8")
    result = _run(pipe, ev)
    assert result["outcome"] == "accepted", result
    assert any(s.startswith("repair=ok") for s in result["stages"])
    sft = (tmp / "dataset" / "math" / "sft.jsonl").read_text(encoding="utf-8")
    rec = json.loads(sft.splitlines()[-1])
    assert rec["meta"]["lineage"] == "repaired"


def test_reaudit_failure_prevents_positive_and_lema(pipeline, monkeypatch):
    pipe, tmp = pipeline
    calls = []

    async def repeated_suspect(*args):
        calls.append(1)
        return {"verdict": "suspect", "first_bad_step": 1, "error_type": "跳步"}

    monkeypatch.setattr(roles, "audit", repeated_suspect)
    event = new_event("zcode", "计算 (3+5)*2 的值", GOOD_THINKING, conclusion="16")
    result = _run(pipe, event)
    assert len(calls) == 2
    assert result["drop_reason"] == "reaudit_failed:跳步"
    assert not (tmp / "dataset/math/sft.jsonl").exists()
    assert not (tmp / "dataset/math/lema.jsonl").exists()


def test_replay_uses_same_published_sft_id(pipeline):
    pipe, tmp = pipeline
    event = new_event("zcode", "计算 (3+5)*2 的值", GOOD_THINKING, conclusion="16")
    first = _run(pipe, event)
    replay = _run(pipe, {**event, "replaying": True})
    assert first["doc_ids"] == replay["doc_ids"] and replay["outcome"] == "accepted"
    assert len((tmp / "dataset/math/sft.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert len((tmp / "logs/sme_backlog.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_short_thinking_health_drop(pipeline):
    pipe, tmp = pipeline
    ev = new_event("zcode", "随便聊聊", "想了一下。", conclusion="嗯")
    result = _run(pipe, ev)
    assert result["outcome"] == "dropped"
    assert "思考过短" in result["drop_reason"]


def test_value_gate_drop(pipeline):
    pipe, tmp = pipeline
    filler = ("今天天气不错适合出门散步，顺便闲聊寒暄几句。"
              "最近街角开了家新的面馆，据说味道还行。"
              "傍晚的风很舒服，广场上有人在放风筝。")
    ev = new_event("zcode", "今天天气寒暄闲聊", filler, conclusion="哈哈")
    result = _run(pipe, ev)
    assert result["outcome"] == "dropped"
    assert result["drop_reason"].startswith("value_gate")


def test_conclusion_mismatch_dropped(pipeline):
    pipe, tmp = pipeline
    ev = new_event("zcode", "计算 (3+5)*2 的值", GOOD_THINKING, conclusion="15")
    result = _run(pipe, ev)
    assert result["outcome"] == "dropped"
    assert "结论不一致" in result["drop_reason"]


def test_stats_and_funnel_written(pipeline):
    pipe, tmp = pipeline
    ev = new_event("zcode", "计算 (3+5)*2 的值", GOOD_THINKING, conclusion="16")
    _run(pipe, ev)
    stats = json.loads((tmp / "logs" / "stats.json").read_text(encoding="utf-8"))
    assert stats["events_total"] >= 1
    assert stats["accepted_sft"] >= 1
    funnel = (tmp / "logs" / "funnel.jsonl").read_text(encoding="utf-8")
    assert "accept" in funnel
    assert (tmp / "dataset" / "_meta" / "stats_report.json").exists()


def test_sme_backlog_flush_on_recovery(pipeline):
    pipe, tmp = pipeline
    ev = new_event("zcode", "计算 (3+5)*2 的值", GOOD_THINKING, conclusion="16")
    _run(pipe, ev)
    backlog_path = tmp / "logs" / "sme_backlog.jsonl"
    assert backlog_path.exists() and backlog_path.read_text(encoding="utf-8").strip()

    # 恢复：起一个桩 SME，flush_backlog 应清空积压
    server = HTTPServer(("127.0.0.1", 0), _StubSME)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    pipe.sme.base_url = f"http://127.0.0.1:{port}"
    flushed = asyncio.run(pipe.sme.flush_backlog())
    assert flushed >= 1
    assert not backlog_path.read_text(encoding="utf-8").strip()
    server.shutdown()
