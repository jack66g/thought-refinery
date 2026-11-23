"""选择模式的开关、两步选择、恢复和人格双线导出。"""
import asyncio
import json
from contextlib import closing, contextmanager
from threading import Event

import httpx

from rsi.choice.scoring import estimates
from rsi.core.export import freeze
from rsi.daemon.engine import RSIDaemon
from rsi.daemon.server import create_app

_MEMORIES = [
    {"memory": {"id": "memory-one", "text": "我习惯在实验可撤销时先进行小范围验证再总结"}},
    {"memory": {"id": "memory-two", "text": "我平时写作前会列几个重点，之后根据反馈调整节奏"}},
]


def daemon_with_memories(base, *, enabled=False, rounds=3):
    daemon = RSIDaemon({
        "llm": {"provider": "fake"}, "choice_mode": {
            "enabled": enabled, "min_profile_updates": 2, "min_memories": 2, "rounds_per_day": rounds},
    }, base)
    daemon.biodata.updates = 2

    def memory(request):
        if request.url.path == "/health":
            return httpx.Response(200)
        payload = json.loads(request.content)
        assert request.url.path == "/memories/search"
        assert payload["metadata_filters"]["ns"] == "rsi.facts"
        return httpx.Response(200, json={"results": _MEMORIES})

    daemon.sme._client = httpx.AsyncClient(transport=httpx.MockTransport(memory))
    return daemon


@contextmanager
def session(daemon, popup=None):
    app = create_app(daemon, popup)
    with closing(asyncio.new_event_loop()) as loop:
        lifespan = app.router.lifespan_context(app)
        loop.run_until_complete(lifespan.__aenter__())
        client = httpx.AsyncClient(
            base_url="http://127.0.0.1",
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 41000)))

        class LocalClient:
            def get(self, *args, **kwargs):
                return loop.run_until_complete(client.get(*args, **kwargs))

            def post(self, *args, **kwargs):
                return loop.run_until_complete(client.post(*args, **kwargs))

            def wait(self, seconds):
                loop.run_until_complete(asyncio.sleep(seconds))

        try:
            yield LocalClient()
        finally:
            loop.run_until_complete(client.aclose())
            loop.run_until_complete(lifespan.__aexit__(None, None, None))
            loop.run_until_complete(loop.shutdown_default_executor())


def settings(client, enabled, *, popup=False):
    return client.post("/choice/api/settings", json={"enabled": enabled, "open_window": popup})


def start(client):
    response = client.post("/choice/api/start", json={})
    assert response.status_code == 200, response.text
    return response.json()


def submit(client, row, step, option="0"):
    return client.post("/choice/api/answer", json={"round_id": row["id"], "step": step, "option": option})


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_opt_in_two_steps_and_freeze(tmp_path):
    daemon = daemon_with_memories(tmp_path)
    popups = []
    with session(daemon, lambda: popups.append(True)) as client:
        assert client.get("/choice").status_code == 200
        assert client.get("/choice/api/state").json()["enabled"] is False
        assert not daemon.choice_mode.path.exists()
        assert client.post("/choice/api/start").status_code == 409
        assert settings(client, True, popup=True).status_code == 200
        assert popups == [True]
        first = start(client)
        assert first["step"] == "decision" and start(client)["id"] == first["id"]
        assert set(first) == {"id", "step", "title", "background", "question", "options"}
        assert submit(client, first, "reason").status_code == 409
        second = submit(client, first, "decision").json()
        assert second["step"] == "reason" and second["decision"] == first["options"][0]["text"]
        assert not (tmp_path / "dataset").exists()
        assert submit(client, first, "decision", "1").status_code == 409
        endpoint = submit(client, first, "reason")
        assert endpoint.status_code == 200 and endpoint.json()["step"] == "complete"
        assert submit(client, first, "reason").json() == endpoint.json()
        status = client.get("/choice/api/state").json()
        assert status["total_answered"] == 1 and status["active_round"] is None
        assert sum(item["observations"] for item in status["estimates"]) == 1
        assert all(item["a_share"] is None for item in status["estimates"])
        assert client.get("/status").json()["stats"]["events_total"] == 0

    pretrain = read_jsonl(tmp_path / "dataset/pretrain/choices.jsonl")
    posterior = read_jsonl(tmp_path / "dataset/style/choices.jsonl")
    assert len(pretrain) == len(posterior) == 1
    assert pretrain[0]["meta"]["event_id"] == posterior[0]["meta"]["event_id"]
    assert pretrain[0]["meta"]["synthetic_scenario"] is True
    assert pretrain[0]["meta"]["subjective_fact"] is False
    assert posterior[0]["meta"]["target_source"] == "confirmed_model_option"
    assert [item["role"] for item in posterior[0]["messages"]] == ["user", "assistant", "user", "assistant"]
    assert not (tmp_path / "dataset/math/sft.jsonl").exists()
    manifest = freeze(tmp_path / "dataset", tmp_path / "export", 0)
    assert manifest["counts"] == {"train/personality_choice": 1, "train/pretrain": 1}
    target = read_jsonl(tmp_path / "export/train/personality_choice.jsonl")[0]
    assert target["meta"]["train_message_indices"] == [1, 3]


def test_gates_and_daily_limit(tmp_path):
    daemon = daemon_with_memories(tmp_path, enabled=True, rounds=1)
    with session(daemon) as client:
        daemon.biodata.updates = 1
        assert "画像样本" in client.post("/choice/api/start").json()["detail"]
        assert daemon.llm.calls == 0
        daemon.biodata.updates = 2
        original = daemon.sme.search

        async def too_few(*args, **kwargs):
            return [_MEMORIES[0], _MEMORIES[0]]

        daemon.sme.search = too_few
        assert "个人记忆" in client.post("/choice/api/start").json()["detail"]
        assert daemon.llm.calls == 0
        daemon.sme.search = original
        first = start(client)
        assert submit(client, first, "decision", "skip").json()["step"] == "skipped"
        assert client.post("/choice/api/start").status_code == 429
        assert client.get("/choice/api/state").json()["total_answered"] == 0
        assert not (tmp_path / "dataset").exists()


def test_pause_restart_and_resume_reasons(tmp_path):
    daemon = daemon_with_memories(tmp_path, enabled=True)
    with session(daemon) as client:
        first = start(client)
        second = submit(client, first, "decision").json()
        assert second["step"] == "reason"
        assert settings(client, False).json()["active_round"] is None
        assert submit(client, first, "reason").status_code == 409

    resumed = daemon_with_memories(tmp_path, enabled=True)
    with session(resumed) as client:
        assert client.get("/choice/api/state").json()["enabled"] is False
        assert settings(client, True).json()["active_round"] == second
        assert submit(client, first, "reason").json()["step"] == "complete"
        assert resumed.llm.calls == 0


def test_reason_generation_failure_preserves_first_choice(tmp_path, monkeypatch):
    daemon = daemon_with_memories(tmp_path, enabled=True)
    real = daemon.llm.chat_json

    async def bad_reasons(prompt, data, **kwargs):
        if prompt.startswith("[ROLE:choice_reasons]"):
            return {"question": "没有选项的输出"}
        return await real(prompt, data, **kwargs)

    monkeypatch.setattr(daemon.llm, "chat_json", bad_reasons)
    with session(daemon) as client:
        first = start(client)
        assert submit(client, first, "decision").status_code == 502
        saved = client.get("/choice/api/state").json()["active_round"]
        assert saved["step"] == "decision_saved" and saved["retry"] == "0"
        assert submit(client, first, "decision", "1").status_code == 409
        monkeypatch.setattr(daemon.llm, "chat_json", real)
        assert submit(client, first, "decision").json()["step"] == "reason"
        assert submit(client, first, "reason").json()["step"] == "complete"


def test_startup_can_open_enabled_page(tmp_path):
    daemon = daemon_with_memories(tmp_path, enabled=True)
    opened = Event()
    with session(daemon, opened.set) as client:
        client.wait(0.8)
        assert opened.is_set()
        assert daemon.llm.calls == 0


def test_choice_only_accessible_from_loopback(tmp_path):
    daemon = daemon_with_memories(tmp_path)
    with session(daemon) as client:
        assert client.get("/choice", headers={"Host": "outside.test"}).status_code == 403
        assert client.post("/choice/api/settings", json={"enabled": True},
                           headers={"Origin": "https://outside.test"}).status_code == 403


def test_estimates_need_breadth_and_shrink_toward_neutral():
    result = {"axis": "risk", "context": "项目与工作", "decision": "a"}
    assert estimates([result, result, result])[0]["a_share"] is None
    diverse = {**result, "context": "日常安排"}
    risk = estimates([result, result, diverse])[0]
    assert risk["observations"] == 3 and risk["contexts"] == 2
    assert 50 < risk["a_share"] < 100


def test_interrupted_export_retries_without_duplicate(tmp_path, monkeypatch):
    daemon = daemon_with_memories(tmp_path, enabled=True)
    write = daemon.choice_mode.formatter.write_personality_choice

    def failed(*args):
        raise OSError("synthetic disk error")

    with session(daemon) as client:
        first = start(client)
        assert submit(client, first, "decision").status_code == 200
        monkeypatch.setattr(daemon.choice_mode.formatter, "write_personality_choice", failed)
        assert submit(client, first, "reason").status_code == 503
        manifest = freeze(tmp_path / "dataset", tmp_path / "incomplete", 0)
        assert manifest["counts"] == {}
        monkeypatch.setattr(daemon.choice_mode.formatter, "write_personality_choice", write)
        assert client.get("/choice/api/state").status_code == 200
        assert submit(client, first, "reason").status_code == 200
        pretrain = read_jsonl(tmp_path / "dataset/pretrain/choices.jsonl")
        posterior = read_jsonl(tmp_path / "dataset/style/choices.jsonl")
        assert len(pretrain) == len(posterior) == 1
