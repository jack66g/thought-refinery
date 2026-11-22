"""常驻队列的接收、分支检查点与恢复。"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from ..core.cleaner import redact


class EventJournal:
    def __init__(self, path: Path, capacity: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.capacity = capacity
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                key TEXT PRIMARY KEY, event_id TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL, payload TEXT, state TEXT NOT NULL,
                created REAL NOT NULL, ready REAL NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                cot TEXT, result TEXT, finished REAL
            );
            CREATE INDEX IF NOT EXISTS events_ready ON events(state, ready, created);
        """)

    def recover(self) -> None:
        with self.db:
            self.db.execute("UPDATE events SET state='pending' WHERE state='running'")

    def enqueue(self, event: dict[str, Any]) -> bool:
        if event.get("source_event_id"):
            identity = [event.get("source"), event.get("session_id"), event["source_event_id"]]
        else:
            identity = [event.get(key) for key in (
                "source", "session_id", "event_kind", "task", "thinking", "conclusion", "transcript")]
        key = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            old = self.db.execute("SELECT event_id FROM events WHERE key=?", (key,)).fetchone()
            if old is not None:
                event["id"] = old["event_id"]
                return True
            if self.count_pending() >= self.capacity:
                return False
            now = time.time()
            self.db.execute(
                "INSERT INTO events(key,event_id,kind,payload,state,created,ready) VALUES(?,?,?,?,'pending',?,?)",
                (key, event["id"], event["event_kind"], json.dumps(redact(event), ensure_ascii=False), now, now))
        return True

    def count_pending(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM events WHERE state IN ('pending','running','budget')").fetchone()[0]

    def count_today(self, kind: str | None = None) -> int:
        midnight = time.mktime(time.strptime(time.strftime("%Y-%m-%d"), "%Y-%m-%d"))
        if kind is None:
            return self.db.execute("SELECT COUNT(*) FROM events WHERE created>=?", (midnight,)).fetchone()[0]
        return self.db.execute(
            "SELECT COUNT(*) FROM events WHERE created>=? AND kind=?", (midnight, kind)).fetchone()[0]

    def claim(self) -> tuple[dict[str, Any], dict[str, Any] | None, int] | None:
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row = self.db.execute(
                "SELECT * FROM events WHERE state IN ('pending','budget') AND ready<=? ORDER BY created LIMIT 1",
                (time.time(),)).fetchone()
            if row is None:
                return None
            self.db.execute(
                "UPDATE events SET state='running', attempts=attempts+1 WHERE key=?", (row["key"],))
        event = json.loads(row["payload"])
        event["replaying"] = row["attempts"] > 0
        return event, json.loads(row["cot"]) if row["cot"] else None, row["attempts"] + 1

    def save_cot(self, event_id: str, result: dict[str, Any]) -> None:
        with self.db:
            self.db.execute("UPDATE events SET cot=? WHERE event_id=?",
                            (json.dumps(redact(result), ensure_ascii=False), event_id))

    def retry(self, event_id: str, ready: float, *, count_attempt: bool = True) -> None:
        with self.db:
            self.db.execute(
                "UPDATE events SET state=?, ready=?, attempts=attempts-? WHERE event_id=?",
                ("pending" if count_attempt else "budget", ready, int(not count_attempt), event_id))

    def release_budget_wait(self) -> None:
        with self.db:
            self.db.execute("UPDATE events SET ready=? WHERE state='budget'", (time.time(),))

    def finish(self, event_id: str, result: dict[str, Any], *, failed: bool = False) -> None:
        with self.db:
            self.db.execute(
                "UPDATE events SET state=?, result=?, finished=?, payload=NULL, cot=NULL WHERE event_id=?",
                ("failed" if failed else "done", json.dumps(redact(result), ensure_ascii=False),
                 time.time(), event_id))

    def recent(self) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT result FROM events WHERE finished IS NOT NULL ORDER BY finished DESC LIMIT 5").fetchall()
        return [json.loads(row["result"]) for row in reversed(rows)]

    def close(self) -> None:
        self.db.close()
