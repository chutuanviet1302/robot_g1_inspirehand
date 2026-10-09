"""SQLite store for evaluation runs and episodes (no database server needed, works on Windows and Linux)."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager

from homehand import paths

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created REAL NOT NULL,
    name TEXT NOT NULL,
    task TEXT NOT NULL,
    controller TEXT NOT NULL,
    perception TEXT NOT NULL,
    randomization TEXT NOT NULL,
    n_episodes INTEGER NOT NULL,
    seed0 INTEGER NOT NULL,
    status TEXT NOT NULL,
    done INTEGER NOT NULL DEFAULT 0,
    summary TEXT,
    elapsed_s REAL
);
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    seed INTEGER NOT NULL,
    outcome TEXT NOT NULL,
    success INTEGER NOT NULL,
    n_cleared INTEGER NOT NULL,
    n_objects INTEGER NOT NULL,
    steps INTEGER NOT NULL,
    replay TEXT,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS episodes_run ON episodes(run_id);
"""


@contextmanager
def connect():
    paths.DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(paths.DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def create_run(name, task, controller, perception, randomization, n_episodes, seed0) -> int:
    with connect() as con:
        cur = con.execute(
            "INSERT INTO runs(created,name,task,controller,perception,randomization,n_episodes,seed0,status) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (time.time(), name, task, controller, perception, randomization, n_episodes, seed0, "running"))
        return int(cur.lastrowid)


def add_episode(run_id: int, ep: dict, replay: str | None) -> None:
    with connect() as con:
        con.execute(
            "INSERT INTO episodes(run_id,seed,outcome,success,n_cleared,n_objects,steps,replay,data) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (run_id, ep["seed"], ep["outcome"], int(ep["success"]), ep["n_cleared"], ep["n_objects"], ep["steps"],
             replay, json.dumps(ep)))
        con.execute("UPDATE runs SET done = done + 1 WHERE id = ?", (run_id,))


def finish_run(run_id: int, summary: dict, elapsed: float, status: str = "done") -> None:
    with connect() as con:
        con.execute("UPDATE runs SET status=?, summary=?, elapsed_s=? WHERE id=?",
                    (status, json.dumps(summary), elapsed, run_id))


def list_runs() -> list[dict]:
    with connect() as con:
        rows = con.execute("SELECT * FROM runs ORDER BY id DESC").fetchall()
    return [_run(r) for r in rows]


def get_run(run_id: int) -> dict | None:
    with connect() as con:
        r = con.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    return _run(r) if r else None


def delete_run(run_id: int) -> None:
    with connect() as con:
        con.execute("DELETE FROM episodes WHERE run_id=?", (run_id,))
        con.execute("DELETE FROM runs WHERE id=?", (run_id,))


def list_episodes(run_id: int | None = None, outcome: str | None = None, limit: int = 500) -> list[dict]:
    q = "SELECT e.*, r.name AS run_name, r.controller, r.perception, r.randomization FROM episodes e " \
        "JOIN runs r ON r.id = e.run_id WHERE 1=1"
    args: list = []
    if run_id is not None:
        q += " AND e.run_id=?"
        args.append(run_id)
    if outcome:
        q += " AND e.outcome=?"
        args.append(outcome)
    q += " ORDER BY e.id DESC LIMIT ?"
    args.append(limit)
    with connect() as con:
        rows = con.execute(q, args).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        data = json.loads(d.pop("data"))
        d["object_outcomes"] = data.get("object_outcomes", {})
        d["log"] = data.get("log", [])
        d["objects"] = data.get("objects", [])
        out.append(d)
    return out


def get_episode(ep_id: int) -> dict | None:
    with connect() as con:
        r = con.execute("SELECT * FROM episodes WHERE id=?", (ep_id,)).fetchone()
    if not r:
        return None
    d = dict(r)
    d["data"] = json.loads(d["data"])
    return d


def _run(r) -> dict:
    d = dict(r)
    d["summary"] = json.loads(d["summary"]) if d["summary"] else None
    return d
