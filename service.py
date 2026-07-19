"""Production-style HTTP service around cs2-coach-agent.

Endpoints:
  POST /v1/coach          -> run the agent, return the final plan + usage as JSON
  POST /v1/coach/stream   -> stream the run as Server-Sent Events (trace-as-product)
  GET  /healthz           -> liveness + readiness
  GET  /                  -> minimal demo page

Config is via environment variables (see README "Run as a service"). The agent core
is untouched; this is a service layer that consumes runner.run_agent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional
from uuid import uuid4

import anthropic
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

import agent
from runner import CONFIGS, TraceEvent, run_agent
from sessions import Call, RoundOutcome, SessionStore, build_grounding

ROOT = Path(__file__).parent

# --- configuration (env vars with defaults) ---------------------------------
MAX_CONCURRENT = int(os.getenv("COACH_MAX_CONCURRENT_RUNS", "2"))
RUN_TIMEOUT = float(os.getenv("COACH_RUN_TIMEOUT_SECONDS", "120"))
MODEL = os.getenv("COACH_MODEL", agent.MODEL)
MAX_ATTEMPTS = int(os.getenv("COACH_MAX_ATTEMPTS", "3"))
RETRY_AFTER = int(os.getenv("COACH_RETRY_AFTER", "10"))
LOG_LEVEL = os.getenv("COACH_LOG_LEVEL", "INFO").upper()
SSE_PING_SECONDS = 15.0
SESSION_WINDOW = int(os.getenv("COACH_SESSION_WINDOW", "5"))
SESSION_TTL = float(os.getenv("COACH_SESSION_TTL", "1800"))
COMPACT_BATCH = int(os.getenv("COACH_COMPACT_BATCH", "5"))

# --- structured logging ------------------------------------------------------
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class JsonLogFormatter(logging.Formatter):
    """One JSON object per line: ts, level, request_id, event, plus any `data` extra."""

    def format(self, record: logging.LogRecord) -> str:
        obj = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "request_id": getattr(record, "request_id", None) or request_id_var.get(),
            "event": record.getMessage(),
        }
        if hasattr(record, "data"):
            obj["data"] = record.data
        if record.exc_info:
            obj["exc"] = self.formatException(record.exc_info)
        return json.dumps(obj, ensure_ascii=False)


log = logging.getLogger("coach.service")


def _setup_logging() -> None:
    if log.handlers:
        return
    (ROOT / "results").mkdir(exist_ok=True)
    fmt = JsonLogFormatter()
    for h in (logging.StreamHandler(sys.stderr),
              logging.FileHandler(ROOT / "results" / "service.log", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.setLevel(LOG_LEVEL)
    log.propagate = False


def _log(level: int, event: str, **data: Any) -> None:
    log.log(level, event, extra={"data": data} if data else None)


# --- concurrency guard -------------------------------------------------------
class RunSlots:
    """Non-blocking, race-free bounded gate. Acquire increments a counter under a lock;
    if capacity is reached it returns False immediately rather than waiting. Using a
    lock-guarded counter (not Semaphore._value) makes the check-and-take atomic, so the
    stream endpoint can decide 429-vs-run *before* it opens a 200 response."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._in_use = 0
        self._lock = asyncio.Lock()

    async def try_acquire(self) -> bool:
        async with self._lock:
            if self._in_use >= self.capacity:
                return False
            self._in_use += 1
            return True

    async def release(self) -> None:
        async with self._lock:
            self._in_use = max(0, self._in_use - 1)


slots = RunSlots(MAX_CONCURRENT)
store = SessionStore(ttl_seconds=SESSION_TTL, window=SESSION_WINDOW, compact_batch=COMPACT_BATCH)


def make_client() -> anthropic.AsyncAnthropic:
    """Indirection so tests can monkeypatch a mock client -- zero real API calls."""
    return anthropic.AsyncAnthropic()


# --- request schema (mirrors sample_match.json + the edge scenarios) ---------
class Score(BaseModel):
    us: int = Field(ge=0)
    them: int = Field(ge=0)


class Economy(BaseModel):
    # extra="allow" keeps optional fields like us_spread (edge_01) instead of dropping them.
    model_config = ConfigDict(extra="allow")
    us_avg: int = Field(ge=0)
    them_avg: int = Field(ge=0)


class Round(BaseModel):
    model_config = ConfigDict(extra="allow")
    round: int
    result: str
    detail: str


class MatchState(BaseModel):
    model_config = ConfigDict(extra="allow")
    map: str = Field(min_length=1)
    side: Literal["T", "CT"]
    score: Score
    economy: Economy
    recent_rounds: list[Round]  # may be empty: the pistol-round probe (edge_03)
    roster: Optional[dict] = None


class ConfigName(str, Enum):
    full = "full"
    no_verifier = "no_verifier"
    no_hard_gate = "no_hard_gate"
    doer_only = "doer_only"


class CreateSession(BaseModel):
    map: str = Field(min_length=1)
    us_team: str = Field(min_length=1)
    window: Optional[int] = Field(default=None, ge=1, le=30)


class RoundIn(BaseModel):
    """A real round outcome submitted to the ledger (a rounds.jsonl record)."""
    model_config = ConfigDict(extra="allow")
    round: int
    our_side: Literal["T", "CT"]
    their_side: Literal["T", "CT"]
    result: Literal["win", "loss"]
    winner: Literal["us", "them"]
    score_after: dict
    economy: dict
    buy: dict
    detail: str


class CoachIn(BaseModel):
    """The known context for the UPCOMING round: our side + freeze-time money."""
    round: int
    our_side: Literal["T", "CT"]
    economy: Economy


# --- app ---------------------------------------------------------------------
_setup_logging()
app = FastAPI(title="cs2-coach-agent", version="0.3.0")


@app.middleware("http")
async def request_id_mw(request: Request, call_next):
    rid = request.headers.get("x-request-id") or str(uuid4())
    token = request_id_var.set(rid)
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = rid
    return response


async def _pump(agen, total: float, ping: Optional[float] = None):
    """Consume an async generator under a hard wall-clock deadline.

    The generator runs as a background task feeding a queue; the reader waits on the
    queue with its own (shorter) timeout so it can emit keepalive pings WITHOUT
    cancelling the in-flight model call. Only the overall deadline cancels the work.
    Yields ("event", ev) and, when ping is set, ("ping", None). Raises TimeoutError
    at the deadline."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + total
    q: asyncio.Queue = asyncio.Queue()

    async def produce():
        try:
            async for ev in agen:
                await q.put(("event", ev))
        except Exception as e:  # surfaced to the reader, which decides how to report it
            await q.put(("error", e))
            return
        await q.put(("done", None))

    task = asyncio.create_task(produce())
    try:
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError
            wait = remaining if ping is None else min(ping, remaining)
            try:
                kind, item = await asyncio.wait_for(q.get(), wait)
            except asyncio.TimeoutError:
                if loop.time() >= deadline:
                    raise
                yield ("ping", None)
                continue
            if kind == "done":
                return
            if kind == "error":
                raise item
            yield ("event", item)
    finally:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


@app.post("/v1/coach")
async def coach(body: MatchState, config: ConfigName = Query(ConfigName.full)):
    rid = request_id_var.get()
    if not await slots.try_acquire():
        _log(logging.WARNING, "rejected_at_capacity", config=config.value)
        raise HTTPException(status_code=429, detail="server at capacity",
                            headers={"Retry-After": str(RETRY_AFTER)})
    _log(logging.INFO, "run_start", config=config.value, endpoint="coach")
    client = make_client()
    events: list[TraceEvent] = []
    final: Optional[TraceEvent] = None
    try:
        agen = run_agent(body.model_dump(exclude_none=True), config=config.value,
                         request_id=rid, client=client, model=MODEL, max_attempts=MAX_ATTEMPTS)
        try:
            async for kind, ev in _pump(agen, RUN_TIMEOUT):
                events.append(ev)
                if ev.event == "final":
                    final = ev
        except asyncio.TimeoutError:
            _log(logging.ERROR, "run_timeout", seconds=RUN_TIMEOUT, events=len(events))
            return JSONResponse(status_code=504, content={
                "request_id": rid, "error": "run exceeded timeout",
                "timeout_seconds": RUN_TIMEOUT,
                "partial_trace": [{"event": e.event, **e.data} for e in events]})
        except Exception as e:
            _log(logging.ERROR, "run_error", error=f"{type(e).__name__}: {e}")
            return JSONResponse(status_code=500, content={
                "request_id": rid, "error": f"{type(e).__name__}: {e}"})
        _log(logging.INFO, "run_done", approved=final.data["approved"],
             attempts=final.data["attempts"], cost_usd=final.data["usage"]["cost_usd"])
        return JSONResponse(final.data)
    finally:
        await client.close()
        await slots.release()


@app.post("/v1/coach/stream")
async def coach_stream(body: MatchState, config: ConfigName = Query(ConfigName.full)):
    rid = request_id_var.get()
    # Acquire the slot BEFORE opening the stream: a rejection is a clean 429, never a
    # 200 event-stream that later emits a saturation error.
    if not await slots.try_acquire():
        _log(logging.WARNING, "rejected_at_capacity", config=config.value)
        raise HTTPException(status_code=429, detail="server at capacity",
                            headers={"Retry-After": str(RETRY_AFTER)})
    _log(logging.INFO, "run_start", config=config.value, endpoint="stream")
    client = make_client()
    match = body.model_dump(exclude_none=True)

    async def gen():
        # Re-set the contextvar: the middleware's reset() already ran by the time this
        # generator streams, so log lines here would otherwise lose the request id.
        request_id_var.set(rid)
        try:
            agen = run_agent(match, config=config.value, request_id=rid,
                            client=client, model=MODEL, max_attempts=MAX_ATTEMPTS)
            async for kind, ev in _pump(agen, RUN_TIMEOUT, ping=SSE_PING_SECONDS):
                if kind == "ping":
                    yield ": ping\n\n"
                    continue
                if ev.event == "final":
                    _log(logging.INFO, "run_done", approved=ev.data["approved"],
                         attempts=ev.data["attempts"], cost_usd=ev.data["usage"]["cost_usd"])
                yield ev.sse()
        except asyncio.TimeoutError:
            _log(logging.ERROR, "run_timeout", seconds=RUN_TIMEOUT)
            yield TraceEvent("error", {"request_id": rid, "error": "run exceeded timeout",
                                       "phase": "timeout", "partial": True}).sse()
        except Exception as e:
            _log(logging.ERROR, "run_error", error=f"{type(e).__name__}: {e}")
            yield TraceEvent("error", {"request_id": rid, "error": f"{type(e).__name__}: {e}",
                                       "phase": "exception", "partial": True}).sse()
        finally:
            await client.close()
            await slots.release()

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "X-Request-ID": rid, "Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _session_state(s) -> dict:
    return {"session_id": s.id, "map": s.map, "us_team": s.us_team, "window": s.window,
            "rounds_submitted": len(s.outcomes), "calls_made": len(s.calls),
            "compacted_batches": s.compacted_batches,
            "score": s.outcomes[-1].score_after if s.outcomes else {"us": 0, "them": 0}}


@app.post("/v1/sessions")
async def create_session(body: CreateSession):
    s = store.create(map=body.map, us_team=body.us_team, window=body.window)
    _log(logging.INFO, "session_created", session_id=s.id, map=s.map)
    return JSONResponse(_session_state(s), status_code=201)


@app.get("/v1/sessions/{sid}")
async def get_session(sid: str):
    try:
        return _session_state(store.get(sid))
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found or expired")


@app.delete("/v1/sessions/{sid}")
async def delete_session(sid: str):
    store.delete(sid)
    return JSONResponse({"deleted": sid}, status_code=200)


@app.post("/v1/sessions/{sid}/round")
async def submit_round(sid: str, body: RoundIn):
    try:
        s = store.submit_round(sid, RoundOutcome(**body.model_dump()))
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found or expired")
    _log(logging.INFO, "round_submitted", session_id=sid, round=body.round, result=body.result)
    return _session_state(s)


@app.post("/v1/sessions/{sid}/coach")
async def coach_session(sid: str, body: CoachIn, config: ConfigName = Query(ConfigName.full)):
    """Stream the agent's call for the UPCOMING round, grounded in the session ledger.
    Emits a `compaction` event (batched, deterministic) when older rounds fold out of the
    detail window, then the normal run trace. Records the call back into the ledger."""
    rid = request_id_var.get()
    try:
        session = store.get(sid)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found or expired")
    if not await slots.try_acquire():
        _log(logging.WARNING, "rejected_at_capacity", session_id=sid)
        raise HTTPException(status_code=429, detail="server at capacity",
                            headers={"Retry-After": str(RETRY_AFTER)})

    recent_rounds, extra_context, compaction_event = build_grounding(session, COMPACT_BATCH)
    score = session.outcomes[-1].score_after if session.outcomes else {"us": 0, "them": 0}
    match = {"map": session.map, "side": body.our_side, "score": score,
             "economy": body.economy.model_dump(), "recent_rounds": recent_rounds}
    client = make_client()

    async def gen():
        request_id_var.set(rid)
        try:
            if compaction_event:
                _log(logging.INFO, "compaction", session_id=sid, **compaction_event)
                yield TraceEvent("compaction", {"session_id": sid, **compaction_event}).sse()
            agen = run_agent(match, config=config.value, request_id=rid, client=client,
                             model=MODEL, max_attempts=MAX_ATTEMPTS, extra_context=extra_context)
            async for kind, ev in _pump(agen, RUN_TIMEOUT, ping=SSE_PING_SECONDS):
                if kind == "ping":
                    yield ": ping\n\n"
                    continue
                if ev.event == "final":
                    p = ev.data["plan"]
                    store.record_call(sid, Call(round=body.round, buy_type=p["buy_type"],
                                                buy=p["buy"], cost=ev.data["cost"],
                                                approved=ev.data["approved"],
                                                attempts=ev.data["attempts"]))
                    _log(logging.INFO, "run_done", session_id=sid, round=body.round,
                         approved=ev.data["approved"], cost_usd=ev.data["usage"]["cost_usd"])
                yield ev.sse()
        except asyncio.TimeoutError:
            _log(logging.ERROR, "run_timeout", session_id=sid, seconds=RUN_TIMEOUT)
            yield TraceEvent("error", {"request_id": rid, "error": "run exceeded timeout",
                                       "phase": "timeout", "partial": True}).sse()
        except Exception as e:
            _log(logging.ERROR, "run_error", session_id=sid, error=f"{type(e).__name__}: {e}")
            yield TraceEvent("error", {"request_id": rid, "error": f"{type(e).__name__}: {e}",
                                       "phase": "exception", "partial": True}).sse()
        finally:
            await client.close()
            await slots.release()

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "X-Request-ID": rid, "Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/healthz")
async def healthz():
    """Liveness is implicit (we answered). Readiness: key present, prices load, and the
    deterministic gate agrees with a known scenario's answer key -- all with no API calls."""
    checks: dict[str, Any] = {}
    checks["api_key"] = bool(os.environ.get("ANTHROPIC_API_KEY"))
    try:
        checks["prices_loaded"] = len(agent.PRICES) > 0
    except Exception:
        checks["prices_loaded"] = False
    try:
        scen = json.loads((ROOT / "scenarios" / "normal_01.json").read_text(encoding="utf-8"))
        kit = agent.FULL_BUY_KIT  # single source: same representative full buy the validator uses
        _, probs = agent.budget_check({"buy": kit, "per_player_spend": agent.kit_cost(kit)},
                                      scen["economy"]["us_avg"])
        checks["validator"] = ("full_buy" in scen["expected"]["buy_type"]) and not probs
    except Exception:
        checks["validator"] = False
    ready = all(checks.values())
    return JSONResponse(status_code=200 if ready else 503,
                        content={"live": True, "ready": ready, "checks": checks,
                                 "config": {"model": MODEL, "max_concurrent": MAX_CONCURRENT,
                                            "run_timeout_s": RUN_TIMEOUT, "max_attempts": MAX_ATTEMPTS}})


@app.get("/", response_class=HTMLResponse)
async def index():
    return (ROOT / "static" / "index.html").read_text(encoding="utf-8")
