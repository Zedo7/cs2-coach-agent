"""Coaching sessions: a structured ledger of real round outcomes and agent calls, a
K-round detail window, deterministic compaction of older rounds, adaptation pressure
derived from the ledger, and sliding TTL expiry.

Storage is behind a SessionStore ABC with two implementations:
  - InMemoryStore: zero-dependency default, single process, used by tests and the CLI.
  - RedisStore:    survives restarts, shared across workers, real TTL.
Selected by COACH_SESSION_BACKEND=memory|redis. Everything above this module is
backend-agnostic.

Concurrency (see README "Session persistence and concurrency"):
  - Ledgers are APPEND-ONLY. Appends are commutative, so two concurrent writers cannot
    lose each other's entry -- no CAS, no lock, no retry.
  - get() returns a SNAPSHOT in both backends. Callers must not mutate it and expect the
    change to persist; a mutation that works in memory and silently vanishes against
    Redis is exactly the bug this abstraction exists to prevent.
  - Exactly-once side effects (compaction events, run admission) use SETNX guards.
"""

from __future__ import annotations

import copy
import json
import os
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Optional
from uuid import uuid4

KEY_PREFIX = "coach:sess"
DEFAULT_TTL = float(os.getenv("COACH_SESSION_TTL", "1800"))   # 30 min inactivity


class SessionNotFound(KeyError):
    """Raised by every backend when a session is absent or expired (never a bare
    KeyError, so callers behave identically regardless of backend)."""


@dataclass
class RoundOutcome:
    """One real round result submitted to the ledger (mirrors a rounds.jsonl record)."""
    round: int
    our_side: str
    their_side: str
    result: str            # win | loss
    winner: str            # us | them
    score_after: dict
    economy: dict
    buy: dict
    detail: str


@dataclass
class Call:
    """One recommendation the agent produced for a round."""
    round: int
    buy_type: str
    buy: list
    cost: int
    approved: bool
    attempts: int


@dataclass
class Session:
    id: str
    map: str
    us_team: str
    window: int
    created_at: float
    outcomes: list = field(default_factory=list)   # list[RoundOutcome]
    calls: list = field(default_factory=list)      # list[Call]

    # --- serialization ----------------------------------------------------------
    # JSON, not pickle: pickle executes arbitrary code on load (a compromised store
    # becomes RCE), couples the payload to exact class definitions (renaming a field
    # breaks every live session on deploy), and is opaque to redis-cli and to any
    # non-Python consumer. Our records are plain scalars/lists/dicts, so the only thing
    # JSON costs us is native datetimes -- and we already use float epochs.
    def meta(self) -> dict:
        return {"id": self.id, "map": self.map, "us_team": self.us_team,
                "window": str(self.window), "created_at": str(self.created_at)}

    @staticmethod
    def from_parts(meta: dict, outcomes: list, calls: list) -> "Session":
        return Session(id=meta["id"], map=meta["map"], us_team=meta["us_team"],
                       window=int(meta["window"]), created_at=float(meta["created_at"]),
                       outcomes=[RoundOutcome(**o) for o in outcomes],
                       calls=[Call(**c) for c in calls])


# --------------------------------------------------------------------------- interface

class SessionStore(ABC):
    """Storage contract. All methods async so the Redis backend is possible; the
    in-memory backend simply never awaits internally."""

    def __init__(self, ttl_seconds: float = DEFAULT_TTL, window: int = 5,
                 compact_batch: int = 5):
        self.ttl = ttl_seconds
        self.window = window
        self.compact_batch = compact_batch

    @abstractmethod
    async def create(self, map: str, us_team: str, window: Optional[int] = None) -> Session: ...

    @abstractmethod
    async def get(self, sid: str) -> Session:
        """Return a snapshot and refresh the sliding TTL. Raises SessionNotFound."""

    @abstractmethod
    async def append_outcome(self, sid: str, outcome: RoundOutcome) -> None: ...

    @abstractmethod
    async def record_call(self, sid: str, call: Call) -> None: ...

    @abstractmethod
    async def delete(self, sid: str) -> None: ...

    @abstractmethod
    async def touch(self, sid: str) -> None:
        """Refresh the sliding TTL without reading the session."""

    @abstractmethod
    async def claim_run(self, sid: str, ttl: float) -> Optional[str]:
        """Admission control for an expensive run. Returns a token, or None if a run is
        already in flight for this session (caller should 409)."""

    @abstractmethod
    async def release_run(self, sid: str, token: str) -> None:
        """Release only if we still hold it -- a run whose guard already expired must not
        release the next holder's guard."""

    @abstractmethod
    async def claim_compaction(self, sid: str, batch: int) -> bool:
        """True exactly once per (session, batch), so N workers emit one event."""

    async def close(self) -> None:
        return None


# --------------------------------------------------------------------------- in-memory

class InMemoryStore(SessionStore):
    """Single-process default. Mirrors the Redis semantics exactly -- including snapshot
    reads -- so the same tests pass against both."""

    def __init__(self, ttl_seconds: float = DEFAULT_TTL, window: int = 5,
                 compact_batch: int = 5, time_fn=time.time):
        super().__init__(ttl_seconds, window, compact_batch)
        self._sessions: dict[str, Session] = {}
        self._expires: dict[str, float] = {}
        self._runs: dict[str, tuple[str, float]] = {}       # sid -> (token, expires_at)
        self._compactions: set[tuple[str, int]] = set()
        self._now = time_fn

    def _sweep(self) -> None:
        now = self._now()
        for sid in [s for s, exp in self._expires.items() if exp <= now]:
            self._sessions.pop(sid, None)
            self._expires.pop(sid, None)

    def _live(self, sid: str) -> Session:
        self._sweep()
        s = self._sessions.get(sid)
        if s is None:
            raise SessionNotFound(sid)
        self._expires[sid] = self._now() + self.ttl
        return s

    async def create(self, map, us_team, window=None) -> Session:
        self._sweep()
        s = Session(id="sesn_" + uuid4().hex[:16], map=map, us_team=us_team,
                    window=window or self.window, created_at=self._now())
        self._sessions[s.id] = s
        self._expires[s.id] = self._now() + self.ttl
        return copy.deepcopy(s)

    async def get(self, sid) -> Session:
        # deepcopy: callers get a snapshot, never a live reference they could mutate
        return copy.deepcopy(self._live(sid))

    async def append_outcome(self, sid, outcome) -> None:
        self._live(sid).outcomes.append(copy.deepcopy(outcome))

    async def record_call(self, sid, call) -> None:
        self._live(sid).calls.append(copy.deepcopy(call))

    async def delete(self, sid) -> None:
        self._sessions.pop(sid, None)
        self._expires.pop(sid, None)

    async def touch(self, sid) -> None:
        self._live(sid)

    async def claim_run(self, sid, ttl) -> Optional[str]:
        now = self._now()
        held = self._runs.get(sid)
        if held and held[1] > now:
            return None
        token = uuid4().hex
        self._runs[sid] = (token, now + ttl)
        return token

    async def release_run(self, sid, token) -> None:
        held = self._runs.get(sid)
        if held and held[0] == token:
            self._runs.pop(sid, None)

    async def claim_compaction(self, sid, batch) -> bool:
        key = (sid, batch)
        if key in self._compactions:
            return False
        self._compactions.add(key)
        return True


# --------------------------------------------------------------------------- redis

class RedisStore(SessionStore):
    """Redis backend. A session is several keys:
        {p}:{sid}:meta        HASH  immutable-ish metadata
        {p}:{sid}:outcomes    LIST  append-only ledger (RPUSH)
        {p}:{sid}:calls       LIST  append-only ledger (RPUSH)
        {p}:{sid}:run         STR   run-admission guard (SET NX PX)
        {p}:{sid}:cmpct:{n}   STR   exactly-once compaction guard (SET NX PX)

    PARTIAL-EXPIRY HAZARD: per-key TTL means the meta hash could expire while a ledger
    list survives, leaving a corrupt half-session. So every access refreshes ALL keys in
    a single pipeline, and a missing meta hash is the authoritative "gone" signal --
    orphaned ledgers self-clean via their own TTL.
    """

    def __init__(self, redis, ttl_seconds: float = DEFAULT_TTL, window: int = 5,
                 compact_batch: int = 5):
        super().__init__(ttl_seconds, window, compact_batch)
        self._r = redis

    def _k(self, sid: str, part: str) -> str:
        return f"{KEY_PREFIX}:{sid}:{part}"

    def _ttl_ms(self) -> int:
        # PEXPIRE (milliseconds), not EXPIRE (integer seconds), so sub-second TTLs are
        # expressible -- tests need them and integer seconds would silently round to 0.
        return max(1, int(self.ttl * 1000))

    def _refresh(self, pipe, sid: str) -> None:
        for part in ("meta", "outcomes", "calls"):
            pipe.pexpire(self._k(sid, part), self._ttl_ms())

    async def create(self, map, us_team, window=None) -> Session:
        s = Session(id="sesn_" + uuid4().hex[:16], map=map, us_team=us_team,
                    window=window or self.window, created_at=time.time())
        async with self._r.pipeline(transaction=True) as pipe:
            pipe.hset(self._k(s.id, "meta"), mapping=s.meta())
            self._refresh(pipe, s.id)
            await pipe.execute()
        return s

    async def get(self, sid) -> Session:
        # MULTI/EXEC: the three reads are one atomic snapshot rather than three
        # independent points in time, so a concurrent append can never be observed
        # half-applied (meta from before, ledger from after).
        async with self._r.pipeline(transaction=True) as pipe:
            pipe.hgetall(self._k(sid, "meta"))
            pipe.lrange(self._k(sid, "outcomes"), 0, -1)
            pipe.lrange(self._k(sid, "calls"), 0, -1)
            self._refresh(pipe, sid)
            meta, outcomes, calls, *_ = await pipe.execute()
        if not meta:
            raise SessionNotFound(sid)
        return Session.from_parts(meta, [json.loads(o) for o in outcomes],
                                  [json.loads(c) for c in calls])

    async def _append(self, sid: str, part: str, item) -> None:
        # Existence check first so appending to a dead session raises rather than
        # resurrecting a ledger with no metadata.
        if not await self._r.exists(self._k(sid, "meta")):
            raise SessionNotFound(sid)
        async with self._r.pipeline(transaction=True) as pipe:
            # RPUSH is atomic and commutative: two concurrent appends both survive.
            pipe.rpush(self._k(sid, part), json.dumps(asdict(item), ensure_ascii=False))
            self._refresh(pipe, sid)
            await pipe.execute()

    async def append_outcome(self, sid, outcome) -> None:
        await self._append(sid, "outcomes", outcome)

    async def record_call(self, sid, call) -> None:
        await self._append(sid, "calls", call)

    async def delete(self, sid) -> None:
        await self._r.delete(*(self._k(sid, p) for p in ("meta", "outcomes", "calls", "run")))

    async def touch(self, sid) -> None:
        if not await self._r.exists(self._k(sid, "meta")):
            raise SessionNotFound(sid)
        async with self._r.pipeline(transaction=True) as pipe:
            self._refresh(pipe, sid)
            await pipe.execute()

    async def claim_run(self, sid, ttl) -> Optional[str]:
        token = uuid4().hex
        ok = await self._r.set(self._k(sid, "run"), token, nx=True, px=max(1, int(ttl * 1000)))
        return token if ok else None

    async def release_run(self, sid, token) -> None:
        # WATCH/MULTI rather than a Lua script: same check-then-delete atomicity without
        # requiring the Lua-enabled server/fake extra.
        key = self._k(sid, "run")
        async with self._r.pipeline(transaction=True) as pipe:
            try:
                await pipe.watch(key)
                if await pipe.get(key) != token:
                    await pipe.reset()
                    return
                pipe.multi()
                pipe.delete(key)
                await pipe.execute()
            except Exception:
                await pipe.reset()

    async def claim_compaction(self, sid, batch) -> bool:
        ok = await self._r.set(self._k(sid, f"cmpct:{batch}"), "1", nx=True, px=self._ttl_ms())
        return bool(ok)

    async def close(self) -> None:
        await self._r.aclose()


def make_store(backend: Optional[str] = None, **kwargs) -> SessionStore:
    """Factory from env: COACH_SESSION_BACKEND=memory|redis, COACH_REDIS_URL."""
    backend = (backend or os.getenv("COACH_SESSION_BACKEND", "memory")).lower()
    if backend == "memory":
        return InMemoryStore(**kwargs)
    if backend == "redis":
        import redis.asyncio as aioredis
        url = os.getenv("COACH_REDIS_URL", "redis://localhost:6379/0")
        return RedisStore(aioredis.from_url(url, decode_responses=True), **kwargs)
    raise ValueError(f"unknown COACH_SESSION_BACKEND: {backend}")


# --------------------------------------------------------------------------- grounding

def _compaction_synopsis(older: list) -> dict:
    """Deterministic roll-up of rounds that have fallen out of the detail window. No model
    call -- pure aggregation. (Roadmap: a model-summary variant would capture tactical
    nuance a counter cannot; see README.)"""
    us_w = sum(1 for o in older if o.winner == "us")
    them_w = len(older) - us_w
    econ = [o.economy["us_avg"] for o in older]
    trend = "flat"
    if len(econ) >= 2:
        trend = "rising" if econ[-1] > econ[0] + 500 else "falling" if econ[-1] < econ[0] - 500 else "flat"
    loss_tails = Counter(o.detail.split("— ", 1)[-1] for o in older if o.result == "loss")
    common = loss_tails.most_common(1)
    return {
        "type": "deterministic",
        "rounds": [older[0].round, older[-1].round],
        "record": f"us {us_w}-{them_w} them",
        "economy_trend": trend,
        "recurring_loss": (f"{common[0][0]} (x{common[0][1]})" if common else None),
    }


def _adaptation_pressure(session: Session) -> tuple[int, Optional[str]]:
    """Escalating signal from the ledger: losing while repeating yourself. Level 0/1/2."""
    outs = session.outcomes
    if not outs or outs[-1].result != "loss":
        return 0, None
    streak = 0
    for o in reversed(outs):
        if o.result == "loss":
            streak += 1
        else:
            break
    if streak >= 2:
        recent_calls = session.calls[-streak:] if session.calls else []
        repeated = (len({c.buy_type for c in recent_calls}) == 1 and recent_calls)
        msg = (f"ADAPTATION PRESSURE (high): you have lost {streak} rounds in a row"
               + (" while calling the same buy each time" if repeated else "")
               + ". The previous approach is not working — change the buy and the plan, "
                 "do not repeat what just failed.")
        return 2, msg
    return 1, ("ADAPTATION PRESSURE (low): last round was a loss. Weigh whether to adjust "
               "the approach rather than repeat it.")


def build_grounding(session: Session, compact_batch: int = 5):
    """PURE. Assemble what the session feeds the agent for the upcoming round.

    Returns (recent_rounds, extra_context, compaction_or_None). It does NOT mutate the
    session and does NOT decide whether the compaction event is emitted -- it reports
    which batch is pending, and the caller asks the store (claim_compaction) so exactly
    one worker emits it. Mutating a snapshot here would silently vanish under Redis."""
    outs = session.outcomes
    window = session.window
    recent = outs[-window:]
    older = outs[:-window] if len(outs) > window else []

    recent_rounds = [{"round": o.round, "result": o.result, "detail": o.detail} for o in recent]

    parts, compaction = [], None
    if older:
        synopsis = _compaction_synopsis(older)
        parts.append(
            f"EARLIER ROUNDS (deterministic compaction of rounds "
            f"{synopsis['rounds'][0]}-{synopsis['rounds'][1]}): {synopsis['record']}, "
            f"economy {synopsis['economy_trend']}"
            + (f", recurring loss: {synopsis['recurring_loss']}" if synopsis['recurring_loss'] else ""))
        batch = len(older) // max(1, compact_batch)
        if batch > 0:
            compaction = {"batch": batch, "compacted_rounds": len(older),
                          "window": window, **synopsis}

    _, pmsg = _adaptation_pressure(session)
    if pmsg:
        parts.append(pmsg)

    return recent_rounds, ("\n\n".join(parts) if parts else None), compaction
