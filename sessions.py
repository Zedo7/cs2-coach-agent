"""In-memory coaching sessions: a structured ledger of real round outcomes and agent
calls, a K-round detail window, deterministic compaction of older rounds, adaptation
pressure derived from the ledger, and TTL expiry. No database -- state lives in process
and is lost on restart, which is fine for a demo/eval surface.

The session layer grounds the (stateless) agent: it assembles the MatchState the agent
sees and injects extra context (compaction synopsis + adaptation pressure) WITHOUT
touching the agent's own prompts.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4


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
    last_active: float
    outcomes: list = field(default_factory=list)   # list[RoundOutcome]
    calls: list = field(default_factory=list)       # list[Call]
    compacted_batches: int = 0                       # how many COMPACT_BATCH folds emitted

    def touch(self) -> None:
        self.last_active = time.time()


class SessionStore:
    def __init__(self, ttl_seconds: float = 1800.0, window: int = 5,
                 compact_batch: int = 5):
        self.ttl = ttl_seconds
        self.window = window
        self.compact_batch = compact_batch
        self._sessions: dict[str, Session] = {}

    # --- lifecycle ----------------------------------------------------------
    def _sweep(self) -> None:
        now = time.time()
        dead = [sid for sid, s in self._sessions.items() if now - s.last_active > self.ttl]
        for sid in dead:
            del self._sessions[sid]

    def create(self, map: str, us_team: str, window: Optional[int] = None) -> Session:
        self._sweep()
        now = time.time()
        s = Session(id="sesn_" + uuid4().hex[:16], map=map, us_team=us_team,
                    window=window or self.window, created_at=now, last_active=now)
        self._sessions[s.id] = s
        return s

    def get(self, sid: str) -> Session:
        self._sweep()
        s = self._sessions.get(sid)
        if s is None:
            raise KeyError(sid)
        s.touch()
        return s

    def delete(self, sid: str) -> None:
        self._sessions.pop(sid, None)

    def __len__(self) -> int:
        return len(self._sessions)

    # --- ledger mutation ----------------------------------------------------
    def submit_round(self, sid: str, outcome: RoundOutcome) -> Session:
        s = self.get(sid)
        s.outcomes.append(outcome)
        s.touch()
        return s

    def record_call(self, sid: str, call: Call) -> None:
        s = self.get(sid)
        s.calls.append(call)
        s.touch()


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
    # cheapest "recurring pattern" signal: the most common detail tail among our losses
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
    # count the current losing streak
    streak = 0
    for o in reversed(outs):
        if o.result == "loss":
            streak += 1
        else:
            break
    if streak >= 2:
        # did our recent calls repeat the same buy_type through the streak?
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
    """Assemble everything the session feeds the agent for the upcoming round, from the
    ledger. Returns (recent_rounds, extra_context, compaction_event_or_None).

    - recent_rounds: the last K outcomes, in the agent's {round,result,detail} shape.
    - extra_context: compaction synopsis + adaptation pressure, appended to the doer
      prompt (the agent's own prompts are untouched).
    - compaction_event: emitted (batched) the first time each COMPACT_BATCH of rounds
      falls out of the detail window."""
    outs = session.outcomes
    window = session.window
    recent = outs[-window:]
    older = outs[:-window] if len(outs) > window else []

    recent_rounds = [{"round": o.round, "result": o.result, "detail": o.detail} for o in recent]

    parts, compaction_event = [], None
    synopsis = None
    if older:
        synopsis = _compaction_synopsis(older)
        parts.append(
            f"EARLIER ROUNDS (deterministic compaction of rounds "
            f"{synopsis['rounds'][0]}-{synopsis['rounds'][1]}): {synopsis['record']}, "
            f"economy {synopsis['economy_trend']}"
            + (f", recurring loss: {synopsis['recurring_loss']}" if synopsis['recurring_loss'] else ""))
        # batched emission: one event per COMPACT_BATCH rounds folded
        batches = len(older) // max(1, compact_batch)
        if batches > session.compacted_batches:
            session.compacted_batches = batches
            compaction_event = {"type": "deterministic", **synopsis,
                                "compacted_rounds": len(older), "window": window}

    level, pmsg = _adaptation_pressure(session)
    if pmsg:
        parts.append(pmsg)

    extra_context = "\n\n".join(parts) if parts else None
    return recent_rounds, extra_context, compaction_event
