"""v0.4 tests: demo-ingestion mapping, session mode, and replay -- all $0.

No real demo is parsed here (demoparser2 is never imported): the parser-facing code is
covered by exercising the pure round-assembly and schema layers against a hand-made
fixture, and the session/replay layers against the mocked client from conftest.
"""

import asyncio
import json
import time
from pathlib import Path

import httpx

import agent
import ingest
import service
from conftest import SAMPLE_MATCH, FakeClient
from sessions import Call, RoundOutcome, SessionStore, build_grounding
from replay import run_replay
from runner import run_agent
from conftest import FakeMessages

FIXTURE = Path(__file__).parent / "fixtures" / "mini_rounds.jsonl"


def _client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=service.app), base_url="http://test")


def _parse_sse(text):
    events = []
    for frame in text.split("\n\n"):
        frame = frame.strip("\n")
        if not frame or frame.startswith(":"):
            continue
        ev = dl = None
        for line in frame.splitlines():
            if line.startswith("event: "):
                ev = line[7:]
            elif line.startswith("data: "):
                dl = line[6:]
        if ev and dl:
            events.append((ev, json.loads(dl)))
    return events


# --------------------------------------------------------------------------- ingestion

def test_fixture_loads_and_validates():
    rounds = ingest.load_rounds(FIXTURE)
    assert len(rounds) == 12
    assert all(not ingest.validate_round_record(r) for r in rounds)


def test_validate_round_record_catches_errors():
    good = ingest.load_rounds(FIXTURE)[0]
    assert ingest.validate_round_record(good) == []
    # winner/result disagreement
    bad = dict(good, winner="them")
    assert any("winner/result" in p for p in ingest.validate_round_record(bad))
    # same side both teams
    bad2 = dict(good, their_side="CT")
    assert any("our_side == their_side" in p for p in ingest.validate_round_record(bad2))
    # bad buy label
    bad3 = json.loads(json.dumps(good)); bad3["buy"]["us"] = "banana"
    assert any("bad buy.us" in p for p in ingest.validate_round_record(bad3))


def test_build_round_record_maps_sides_and_classifies():
    # us on CT (3), CT wins -> we win; equip 4300 -> full_buy via the shared price table
    rec = ingest.build_round_record(
        round_no=1, us_side=3, winner_side=3,
        us_balances=[800] * 5, them_balances=[800] * 5,
        us_equip=[4300] * 5, them_equip=[250] * 5,
        score_after={"us": 1, "them": 0}, bomb="defused", site="B")
    assert rec["our_side"] == "CT" and rec["their_side"] == "T"
    assert rec["result"] == "win" and rec["winner"] == "us"
    assert rec["buy"]["us"] == "full_buy" and rec["buy"]["them"] == "pistol"
    assert rec["economy"]["us_avg"] == 800
    assert rec["detail"] == "CT win — bomb defused"
    assert ingest.validate_round_record(rec) == []


def test_build_round_record_loss_when_other_side_wins():
    rec = ingest.build_round_record(
        round_no=5, us_side=3, winner_side=2,          # us CT, T wins
        us_balances=[2000] * 5, them_balances=[4000] * 5,
        us_equip=[1000] * 5, them_equip=[4500] * 5,
        score_after={"us": 3, "them": 2}, bomb="exploded", site="A")
    assert rec["result"] == "loss" and rec["winner"] == "them"
    assert rec["detail"] == "T win — bomb detonated at A"


def test_detail_string_variants():
    assert ingest.detail_string("CT", "defused", None) == "CT win — bomb defused"
    assert ingest.detail_string("T", "exploded", "A") == "T win — bomb detonated at A"
    assert ingest.detail_string("CT", None, None) == "CT win — elimination / time"


# --------------------------------------------------------------------------- session store

def test_session_store_ttl_expiry():
    store = SessionStore(ttl_seconds=0.05)
    s = store.create(map="Inferno", us_team="FURIA")
    assert store.get(s.id).id == s.id
    time.sleep(0.1)
    try:
        store.get(s.id)
        assert False, "expected expiry"
    except KeyError:
        pass


def _outcomes_from_fixture():
    return [RoundOutcome(**{k: r[k] for k in (
        "round", "our_side", "their_side", "result", "winner",
        "score_after", "economy", "buy", "detail")}) for r in ingest.load_rounds(FIXTURE)]


def test_build_grounding_window_and_compaction():
    store = SessionStore(window=5, compact_batch=5)
    s = store.create(map="Inferno", us_team="FURIA")
    s.outcomes = _outcomes_from_fixture()          # 12 rounds
    recent, extra, compaction = build_grounding(s, compact_batch=5)
    assert len(recent) == 5                          # K-window
    assert recent[0]["round"] == 8 and recent[-1]["round"] == 12
    assert compaction is not None and compaction["type"] == "deterministic"
    assert "EARLIER ROUNDS (deterministic compaction" in extra
    # second call: same batch already emitted -> no new compaction event
    _, _, again = build_grounding(s, compact_batch=5)
    assert again is None


def test_build_grounding_adaptation_pressure():
    store = SessionStore(window=5, compact_batch=5)
    s = store.create(map="Inferno", us_team="FURIA")
    s.outcomes = _outcomes_from_fixture()          # ends on a long loss streak
    # simulate the agent having called the same buy through the streak
    s.calls = [Call(round=r, buy_type="full_buy", buy=["ak47"], cost=4200,
                    approved=True, attempts=1) for r in range(7, 13)]
    _, extra, _ = build_grounding(s, compact_batch=5)
    assert "ADAPTATION PRESSURE (high)" in extra
    assert "same buy" in extra


# --------------------------------------------------------------------------- session HTTP

def test_session_create_submit_get(monkeypatch):
    monkeypatch.setattr(service, "make_client", lambda: FakeClient())

    async def body():
        async with _client() as c:
            r = await c.post("/v1/sessions", json={"map": "Inferno", "us_team": "FURIA"})
            assert r.status_code == 201
            sid = r.json()["session_id"]
            round0 = ingest.load_rounds(FIXTURE)[0]
            r = await c.post(f"/v1/sessions/{sid}/round", json=round0)
            assert r.status_code == 200 and r.json()["rounds_submitted"] == 1
            r = await c.get(f"/v1/sessions/{sid}")
            assert r.json()["score"] == {"us": 1, "them": 0}
    asyncio.run(body())


def test_session_coach_streams_and_records_call(monkeypatch):
    monkeypatch.setattr(service, "make_client", lambda: FakeClient())

    async def body():
        async with _client() as c:
            sid = (await c.post("/v1/sessions", json={"map": "Inferno", "us_team": "FURIA"})).json()["session_id"]
            coach_in = {"round": 1, "our_side": "T", "economy": {"us_avg": 800, "them_avg": 800}}
            async with c.stream("POST", f"/v1/sessions/{sid}/coach", json=coach_in) as r:
                assert r.status_code == 200
                text = "".join([chunk async for chunk in r.aiter_text()])
            names = [e for e, _ in _parse_sse(text)]
            assert names[0] == "run_started" and names[-1] == "final"
            # the call was recorded into the ledger
            st = (await c.get(f"/v1/sessions/{sid}")).json()
            assert st["calls_made"] == 1
    asyncio.run(body())


def test_session_404_on_unknown_id(monkeypatch):
    monkeypatch.setattr(service, "make_client", lambda: FakeClient())

    async def body():
        async with _client() as c:
            assert (await c.get("/v1/sessions/nope")).status_code == 404
            assert (await c.post("/v1/sessions/nope/round",
                                 json=ingest.load_rounds(FIXTURE)[0])).status_code == 404
            assert (await c.post("/v1/sessions/nope/coach",
                                 json={"round": 1, "our_side": "T",
                                       "economy": {"us_avg": 800, "them_avg": 800}})).status_code == 404
    asyncio.run(body())


# --------------------------------------------------------------------------- model routing

def test_verifier_pinned_to_strong_model_when_doer_is_weak():
    """A weak doer must not drag the verifier down to its model -- the auditor's
    independence is the point. Regression for the Haiku-doer 400-on-effort bug."""
    seen = []

    class RecMessages(FakeMessages):
        async def create(self, **kw):
            seen.append(kw.get("model"))
            return await super().create(**kw)

    class RecClient:
        def __init__(self):
            self.messages = RecMessages()

        async def close(self):
            pass

    async def body():
        async for _ in run_agent(SAMPLE_MATCH, config="full", client=RecClient(),
                                 model="claude-haiku-4-5-20251001"):
            pass

    asyncio.run(body())
    assert any(m and m.startswith("claude-haiku-4-5") for m in seen), "doer should run on haiku"
    assert agent.MODEL in seen, "verifier should stay pinned to the strong MODEL (opus)"


# --------------------------------------------------------------------------- replay

def test_replay_over_fixture_with_mock():
    rounds = ingest.load_rounds(FIXTURE)
    events = []

    async def body():
        return await run_replay(rounds, FakeClient(), config="full", model="claude-opus-4-8",
                                map_name="Inferno", us_team="FURIA", window=5,
                                max_rounds=6, on_event=lambda k, d: events.append((k, d)))

    rows, s = asyncio.run(body())
    assert len(rows) == 6
    assert all("agent_buy_type" in r and isinstance(r["match"], bool) for r in rows)
    # the ledger accumulated the revealed outcomes and the recorded calls
    assert len(s.outcomes) == 6 and len(s.calls) == 6
    assert [k for k, _ in events].count("round") == 6
