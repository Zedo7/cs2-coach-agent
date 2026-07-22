"""v0.5: storage abstraction, TTL policy, and concurrency -- $0, no live Redis.

Every store test is parametrized over BOTH backends, so backend parity is literally the
same assertions rather than a separate suite that can drift. fakeredis provides an
in-process Redis; CI needs no server.
"""

import asyncio
import json
import time

import fakeredis.aioredis
import httpx
import pytest

import service
from conftest import SAMPLE_MATCH, FakeClient
from sessions import (
    Call, InMemoryStore, RedisStore, RoundOutcome, Session, SessionNotFound,
    build_grounding, make_store,
)

OUTCOME = RoundOutcome(round=1, our_side="T", their_side="CT", result="win", winner="us",
                       score_after={"us": 1, "them": 0},
                       economy={"us_avg": 800, "them_avg": 800},
                       buy={"us": "eco", "them": "eco"}, detail="T win — bomb detonated")
CALL = Call(round=1, buy_type="eco", buy=["p250"], cost=300, approved=True, attempts=1)


def make_backend(kind, **kw):
    """Fresh store of the requested kind. fakeredis instance is per-store, so tests are
    isolated without flushing a shared server."""
    if kind == "memory":
        return InMemoryStore(**kw)
    return RedisStore(fakeredis.aioredis.FakeRedis(decode_responses=True), **kw)


BACKENDS = ["memory", "redis"]


# --------------------------------------------------------------------------- parity

@pytest.mark.parametrize("backend", BACKENDS)
def test_crud_and_ledger_roundtrip(backend):
    """Serialization round-trip: what goes into the ledger comes back identical."""
    async def body():
        store = make_backend(backend)
        s = await store.create(map="Inferno", us_team="FURIA", window=5)
        got = await store.get(s.id)
        assert (got.id, got.map, got.us_team, got.window) == (s.id, "Inferno", "FURIA", 5)

        await store.append_outcome(s.id, OUTCOME)
        await store.record_call(s.id, CALL)
        got = await store.get(s.id)
        assert len(got.outcomes) == 1 and len(got.calls) == 1
        assert got.outcomes[0] == OUTCOME          # dataclass equality after JSON round-trip
        assert got.calls[0] == CALL
        assert got.outcomes[0].score_after == {"us": 1, "them": 0}   # nested dicts survive

        await store.delete(s.id)
        with pytest.raises(SessionNotFound):
            await store.get(s.id)
        await store.close()
    asyncio.run(body())


@pytest.mark.parametrize("backend", BACKENDS)
def test_missing_session_raises_everywhere(backend):
    async def body():
        store = make_backend(backend)
        for op in (store.get("nope"), store.touch("nope"),
                   store.append_outcome("nope", OUTCOME), store.record_call("nope", CALL)):
            with pytest.raises(SessionNotFound):
                await op
        await store.close()
    asyncio.run(body())


@pytest.mark.parametrize("backend", BACKENDS)
def test_get_returns_snapshot_not_live_reference(backend):
    """Mutating the returned object must NOT affect stored state. In-memory used to hand
    back a live reference; that mutation would silently vanish against Redis."""
    async def body():
        store = make_backend(backend)
        s = await store.create(map="Inferno", us_team="FURIA")
        snap = await store.get(s.id)
        snap.outcomes.append(OUTCOME)
        snap.map = "Mirage"
        fresh = await store.get(s.id)
        assert fresh.outcomes == [] and fresh.map == "Inferno"
        await store.close()
    asyncio.run(body())


# --------------------------------------------------------------------------- concurrency

@pytest.mark.parametrize("backend", BACKENDS)
def test_concurrent_appends_do_not_lose_updates(backend):
    """THE lost-update test: many concurrent appends, every one survives. This is what
    append-only ledgers buy us -- a read-modify-write would drop most of these."""
    async def body():
        store = make_backend(backend)
        s = await store.create(map="Inferno", us_team="FURIA")
        outcomes = [RoundOutcome(**{**OUTCOME.__dict__, "round": i}) for i in range(1, 21)]
        calls = [Call(**{**CALL.__dict__, "round": i}) for i in range(1, 21)]
        await asyncio.gather(
            *[store.append_outcome(s.id, o) for o in outcomes],
            *[store.record_call(s.id, c) for c in calls])
        got = await store.get(s.id)
        assert len(got.outcomes) == 20, "an append was lost"
        assert len(got.calls) == 20, "a call was lost"
        assert {o.round for o in got.outcomes} == set(range(1, 21))
        await store.close()
    asyncio.run(body())


@pytest.mark.parametrize("backend", BACKENDS)
def test_run_guard_admits_one_and_releases(backend):
    async def body():
        store = make_backend(backend)
        s = await store.create(map="Inferno", us_team="FURIA")
        first = await store.claim_run(s.id, ttl=5)
        assert first is not None
        assert await store.claim_run(s.id, ttl=5) is None      # second is rejected
        await store.release_run(s.id, first)
        assert await store.claim_run(s.id, ttl=5) is not None   # free again
        await store.close()
    asyncio.run(body())


@pytest.mark.parametrize("backend", BACKENDS)
def test_run_guard_release_is_token_checked(backend):
    """A run whose guard already expired must not release the next holder's guard."""
    async def body():
        store = make_backend(backend)
        s = await store.create(map="Inferno", us_team="FURIA")
        held = await store.claim_run(s.id, ttl=5)
        await store.release_run(s.id, "some-other-token")       # wrong token: no-op
        assert await store.claim_run(s.id, ttl=5) is None, "guard was wrongly released"
        await store.release_run(s.id, held)
        await store.close()
    asyncio.run(body())


@pytest.mark.parametrize("backend", BACKENDS)
def test_run_guard_expires_so_a_crash_cannot_wedge_a_session(backend):
    async def body():
        store = make_backend(backend)
        s = await store.create(map="Inferno", us_team="FURIA")
        assert await store.claim_run(s.id, ttl=0.15) is not None
        assert await store.claim_run(s.id, ttl=0.15) is None
        await asyncio.sleep(0.25)                                # holder "crashed"
        assert await store.claim_run(s.id, ttl=0.15) is not None
        await store.close()
    asyncio.run(body())


@pytest.mark.parametrize("backend", BACKENDS)
def test_compaction_claimed_exactly_once_under_concurrency(backend):
    async def body():
        store = make_backend(backend)
        s = await store.create(map="Inferno", us_team="FURIA")
        results = await asyncio.gather(*[store.claim_compaction(s.id, 1) for _ in range(10)])
        assert sum(1 for r in results if r) == 1, "compaction event would be emitted twice"
        assert await store.claim_compaction(s.id, 2) is True     # a different batch is free
        await store.close()
    asyncio.run(body())


# --------------------------------------------------------------------------- expiry

@pytest.mark.parametrize("backend", BACKENDS)
def test_ttl_expires_idle_session(backend):
    async def body():
        store = make_backend(backend, ttl_seconds=0.2)
        s = await store.create(map="Inferno", us_team="FURIA")
        await asyncio.sleep(0.35)
        with pytest.raises(SessionNotFound):
            await store.get(s.id)
        await store.close()
    asyncio.run(body())


@pytest.mark.parametrize("backend", BACKENDS)
def test_ttl_slides_on_access(backend):
    """A live match must never expire: each access pushes the deadline out."""
    async def body():
        store = make_backend(backend, ttl_seconds=0.3)
        s = await store.create(map="Inferno", us_team="FURIA")
        for _ in range(4):                       # total elapsed > ttl, but never idle > ttl
            await asyncio.sleep(0.15)
            await store.touch(s.id)
        assert (await store.get(s.id)).id == s.id
        await store.close()
    asyncio.run(body())


@pytest.mark.parametrize("backend", BACKENDS)
def test_append_also_refreshes_ttl(backend):
    """All session keys are refreshed together -- otherwise the meta hash can expire out
    from under a surviving ledger list (the partial-expiry hazard)."""
    async def body():
        store = make_backend(backend, ttl_seconds=0.3)
        s = await store.create(map="Inferno", us_team="FURIA")
        for i in range(4):
            await asyncio.sleep(0.15)
            await store.append_outcome(s.id, RoundOutcome(**{**OUTCOME.__dict__, "round": i}))
        got = await store.get(s.id)
        assert len(got.outcomes) == 4
        await store.close()
    asyncio.run(body())


def test_make_store_factory_selects_backend(monkeypatch):
    monkeypatch.setenv("COACH_SESSION_BACKEND", "memory")
    assert isinstance(make_store(), InMemoryStore)
    monkeypatch.setenv("COACH_SESSION_BACKEND", "nonsense")
    with pytest.raises(ValueError):
        make_store()


# --------------------------------------------------------------------------- HTTP layer

def _client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=service.app),
                             base_url="http://test")


@pytest.mark.parametrize("backend", BACKENDS)
def test_concurrent_coach_on_same_session_returns_409(monkeypatch, backend):
    """Two /coach requests on one session: one runs, the other is rejected -- never a
    duplicate agent run billed for a question already in flight."""
    monkeypatch.setattr(service, "store", make_backend(backend))
    started = asyncio.Event()

    class SlowFake(FakeClient):
        async def _gate(self):
            started.set()
            await asyncio.sleep(0.4)

    fake = SlowFake()
    orig_create = fake.messages.create

    async def slow_create(**kw):
        await fake._gate()
        return await orig_create(**kw)

    fake.messages.create = slow_create
    monkeypatch.setattr(service, "make_client", lambda: fake)

    async def body():
        async with _client() as c:
            sid = (await c.post("/v1/sessions",
                                json={"map": "Inferno", "us_team": "FURIA"})).json()["session_id"]
            payload = {"round": 1, "our_side": "T", "economy": {"us_avg": 800, "them_avg": 800}}

            async def coach():
                async with c.stream("POST", f"/v1/sessions/{sid}/coach", json=payload) as r:
                    await r.aread()
                    return r.status_code

            first = asyncio.create_task(coach())
            await asyncio.wait_for(started.wait(), timeout=5)   # ensure the run is in flight
            second = await coach()
            assert second == 409, f"expected 409 while a run is in flight, got {second}"
            assert await first == 200
            # after the first finishes the guard is released
            assert await coach() == 200
    asyncio.run(body())


@pytest.mark.parametrize("backend", BACKENDS)
def test_session_endpoints_work_on_both_backends(monkeypatch, backend):
    monkeypatch.setattr(service, "store", make_backend(backend))
    monkeypatch.setattr(service, "make_client", lambda: FakeClient())

    async def body():
        async with _client() as c:
            sid = (await c.post("/v1/sessions",
                                json={"map": "Inferno", "us_team": "FURIA"})).json()["session_id"]
            r = await c.post(f"/v1/sessions/{sid}/round", json={
                "round": 1, "our_side": "T", "their_side": "CT", "result": "win",
                "winner": "us", "score_after": {"us": 1, "them": 0},
                "economy": {"us_avg": 800, "them_avg": 800},
                "buy": {"us": "eco", "them": "eco"}, "detail": "T win — bomb detonated"})
            assert r.status_code == 200 and r.json()["rounds_submitted"] == 1
            assert (await c.get(f"/v1/sessions/{sid}")).json()["score"] == {"us": 1, "them": 0}
            assert (await c.delete(f"/v1/sessions/{sid}")).status_code == 200
            assert (await c.get(f"/v1/sessions/{sid}")).status_code == 404
    asyncio.run(body())
