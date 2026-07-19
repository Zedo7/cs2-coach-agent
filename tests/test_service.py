"""Service tests. Zero real API calls: make_client is monkeypatched to a fake.

Tests are plain sync functions that drive an async body with asyncio.run, so the only
new test dependencies are pytest + httpx (no pytest-asyncio)."""

import asyncio
import json

import httpx

import service
from conftest import SAMPLE_MATCH, FakeClient, SlowClient
from service import RunSlots


def _client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=service.app),
                             base_url="http://test")


def _parse_sse(text):
    events = []
    for frame in text.split("\n\n"):
        frame = frame.strip("\n")
        if not frame or frame.startswith(":"):        # keepalive comment
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


def test_valid_request_returns_plan(monkeypatch):
    monkeypatch.setattr(service, "make_client", lambda: FakeClient())

    async def body():
        async with _client() as c:
            r = await c.post("/v1/coach", json=SAMPLE_MATCH)
        assert r.status_code == 200
        j = r.json()
        assert j["approved"] is True
        assert j["plan"]["buy_type"] == "eco"
        assert j["attempts"] == 1
        assert j["usage"]["cost_usd"] > 0
        assert r.headers["X-Request-ID"]
    asyncio.run(body())


def test_invalid_request_returns_422(monkeypatch):
    monkeypatch.setattr(service, "make_client", lambda: FakeClient())
    bad = {"map": "Mirage", "side": "X",                 # side not T/CT
           "score": {"us": -1, "them": 7},               # negative
           "economy": {"us_avg": 2400}}                  # missing them_avg + recent_rounds

    async def body():
        async with _client() as c:
            r = await c.post("/v1/coach", json=bad)
        assert r.status_code == 422
        locs = {tuple(e["loc"]) for e in r.json()["detail"]}
        assert ("body", "side") in locs
        assert ("body", "recent_rounds") in locs
    asyncio.run(body())


def test_invalid_config_returns_422(monkeypatch):
    monkeypatch.setattr(service, "make_client", lambda: FakeClient())

    async def body():
        async with _client() as c:
            r = await c.post("/v1/coach?config=nonsense", json=SAMPLE_MATCH)
        assert r.status_code == 422
    asyncio.run(body())


def test_sse_event_sequence(monkeypatch):
    monkeypatch.setattr(service, "make_client", lambda: FakeClient())

    async def body():
        async with _client() as c:
            async with c.stream("POST", "/v1/coach/stream", json=SAMPLE_MATCH) as r:
                assert r.status_code == 200
                assert r.headers["content-type"].startswith("text/event-stream")
                text = ""
                async for chunk in r.aiter_text():
                    text += chunk
        names = [e for e, _ in _parse_sse(text)]
        # exact happy-path shape
        assert names == ["run_started", "attempt_started", "tool_call",
                         "doer_plan", "budget_check", "verifier_verdict", "final"]
        final = _parse_sse(text)[-1][1]
        assert final["approved"] is True and final["plan"]["buy_type"] == "eco"
    asyncio.run(body())


def test_no_verifier_config_skips_verdict(monkeypatch):
    monkeypatch.setattr(service, "make_client", lambda: FakeClient())

    async def body():
        async with _client() as c:
            async with c.stream("POST", "/v1/coach/stream?config=doer_only", json=SAMPLE_MATCH) as r:
                text = "".join([chunk async for chunk in r.aiter_text()])
        names = [e for e, _ in _parse_sse(text)]
        assert "verifier_verdict" not in names
        assert names[-1] == "final"
    asyncio.run(body())


def test_saturation_returns_429(monkeypatch):
    # Capacity 1, already filled -> the request must be cleanly rejected, not queued.
    slots = RunSlots(1)
    monkeypatch.setattr(service, "slots", slots)
    monkeypatch.setattr(service, "make_client", lambda: FakeClient())

    async def body():
        assert await slots.try_acquire() is True          # occupy the only slot
        async with _client() as c:
            r = await c.post("/v1/coach", json=SAMPLE_MATCH)
        assert r.status_code == 429
        assert r.headers["Retry-After"] == str(service.RETRY_AFTER)
        await slots.release()
    asyncio.run(body())


def test_stream_saturation_returns_429_not_open_stream(monkeypatch):
    # The slot is taken before the 200 stream opens, so saturation is a clean 429.
    slots = RunSlots(1)
    monkeypatch.setattr(service, "slots", slots)
    monkeypatch.setattr(service, "make_client", lambda: FakeClient())

    async def body():
        assert await slots.try_acquire() is True
        async with _client() as c:
            r = await c.post("/v1/coach/stream", json=SAMPLE_MATCH)
        assert r.status_code == 429                        # never a 200 event-stream
        await slots.release()
    asyncio.run(body())


def test_timeout_returns_504_with_partial_trace(monkeypatch):
    monkeypatch.setattr(service, "make_client", lambda: SlowClient())
    monkeypatch.setattr(service, "RUN_TIMEOUT", 0.2)

    async def body():
        async with _client() as c:
            r = await c.post("/v1/coach", json=SAMPLE_MATCH)
        assert r.status_code == 504
        j = r.json()
        assert j["timeout_seconds"] == 0.2
        # run_started fires before any model call, so a partial trace exists
        assert any(e["event"] == "run_started" for e in j["partial_trace"])
    asyncio.run(body())


def test_stream_timeout_emits_error_event(monkeypatch):
    monkeypatch.setattr(service, "make_client", lambda: SlowClient())
    monkeypatch.setattr(service, "RUN_TIMEOUT", 0.2)

    async def body():
        async with _client() as c:
            async with c.stream("POST", "/v1/coach/stream", json=SAMPLE_MATCH) as r:
                assert r.status_code == 200                # stream already opened
                text = "".join([chunk async for chunk in r.aiter_text()])
        events = _parse_sse(text)
        assert events[-1][0] == "error"
        assert events[-1][1]["phase"] == "timeout"
    asyncio.run(body())


def test_healthz_ready_and_unready(monkeypatch):
    async def body():
        async with _client() as c:
            monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
            r = await c.get("/healthz")
            assert r.status_code == 200 and r.json()["ready"] is True

            monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
            r = await c.get("/healthz")
            j = r.json()
            assert r.status_code == 503 and j["ready"] is False
            assert j["checks"]["api_key"] is False
            assert j["checks"]["prices_loaded"] is True     # non-key checks still pass
            assert j["checks"]["validator"] is True
    asyncio.run(body())
