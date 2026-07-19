"""Test fixtures: a fake Anthropic async client so the whole suite costs $0.

The fake inspects create() kwargs to decide what to return -- a tool_use on the first
doer turn, the plan JSON on the format-constrained final turn, the verdict JSON when
the verifier system prompt is present. That reproduces a real run's call sequence
without any network access.
"""

import asyncio
import json
from types import SimpleNamespace

import agent

SAMPLE_MATCH = {
    "map": "Mirage", "side": "T", "score": {"us": 3, "them": 7},
    "economy": {"us_avg": 2400, "them_avg": 5200},
    "recent_rounds": [
        {"round": 8, "result": "loss", "detail": "A执行被守枪+连狙拒之门外, 白给4人"},
        {"round": 9, "result": "loss", "detail": "eco局rush B, 被双架点交叉火力清光"},
        {"round": 10, "result": "loss", "detail": "半起局中路慢摸, 被CT前顶打乱, 残局1v3失败"},
    ],
}

# A feasible, honestly-labelled eco plan at $2400, and an approving verdict, so the
# happy path terminates on attempt 1.
PLAN = {
    "loss_reason": "R8 A execute walled off by AWP+rifle hold at the choke; repeated pattern.",
    "buy_type": "eco", "buy": ["p250"], "per_player_spend": 300,
    "instruction": "Save this round; stack B next round with a real buy.",
}
VERDICT = {"econ_feasible": True, "reason_is_specific": True,
           "instruction_addresses_reason": True, "approved": True, "issues": []}


def _usage(i=1200, o=200, cw=0, cr=0):
    return SimpleNamespace(input_tokens=i, output_tokens=o,
                           cache_creation_input_tokens=cw, cache_read_input_tokens=cr)


def _resp(blocks, usage=None):
    return SimpleNamespace(content=blocks, stop_reason="end_turn", usage=usage or _usage())


def _text(s):
    return SimpleNamespace(type="text", text=s)


def _tool(name, inp, id="tu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=inp, id=id)


class FakeMessages:
    def __init__(self, plan=PLAN, verdict=VERDICT):
        self.plan, self.verdict, self.tool_turns = plan, verdict, 0

    async def create(self, **kw):
        if kw.get("system") == agent.VERIFIER_SYSTEM:
            return _resp([_text(json.dumps(self.verdict, ensure_ascii=False))])
        if "format" in (kw.get("output_config") or {}):        # final plan turn
            return _resp([_text(json.dumps(self.plan, ensure_ascii=False))])
        self.tool_turns += 1                                    # doer tool turn
        if self.tool_turns == 1:
            return _resp([_tool("check_budget", {"items": self.plan["buy"], "avg_money": 2400})])
        return _resp([_text("")])                               # no tool_use -> loop breaks


class FakeClient:
    def __init__(self, plan=PLAN, verdict=VERDICT):
        self.messages = FakeMessages(plan, verdict)

    async def close(self):
        pass


class SlowMessages:
    async def create(self, **kw):
        await asyncio.sleep(30)                                 # never completes before the test timeout
        return _resp([_text("{}")])


class SlowClient:
    def __init__(self):
        self.messages = SlowMessages()

    async def close(self):
        pass
