# cs2-coach-agent

Pause-coach agent for CS2: reads a match state JSON, proposes the next-round
call via a Doer-Verifier loop (economic feasibility gated by deterministic math,
not LLM agreement).

## Run

```sh
pip install anthropic && cp .env.example .env  # add your key
python agent.py
```

## Sample output

```
=== NEXT ROUND CALL ===
ECO: p250, flash, smoke
本回合放弃对枪, 全员轻眼投资保枪, 目标是攒到R12的完整装备。打法: 5人集合快下B, 一颗闪+一颗烟先扔进匪道压CT视野, 用p250近距离贴脸打, 抢不到就撤保枪, 绝不散点各自为战、绝不中路慢摸送人头。
```

## Architecture

Doer(战术指令) → budget_check(确定性价格表校验) → Verifier(败因对应性审计) → 最多3次打回重出

## Roadmap (do not touch until v0.1 has users)

- CS2 GSI 实时数据接入
- 道具使用追踪
- 点位预测 (MCTS)
