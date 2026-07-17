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

Doer 是一个 tool-use agent loop，不是单次调用：模型自己决定查什么，查完再出方案。

```
Doer agent loop (最多 8 轮)
  ├─ get_round_details(round_numbers)  读具体回合的败因原文
  ├─ get_item_prices(items)            查价格，不许凭记忆猜
  └─ check_budget(items, avg_money)    候选买法是否买得起，不可行就改
        ↓ 模型不再调工具
  最终一轮: structured output → DOER_SCHEMA (loss_reason/buy_type/buy/per_player_spend/instruction)
        ↓
budget_check(确定性价格表校验)  ← 经济可行性由算术裁定，不由模型自评
        ↓
Verifier(败因对应性审计)  ← 无工具、低 effort、只看事实+方案+独立算出的花费
        ↓
未通过 → 把问题回传给 Doer 重出 (最多 3 次)
```

Verifier 故意不给工具：它的价值在于独立性，工具会让它去复现 Doer 的推理路径。

**闸门的确定性只到它的数据源为止。** 价格不写死在代码里，而是钉在
[`prices.json`](prices.json)，带 `source`（https://op.gg/cs2/weapons）和 `as_of` 日期；
CS2 经济改动后需要重新核对并更新 `as_of`。默认手枪（glock/usp/p2000）**故意不在**价格表里，
`excluded` 块记录了原因 —— 它们不是购买项，`budget_check` 会把它们当作 unknown item 报错
（不写原因的话，下一个人会"好心"地按 op.gg 把它们加回来）。改动价格或场景后跑
`python validate_scenarios.py`。

**踩过的坑：缓存 key 必须包含"经工具到达模型"的东西。** eval 的磁盘缓存按 prompt hash 做 key，
但**价格从不出现在 prompt 里** —— 模型是通过 `get_item_prices` 工具拿到的。所以第一版改完价格
重跑，会原样返回旧价格下算出的旧方案，delta 显示为 0 而看不出任何问题。现在 doer 的 cache key
里显式包含了 `PRICES` 和 `doer_model`。凡是通过工具结果影响模型的输入，都得进 key。

## Does the Verifier actually help?

24 场景 × 4 配置 = 96 次运行，Doer = Opus 4.8，价格来自 `prices.json` (op.gg, 2026-07-17)。
完整表在 `results/`，改价前的旧结果保留在 `results/archive/` 作对照。

| config | n | buy_type acc | budget viol. | generic reason | tag cov. | retries/scen | tokens/scen | failed |
|---|---|---|---|---|---|---|---|---|
| `full` (doer+gate+verifier) | 24 | 92% | **0** | 4% | 100% | 0.08 | 14,717 | 0 |
| `no_verifier` (doer+gate) | 24 | **96%** | **0** | 4% | 100% | 0.04 | 13,842 | 0 |
| `no_hard_gate` (doer+verifier) | 24 | 92% | 1 | 4% | 96% | 0.04 | 15,736 | 0 |
| `doer_only` | 24 | **96%** | 1 | 4% | 100% | 0.00 | 13,130 | 0 |

> ⚠️ **`edge_03` 这一格的数据有已知瑕疵。** 上表跑在一版措辞含糊的 DOER_SYSTEM 上（"label by
> intent and spend level"），"spend level" 被读成了相对含义，于是 4 个配置里有 3 个把手枪局
> 花光 800 块标成了 `full_buy`。prompt 已修（buy_type 改为绝对定义 + ~$4200 full_buy 下限），
> 并做了单场景 spot check：**修复后 4 个配置全部不再输出 `full_buy`**（`full`/`no_verifier`/
> `no_hard_gate` = `eco`，`doer_only` = `half_buy`）。但受预算限制**整张表没有重跑** —— 上表
> 的 `edge_03` 行仍是旧 prompt 的产物。这是一个已知且被记录的不一致：披露比隐藏便宜。

**诚实的结论：干活的是那个确定性算术闸门，不是 Verifier。** 去掉 Verifier 反而略好
（96% vs 92%），更便宜也更快。Verifier 唯一可测量的贡献是 0.08 次/场景的重试，而那些重试
没有换来任何指标提升。**Verifier 在 Opus Doer 上不值它的成本。**

**但闸门的功劳这一版也缩水了。** 改价并修正 eco 语义后 `doer_only` 的准确率从 83% 跳到 96%
（旧表见 `results/archive/`），所以"闸门救回准确率"的说法站不住了。闸门剩下的确凿价值很窄
但很硬：`edge_01` 里模型报了一个 **$15,350** 的买法（它按五个人总价算而不是单人），只有带
hard gate 的配置抓住了。

**`no_hard_gate` 的混淆项测清楚了：Verifier 不能替代算术。** 它拿到了独立算出的 cost，仍然
放行了 `edge_01` 里 `per_player_spend` 与实际花费矛盾的方案。

**`edge_03` 是探针，不是 bug。** 0-0 手枪局，`recent_rounds` 为空，但 DOER_SCHEMA 强制要求
`loss_reason`。所有配置都诚实写了 "N/A — 没有前置回合可引用"，**没有一个编造败因** ——
schema 压力没有诱发幻觉。judge 把这种回答判成 GENERIC（4% 那一格几乎全来自它），这是判分
口径问题：一个正确的"无可引用"在字面上确实不 specific。

### Doer 能力依赖实验（Haiku 4.5 Doer）

> ⚠️ **幸存者偏差，各配置 n 不同，仅供定性参考。** 48 次运行中 10 次 FAILED（6 次 HTTP 529
> 过载；4 次是 harness bug —— 弱 Doer 在最终回合仍在调工具导致没有 text block，已修）。
> 死掉的恰恰是"停不下来"的那些，很可能是它表现最差的场景，所以下面的准确率是**偏乐观**的；
> 且 `full` (n=18) 与 `no_verifier` (n=20) 根本不是同一批场景，**两者的准确率不可比**。

| doer | config | buy_acc | budget viol. | retries/scen | n | failed |
|---|---|---|---|---|---|---|
| Opus 4.8 | `full` | 92% | 0 | 0.08 | 24 | 0 |
| Opus 4.8 | `no_verifier` | 96% | 0 | 0.04 | 24 | 0 |
| Haiku 4.5 | `full` | 89% | 0 | **0.33** | 18 | 6 |
| Haiku 4.5 | `no_verifier` | 95% | 0 | **0.15** | 20 | 4 |

**站得住的结论**（不依赖准确率）：Verifier 在弱 Doer 上触发频率是强 Doer 的 **4 倍**
（0.33 vs 0.08 次/场景），另有 **10 次撞上 8 轮工具迭代上限**（全部来自 Haiku）。也就是说
假设的前半段成立：**Doer 越弱，确实越会产出 Verifier 想拦的东西。** 但后半段没有证据支持 ——
这些额外的拦截没有转化成更好的结果。**准确率对比不可用**，见上方偏差说明。

### Limitations

- **答案键和 prompt 里的领域假设必须由领域专家审计 —— 这不是可以外包给模型的部分。**
  本项目已被抓到三次，每次都是模型（包括写这段的模型）自信地写错：
  1. **M4A4 价格**写死成 3100，实际 2900（现已钉到 `prices.json`，带 source + as_of）；
  2. **CT 有两把默认手枪**（USP-S 和 P2000），默认手枪根本不是购买项，不该出现在词表和价格表里；
  3. **eco 不等于不买枪** —— eco 是"少花钱攒钱"，买 p250/deagle 是标准打法；把 eco 定义成
     "空买法"会同时污染 DOER_SYSTEM、VERIFIER_SYSTEM 和 validator 的代表性 kit 假设。

  第 3 条尤其说明问题：那个错误假设**同时**存在于 prompt、审计员 prompt 和验证器里 —— 三处
  互相印证，全错。只有领域专家能打破这种一致的错误。
- **最终 prompt 修复后没有做全量重跑**（预算限制），只做了 `edge_03` spot check。所有 prompt
  与数据修正都已提交，因此任何人拿自己的 API key 花 **~$10** 就能完整复现这张表：
  `python eval.py`（`--estimate` 先看账单）。
- 单次采样、24 个场景。92% 与 96% 的差距约等于 1 个场景，**统计上不显著**，不要过度解读。
- judge 的口径把诚实的 "N/A" 判成 GENERIC，见 `edge_03`。

## Roadmap

- CS2 GSI 实时数据接入
- 道具使用追踪
- 点位预测 (MCTS)
