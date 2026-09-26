# 強化学習基盤 Phase 3: RandomAgent評価

`app.rl.evaluate.evaluate_agent()` は既存AgentをP1の方策として、固定seed群で `CatanEnv` 上の対局を完走させる。

```python
from app.agents.random_agent import RandomAgent
from app.rl.evaluate import evaluate_agent

report = evaluate_agent(RandomAgent(), seeds=range(1000, 1100))
print(report.to_dict()["summary"])
```

既定の対局はP1が渡したAgent、P2〜P4が `HeuristicAgent`。Agentが選んだActionは必ず固定Action IDと現在のMaskを通して検証し、Mask外なら例外にする。したがって正常なRandomAgent評価では `illegal_action_count == 0` である。

評価Reportには以下を含める。

- seedごとの勝者、P1の実得点、ターン数、P1の操作数、累積報酬
- P1の公開行動種別ごとの回数
- win rate、平均得点、平均ターン数、非法Action数

CSV / SQLite / Parquetへの永続化と大規模benchmarkはまだ行わない。Phase 5以降で評価結果をファイルへ保存する。
