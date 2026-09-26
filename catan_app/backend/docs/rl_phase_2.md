# 強化学習基盤 Phase 2: CatanEnv

`app.rl.env.CatanEnv` は、FastAPI・Next.jsを経由せずにゲームエンジンを直接動かす、単一学習者向けのGymnasium Environmentである。

```python
from app.rl.env import CatanEnv

env = CatanEnv()
observation, info = env.reset(seed=42)

while True:
    action_id = info["action_mask"].nonzero()[0][0]
    observation, reward, terminated, truncated, info = env.step(int(action_id))
    if terminated or truncated:
        break
```

## 設定

既定ではP1が学習者、P2〜P4が既存の `HeuristicAgent` である。

```python
from app.rl.env import CatanEnv, CatanEnvConfig

config = CatanEnvConfig(
    learning_player_id=1,
    opponent_controller="heuristic",  # "random" も可
    controller_by_player={4: "external"},
)
env = CatanEnv(config)
```

席ごとの値は `learner` / `heuristic` / `random` / `external`。学習者は1人だけに限定する。

- `heuristic` / `random`: Environmentが既存Agentで自動進行する。
- `external`: Environmentが停止し、`pending_external_player_id` に操作席を示す。`step_external(player_id, action)` で操作を渡す。`action` はPPO v1のAction ID、または通常ゲームと共通の `Action` を使える。

`external` は人間対戦のデバッグ・教材向けであり、無人PPO学習には向かない。人が考える時間でEpisodeが遅くなり、対戦相手の戦略も固定されないため、最初の性能比較ではP2〜P4を固定Heuristicにする。

## PPO v1の簡略化

Phase 2のEnvironmentではプレイヤー間交渉を無効化する。Opponentが対人交渉を提案しそうなときは手番終了に置き換える。学習者の377 Action IDにも交渉は含まれない。

これはEnvironmentだけの制約であり、FastAPI・Next.js・通常の人間対局にある交渉機能は変更しない。

7での破棄はPhase 1の決定的な `DiscardHalfAction` を使用する。

## Reward v1

`RewardConfig` は以下の既定値を持つ。

| イベント | 報酬 |
| --- | ---: |
| 勝利 | +1.0 |
| 敗北 | -1.0 |
| 実得点の増減 | `+/- 0.05 × ΔVP` |

道路・資源枚数などへの報酬は入れず、Reward hackingを避ける。`info["reward_components"]` には遷移ごとの内訳が入る。

## 再現性と終了

- `reset(seed=42)` は同じ盤面とゲーム乱数系列を作る。
- `step()` はGymnasium形式の `(observation, reward, terminated, truncated, info)` を返す。
- 勝者が決まると `terminated=True`。
- `max_policy_steps` 到達時は `truncated=True`。既定値は5000。
- `info["action_mask"]` と `action_masks()` は次の合法Actionを返す。

## 未実装

Gymnasium互換のEnvironmentとRewardは追加したが、PPO、PyTorch、学習スクリプト、評価、モデル保存、Vectorized Environmentはまだ追加していない。次のPhaseでMaskablePPOを導入する。
