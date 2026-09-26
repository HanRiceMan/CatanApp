# Phase 5: 学習済みAIを通常対局で選ぶ入口

通常の対局UIと、PPO学習用の`CatanEnv`は別物のまま維持する。通常対局では既存のFastAPI → Agent → `Action` → `apply_action()`経路にPPOを追加するだけで、ゲームルールを複製しない。

```text
Next.js の席選択
  -> POST /api/games (ai_agents) または controller変更
  -> PPOAgent / HeuristicAgent / RandomAgent
  -> Action
  -> apply_action()
```

## 追加したAPI

- `GET /api/ai-models`
  - 現在のObservation / Action Spaceと互換性がある保存済みモデルを返す。
  - `ppo_runtime_available` と理由も返す。Python 3.13で通常サーバーを動かしている場合は、モデルを選べないことを明示する。
- `POST /api/games`
  - 既存の`ai_player_ids`は維持。
  - 任意で`ai_agents: {"2": "random", "3": "ppo:ppo_v001"}`を追加できる。
- `POST /api/games/{game_id}/players/{player_id}/controller`
  - 既存の`is_ai`だけのリクエストを維持。
  - 任意で`agent_name`を追加して、途中からAI種別を切り替えられる。

## UI

新規対局の各AI席で、ルールベースAI・ランダムAI・互換性のあるPPOモデルを選べる。全体表示の各プレイヤー行でも、AI席の種類を途中変更できる。モデル一覧には学習step数と保存済み評価勝率を表示する。

## PPOモデルを対局に使う起動方法

PPOはPyTorchのWindows対応に合わせてPython 3.12の`backend/.venv-rl`へ導入している。そのため、PPOをUI対局で選ぶ時だけ、バックエンドもこちらで起動する。

```powershell
cd C:\Users\okome\Documents\CATAN\catan_app\backend
.\.venv-rl\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8003
```

Next.jsの起動方法とポート3002は今まで通り。通常の人間・ルールベースAI対局だけなら、従来のPython 3.13起動も使える。

`ppo_smoke_v001`は接続確認用の512 stepモデルで、強さを示すものではない。比較可能なモデルを作る段階では、十分な学習step数と複数seed評価を保存する。

実際の30万・100万step学習と同一seed比較の結果は[`rl_training_2026-09-22.md`](rl_training_2026-09-22.md)に記録した。
