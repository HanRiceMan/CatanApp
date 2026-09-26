# Phase 4: MaskablePPOによる最初の実学習

このPhaseで、PPOを実際に学習・保存・評価する入口を追加した。ゲームエンジン、通常のFastAPI、Next.js UIにはPPO固有の依存を入れていない。

## 実装物

- `app/agents/ppo.py`: 保存済み`MaskablePPO`からObservationとAction Maskを使い、合法な`Action`だけを返す`PPOAgent`。
- `app/rl/train.py`: `CatanEnv`でP1を学習し、`model.zip`、`metadata.json`、`config.json`、TensorBoardログを保存する学習コマンド。
- `requirements-rl.txt`: PPO用の追加依存。通常のFastAPI起動には不要。
- `tests/test_rl_phase_4.py`: 小さなPPOを学習し、保存・読み込み・合法Action選択までを確認するスモークテスト。

## Python環境

通常アプリは既存のPython 3.13環境のまま使う。PyTorchのWindows対応に合わせ、PPO学習だけは`backend/.venv-rl`（Python 3.12）で実行する。

```powershell
cd C:\Users\okome\Documents\CATAN\catan_app\backend
.\.venv-rl\Scripts\python.exe -m app.rl.train --model-id ppo_v001 --timesteps 100000 --seed 42
```

短い動作確認だけなら、別名のモデルIDでstep数を小さくする。

```powershell
.\.venv-rl\Scripts\python.exe -m app.rl.train --model-id ppo_smoke_v001 --timesteps 512 --seed 42 --eval-seeds 1001
```

保存先は`backend/models/<model_id>/`。`model.zip`とTensorBoardログはGit管理外で、`metadata.json`と`config.json`には学習条件・Observation版・Action Space版・評価結果を残す。

長い学習では`--checkpoint-interval`で指定したstepごとに途中の重みを保存する。途中で停止した場合は同じモデルIDと`--resume-checkpoint-steps`で続行できる。元の完成モデルを残して追加学習する場合は、新しいモデルIDと`--resume-from-model-id`を併用する。再開時は盤面の乱数系列が新しく始まるため、完全なビット単位の再現ではない。

## 今回の範囲

- P1のみを学習し、P2〜P4はHeuristicAgentまたはRandomAgent。
- プレイヤー間交渉はPPO v1の`CatanEnv`では無効（通常UIの交渉機能は維持）。
- 報酬は勝敗（+1/-1）と小さな勝利点差分（既定で±0.05）。
- `PPOAgent`は状態を直接更新せず、必ず既存の`apply_action()`を通る。

## 次のPhase

Phase 5で、固定seed評価、Heuristic/Randomとの比較レポート、および互換性のある保存モデルをFastAPI/Next.jsの席選択へ接続する。学習と通常UI対局は同じ`PPOAgent -> Action -> apply_action()`を使うが、Environment自体をブラウザには載せない。
