# 学習済みAIを通常UIで使うための方針

学習は `CatanEnv` でUIを通さず高速に実行する。一方、学習済みモデルとの人間対局・観戦は、既存のNext.js → FastAPI → Game Engine経路で行う。学習用Environmentをブラウザの対局状態として再利用しない。

```text
学習:     PPO -> CatanEnv -> Game Engine
公開:     model.zip + metadata.json -> ModelRegistry
対局UI:   Next.js -> FastAPI -> PPOAgent -> Action -> apply_action() -> Game Engine
```

## モデル保存形式

モデルごとに次の構造を使う。重いモデル本体はGit管理外、metadataは必要ならGit管理できる。

```text
backend/models/
  ppo_v001/
    model.zip
    metadata.json
```

`metadata.json` の必須項目は次のとおり。

- `model_id`, `algorithm`, `artifact_filename`
- `observation_version`, `observation_size`
- `action_space_version`, `action_space_size`
- `training_steps`, `training_seed`, `created_at`
- `evaluation`

`ModelRegistry` はモデル読み込み時にObservation / Action Spaceの版と次元数を検証する。互換性がない古いモデルはUIの選択肢に出さない。これにより、仕様変更後に誤ったモデルを対局で使う事故を防ぐ。

## 現在のPPO導入状況とUI接続

Phase 4で、`PPOAgent`が`get_observation()`と`get_action_mask()`からAction IDを推論し、既存AIと同じくActionだけを返すようになった。状態変更は常に`apply_action()`が担当する。

Phase 5で、次を実装した。

1. `GET /api/ai-models` が互換性のあるModelManifest一覧とPPO実行可否を返す。
2. 新規対局と途中切替で、席ごとに `human` / `heuristic` / `random` / `ppo:<model_id>` を選べる。

この構成なら、人間P4・PPO P1・Heuristic P2/P3などの混合対局、観戦、途中の人⇔AI切替を既存UIの設計に沿って扱える。
