# 強化学習基盤 Phase 1

Phase 1では、ゲームルールを変更せずに、強化学習側へ渡す入力と出力の境界を追加した。

```text
GameState -- get_observation(player_id) --> Observation v1 -- encode_observation --> float32 vector
     ^                                                                         |
     |                              Action <--- id_to_action <--- Action ID --+
     +----------------------------------------- apply_action() <--- legal mask
```

## Observation v1

`app.rl.observation.get_observation(game, player_id)` はimmutableな `Observation` を返す。これは `GameState` の参照を持たない。

- 自分: 資源の種類別枚数、発展カード、購入したばかりで使用できない発展カード、使用済み発展カード、公開得点と実得点、建物・道路・騎士・港・称号
- 相手: 資源と発展カードの**合計枚数**、公開得点、建物・道路・騎士・称号、使用済み発展カードだけ
- 盤面: 19タイル、54交差点、72辺、9港、盗賊
- 進行: phase、相対座席順、手番、ダイス、破棄・略奪・交渉の保留状態、銀行、山札残数

相手の資源内訳、未公開発展カード内訳、非公開勝利点はObservationに含めない。相手の発展カードの使用履歴は公開情報として含める。

`encode_observation(observation)` は次の固定長ベクトルを返す。

| 部分 | 次元 |
| --- | ---: |
| 自分の非公開・公開情報 | 29 |
| 他3人の公開情報 | 42 |
| 19タイル | 342 |
| 54交差点 | 432 |
| 72辺 | 360 |
| 9港 | 72 |
| ゲーム進行 | 66 |
| **合計** | **1343** |

地形・数字・建物・所有者・phaseはone-hot、枚数やターン数は0〜1へ正規化する。所有者は、将来の座席回転・Parameter Sharingに備え、エンコード時に「自分・時計回りの相対プレイヤー」へ変換する。

## Action Space v1

`app.rl.action_space` の `ACTION_CATALOG` は377個の固定Action IDを持つ。`action_to_id()` と `id_to_action()` はカタログ内のActionで可逆である。

| 種別 | 数 |
| --- | ---: |
| 順番決め／通常ダイス | 2 |
| 初期開拓地・初期道路 | 126 |
| ゲーム開始 | 1 |
| 盗賊移動・略奪対象 | 23 |
| 道路・開拓地・都市 | 180 |
| 発展カード購入 | 1 |
| 銀行・港交易 | 20 |
| 発展カード使用 | 22 |
| 7の破棄・手番終了 | 2 |
| **合計** | **377** |

`get_action_mask(game, player_id)` は長さ377の `numpy.bool_` 配列を返す。TrueのIDだけが `legal_policy_actions()` の合法手である。

サイコロの目は方策が指定しない確率イベントなので、Action IDでは `dice=None` とする。豊作は資源順の組合せ15通りで表す。

### PPO v1で意図的に簡略化した操作

既存の人間対戦・FastAPI・AI観戦では、以下を従来どおり使える。PPO v1用カタログだけから除外・単純化した。

- プレイヤー間交渉、交渉返答、逆交渉: 方策の出力空間から除外。Phase 2のEnvironmentでは交渉を無効化するか、固定Opponentに任せる。
- 7で捨てる資源の任意内訳: `DiscardHalfAction` は既存の決定的な候補を使う。通常UI/APIの `DiscardResourcesAction` は任意内訳のまま。

この制限により、数百万規模になり得る組合せAction Spaceを避け、最初のMaskablePPOで扱える377出力に保つ。

## 未実装（次のPhase）

Phase 1にはGymnasium、Reward、PPO、PyTorch、DB、MCTSは入れない。

Phase 2では `CatanEnv.reset(seed)` / `step(action_id)` を追加し、学習対象P1がAction IDを選び、他プレイヤーは既存Agentで自動進行する。EnvironmentだけがRL v1の交渉無効化と決定的破棄を選び、Game Engineの通常ルールは変更しない。
