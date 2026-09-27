# Macro Goal Step 2A-3: 交渉目的のsource-level診断

## 結論

`choose_proactive_trade()`が交渉候補を選択した時点で保持している建設目標と
資源不足を、Actionを変えずに診断情報として取得できるようにした。

300局・13,898 DecisionをStep 2A-2と同じseedで再収集した結果、PlayerTrade
3,703件のうち2,679件（72.35%）にはsource-levelでSETTLEMENTまたはCITYを
断定できた。これらは全件、従来のAction後診断と一致した。

一方、`objective_goal=None`は1,213件（8.73%）のままで、新たに安全に分類できた
交渉は0件だった。残るPlayerTrade 1,024件は、現在のactive goalに必要な資源を
直接補う交換ではなく、将来価値・資源再配分・手札超過対策として選ばれていた。
したがって、Noneを減らすための推測分類は行っていない。

これは失敗ではなく、従来の交渉ラベル2,679件がsource-levelの根拠と完全一致し、
残るNoneが本当に曖昧であることを確認できた結果である。taxonomyへTRADEを追加する
必要はなく、現行分類を維持できる。

## 変更範囲

- `trade_strategy.py`
  - `TradeDecisionDiagnostic`を追加した。
  - 交渉選択時のactive goal、必要資源、wanted/offered resource、交換比率、対象player、
    交換前後の不足数、即時到達性、理由codeを記録する。
  - 通常の`choose_proactive_trade()`は従来どおりActionだけを返す。
  - 診断版は同一の選択処理からActionと診断を同時に返し、交渉ロジックを再実行しない。
- `settlement_planning_agent.py`
  - 既存Action選択フローから診断を受け渡す内部経路を追加した。
  - `select_action()`の公開仕様と最終Actionは変更していない。
- `macro_goal.py`
  - schemaを`macro_goal_diagnostics_v2`へ更新した。
  - source-levelで断定可能なPlayerTradeだけをSETTLEMENT/CITYへ分類する。
  - source-levelで曖昧と判定されたPlayerTradeは、後段推測で上書きせずNoneを維持する。
- `macro_teacher_dataset.py`
  - schemaを`macro_teacher_v3`へ更新した。
  - optionalな交渉診断fieldとsource objective集計を追加した。
- 関連unit testを追加した。

変更していないもの:

- ChampionのActionロジック、候補順位、提案倍率、対象player
- Observation v2（1667次元）
- 377 Action Space
- PPO/GNN、Macro Head、Distillation、Shadow、DAgger、League PPO
- 既存モデルおよびconfig

## source-level情報の取得位置

交渉Actionを生成した後から目的を推測するのではなく、
`choose_proactive_trade()`内部の以下2経路から取得した。

1. 建設費用を直接完成させる標準交渉
   - activeな開拓地計画または都市計画
   - その目標の不足資源
   - 交換後に建設費用へ到達するか
2. 余剰資源を使うstrategic trade
   - active goalとその必要資源budget
   - wanted resourceがactive goalの不足を実際に減らすか
   - 手札超過、資源再配分、将来価値の理由

strategic tradeでwanted resourceがactive goalの不足資源ではない場合、
active goalが存在しても`trade_objective=None`とした。例えば開拓地計画中に、
将来の都市用として鉱石を得る提案をSETTLEMENTへ誤分類しない。

現在のproactive trade実装には発展カード購入または最大騎士力を直接目的とする
source経路がないため、DEVELOPMENT / KNIGHT_TITLEは新たに生成していない。

BankTradeは今回のsource-level対象外である。従来の安全なAction後分類は維持するが、
`source_level_objective_available=False`として区別する。

## 新規診断field

`PlannerDecisionReport`とDataset recordへ、後方互換なoptional fieldとして追加した。

- `trade_objective`
- `trade_reason_codes`
- `trade_target_player`
- `wanted_resource`
- `offered_resource`
- `offered_resources`
- `offer_ratio`
- `source_level_objective_available`
- `target_goal`
- `immediate_goal_reachable`
- `goal_after_trade`
- `shortage_before_trade`
- `shortage_after_trade`
- `shortage_basis`

追加した主な`PlannerReasonCode`:

- `TRADE_FOR_SETTLEMENT`
- `TRADE_FOR_CITY`
- `TRADE_HAND_OVERFLOW`
- `TRADE_RESOURCE_REBALANCE`
- `TRADE_FUTURE_VALUE`
- `TRADE_OBJECTIVE_AMBIGUOUS`

自由文をTeacher labelには使用していない。

## Dataset

- experiment: `macro_teacher_step2a3_300_s01_20260928`
- schema: `macro_teacher_v3`
- Observation: v2、1667次元
- games: 300（Rule / Champion / Mixed各100）
- Decisions: 13,898
- Observation shape: `(13898, 1667)`
- collection seed:
  - Rule: 2370001〜2370100
  - Champion: 2371001〜2371100
  - Mixed: 2372001〜2372100
- 各profileで対象Championのseatを25局ずつ割り当てた。
- 生Datasetは`experiments/`配下のgitignore対象ローカルartifactである。

Step 2A-2と同じseed、profile、seat配置を用いたため、Decision単位で直接比較した。

## Before / After

### 全体

| 指標 | Before | After |
|---|---:|---:|
| objective_goal=None | 1,213 / 13,898 (8.73%) | 1,213 / 13,898 (8.73%) |
| PlayerTrade未分類 | 1,024 / 3,703 (27.65%) | 1,024 / 3,703 (27.65%) |
| BankTrade未分類 | 164 / 1,187 (13.82%) | 164 / 1,187 (13.82%) |
| BuildRoad未分類 | 25 / 1,771 (1.41%) | 25 / 1,771 (1.41%) |
| 新規分類 | - | 0 |
| 既存Goalから別Goalへの変更 | - | 0 |

### profile別

| profile | None率 Before / After | PlayerTrade未分類 | BankTrade未分類 | BuildRoad未分類 |
|---|---:|---:|---:|---:|
| Rule | 6.82% / 6.82% | 228 / 941 (24.23%) | 60 / 335 (17.91%) | 10 / 640 (1.56%) |
| Champion | 10.12% / 10.12% | 464 / 1,641 (28.28%) | 50 / 479 (10.44%) | 9 / 587 (1.53%) |
| Mixed | 9.00% / 9.00% | 332 / 1,121 (29.62%) | 54 / 373 (14.48%) | 6 / 544 (1.10%) |

## source-levelで確認できた交渉目的

| profile | source objectiveあり | PlayerTrade中の割合 | SETTLEMENT | CITY |
|---|---:|---:|---:|---:|
| Rule | 713 / 941 | 75.77% | 313 | 400 |
| Champion | 1,177 / 1,641 | 71.72% | 862 | 315 |
| Mixed | 789 / 1,121 | 70.38% | 513 | 276 |
| 全体 | 2,679 / 3,703 | 72.35% | 1,688 | 991 |

source objectiveと従来の`objective_goal`の不一致は0件だった。
DEVELOPMENT、KNIGHT_TITLE、ROAD_TITLEへ新規分類された交渉も0件である。

Rule戦ではCITY用交渉の比率が高く、Champion/Mixed戦ではSETTLEMENT用交渉が
多い。これはStep 2A-2で観測した到達局面の差と整合する。source objectiveを
取得できる割合は70.38〜75.77%で、profileを変えても極端には崩れていない。

## Noneとして残るPlayerTrade

source-levelで曖昧なPlayerTradeは全1,024件だった。全件に次の理由が付いた。

| reason | 件数 |
|---|---:|
| TRADE_FUTURE_VALUE | 1,024 |
| TRADE_RESOURCE_REBALANCE | 1,024 |
| TRADE_OBJECTIVE_AMBIGUOUS | 1,024 |
| TRADE_HAND_OVERFLOW | 644 |

代表例は次のとおり。

- 開拓地計画はあるが、その開拓地に不要な鉱石を将来の都市用として得る。
- 余剰小麦等を希少資源へ変えるが、交換直後の建設目標を一意に断定できない。
- 8枚以上の手札を減らしながら資源価値を改善する。
- 現在の建設費用完成ではなく、複数の将来候補に使える資源へ再配分する。

これらへSETTLEMENTやCITYを割り当てると、active planと実際に得たい資源が
一致しないTeacher labelを作るため、Noneのまま保持した。

## Action parityと副作用

### Step 2A-2 Datasetとの完全比較

同一300局について次を確認した。

- 13,898 Decisionの識別情報と`selected_action`が完全一致
- 13,898件のObservation v2がbitwise一致、最大差0
- 300局すべてでwinner、対象Championの勝敗、final VP、rank、turn数、
  全Action数、Macro Decision数が一致

`game_id`は再収集時に生成し直すため比較対象外とした。

### 診断あり / なしの固定seed parity

| profile | games | 全Action数 | 結果 |
|---|---:|---:|---|
| Rule | 5 | 2,088 | 完全一致 |
| Champion | 5 | 2,339 | 完全一致 |
| Mixed | 5 | 1,773 | 完全一致 |
| 合計 | 15 | 6,200 | 完全一致 |

比較対象は全Action列、winner、final score、final rank、最終GameState。
すべて完全一致した。

またunit testで、source診断を取得しても`game.random_source.counter`が変化しない
ことを確認した。診断のためのtradeロジック再実行や追加乱数消費はない。

## テスト結果

- 全テスト: 244件成功
- subtest: 1,161件成功
- 既存のdeprecation warning 2件のみ
- 関連テストでは以下を確認した。
  - 通常Actionと診断Actionの一致
  - RNG counter不変
  - source-level CITY / SETTLEMENTの伝播
  - 将来価値交渉を誤分類しないこと
  - Datasetのsource trade集計

## Observation v2からの再現可能性

### v2から学習できる見込みがある情報

- 自分の資源、盤面、建設状況
- activeな開拓地・都市候補を再計算するための公開状態
- SETTLEMENT / CITYの不足資源
- 交換後に建設費用へ到達するかという短期計算

したがって、source-levelで明確だったSETTLEMENT/CITYというObjective Goal自体は、
Observation v2から学習できる見込みがある。

### v2だけでは再現困難な情報

- `resource_events`から作る相手手札のベイズ推定
  - 誰へ提案するか、相手がwanted resourceを持つ確率に影響する。
- 当該手番の`activity_log`に残る交渉履歴
  - すでに断られた相手・offerを避ける処理に使われる。
- 同一手番内のattempted target / offer
  - Observation v2にはtrade countはあるが、具体的な相手・内容はない。

Planner固有の永続的な隠し状態はなく、開拓候補などの導出値は公開状態から再計算可能。
ただし、PlayerTradeを「今実行するか」、対象player、提案倍率までStudentへ移す場合、
ベイズbeliefと手番内trade historyがないObservation v2では情報不足になる。

今回の`trade_objective=None`を独立した完全な学習クラスにして、交渉実行タイミングまで
模倣させる場合も同じ問題がある。似た公開盤面でも、v2にないbelief/historyにより
Teacher Actionが変わり得る。

## Distillation前の提案

taxonomyやObservationを今すぐ変更する必要はない。

- TRADEをObjective Goalへ追加しない。
- NoneをSETTLEMENT/CITYへ推測再分類しない。
- HOLDは引き続き実質未使用として扱う。
- 最初の限定的Distillationでは、明示ラベルのあるGoalを教師対象にする。
- Noneはunknown/fallbackとしてsupervised goal lossから除外するか、別maskで扱う。
- train / validation / testはDecision単位でなくgame/seed単位、かつprofile×seatで層化する。
- TRADE_PLAYERの具体的実行は当面Plannerへ残す。
- 将来、交渉実行そのものをStudentへ委譲する段階でBayesian beliefと手番内trade historyを
  Observationへ追加する。

以上から、Teacher Goalラベル品質のための追加taxonomy変更は不要である。
Observation v2のまま明示Goalに限定したDistillationへ進むことは可能だが、
交渉Actionの詳細な委譲はまだ行わないのが安全である。
