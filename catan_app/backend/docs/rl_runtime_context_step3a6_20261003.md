# Step 3A-6: Runtime Context付きObservation v7とzero-impact Policy

## 結論

Step 3A-5の診断値から、Action選択前に決定計算できる68 fieldを選び、11個の独立valid maskを付けて79次元のRuntime Contextを定義した。Observation v7は**v2の1667次元をbitwiseでそのまま先頭に置き**、Contextを末尾に付けた1746次元である。旧Observation、Championモデル、Planner、377 Actionと合法maskは変更していない。旧GNN/Actor/Critic重みをコピーした実験Policyでは、Context残差のActor/Critic出口を0初期化する。学習前はContextが入力されてもChampionと同一の出力になる。

Rule / Champion / Mixed各2局、計6局・3,052 Action・755 Context入力で、全Action列、勝者、得点、順位、全GameState、RNG counterが一致した。固定probeのActor logits、masked distribution、Critic valueもbitwise一致。全267テスト成功。**PPO学習・Planner override削除・300局Teacher再収集は行っていない。**

## 実装・machine-readable schema

| ファイル | 役割 |
|---|---|
| `app/rl/runtime_context.py` | field order、単位、raw範囲、固定正規化、valid mask、provenance、意味、非介入encoder。`runtime_context_definition()`が機械可読な唯一の仕様 |
| `docs/rl_runtime_context_v7_definition.json` | 上記関数から生成した固定schema snapshot。テストでsourceとの一致を確認 |
| `app/rl/observation.py` | 未使用version `v7`。v2 prefix +79次元。既存v1～v6は不変 |
| `app/rl/context_aware_policy.py` | 既存GNN + 小型Context encoder + Actor/Critic zero residual。Champion重みの完全コピーと検査。実験用in-memory Agent |
| `app/rl/verify_context_aware_policy.py` | モデルprobeとRule / Champion / Mixed固定seed parity、Context値域検証 |
| `tests/test_rl_runtime_context.py` | schema、非変更、v2 prefix、invalid mask、本人新規カード、Actor/Critic parity |

`runtime_context_definition()`のJSONはリポジトリ内のdefinition snapshotに加え、検証実行時に`experiments/context_aware_step3a6_final_20261003/runtime_context_definition.json`にも保存した。実験フォルダはGit対象外。`field_order`は各値の`vector_index`と、必要なら直後の`.valid` maskの位置を固定する。無効値は**0 + valid=0**であり、99のようなsentinelを入力しない。

## 入力fieldの全体像

以下の順序は各section内でも`CONTEXT_FIELDS`の定義順に固定。`shortage`や`blocked`は括弧内の資源順でそれぞれ独立fieldとする。

| Section | field | 個数 |
|---|---|---:|
| Settlement | candidate_exists, candidate_count, min_additional_roads, shortage.(wood/brick/sheep/wheat), immediate_build, eta | 9 |
| City | upgradeable_count, best_production_gain, shortage.(wheat/ore), immediate_build, eta, pieces_remaining | 7 |
| Road Title | current_length, acquisition_gap, defense_gap, structurally_extendable_count, affordable_title_directed_count, immediate_acquisition, immediate_win, shortage.(wood/brick) | 9 |
| Knight Title | used, usable, plays_needed, draws_needed, deck_remaining, development_affordable, playable_now, immediate_acquisition, immediate_win | 9 |
| Robber | blocked_pips, blocked_ratio, blocked.(wood/brick/sheep/wheat/ore), playable_knight, removable_now | 9 |
| Resource/Risk | hand_size, discard_exposure, discard_count_if_seven, resource_diversity, development_shortage.(sheep/wheat/ore), bank_trade_options, port_advantage.(wood/brick/sheep/wheat/ore), legal_family.(road/settlement/city/development/bank_trade/play_development), immediate_construction | 20 |
| 本人新規発展カード | new_development.(knight/road_building/year_of_plenty/monopoly/victory_point) | 5 |

11 maskはSettlementの`min_additional_roads`、4資源不足、`eta`、Cityの`best_production_gain`と`eta`、Roadの`acquisition_gap`と`defense_gap`、Robberの`blocked_ratio`。開拓候補がないときの不足量は「0不足」ではなく無効値になる。City候補なし、称号保持中でacquisition gapが非適用等も区別する。

最短道路候補の不足・ETAは、候補を追加道路数→ETA→vertex IDの順で**決定的に**選ぶ。Plannerの`selected_target`、`site_value`、`plan_value`、異Goal scoreは読まない。vertex/edgeの推奨IDも渡さず、GNN/PPOに判断させる。

## 正規化・provenance・削減

全encoded値はfloat32の0～1。boolは0/1。cards・pips・道路長・Action件数などはDataset由来min-maxでなく、定義JSONの固定`scale`で割る。例: hand_size/95、road.current_length/15、used knights/14、blocked_pips/30、各資源不足/当該建設の最大費用。港だけはraw交換比2～4を`(4 - ratio)/2`に変換する。ETAは0～12 roundのみ有効とし、到達不能・未定義・12超はmask=0とする。異Goalの価値scoreへ正規化していない。

`RAW_V2_DUPLICATE`は騎士使用数、山札残数、本人手札合計など少数の要約であり、ablation時の削減候補。`DETERMINISTIC_FROM_V2`は資源不足・7曝露・港比・称号gap等、`GNN_BOARD_SUMMARY`は候補数・道路グラフ・生産pip、`MISSING_FROM_V2`は新規カード使用可否やphase別合法Action数等。v2の盤面と本人手札から計算できる値でも、ルール上の厳密計算をNNに再学習させる代わりに与える。各fieldの個別分類は定義JSONの`provenance`を参照。

118 numeric leafをそのまま移植しなかった。除外した主な値は、Planner選択vertex/edge、`site_value`/`plan_value`、選択候補に条件づけたPlanner固有評価、公開値と完全同じ5種の本人資源ベクトル、盤面tile ID、称号holder ID、単純な合法Action総数、戦略Intent/Reason/Action attribution、対局結果。カード種類別の`new_development_cards`は本人の既知状態でv2に欠けるため5 field追加した。相手の秘匿カードは入力しない。

Bayesian hand belief、resource_events、当該手番の交渉・拒否履歴は**今回のCore Contextに含めない**。これらを要する相手選択・交換倍率・再提案・盗賊の配置/盗みはPlanner/既存algorithm側へ残す。`FUTURE_VALUE`、`STALL_BREAK`、`HAND_OVERFLOW`等のTeacher Reasonも直接入力せず、手札数・不足・合法候補数など構成事実だけを渡す。

## Policyとparity

v7 encoder → 既存GNNはv2 prefixのみ処理 → 旧Actor/Critic latent。追加79次元は`79→64→64`のContext encoderを通り、Actorの377 Action + family logitsおよびCritic latentへ残差として加算する。残差の最終projectionの重みとbiasを**すべて0**で開始し、旧重みは`state_dict`でコピー後、各tensorのbitwise一致を検査した。Context branch自体はまだ学習していない。旧Champion artifactは上書き・登録していない。

| Profile | seeds | 席 | Action数 | Context入力数 | 結果 |
|---|---|---|---:|---:|---|
| Rule | 2353001–2353002 | 1, 2 | 1,205 | 277 | 完全一致 |
| Champion | 2354001–2354002 | 1, 2 | 999 | 247 | 完全一致 |
| Mixed | 2355001–2355002 | 1, 2 | 848 | 231 | 完全一致 |

全入力79次元がNaN/infなし、0～1、fieldとmaskの整合検査を通過。Rule / Champion / Mixedの各phase probeでlogitsとCritic valueがbitwise一致し、合法maskがあるphaseではmasked Action distributionも一致した。trade response/counterは377 maskの対象外で全falseのため分布比較は非適用とし、Action一致を確認した。Seat 3/4を含む広い評価は**学習前の次段階**で追加すべきで、今回の6局を強さや採用の根拠にはしない。

参考として初期局面50回平均のObservation生成はv2 0.261 ms、v7 2.104 ms（追加1.842 ms）。これは開発PCの一局面だけの値で、盤面が混雑した終盤や学習時のthroughputを保証しない。Structured Context計算と到達計算の共有・キャッシュは次段階の性能確認対象。

## 次のPlanner移行対象とGate

最初の狭い委譲候補は、`action` phaseで**即時合法なBUILD_SETTLEMENT / BUILD_CITYとEND_TURNの間の価値判断**。開拓地/都市の建設コスト、piece、候補、生産改善、資源不足はv7で観測でき、交渉相手やBayesian beliefへの依存が小さい。即時勝利・称号獲得等の明確な例外は引き続きPlanner側に残し、選択先vertexはGNNの377 Action logitsが選ぶ。実際の委譲前には十分な状態数と合法性・勝率・建設数の対照評価が必要で、今回overrideを外していない。

次候補は道路延伸とEND_TURN、発展購入と建設/END_TURN、銀行交換である。ただしこれらはSettlement/Titleとの機会費用、停滞や7リスクを含むため、最初から一括移行しない。PLAYER_TRADEの相手/資源/倍率・応答/逆提案、盗賊配置と盗みはbelief/history専用Contextを設計するまで現行Planner側へ残す。カード使用もphase別合法性と効果を検証してから扱う。

**小規模PPO fine-tuningの基盤としては準備完了**。ただし開始前にv7モデルを別artifact IDで保存/再読込できること、Context計算のruntime overhead、Seat 1～4のparity、学習/評価seed分離、初回委譲surfaceのmaskと安全fallbackを確認する。ここではPPO更新、Shadow、DAgger、300局再収集は行わない。
