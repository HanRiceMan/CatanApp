# Step 3A-7: Construction Decision Surfaceとv7 Shadow

## 結論・Gate

`CONSTRUCTION_READY_CHOICE`を**Action前GameState + 377合法mask**だけから検出する診断Surfaceとして実装した。ChampionのActionを選ぶ前にSurfaceを判定し、Championは従来PlannerのActionをそのまま実行する。v7 Policyは同じ状態で提案を出すだけで、提案をGameStateへ適用しない。従って今回測ったのは「Planner override前のzero-impact PPOが、建設可能局面で何を選ぶか」であり、強さや新Policyの優劣ではない。

Rule / Champion / Mixed各12局、席1～4を各3局、計36局・15,536 Actionで、Shadowあり/なしの全Action列、勝者、最終VP/順位、全GameState、RNG counterが完全一致した。Surfaceは166 Decision。save/reload後の全重み、Actor logits、Critic value、masked distribution、選択Actionも一致。**PPO更新、Planner override削除、Champion artifact変更は未実施。**

Step 3A-8の**全面開始はまだ保留**を推奨する。Surface自体・Shadow・parityは合格だが、`SETTLEMENT_AND_CITY`は36局で2件しかなく、都市と開拓地の真の機会費用比較を評価できない。実対局中のRuntime Context生成は平均22.61 ms / p95 34.67 msで、初期局面だけの2.10 msとは大きく違う。先にContext計算の共有/高速化と、両方建設可能な局面を含む評価設計を行い、その後に狭いSurfaceで案Aを試すのが安全。

## 正確なSurface定義とhard exception

`game.phase == "action"`かつ当該playerが手番中で、合法maskに`BuildSettlementAction`または`BuildCityAction`が1つ以上存在する。Championの最終Action、`objective_goal`、Planner候補順位はpredicateへ使わない。`SETTLEMENT_ONLY` / `CITY_ONLY` / `SETTLEMENT_AND_CITY`を即時合法建設familyで決める。候補頂点の良し悪し、将来ETA、交易での完成、道路延伸は**Surface参加条件にしない**。このため「今資源を投入するか、他の価値を選ぶか」が比較対象になる。

除外条件は、setup・pre-roll・robber・trade応答/逆提案・discard等の`action`以外、無料道路の連続処理、未処理の交渉、合法Actionが1個だけのforced状態、既にゲーム終了、自己得点9点以上で建設1点が即勝利になる状態、道路1本/騎士1枚による称号獲得が即勝利になる状態。即時Title勝利は既存の決定的Structured Contextから判定する。36局で対象建設候補から除外したものは`IMMEDIATE_BUILD_WIN`が15 Decision、他のhard exceptionは0。これは最初の移行範囲からの除外であり、永久にPlannerへ固定する意味ではない。

## 保存したShadow recordとartifact

`app/rl/construction_shadow.py`はpre-actionのSurface subtype、ゲームseed/profile/seat/turn、本人score/site/city数、selected Champion Action、source-level `execution_source`、確認済みReason、v7 Context、Shadow selected Action/family、top-5、selected probability、**377個のraw Action logits**、family logits、**377個のlegal mask**を保存する。ChampionがPlayerTradeを選んだ場合も、377 Action外の実Actionとして区別する。`BASE_PPO`のActionにはPlanner Reasonを後付けしない。全recordとゲーム別parityはGit対象外の`experiments/construction_shadow_step3a7_final_20261003/`に保存し、集計は本書に残す。

保存モデルは同フォルダの`context_policy/model.zip`、`manifest.json`。manifestは親Champion `ppo_gnn_board_65k_s03_exp_v003`、Observation v7 / 1746次元、Runtime Context `runtime_context_v1` / 79次元とschema SHA-256、377 Action、Policy classを固定する。既存artifactを上書きせず、新規フォルダ作成のみ許す。親またはschema不一致ではloadを拒否する。保存前後の全Policy tensor、Context-aware actor/critic、mask後分布、Actionはbitwise一致した。Championの既存モデルとconfigは不変。

## Surface数とAction分布

| 対戦条件 | 局数 | Surface数 | Settlement only | City only | 両方 |
|---|---:|---:|---:|---:|---:|
| Rule | 12 | 64 | 36 | 27 | 1 |
| Champion | 12 | 53 | 32 | 21 | 0 |
| Mixed | 12 | 49 | 32 | 16 | 1 |
| 合計 | 36 | **166** | **100** | **64** | **2** |

| Family | Champion実Action | v7 Shadow提案 |
|---|---:|---:|
| BUILD_SETTLEMENT | 99 | 102 |
| BUILD_CITY | 53 | 64 |
| BUILD_ROAD | 6 | 0 |
| END_TURN | 3 | 0 |
| TRADE_BANK | 2 | 0 |
| TRADE_PLAYER | 3 | 0 |
| BUY_DEVELOPMENT / その他 | 0 | 0 |

Shadowはこの標本の建設可能局面で**常に即時建設**を提案した。Championの非建設14件は称号を狙う道路6、開拓地予約等に伴う手番終了3、銀行交換2、プレイヤー交渉3。END_TURNは単なる負例ではなく、少なくともこの3件では`SETTLEMENT_PLANNER`の資源予約経路で選ばれている。PlayerTradeの相手・内容は377 Action外で、将来の委譲時もbelief/history依存Executionとして保護する必要がある。

## 一致・Planner介入

| 指標 | 結果 | 分母の意味 |
|---|---:|---|
| exact Action一致 | 111/166 = **66.9%** | 頂点IDまで同一 |
| Action family一致 | 152/166 = **91.6%** | 建設地点の違いは許容 |
| 対象建設(開拓/都市) vs 他 | 152/166 = **91.6%** | 道路は「他」扱い |
| 双方が開拓/都市を選んだ時のfamily一致 | 152/152 = **100%** | ほぼ片方しか合法でない状態を含むため、CITY vs SETTLEMENT能力の証拠ではない |
| Planner経路 | 165/166 = **99.4%** | `execution_source != BASE_PPO` |
| 実Actionを書き換えた割合 | 55/166 = **33.1%** | Champion Action != zero-impact PPO提案 |

source内訳は`SETTLEMENT_PLANNER` 102、`CITY_PLANNER` 54、`TITLE_PLANNER` 6、`TRADE_PLANNER` 3、`BASE_PPO` 1。`BASE_PPO`の1件はShadow Actionと一致し、Planner Reason誤帰属0件。exact不一致55件のうち**41件は同じfamilyで建設地点だけが異なる**（CITY 40、SETTLEMENT 1）。family不一致14件のうちCITY_ONLYで11、SETTLEMENT_ONLYで3。両方合法の2件はChampion/Shadowとも開拓地を選んだが、サンプル不足で一般化できない。

## Context別の不一致とEND_TURN

値は全てAction前状態の平均。score/site/city数は本人のみ、その他はv7 Runtime Contextを固定scaleから復元した値。欠測maskの値は平均から除外。因果関係や新しい固定ruleは推定しない。

| 局面群 | n | score | site数 | 都市数 | 手札枚数 | 7破棄対象率 | 盗賊阻害pip | 都市生産増分 | 開拓ETA | 都市ETA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Surface全体 | 166 | 4.50 | 3.51 | 0.43 | 7.87 | 52% | 3.24 | 10.45 | 0.20 | 0.79 |
| family不一致 | 14 | 5.29 | 3.86 | 0.86 | 8.86 | 79% | 4.00 | 9.57 | 0.36 | 0.05 |
| 同family・地点のみ不一致 | 41 | 5.78 | 4.46 | 0.59 | 8.46 | 59% | 4.85 | 10.22 | 0.71 | 0.03 |

family不一致には、TITLE_PLANNER道路6、SETTLEMENT_PLANNERのEND_TURN/BankTrade等4、TRADE_PLANNER交渉3、CITY_PLANNERのBankTrade1が含まれる。手札や盗賊被害の平均に差はあるが、この小標本で「原因」とは断定しない。END_TURNの3件はscore 2～3、全てCITY_ONLYでShadowはCITYを提案した。Champion sourceは`SETTLEMENT_PLANNER`で、開拓地計画用資源を守る経路を通っている。1件は手札8枚で7破棄リスクもあったが、`HAND_OVERFLOW`がEND_TURN選択理由だったとは記録していない。将来の学習でこの3件を一律negative扱いしない。

## 実対局runtime

各phaseを別計測した。Observationはv2 prefix生成、Contextは79次元決定計算、Policyはtensor化・Actor forward・合法mask分布・top-k生成を含む。environment stepは全4人の`apply_action` 15,536回で、Shadow対象166件だけではない。単位ms。

| 計測 | 回数 | mean | median | p95 |
|---|---:|---:|---:|---:|
| Surface判定 | 1,581 | 0.421 | 0.270 | 0.345 |
| Observation v2 prefix | 166 | 0.579 | 0.565 | 0.654 |
| Runtime Context | 166 | **22.610** | **20.866** | **34.674** |
| v7 Policy提案・分布 | 166 | 8.035 | 7.995 | 8.401 |
| Environment step | 15,536 | 0.040 | 0.025 | 0.165 |

Runtime Contextは初期局面だけの約2.1 msより重い。今回のShadowはSurface166回のみ追加したため36局収集可能だったが、毎Decisionでv7を用いるPPO学習ではthroughputの懸念がある。`runtime_context._source_values()`はStructured Context全体を作った後に開拓候補解析をもう一度行い、道路称号では仮設edgeごとに最長道を再計算している。**値を削らず**、同一Action前状態に限定した構造・到達計算の共有、必要fieldだけの計算、per-decision read-only workspaceを比較する。GameStateのrevisionだけを鍵にした広域cacheは副作用・陳腐化リスクがあり避ける。

## Step 3A-8への提案

まずは**案A: Champion backbone/旧Actor/旧Criticをfreezeし、Context encoderとActor residualを学習**する。ただしCITY_ONLYで64件中40件が同じCITYでも頂点を違えており、Contextのみの残差で位置選択まで一気に学習させるのは危険。最初の移行対象は「今建設familyへ投資するか」の価値判断とし、建設地点の選択は別の下位選択として現行候補選択を維持するか、GNN地点選択を独立評価する。これはPlannerに「建てる/建てない」の先行判断を残すことではない。案B（Context + Actor後段更新）は地点選択まで学習する段階で比較、案C（全体fine-tuning）は破壊的忘却のリスクが高く最初に採用しない。

PPO rolloutではSurface内でStudentが実際に選択したDecisionに限りActor policy/entropy lossを計算し、Surface外でPlannerが実行したDecisionのpolicy lossを混ぜない。Criticは旧値を初期基準にし、mixed-policy軌跡からのvalue lossを**Actorとは独立**に設計する。現行`MaskablePPO`をそのまま全局面へ適用して済ませず、委譲mask・rollout記録・合法性fallback・価値推定を別検証する。PlayerTradeとbelief/history依存盗賊Executionは保護する。TITLE_PLANNERとの競合も非即勝利6件を確認したため、初回は明示的な安全境界を置く。

Gateを満たすための次の作業は(1) 79次元を変えずRuntime Context計算を高速化して再parity、(2) CITY_ONLYの地点選択を投資判断と切り分ける実験設計、(3) 両方合法な局面を含む追加の**評価**状態（大量Teacher収集ではない）の確保、(4) Surface-only PPO lossとCritic設計。まだPPO学習・DAgger・League PPO・PlayerTrade/Robber移行は実施しない。
