# Step 3A-14: Strategic Gate 実委譲 integration smoke

2026-10-04。PPO/Gate/Criticのparameter update、Reward変更、Champion artifact変更は行っていない。実験コードは `strategic_gate_rollout_step3a14.py`、実行・集計は `run_strategic_gate_step3a14.py`、データは `experiments/strategic_gate_step3a14_20261004/summary.json` と `games.json`。全Action列等を含む一局ごとのcheckpointはローカル実験フォルダに置き、Gitには集計とDecision/transitionを含む `games.json` を保存した。

## Controllerと実委譲境界

Actionの仲裁順は (1) ImmediateWinResolver、(2) setup/robber/discard/trade response/counter/free-road等の従来phase controller、(3) protected PlayerTrade、(4) state-based Strategic Surface、(5)外部RNGによるdelegation抽選、(6)凍結native 15-family logit＋ゼロ残差の6-family Gate、(7)選択familyだけのFrozen B Executor、(8)`env.step_action()`。Resolverはaction/turn_pre_rollの確定単一Action勝利だけをコピー適用で検出し、発展カード購入の不確定な勝利点抽選を含めない。Resolver実行Actionとprotected交渉はGate sampleではない。

Legacy modeはResolver OFF・委譲0%。Safety modeはResolver ON・委譲0%。Strategic modeはResolver ON・protected PlayerTrade一時保護・委譲5%または10%。3 variantを混ぜて評価しない。既存ChampionのPlayerTrade選択をprotected preemptionとし、提案候補のsource-level存在とは別fieldに記録する。今回の観測では両者が一致したが、永久的なPlayerTrade優先ruleではない。

Gateの残差とStrategic Critic残差はゼロのまま。ExecutorはSettlement `_best_settlement()`、City `_best_city()`、Road/BankTradeは合法family内の凍結377 candidate logit最大、BuyDevelopment/EndTurnは直接Action。Road A、BankTrade A、uniform/max/log-mean-exp priorは実Actionに使用していない。抽選用のdelegation seedとfamily sampling seedを各episodeへ保存し、GameState RNGを使用しない。Champion counterfactualは診断だけで教師Actionにしない。

## 先行Gateと強制適用

Rule/Champion/Mixed × 席1〜4の12局で、Legacy対「Resolver OFF・Gate抽選0%」の全Action列、winner、最終VP、rank、全GameState、RNG counter、policy_stepsが12/12完全一致。さらに10%実験用の別seed36局も同じ0% parityが36/36成立した。

Safety-onlyの12局ではResolverに由来する行動差分が2局、別seed36局では7局。差分が生じた最初のActionはすべてResolverが検出した確定勝利Actionで、そのstepで本人が勝ち終局した。既知の seed `2941006` / Champion相手 / 席3 / turn 63 / revision 365では、Champion counterfactualはPlayerTradeだったが、Resolverが `BuildSettlementAction(vertex_id=0)` を実行し、そのstepでP3が10点勝利した。Stage 3A-13で見えた同一試合の3連続missを、最初の局面で止める。

6 familyを各1局でGate抽選を迂回して強制指定し、Frozen Executor→`env.step_action()`→次learner boundary→終局まで通した。全6局が合法、family一致、終局到達、truncationなし。forced recordはon-policy sampleへ混ぜていない。Road Bの例は `BuildRoadAction(edge_id=14)`、所有者4、最長道路長1→2、接続後の合法道路候補7→8、開拓地候補0→1、木/レンガ各1消費、相手開拓地endpointなし。BankTrade Bの例は木4→0、羊2→3、銀行の木7→11・羊17→16、交換比率4:1で資源保存。別unit testでは2:1港と銀行在庫0によるAction mask抑制、相手開拓地endpointへ伸びる道路の記録も検証した。

## 実委譲smoke

5%・native prior argmaxはRule/Champion/Mixed各4局、計12局を完走。Gate Surface 371、protected preemption 128、抽選対象243、実委譲13。familyはSettlement1、City2、Road3、Development4、BankTrade3、EndTurn0。全て終局し、illegal、Executor failure、deadlock、truncationは0。

10%・native prior categoricalは各profile12局、席1〜4各3局、計36局を完走。Gate Surface 1,126、protected PlayerTrade候補あり380、Championが実際に保護交渉を選択380、抽選対象746、実委譲66。family選択は以下。

| family | 選択数 | 選択時の平均確率 | available時の平均確率 | 選択時top-1率 |
|---|---:|---:|---:|---:|
| BUILD_SETTLEMENT | 9 | 0.999 | 0.999 | 100% |
| BUILD_CITY | 5 | 0.997 | 0.997 | 100% |
| BUILD_ROAD | 21 | 0.798 | 0.475 | 100% |
| BUY_DEVELOPMENT | 18 | 0.995 | 0.747 | 100% |
| BANK_TRADE | 10 | 0.695 | 0.334 | 70% |
| END_TURN | 3 | 0.157 | 0.059 | 0% |

確率の集計母集団は「実委譲した66局面」であり、全Surfaceのprior分布ではない。entropy平均（nat）は候補2が0.303（30件）、候補3が0.318（27件）、候補4以上が0.018（9件）。native priorは特に多候補時もかなりsharpであり、value qualityの保証ではない。Development18/66、BankTrade10/66を「過多」と断定するには局数不足だが、次工程で監視対象とする。実委譲局面のChampion family一致40/66 = 60.6%、具体Action一致28/66 = 42.4%。これは挙動差分の診断であり、成功指標ではない。

66委譲に対し `StrategicTransition` 66件。次に実委譲されたGateでcloseしたもの37、terminal close29。途中でprotected tradeをまたいだ延べ回数239、非委譲Surfaceをまたいだ延べ回数419。macro_durationは既存学習者`policy_steps`単位で平均23.12、中央値17、p95 64、最大85。割引報酬 `Σ gamma^i r_i` を全transitionで元reward列から再計算し一致、`gamma=0.99` の `gamma^duration` とterminal bootstrap 0も確認。macro rewardは平均0.0945、中央値0.0468、範囲 -0.9227〜1.05。Actor sampleは実際にGateが選んだ66件だけで、protected/Resolver/非委譲Actionは入れていない。

全36局で終局到達。illegal Action、family外Action、Executor failure、deadlock、truncation、非有限rewardは0。Rule/Champion/Mixed各12局、席1〜4各9局。Strategic Gate用optimizerを作らず、Champion重みSHA-256は前後 `4225902d41a83540eb1f1552826ec623706e778b99ad4ae8ec015f976644392a` で不変。

最終コードで全313テスト成功。収集完了後に追加したGameState RNG非消費assertと、ROAD/BANK_TRADE適用後maskの診断も、BankTrade Bの再実行smokeおよび専用unit testで確認した。36局の保存済みデータはこれら追加field導入前の同じ実委譲結果であり、`next_legal_*` fieldは含まない。

## 同seed比較と判断

36局のvariant別指標（対局差の分離用であり、有意な強弱判定ではない）：

| variant | 勝数 | 平均VP | 平均順位 | 平均policy steps | 開拓/都市/道路 | 発展購入/銀行交換/EndTurn/PlayerTrade |
|---|---:|---:|---:|---:|---|---|
| Legacy | 17/36 | 8.03 | 1.92 | 72.5 | 111 / 71 / 226 | 62 / 155 / 584 / 401 |
| Safety | 18/36 | 8.06 | 1.89 | 72.1 | 109 / 70 / 228 | 62 / 154 / 580 / 396 |
| Strategic 10% | 17/36 | 7.86 | 1.94 | 72.7 | 104 / 64 / 227 | 73 / 134 / 586 / 402 |

建設/その他Action数は対象学習者のaction phaseで選択した回数で、初期配置を含まない。SafetyのResolverは18回発火し、うち7局でLegacyのActionを確定勝利Actionに変更。Strategic実験ではResolverが16回発火。A→BはResolver効果、B→CはGate導入差だが、Gateにより到達状態も変わるため単一Actionの因果効果ではない。36局では微小な平均値差から性能改善・悪化を確定しない。

Step 3A-15の**初回Strategic Gate RL設計へ進むためのintegration gateは通過**。ただし自動学習へ進まず、native priorのsharpさ、Developmentへの偏り、BankTrade Bの実価値、66件という少量sample、長いsemi-MDP区間の分散を先に考慮する。次工程ではChampion GNN/377 Actor/Executorを固定し、実委譲6-family sampleだけでGate residualと別Strategic Criticを更新する設計を検討する。学習用rolloutでは、今回の診断recordに加えてpre-action v2 Observationと79次元Runtime Contextの保存が必要。安全層とprotected arbitrationは維持し、まず新seed・低委譲率・KL/entropy監視を定める。Champion一致をrewardへ入れない。
