# Step 3A-13: Strategic Gate zero-impact / Frozen Executor / prior 監査

2026-10-04。学習・Gate実委譲・Champion変更は行っていない。Rule / Champion / Mixed 各12局、対象席1〜4各3局（各profile）を固定seed `2940000〜2940011`、`2941000〜2941011`、`2942000〜2942011` で監査した。生データと集計は `experiments/strategic_gate_shadow_step3a13_20261004/` に保存した。

## 設計と安全境界

- 上位候補は固定順 `BUILD_SETTLEMENT / BUILD_CITY / BUILD_ROAD / BUY_DEVELOPMENT / BANK_TRADE / END_TURN` の6 family。既存のaction前GameState、合法377 mask、Step 3A-12 surface predicateからcandidate maskを決める。2候補以上のみGate対象。Champion最終Actionをmask作成へ使わない。
- Gate入力は凍結Champion Actor latent、v7 Runtime Context 79値、6候補mask。既存15 family headの対応logitに、別encoderの6-way residualを加える。残差最終層をゼロ初期化したため、利用可能familyの初期logitは既存family logitとbitwise一致。377 Actor、GNN、Critic、v7の377-way residualは変更しない。
- Gate proposalはShadow記録のみ。実Actionは従来のChampionが一度だけ選択する。ExecutorにはChampion actual Action、execution source、Intent、Reasonを渡さない。Executorはfamilyを選ぶ価値判断をせず、同じfamilyの具体Actionまたは明示的failureを返す。
- Settlementは `_best_settlement()` の合法vertex順位、Cityは `_best_city()` の合法vertex順位を使用。`_city_action()` 全体は使用しない。BuyDevelopmentとEndTurnはそれぞれ直接Action生成。
- Road Aは既存の拡張計画から有目的edgeを抽出して、拡張・最長路伸長・凍結候補logitの順でfamily内順位付けする。Road Bは合法BuildRoadActionだけに凍結377候補logitをmaskして選ぶ。Aは既存計画への依存があり、将来「道路の目的別価値」をPPOへ移す際には再監査が必要。Bはよりcleanなfamily-only baseline。
- BankTrade Aは現行helperがgoal/budgetと結合しており、純粋なgive/get順位だけを安全に抽出できないため `UNSAFE_GOAL_BUDGET_DEPENDENCY` を明示する。BankTrade Bは合法BankTradeActionだけを凍結377候補logitで選ぶ。A失敗時もChampion ActionやEndTurnへfallbackしない。
- 各Executorを同一入力で2回実行して決定性を確認し、377合法mask・family一致・GameState非変更・コピーへの `apply_action()` 成功を検査。BankTradeは各資源の銀行＋全員手札の保存、在庫・手札非負も検査する。
- PlayerTradeは6 familyへ追加しない。source-levelの提案候補有無とChampionによる実選択を別fieldへ保存し、overlap局面でもShadow GateとExecutorを評価する。初回実委譲時は、belief/history入力が不足する間だけ、このoverlapを一時的に委譲対象外とする案を推奨する。交渉の価値を常に上位とする永久ルールではない。
- ImmediateWinResolverはChampion判断を参照せず、合法な建設・道路・騎士ActionをGameStateコピーへ適用し、勝者が本人かつ得点10以上を確認する。同時に複数あれば最小377 catalog ID。カード内容が不確定な発展購入など、単一Actionで確定しない勝利は対象外。今回は検出するだけで実Actionを上書きしない。
- 別encoderを持つStrategic Criticは `frozen Champion base value + zero-initialized residual`。旧Critic valueは新Gate policyの真値ではない。6-family `StrategicTransition` 型は、実委譲時だけActor sampleを持ち、次の実委譲Gateまたは終局でcloseするsemi-MDPを表す。`gamma ** macro_duration`、終局bootstrap 0。今回transitionは収集していない。

## 36局 Shadow結果

| profile | games | Gate decision | protected PlayerTrade available / selected | 全候補のB Executor成功state |
|---|---:|---:|---:|---:|
| Rule | 12 | 393 | 126 / 126 | 393 |
| Champion | 12 | 383 | 144 / 144 | 383 |
| Mixed | 12 | 347 | 118 / 118 | 347 |
| 合計 | 36 | 1,123 | 388 / 388 | 1,123 |

各profileの席1〜4はそれぞれ3局。Decision数は席1/2/3/4が295/262/327/239。候補数分布は2候補570、3候補399、4候補132、5候補20、6候補2。protected overlapなし735、あり388。前者のChampion実Actionは道路212、EndTurn202、BankTrade111、Settlement94、Development59、City57。overlapあり388局面ではChampionの実ActionはすべてPlayerTradeだった。proactive trade候補をaction前のコピーから別途診断しているため、`available` と `selected` は別概念だが、今回のsampleでは数が一致した。

| family | available | B成功 | A成功 | 備考 |
|---|---:|---:|---:|---|
| BUILD_SETTLEMENT | 97 | 97 | — | 合法vertexのみ |
| BUILD_CITY | 72 | 72 | — | 合法vertexのみ |
| BUILD_ROAD | 616 | 616 | 616 | A/Bとも合法・決定的 |
| BUY_DEVELOPMENT | 403 | 403 | — | 価値判断なし |
| BANK_TRADE | 666 | 666 | 0 | Aは666件すべて明示的なgoal/budget依存failure |
| END_TURN | 1,123 | 1,123 | — | 価値判断なし |

Bを採用する場合、全available familyを具体Actionへ変換できたstateは1,123/1,123。illegal、family外Action、非決定性、GameState/RNG変更は0。BankTrade Bの資源保存・銀行在庫検査も全666件通過。AのBankTrade failureは隠さず記録した。

Road Aの616提案中、拡張edge 551、最長路伸長edge 494、両方429、どちらでもないedge 0。Road Bはそれぞれ422/336/254/112。同じ局面でChampionがROADを選んだ212件のうち、edge完全一致はA 194、B 87。これはChampion模倣度であり、Aが対局価値で優れる証拠ではない。BankTrade BはChampionがBankTradeを選んだ111件中57件で具体Action一致。Aは安全な純粋rankingを抽出できず比較不能。

## family prior監査

既存hierarchical Actorの15 familyは、`roll_order / roll_turn / initial_settlement / initial_road / start_game / robber_move / robber_steal / road / settlement / city / development_purchase / bank_trade / development_use / discard / end_turn`。今回の6 familyは末尾の該当6分類へ一対一対応する。元の分布はfamilyを選んでからfamily内Actionを選ぶ構造なので、377候補logitを異family間でそのまま比較するmax・log-mean-expより、既存family logitを上位priorに使う方が意味論上自然。ただしこれもPlanner正解ではなく初期prior候補である。

| prior | 平均entropy（nat） | protectedなし735件のChampion family一致 | 全1,123件の最多top |
|---|---:|---:|---|
| uniform | 0.937 | 65.7% | Road 494、Bank 257、Development 206 |
| max legal logit | 0.541 | 39.2% | Bank 666、Road 284、EndTurn 173 |
| legal log-mean-exp | 0.661 | 27.1% | Bank 666、EndTurn 451 |
| native 15-family logit | 0.328 | 58.4% | Road 332、Development 332、Bank 290 |

Uniformのtopは同確率時の固定tie-breakが強く反映され、一致率65.7%を品質指標と解釈できない。max・log-mean-expはBankTradeが候補にある全666件でtopになり、familyをまたぐlogitの尺度が不適切な疑いが強い。これらを初回実委譲priorには採用しない。native priorはBank一辺倒でなく、候補数を増減させても同じfamily logitが変わらない構造のため、初回候補として推奨する。ただしnativeは平均entropyが低く、Developmentを過度に選ぶ可能性もあるため、実委譲前に安全性を別検証する。

合法Action数と当該family確率のPearson相関は、Roadではuniform -0.055、max -0.049、log-mean-exp -0.103、native -0.010。BankTradeではuniform -0.186、max +0.160、log-mean-exp -0.026、native -0.159。これは候補の共起や局面状態との相関も含み、因果的なcardinality biasの推定ではない。native priorは候補数を直接集計しないという構造上の利点がある。profile別native一致はRule 58.1%、Champion 54.8%、Mixed 62.4%。席1〜4別は53.6% / 60.0% / 63.7% / 55.8%。分布の全表は `summary.json` の `by_profile` / `by_seat` と生recordに保存した。

## 即時勝利とparity

ImmediateWinResolverは19 Decisionで確定勝利Actionを検出した。内訳はSettlement 10、City 7、Road 2。Championがどの勝利Actionも選ばなかったのは3 Decisionで、いずれもChampion profileの同一試合 seed `2941006`、seat 3、turn 63（revision 365/367/369）。合法な `BuildSettlementAction(vertex=0)` で即10点にできたが、Championは異なる相手へのPlayerTradeを連続選択した。したがって単純な「GateでなければChampionへ戻す」だけでは即時勝利を保証しない。今回Championを修正せず、次工程でResolverをGate/temporary protected arbitrationより上位に置くべきと判断する。3件は独立した3試合ではない。

36/36局でShadow有無の全Action列、winner、最終VP、順位、全GameState、RNG counterが完全一致。1,123局面で凍結Actor candidate/family logits・Critic valueの前後一致と、zero residualのfamily logits・Critic一致を確認した。Champion重みのSHA-256は前後 `4225902d41a83540eb1f1552826ec623706e778b99ad4ae8ec015f976644392a` で不変。生recordの重複decision、候補数/mask不一致、確率和逸脱、protected以外のChampion Action候補漏れは各0。全テスト308件成功。

## Step 3A-14への判断

Shadow基盤・mask・B Executor全family・parityは通過したため、**学習ではなく限定的な実委譲integration smokeの設計へ進んでよい**。次工程では先にImmediateWinResolverを実Actionへ適用する安全層、次にprotected PlayerTrade overlapの一時除外、次にB Executorによる6-family Gateを置く。BankTrade Aは不採用、Road A/Bの実価値差は未確定なので初回はcleanなBを基準とする。native family logit priorを凍結初期分布として用い、uniformをそのまま委譲しない。委譲0%の完全parity、低委譲率での合法性・完走、同一family concrete executorの状態不変性を確認してから、初めてon-policy semi-MDP収集を検討する。Champion agreementを教師報酬や採用条件にしない。
