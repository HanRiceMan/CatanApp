# Step 3B-4 — Hierarchical Shadow Controller / Routing Architecture

## 実装範囲と結論

Step 3B-4では、3B-2の未学習Victory Plan / Strategic Priority Headを使い、**現在どの判断境界にいるか**を副作用なしで表すShadow Controllerを追加した。Plan選択、Action推薦、Action適用、PPO更新は行わない。現行ChampionのActionはそのまま実行する。

新規moduleは [`hierarchical_shadow_controller_step3b4.py`](../app/rl/hierarchical_shadow_controller_step3b4.py)、300局再生・照合は [`run_hierarchical_shadow_step3b4.py`](../app/rl/run_hierarchical_shadow_step3b4.py)、単体テストは [`test_hierarchical_shadow_controller_step3b4.py`](../tests/test_hierarchical_shadow_controller_step3b4.py)。圧縮episode recordと集計は [`rl_hierarchical_shadow_step3b4_20261013_evidence`](rl_hierarchical_shadow_step3b4_20261013_evidence/summary.json) に保存した。

## Routingの意味論

`HierarchicalRoute` は `IMMEDIATE_WIN`, `FORCED_PHASE`, `PRE_ROLL_DEV_USAGE`, `ACTION_DEV_USAGE`, `CONSTRUCTION_COMMIT`, `CITY_VALUE_COMPARISON`, `DUAL_CONSTRUCTION_COMPARISON`, `INVESTMENT` の8種。優先順は次のとおり。

```text
ImmediateWinResolver が確定勝利を検出 → IMMEDIATE_WIN
free-road / setup / discard / robber / trade pending 等 → FORCED_PHASE
turn_pre_roll で旧Dev Card使用可能 → PRE_ROLL_DEV_USAGE
通常 action で旧Dev Card使用可能 → ACTION_DEV_USAGE
    NO_USE の先 → 通常action route（fallback_route）
通常 action:
    Settlementのみ建設可能 → CONSTRUCTION_COMMIT
    Cityのみ建設可能       → CITY_VALUE_COMPARISON
    両方建設可能           → DUAL_CONSTRUCTION_COMPARISON
    どちらも不可           → INVESTMENT
```

`CONSTRUCTION_COMMIT` は「このturnに何らかの得点建設へ向かう」意味であり、**次ActionをBuild Settlementへ強制しない**。BankTrade / PlayerTrade先行やCityへの切替えを妨げない。今回の`construction_commit_active`は当該stateの分類factであり、turnを跨ぐ持続的な命令・Planではない。City-onlyはBuild/No-Buildを価値比較へ残し、BOTHはSettlement/City比較候補、NONEはRoad/BuyDev/BankTrade/PlayerTrade/EndTurnのInvestment候補とした。pre-rollとaction-phaseのDev使用には必ず`NO_USE`を含み、後者は`fallback_route`へ戻れる。free-road、pending trade response/counter等はStrategic routingへ混ぜない。

Immutable `HierarchicalDecisionContext`はroute、phase、build class、建設可否、Dev使用mask、7-familyの`legal_family_mask`と`comparison_family_mask`、PlayerTradeの**ルール上の提案可能性**、construction commit fact、forced理由、soft Planを保持する。PlayerTrade factはprotected controllerが実際にTradeを選ぶかとは区別する。maskは**将来の比較境界を示すShadow診断**であり、実環境のAction maskには接続しない。DUALのcomparison maskは当面Settlement/Cityだけ、他の合法familyは別のlegal maskに残す。COMMITでもEndTurn等の合法familyを診断上消さない。

## Soft Plan / Priority bridge

未学習Planをsample/argmaxしない。各valid Plan候補の確率と決定的candidate factの加重平均で、`(Settlement予定増分, City予定増分, Longestを含む確率, Armyを含む確率, planned gain, overshoot)` の6次元`weighted_features`を生成する。`expected_priorities`は候補別4独立Priorityの同じ確率加重平均である。FEASIBLE/UNKNOWNはvalid、IMPOSSIBLEは確率0。Planがunresolvedまたは残り点0なら`available=false`でaggregateは`None`とし、架空のzero Planを作らない。

Headは3B-2のzero initializationのままで、valid Plan間の分布はuniform、expected Priorityは4軸とも約0.5。これは**未学習の中立出力**であり、戦略選択の結果ではない。Planner Intent、Champion Action、native 6-family logits、手作りPlan価値係数は入力・教師にしない。候補順を入れ替えてもID対応後のsoft featureとexpected Priorityが一致することをunit testで検証した。

## 300局Shadow結果

Rule `34000000–34000099`、Champion `34001000–34001099`、Mixed `34002000–34002099`、各profile100局・各seat25局。各局をLegacy baselineとobserver付きLegacyで再実行し、Action trace、winner、final VP、rank、policy steps、final GameState、Game RNGを完全照合した。さらに3B-3保存済みのaction/pre-roll learner boundaryとAction、turn、revision、policy step、build class、ImmediateWinを突き合わせた。

全300局、Rule/Champion/Mixed各100局・seatごとに25局。learner boundary総数22,355。route件数は次のとおり。

| route | Rule | Champion | Mixed | 合計 |
|---|---:|---:|---:|---:|
| IMMEDIATE_WIN | 100 | 27 | 46 | 173 |
| FORCED_PHASE | 2,936 | 3,078 | 2,743 | 8,757 |
| PRE_ROLL_DEV_USAGE | 160 | 150 | 137 | 447 |
| ACTION_DEV_USAGE | 0 | 0 | 0 | 0 |
| CONSTRUCTION_COMMIT | 299 | 252 | 265 | 816 |
| CITY_VALUE_COMPARISON | 218 | 166 | 159 | 543 |
| DUAL_CONSTRUCTION_COMPARISON | 6 | 5 | 8 | 19 |
| INVESTMENT | 3,491 | 4,398 | 3,711 | 11,600 |

通常actionのImmediateWinは166件、pre-rollの確定Knight勝利などが7件。3B-3のImmediateWin除外後のbuild classとrouteは、`SETTLEMENT_ONLY→COMMIT 816/816`、`CITY_ONLY→CITY_VALUE 543/543`、`BOTH→DUAL 19/19`、`NONE→INVESTMENT 11,600/11,600`。free-road 64件は`FORCED_PHASE`へ入り、InvestmentのRoadへは入らない。`FORCED_PHASE`の内訳はpre-roll roll-only 5,178、rolling order 412、ready-to-play 75、robber move 1,260、robber steal 1,253、discard 515、free-road 64。

pre-roll Dev境界447件では`NO_USE`が全件で利用可能。使用可能cardはKnight 334、Monopoly 45、Year of Plenty 38、Road Building 34件（複数cardが同時利用可能なstateあり）。pre-rollでDev使用可能だった計454件のうち7件は確定勝利が上位routeへ移った。通常action中の使用可能Devはこの300局で0件だが、合成unit stateで`ACTION_DEV_USAGE`、`NO_USE`、`fallback_route`を確認した。

Shadow family maskでは、COMMIT 816件の全てにBuild Settlementが合法。Road/EndTurnも合法だが、**Championの次Action**はBuild Settlement 802、BankTrade 9、PlayerTrade 5で、Road/BuyDev/EndTurnは0だった。これはAction強制の根拠ではなく、合法maskと実際の選択を区別する診断である。City Value 543件の次ActionはBuild City 418、Road 32、BankTrade 24、PlayerTrade 28、EndTurn 41。Dual 19件はSettlement 15、City 4。Investment 11,600件の次ActionはRoad 1,603、BuyDev 597、BankTrade 1,022、PlayerTrade 3,389、EndTurn 4,989。これらはChampion行動の記述であり教師ラベルではない。

soft Plan forwardを行った13,425 stateで全てPlanがavailable、valid candidate確率和1、IMPOSSIBLE確率0、expected Priorityの平均は各軸`0.500000013`（浮動小数誤差のみ）。Plan-unavailableは実300局では0件だが、score≥10のunit stateで`available=false`、aggregateなしを確認した。代表state（seed `34000000`）では、turn1のInvestmentでsoft featureはおよそ`(3.0, 2.82, 0.545, 0.545, 8.0, 0.0)`、expected Priorityは全軸0.5。これはuniform未学習Headからの**候補平均**であり、選択済みPlanを意味しない。

全300/300局でLegacy baselineとのAction列、winner、final VP、rank、policy steps、final GameState、RNGが完全一致。3B-3保存済みaction/pre-roll learner boundaryとのAction・turn・revision・build class照合も全件一致。ObserverのGameState/RNG mutation 0、Action変更 0。収集前後のSHA-256 digestはChampion `4225902d41a83540eb1f1552826ec623706e778b99ad4ae8ec015f976644392a`、Robber PPO `ecde7fa9e9d67370bb76686ecd767d2d532a76d467c0051aaebdf8d4aff702bf`、3B-2 Head `2000b1390fcf8135ba819ba40b1ee1720f0596327c2c658ec1e2f24a29dd1887` で各々前後一致。optimizerは未生成。

検証はStep 3B-4 unit **7 passed**、backend全体 **389 passed / 1,172 subtests passed**（既存deprecation warning 2件とpytest cache権限warning 1件）。候補順序不変、weighted soft集約、Plan unavailable、Dev `NO_USE` fallback、forced/trade/free-road分離、family maskを含む。全件数・profile別件数・mask別件数・代表例・各episode checkpointは集計JSONと圧縮recordに保存した。

`ACTION_DEV_USAGE`は本sampleでは0件だが、使用可能な人工action-phase stateで`NO_USE`と通常routeへのfallbackをunit testした。`PRE_ROLL_DEV_USAGE`は旧Dev Card使用可能なpre-roll stateのみ。ImmediateWinは通常の建設/Investment集計から分離した。Shadow observerのGameState mutation、RNG consumptionはいずれも0。Frozen Champion、Robber PPO、3B-2 Plan/Priority Headのdigestは収集前後一致し、optimizerは生成していない。

## Step 3B-5へ

次はこの`HierarchicalDecisionContext`とsoft Plan/PriorityをL3 Shadow Policyへ入力し、`INVESTMENT`、`CITY_VALUE_COMPARISON`、`DUAL_CONSTRUCTION_COMPARISON`で**比較可能なfamily head**を設計する。`CONSTRUCTION_COMMIT`は同turn建設へのcommitとその前のTrade等のAction順序を分ける。Development Usageの`NO_USE`→fallbackを維持し、PlayerTradeは当面protected。初期段階はActionに接続せず、既存Championとの完全parityを継続する。Plan sample/persistence、Reward変更、PPO updateは別工程とする。
