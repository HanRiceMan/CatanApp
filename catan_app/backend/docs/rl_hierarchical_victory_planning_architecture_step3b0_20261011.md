# Step 3B-0 — Hierarchical Victory Planning 設計監査

2026-10-11。**設計・コード監査のみ**。PPO/Critic update、Reward、Observation、candidate mask、Executor、Champion Action・weightの変更はない。Step 3A-23のE/F収集は依頼変更時点で停止した（Eは64ペア、Fは47ペア＋Safetyのみ1件、計223個の正常に読める対局checkpoint）。部分証跡は [`Step 3A-23 PAUSED archive`](rl_strategic_stage0_final_step3a23_PAUSED_20261011.zip) に保存した。432 paired gameのqualificationは未実施・未判定であり、Step 3A-22の10%判定 **BORDERLINE** を「合格」に読み替えない。Step 3A-21 Post1は技術検証artifactとして保存し、Step 3Bの教師・parentとは定めない。

## 1. 現行3Aの構造と、3Bで変える責務

現行の [`strategic_action_surface.py`](../app/rl/strategic_action_surface.py) はaction phaseの合法377 maskから `BUILD_SETTLEMENT / BUILD_CITY / BUILD_ROAD / BUY_DEVELOPMENT / BANK_TRADE / END_TURN` の6候補を作る。`candidate_count >= 2` などでSurfaceを定義し、確定勝利・free-road・pending trade等を除外する。[`strategic_gate_step3a13.py`](../app/rl/strategic_gate_step3a13.py) のGateはFrozen Championの15-family logitsを6 familyへ写し、T=2 priorと学習可能residualから分布を作る。Executorは選ばれたfamilyを具体Actionへ変換する。Road/BankTradeは合法候補内のFrozen 377 logit argmax、Settlement/CityはChampionのfamily内地点選択、Development/EndTurnは直接Actionである。[`strategic_gate_rollout_step3a14.py`](../app/rl/strategic_gate_rollout_step3a14.py) はImmediateWinResolver→phase/forced→protected PlayerTrade→Surface→外部RNGによる委譲→Gate→Executorの順でActionを適用し、**実委譲**の間だけsemi-MDP transitionを作る。

この構造が選ぶのは「今回どのfamilyか」であり、(a)10点への得点構成、(b)複数得点源への持続的な優先度、(c)その文脈での資源投資を別変数として持たない。Frozen native family logitは既存377 Action分布の一部で、長期得点源間の価値尺度ではない。Road/BankTradeの具体Action品質もfamily判断のreturnに混入する。従って6-family Gateは接続・合法性・on-policy学習の**技術prototype**として保持するが、最終階層とはみなさない。Step 3A-18/19のFull Surfaceで見えた建設減少、Dev増加、累積的trajectory変化は、PlanやPriorityを後付けTeacherラベル化して解く問題でもない。

| 階層 | 出力・意味 | 決定論的層が担当 | PPOに委ねる価値判断 | 下位への作用 |
|---|---|---|---|---|
| L1 Victory Plan | 本人の**現在点から**10点へ至る暫定的な得点増分構成。軸は Settlement gain、City upgrade、Longest Road、Largest Army | 本人既知の実得点、残点、駒・盤面の物理上限、称号保有、公開相手状況、候補feasibility/unknown mask | 複数の実現可能な勝ち筋のうち、どの組合せを今追うか。いつ再計画するか | 文脈・記憶。下位Actionの禁止maskではない |
| L2 Strategic Priorities | 4得点源への**独立した**優先度。複数同時に高くてよい | title gap、資源不足、盤面到達性、生産、コマ数、相手の公開進捗 | 得点源ごとの重要度、切替え費用、相乗効果、現在の機会費用 | 投資・カード使用の条件付き入力。単独では命令でない |
| L3 Current Investment | 今の資源をRoad/Dev/BankTrade/将来PlayerTrade/Waitまたは即時建設へ配分 | phase、合法mask、費用、候補、具体Actionの合法性 | 資源保持と消費、目的物完成までの寄り道、Action family間の機会費用 | family選択後に具体Executorへ |

勝利点は [`reward.actual_score`](../app/rl/reward.py) / [`game._check_winner`](../app/domain/game.py) に整合させる。`remaining=max(0,10-self_actual_score)`。開拓地新設は+1、既存SettlementのCity化は**純増+1**、未保有Long/Army称号の新規取得は各+2。現在保有する称号の2点や本人の既知VPカード点は既に `self_actual_score` に入るため、Planの将来増分へ二重計上しない。称号防衛の必要性は、増分+2ではなく保持リスク/priorityとして表す。VPカードの**将来ドロー**はPlan軸に置かず、引いた事実が本人に分かった時点で現在点を更新する。相手の非公開VP・将来ダイス・山札の次カード・相手手札の真値は入力しない。

Planは契約でもmaskでもない。例えば `planned_settlement_gain=0` でも有利な合法Settlementを建てられる。その後現在点、駒、到達性が変わるため再計画する。確率的な将来まで「厳密合法」と断言せず、物理的に不可能な候補だけ除外し、到達性が不確かな候補にはvalid/unknownを残す。Settlement駒はCity化で戻るため、現在の残駒数だけで将来Settlementの総上限を切らない。候補集合が空なら架空のPlanを選ばず `unresolved` とする。

例として現在2点なら、`Settlement +2、City純増 +2、Longest +2、Army +2` は8点分の暫定経路になり得る。途中でPlan外のSettlementを建てて3点になれば、次のPlanは残り7点から作り直す。Cityへのupgradeは建物表示の2点でなく、現在のSettlementからの**増分1点**だけをPlanに積む。

```text
本人既知状態 + 公開盤面 + 既存Context79
    ├─ ルール・スコア・駒/グラフ・候補/valid mask（決定論的）
    └─ Frozen GNN/表現
          ↓
L1 Victory Plan（再計画時に選択、保持状態・ageを記録）
          ↓
L2 4独立Priority（Plan/現状態に条件付け、同時に高くてもよい）
          ↓
phase routing + 即時勝利/forced safety
          ├─ turn_pre_roll: roll と使用可能Devのタイミング
          └─ action: Dev使用機会・即時建設・Investmentを整合的に比較
                    ↓
L3 Investment / Dev Usage / 将来Trade Policy
          ↓
候補mask + Concrete Executor → 既存377 Action → Environment
          ↓
本人得点/称号/盤面変化でPlan候補を更新。Plan外行動は許す
```

### Victory Plan表現案

| 案 | 利点 | 制約・学習上の問題 | 暫定判断 |
|---|---|---|---|
| A. 離散候補 | `(追加Settlement, 追加City upgrade, 新規Road称号, 新規Army称号)` を列挙しmask。1候補の意味が明瞭、PPO categoricalと整合 | 地平が長く候補数が増える。駒再利用、title奪還/喪失、VP獲得で頻繁に再計画。現時点で未到達を「永久不可能」と誤判定しやすい | 小さい候補集合を診断できるか3B-1で評価 |
| B. 4独立target score | 固定4次元で柔軟、ActionがPlan外でも不自然でない | 「残り何点」を直接満たさず、称号2点・City純増1点や物理上限との整合が見えにくい。Planの解釈が曖昧 | Priorityには適するがL1単独案として弱い |
| C. hybrid | 残点と決定論的feasibility/unknown、暫定4軸の得点増分Plan、および別の4独立priorityを併用。解釈性と柔軟性を両立 | Plan候補生成と保持状態の設計が必要。PlanとPriorityの重複を避ける必要 | **第一候補**。Aの候補生成が過大なら疎なPlan表現へ切替え |

Plan候補の合計点は現在点からの「見通し」であり、将来カード運・相手の妨害・title喪失まで決定できない。残点を満たす候補を優先的に表現するが、境界の丸めや保険を勝手なheuristic scoreにしない。Plan選択後も4軸以外の合理的Action（盗賊解除、資源調整など）を許す。

### Strategic Priority表現案

| 方式 | 長所 | 問題 |
|---|---|---|
| 排他的Softmax | 実装容易 | 1軸を上げると他軸が必ず下がり、道路のSettlement+Longest相乗効果を表しにくい。**第一候補にしない** |
| 独立Sigmoid | boundedで同時高・同時低を表せる | 端で飽和し、絶対的機会費用の尺度は決まらない |
| 独立unbounded score | 相対価値幅を自由に表せる | scale/offsetが同定不能になりやすい。学習時の分布/安定化設計が必要 |
| 正値weightの正規化 | 比率が読みやすい | 合計1等なら事実上競合させてしまう。非正規化ならscale問題が残る |

第一候補は**4個の独立score**。実装時に bounded Sigmoid か unbounded stochastic/latent を比較する。PriorityをPPOの明示Actionにするならそのsample分布とlog-probをon-policyで保存する。deterministic内部表現として下位Actorに渡すなら、Priority自体に独立したPPO ratioを付けたと称さず、下位Actionのreturnからend-to-endに学ぶ設計と区別する。Priority値をPlanner Intentの教師ラベルへ合わせない。

## 2. 即時建設・Development Usage・Tradeの境界

[`game.available_actions`](../app/domain/game.py) 相当の分岐では、`turn_pre_roll` に建設はなく、Rollまたは旧カードの使用が可能。`action` では建設・購入・銀行交換・カード使用・EndTurnが同時に合法になり得る。新規購入カードはその手番では使用できず、1手番に使用できるDevは1枚。Road Building Card後のfree-roadは先に処理する。従って「Development Usageを必ず建設より先に実行」という**固定優先順位は不正確**。phaseを先に分け、action phaseではカード使用機会と即時建設の競合を表現する。

即時建設の初期baselineは、Resolverで確定10点勝利を先に取得し、free-road/forced等を除いた通常actionで、Settlementだけ即時合法ならSettlement、CityだけならCity、両方ならPlan+Priority+地点/生産情報による比較、どちらも不可ならInvestmentへ進む、というもの。ただし「一方だけ即時可能」を**永久のhard safety ruleにしない**。先にKnightでtitle/盗賊、Monopoly/Year of Plentyで生産・資源獲得、Road Buildingで相乗効果を得るなど、合法な価値競合はあり得る。3B-3で実ゲーム事例を調べ、必要な例外を憶測で追加せず、baseline/Shadowと将来のPPO判断境界を固定する。Plan値が0の得点源でも建設をmaskしない。

Development Usageは購入と完全に分離する。使用Policyは `NO_USE / KNIGHT / MONOPOLY / YEAR_OF_PLENTY / ROAD_BUILDING` の候補を本人所有・新規カード・1手番1枚・phase・銀行在庫・free-roadでmaskする。`NO_USE` はpre-rollではRollへ、actionでは後続建設/Investmentへ渡す。選んだ後のMonopoly資源1種、Year of Plenty 2資源、Road Buildingのedge、Knightの盗賊tile/steal相手は別のtarget選択境界を置く。これらのtargetにも価値判断があるため、単なる「具体Actionへ機械的変換」で済むとは仮定しない。Monopolyの相手手札真値は見せず、後段で公開履歴/Bayesian beliefを用いる。Road BuildingはSettlement/Longestの両方に寄与し得る。KnightはArmy進捗・Robber relief・妨害を同時に見られるようにする。

PlayerTradeは最終的にInvestment候補へ追加し、選択後の相手・give/get・倍率・再提案をTrade Policy/Executorへ分ける。現3Aでは [`trade_strategy.py`](../app/rl/trade_strategy.py) の相手資源belief、今ターン提案履歴、予算・資源予約に依存し、[`strategic_gate_rollout_step3a14.py`](../app/rl/strategic_gate_rollout_step3a14.py) が一時的に保護している。保護は「Tradeが常に他投資より優先」という価値ruleではなく**情報不足の境界**である。belief/historyをCore79へ一括混入せず、Trade選択とtrade tuple実行のどちらに必要かを3B-6で分ける。

## 3. 既存Runtime Context79の再利用先

[`runtime_context.py`](../app/rl/runtime_context.py) は68 field + 11個のvalid mask = 79次元。以下の `{resource}` は実在する各fieldをすべて表す。`P`=Plan feasibility、`R`=Priority、`I`=Investment、`D`=Development Usage、`S`=safety/legality、`—`=初期階層では追加利用不要。単一fieldが複数の層へ届くのは同質な教師目標を増やす意味ではない。invalid maskは各利用層へ同時に渡す。

| field（展開を含む） | 主な利用先 | 理由・注意 |
|---|---|---|
| `settlement.candidate_exists`, `.candidate_count`, `.min_additional_roads` | P,R,I | 物理的な候補/距離。vertex選択やPlanner `plan_value` の正解ではない |
| `settlement.shortage.{wood,brick,sheep,wheat}` | I,R | canonical最少道路候補への不足。最良候補の価値ではない |
| `settlement.immediate_build` | I,S | 即時合法性/建設routing |
| `settlement.eta` | P,R,I | 現生産・港に基づく概算。無効時のmask必須 |
| `city.upgradeable_count`, `.best_production_gain`, `.pieces_remaining` | P,R,I | 将来候補数、最大pip純増、物理駒制約。具体city vertexの教師ではない |
| `city.shortage.{wheat,ore}` | R,I | 資源の機会費用 |
| `city.immediate_build` | I,S | 即時建設候補 |
| `city.eta` | P,R,I | 概算待ち。valid mask必須 |
| `road.current_length`, `.acquisition_gap`, `.defense_gap`, `.structurally_extendable_count` | P,R | 現称号の取得/保持可能性。相手title進捗も別途必要 |
| `road.affordable_title_directed_count` | R,I | 今置ける称号進捗edge数。どのedgeが最善かではない |
| `road.immediate_acquisition`, `.immediate_win` | I,S | 即時得点/確定勝利の候補事実。Resolverを優先 |
| `road.shortage.{wood,brick}` | R,I | Road投資の資源不足 |
| `knight.used`, `.usable`, `.plays_needed`, `.draws_needed`, `.deck_remaining` | P,R,D | Army新規取得または防衛の制約。plays/drawsは別意味 |
| `knight.development_affordable` | I,R | Dev購入はInvestmentで、Army専用Actionではない |
| `knight.playable_now`, `.immediate_acquisition`, `.immediate_win` | D,S | 使用可否/確定title。新規カードと1枚/手番制約を反映 |
| `robber.blocked_pips`, `.blocked_ratio`, `.blocked.{wood,brick,sheep,wheat,ore}` | R,D,I | 盗賊被害/資源阻害。5番目のVictory Plan得点源にはしない |
| `robber.playable_knight`, `.removable_now` | D,S | 解除の実行可能性で、解除すべき価値ではない |
| `risk.hand_size`, `.discard_exposure`, `.discard_count_if_seven`, `.resource_diversity` | I,R | 保持/消費・7リスクの客観事実。HAND_OVERFLOW Reason labelは入力しない |
| `risk.development_shortage.{sheep,wheat,ore}`, `.bank_trade_options` | I | Development/銀行交換の費用・可用性 |
| `risk.port_advantage.{wood,brick,sheep,wheat,ore}` | P,R,I | 港変換可能性。獲得予定港の将来価値は別判断 |
| `risk.legal_family.{road,settlement,city,development,bank_trade}` | I,S | 377合法maskの集計。候補maskを置換しない |
| `risk.legal_family.play_development` | D,S | 使用可能Action総数。カード種類別maskは別に必要 |
| `risk.immediate_construction` | I,S | Roadも含む集約flag。Settlement/Cityの個別flagを優先して使う |
| `new_development.{knight,road_building,year_of_plenty,monopoly}` | D,S | **今ターン購入分**であり、使用可能枚数ではない。既存手札から差引く |
| `new_development.victory_point` | —（本人得点整合の診断） | VPカードは使用Actionなし。本人実得点に既に反映されるのでPlan点数へ再加算しない |

追加候補はまだ**未実装**。既存v2/GNNにあるraw情報と重複する場合は、必要なら決定的集計するが無意味な同値copyを増やさない。

| 候補 | 必要な層 | 入手可能性・責務 |
|---|---|---|
| 本人実得点/残点、所有title、相手公開点・公開title保持 | P,R,S | v2/rawで既知。Plan作成時に算術計算。相手隠しVPは不可 |
| Settlement/Cityの将来物理上限、駒返却を考慮したfeasibility、道路接続/相手blockage | P | 公開盤面+ルールから決定。到達不能と未確定を分離 |
| **候補vertex別**資源pip、港、必要道路、候補edge別Settlement到達改善/最長路長増分、共有効果 | P,R,I,D(target) | 現Context79は個数/最大値等の集計で、候補ごとの構造は入らない。Planner `site_value/plan_value` をコピーしない。GNN側盤面表現との重複を検証 |
| 自分/相手の資源別生産pip、相手最長路長/使用済騎士、称号防衛差 | P,R | 公開盤面/カード使用履歴から決定。相手未来行動価値はPPO判断 |
| 旧カード種類別**今使える**枚数、銀行在庫、Monopoly/Year of Plenty合法target | D,S | 本人既知手札−newカード、1枚/手番、銀行公開在庫から決定。79にはnew枚数のみ |
| 今ターンtrade履歴、資源event/Bayesian相手手札belief | 将来I(Trade)/D(Monopoly・Knight steal) | 3A Core79から意図的に除外。履歴/推定であり、相手手札の真値ではない |
| 現在のPlan・plan age・再計画理由 | P→R/I/D | **Policy自身の過去出力/状態**。外部Teacher labelではない。保存・復元してMarkov性を保つ |

## 4. 現行Plannerで勝利方向を判断している所在

| コード | 現在の判断/依存 | 3Bでの行先 |
|---|---|---|
| [`settlement_planning_agent._select_planned_action`](../app/rl/settlement_planning_agent.py) | 既存PPO後にsetup/robber、pre-roll Knight、9点即建設、9点Settlement vs City ETA、3拠点City vs connected Settlement、目標拠点数後のCity/title/予約、first/second/third City deficit、resource reserve、最後にbase PPOへ分岐 | phase/即時勝利はA、各ETA/合法性はB、family内地点はC、目標数・City先行・title先行・予約閾値等はD。L1/L2/L3に分割 |
| [`_prefer_city_over_connected_settlement`, `_city_upgrade_value`, `_near_city_reservation`, `_guard_near_city_resources`](../app/rl/settlement_planning_agent.py) | pip/ETA/Planner比率でCityを優先し麦・鉱石を予約 | raw pip/ETA/費用はB。City投資と他用途の比較/予約価値はD。単純にExecutorへ移さない |
| [`_best_settlement`, `_best_city`, `_city_vertex_value`](../app/rl/settlement_planning_agent.py) | 合法頂点内の生産pip＋既存PPO logitによる地点選択 | 初期はCとして凍結。pipはBにも提供可。ただし将来vertex choice移行候補 |
| [`_longest_road_progress`, `_largest_army_progress`](../app/rl/settlement_planning_agent.py) | title gap/候補計算とscore/horizon/防衛thresholdを混在 | 長さ・gap・合法edge/usable KnightはB、今追う価値はD(L2/L3)、具体edge/使用ActionはC/別Policy |
| [`expansion_planner.analyze_expansion_plan`](../app/rl/expansion_planner.py) | 接続候補・pip・港・ETAを計算する一方、`site_value`、道路費用/待ち時間を含む`plan_value`でselected targetを選ぶ | 候補・構造・不足/ETAはB。`site_value/plan_value`はPlanner固有DなのでPPO入力/正解へコピーしない。暫定Executor依存は監査 |
| [`development_strategy.assess_development_purchase`](../app/rl/development_strategy.py) | 停滞/盗賊/Army近接、未知カード期待値、建設遅延閾値で購入を判定。VP狙いは購入理由にしない | 山札残・本人既知カード・資源等はB。購入の期待価値・City資源との機会費用はD(L3)。`BUY_DEVELOPMENT`はPlan軸でなくInvestment |
| [`trade_strategy.construction_trade_goal`, `_strategic_trade_budget`, `choose_proactive_trade...`](../app/rl/trade_strategy.py) | 直近建設goalと将来City資源予約、手札リスク、相手/提案履歴でtrade提案を選ぶ | 費用/在庫/履歴・beliefはB。tradeを今選ぶかはD(L3)、tuple内容は将来Trade Policy/Executor |
| [`victory_race.py`](../app/rl/victory_race.py) | 相手公開点、盗賊tile/stealの機会価値、belief | 公開点/盤面はB。盗賊妨害・steal価値は将来Dev/Robber下位Policy。Plan得点源ではない |
| [`strategic_action_surface.py`, `strategic_gate_step3a13.py`](../app/rl/strategic_action_surface.py) | 377合法mask→6 family可用性、即勝利除外、native prior＋residual、family内Executor | ルール/候補はB、具体ActionはC、今familyを選ぶ部分はD(L3)。現prior magnitudeはVictory Priorityの尺度にしない |

A=rule/hard constraint、B=決定的計算、C=具体Action execution、D=価値判断。AとBがDを先取りしないことが移行の核心である。特に`construction_before_titles`、`target_sites`、City deficit閾値、title horizon、paid-road/資源予約を「安全rule」として残さず、診断/legacy比較にとどめる。

## 5. 学習境界・Plan persistence・資産再利用

再計画候補イベントは本人/相手得点、建設、title holder/騎士数、Dev取得/使用、本人VP取得、港、到達性変化。**イベント発生=必ずPlan変更**ではない。候補/validを更新した上で、保持/再計画をPolicyで選ぶか、制約を最小にしたイベント駆動の再計画にするか3B-1/2のShadowで比較する。Plan age・前Plan・変更理由を記録し、短時間の振動率を測る。最終更新頻度やhysteresis係数は今回決めない。手製Plannerの「CITYを3ターン維持」のような固定契約は避ける。

学習時のActor sampleは、そのPolicyが**実際に選択した境界だけ**。L1なら実再計画時、L2を確率的Priorityとするなら実sample時、L3なら実委譲Investment時、Development Usageなら合法かつそのPolicyが担当した時。protected PlayerTrade/Robberや非委譲Planner Actionを、どのHeadのon-policy sampleにも偽装しない。下位Policyは保持Plan/Priorityを入力するが、未来の結果から作ったPlanを現在入力しない。Plan stateとvalid maskはartifact/rolloutに保存する。異なる時間尺度のL1/L3 valueには別Criticまたは条件付きvalue headが要る。既存 [`StrategicTransition`](../app/rl/strategic_gate_step3a13.py) の実委譲間macro reward `Σγ^j r_j`・`γ^duration` とrollout境界は再利用できるが、Planはさらに長いsemi-MDP区間を持ち得る。durationは既存learner policy step単位とし、terminal bootstrap=0、truncationをterminal扱いしない。

Reward候補の第一基準は現行環境のterminal win/loss ±1 と本人VP差+0.05（[`reward.py`](../app/rl/reward.py)）。L1/L2/L3/Dev Usageが同じ対局returnを別時間尺度で受ける設計をまず検討し、HeadごとのTeacher一致bonusやPlanner score rewardは設けない。Planが長期、即時建設は+0.05と時間尺度が異なるため、VP shaping偏り、title得失、相手turnによる変動、long macro duration、被保護Actionを跨ぐcredit、複数Headのjoint log-prob/ratio、Plan更新後の旧Plan creditを診断する必要がある。報酬係数や追加potential shapingはこのStepで確定・実装しない。

再利用する3A資産: v2/GNN encoderとv7 Context79、377合法mask、Strategic Surfaceと6-family候補mask、Frozen concrete Executor、ImmediateWinResolver、protected PlayerTrade境界、semi-MDP rollout/GAE、Strategic Critic/KL/family-collapse診断、held-out同seed比較、Champion artifactとのfreeze/parity。**再利用=現行6-family priorをPlan/Priorityの正解とする意味ではない**。既存6-family Gate/Stage0 Post1 artifactは保存し、新階層の実装で上書きしない。

新規module（候補名、まだ作成しない）: `victory_plan_context.py`（本人既知点/物理制約/候補とvalid）、`victory_plan_policy.py`（Plan選択と保持/再計画）、`strategic_priority_policy.py`（4独立score）、`investment_policy.py`（Action候補間比較）、`development_usage_policy.py`＋`development_target_policy.py`（使用タイミングとtarget）、後段の`player_trade_policy.py`、`hierarchical_controller.py`（phaseとsafety arbitration）、`hierarchical_rollout.py`（各Headの実委譲境界）、`hierarchical_artifact.py`（version・parent・Plan state永続化）。既存Frozen Executorはfamily選択価値を再判定しないことを条件に利用する。

## 6. 最小実装順序 / Step 3B-1

1. **3B-1**: Action非介入で本人得点・残点と4軸の物理feasibility/unknownを計算する`VictoryPlanContext`/candidate schemaを設計・実装。Planは候補でありTeacher targetでもAction maskでもない。現79 fieldからの重複、候補爆発、title保有時の二重計上、City駒返却、Plan候補ゼロを検査。固定seedのAction/GameState/RNG parityを取る。
2. **3B-2**: Plan/Priority headをzero-impact Shadow接続。Plan age・再計画イベント・churn・score整合・独立priorityの共起を観測。まだActionを変えない。
3. **3B-3**: phase別の即時建設baselineとDevelopment Usage競合を監査し、確定勝利以外を安易なhard safetyへ格上げしない。
4. **3B-4**: Plan/Priorityを条件とするInvestment PolicyをShadow。RoadのSettlement+Longest同時寄与とDev対資源予約の競合を評価。
5. **3B-5/6**: Dev Usage/target、次いでbelief/history付きPlayerTradeの別境界を設計。Robber詳細は引き続き別管理。
6. **3B-7以降**: 各境界のon-policy観測・parity・operating envelopeを確かめてから限定委譲RL。3A-23が未判定なので、既存10%の合格を新構造へ継承しない。

**Step 3B-1への提案**: まず学習しない決定論的なPlan候補生成と記録APIだけに限定する。具体的には `current_score / remaining_points / title_owned / settlement/city piece feasibility / board reachability / candidate tuple / valid mask / reason for unknown` を返し、候補点数の純増・Plan外建設後の再計算・VP取得後の残点更新を単体テストする。Planner `site_value/plan_value`、旧Intent、Champion Action、未来結果はAPI入力/出力に含めない。

検証: backend全テスト **352 passed、1,172 subtests passed**。依存ライブラリ由来のdeprecation warning 2件、失敗0。今回新たに保存した3A-23部分証跡223件は全件gzip/JSONとして読めることを確認したが、完走していないsetがあるためqualification計算には使用していない。
