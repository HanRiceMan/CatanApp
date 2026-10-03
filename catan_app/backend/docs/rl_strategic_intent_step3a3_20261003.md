# Step 3A-3: Strategic Intent source-level診断と小規模coverage監査

## 結論

5 Intentのtri-state診断、Action選択分岐のtrace、Teacher Dataset v4を実装した。Rule / Champion / Mixed各8局（計24局、1,063 Decision）では、複数Intentの同時activeを記録でき、`reason_contributed_to_selection=true`なのにIntent stateがactiveでない矛盾は0件だった。全Actionを診断なしと比較した固定seed parityも一致した。

ただし**300局再収集・Masked Multi-label Head学習にはまだ進まない**。`SETTLEMENT`の確定negativeは0件、`CITY`は5件しかない。`ROAD_TITLE`はactiveが20件で、連続activeは平均1.82 Decision、最大4 Decisionにとどまる。現状のTeacher信号は、5 Intentを同等に学習するには不十分である。`KNIGHT_TITLE`の長い持続も「称号までの距離が閾値内」という状態信号であり、毎回称号を優先している証拠ではない。

## 実装範囲とDataset

- `app/rl/strategic_intent.py`: 5 Intentの独立した`active / inactive / unknown`、構造化reason/evidence、card別Execution、選択分岐のtrace。診断はGameStateのdeep copy上で行う。
- `app/rl/settlement_planning_agent.py`: Actionを選択した分岐で`execution_source`と、その分岐が使用したIntent/Reasonを記録。`select_action_with_intent_report()`を追加し、既存`select_action()`のAction選択は変更しない。
- `app/rl/macro_teacher_dataset.py`: schema `macro_teacher_v4_strategic_intent`、5要素の`strategic_intent_targets`と`strategic_intent_known_mask`、state/reason/evidence、`selected_execution`、`execution_source`、Intent別`reason_contributed_to_selection`をDecisionへ追加。旧Objective/Trade診断は比較用に残した。`game_id`、`game_seed`、`opponent_profile`、`seat`、`observation_index`を維持。
- `app/rl/collect_macro_teacher_dataset.py`: 新集計とparity結果を出力。
- `tests/test_rl_strategic_intent.py`: unknown mask、カード別Execution、BASE_PPO購入の誤帰属防止、Intent stateとAction寄与の区別、source寄与によるactive、全GameState比較を検証。

局データはローカルの `experiments/strategic_intent_step3a3_verified_20261003/` に保存した（Git管理外）。`definition.json`、`decisions.jsonl`、`games.jsonl`、`observations.npz`、`strategic_intent_summary.json`、`parity_summary.json`を含む。Observationは既存v2、形状は`[1063,1667]`。seedはRule `2330001–2330008`、Champion `2331001–2331008`、Mixed `2332001–2332008`で分離し、各profile内で収集対象席を1〜4へ均等に回した。

## source-level診断の定義

| Intent | activeの安全な根拠 | inactiveの安全な根拠 | それ以外 |
|---|---|---|---|
| SETTLEMENT | 到達範囲内の開拓計画、または実際に開拓計画分岐がActionを選択 | 開拓地5・都市4でコマ回収不能 | unknown |
| CITY | 即時都市候補、または建設交換目標がCITY | 都市4個使用済み | unknown |
| ROAD_TITLE | 現行称号分岐が道路候補を返す、または称号分岐がActionを選択 | 道路15本使用済み、または称号保持中でconfig上防衛しない | unknown |
| KNIGHT_TITLE | 称号分岐の騎士候補、またはadaptive assessmentの`largest_army_race` | 保持中で防衛しない、または山札枯渇かつ使用可能な騎士なし | unknown |
| ROBBER_RELIEF | 自分の生産阻害が4 pips以上 | 自分の生産阻害が0 pips | unknown |

Intent stateは、選択Actionへの寄与とは別に算出する。例えば`ROBBER_RELIEF=active`でも都市建設が選ばれたなら、盗賊解除がその都市建設を選ばせたとは記録しない。`BASE_PPO`が独自に発展カードを買った場合も、Planner assessmentが同時に騎士・盗賊理由を示しただけでは寄与をtrueにしない。`DEVELOPMENT`、`WIN_PUSH`、`VP_CARD`はIntentに入れず、9点購入へ`VP_CARD_WIN_CHANCE`も付けない。

`reason_contributed_to_selection=true`は「実際に選択したPlanner分岐の評価にその理由が含まれた」という意味であり、厳密な反実仮想（その理由だけを除けばActionが変わる）の証明ではない。`false`と`unknown`も区別して保存する。

## Coverageとprofile差

数値は`active / inactive / unknown`。known率は`(active+inactive)/Decision数`。positive/negative率はknown内で計算する。

| Intent | 全体（1,063） | known率 | known内positive / negative |
|---|---:|---:|---:|
| SETTLEMENT | 870 / 0 / 193 | 81.8% | 100.0% / 0.0% |
| CITY | 265 / 5 / 793 | 25.4% | 98.1% / 1.9% |
| ROAD_TITLE | 20 / 105 / 938 | 11.8% | 16.0% / 84.0% |
| KNIGHT_TITLE | 345 / 30 / 688 | 35.3% | 92.0% / 8.0% |
| ROBBER_RELIEF | 343 / 681 / 39 | 96.3% | 33.5% / 66.5% |

| profile（Decision数） | SETTLEMENT | CITY | ROAD_TITLE | KNIGHT_TITLE | ROBBER_RELIEF |
|---|---:|---:|---:|---:|---:|
| Rule（370） | 280/0/90 | 115/0/255 | 12/69/289 | 128/5/237 | 156/195/19 |
| Champion（362） | 284/0/78 | 89/0/273 | 7/36/319 | 61/25/276 | 90/270/2 |
| Mixed（331） | 306/0/25 | 61/5/265 | 1/0/330 | 156/0/175 | 97/216/18 |

特にMixedではROAD_TITLEのknownが1件のみで、profileごとのnegativeの有無が不安定である。この24局を母集団全体の比率とみなしてはいけない。

## Multi-Goal同時成立と持続性

1 Decision内のactive数は0個=0、1個=473（44.5%）、2個=426（40.1%）、3個以上=164（15.4%）。2個以上の同時activeは590/1,063（55.5%）で、排他的1クラス表現よりmulti-label表現が自然な例が多い。

主要共起件数は`SETTLEMENT+CITY=72`、`SETTLEMENT+ROAD_TITLE=0`、`CITY+ROAD_TITLE=20`、`KNIGHT_TITLE+ROBBER_RELIEF=131`、`SETTLEMENT+ROBBER_RELIEF=279`、`CITY+ROBBER_RELIEF=92`。`SETTLEMENT+ROAD_TITLE=0`は同時成立を禁止した結果ではなく、今回の候補・分岐・標本で未観測だっただけである。

| Intent | 連続run数 | 平均 / median / 最大Decision | 1 Decisionのみ | 2以上 | 5以上 |
|---|---:|---:|---:|---:|---:|
| SETTLEMENT | 38 | 22.89 / 27.5 / 49 | 2.6% | 97.4% | 78.9% |
| CITY | 43 | 6.16 / 2 / 24 | 41.9% | 58.1% | 34.9% |
| ROAD_TITLE | 11 | 1.82 / 1 / 4 | 54.5% | 45.5% | 0% |
| KNIGHT_TITLE | 19 | 18.16 / 21 / 40 | 5.3% | 94.7% | 78.9% |
| ROBBER_RELIEF | 64 | 5.36 / 4 / 18 | 15.6% | 84.4% | 48.4% |

ROAD_TITLEは現在もほぼAction-local。KNIGHT_TITLEは購入前から継続activeになり得るが、その大半は`largest_army_race`の距離閾値による。activeの意味を「称号を今優先する」と読み替えてはならない。

## Action選択経路と寄与

`execution_source`は`BASE_PPO=355`、`SETTLEMENT_PLANNER=332`、`TRADE_PLANNER=264`、`CITY_PLANNER=60`、`PLANNER_ADAPTIVE_DEVELOPMENT=35`、`TITLE_PLANNER=13`、`PLANNER_GUARD=4`。sourceの判明と5 Intentへの帰属は別物である。

- `BUY_DEVELOPMENT` 52件: `KNIGHT_TITLE`のみ16、`ROBBER_RELIEF`のみ13、両方6、`BASE_PPO`17。Planner評価理由を後付けしてBASE_PPO購入へ帰属していない。5 Intentへの寄与は35/52（67.3%）。
- `BUILD_ROAD` 147件: `SETTLEMENT`125、`ROAD_TITLE`13、両方0、その他9。5 Intentへの寄与は138/147（93.9%）。
- 5 Intentへの寄与率は都市建設39/40、開拓地建設63/65、銀行交換54/74、対人交渉163/264、手番終了111/421。対人交渉はPlanner経路を264/264で特定できても、最終Intentへ帰属できるのは163件である。
- activeでも今回のActionへ寄与しなかった件数は`SETTLEMENT=426`、`CITY=154`、`ROAD_TITLE=7`、`KNIGHT_TITLE=323`、`ROBBER_RELIEF=324`。反対の矛盾、すなわち寄与trueでstateがactive以外の件数は0。

`new_development_cards`により購入または騎士使用の可否へ影響し得た診断は9 Decision（購入側3、騎士使用側6）。これは該当gateの件数であり、Intentラベルが9件変わった、あるいは別Actionになったと証明するものではない。既存v2から「購入直後で今は使えない」区別ができない点は残る。

## Observation v2との情報差

- **直接存在**: 自分の資源・建設物・保有発展カード総数と種類、使用済みカード、盤面・盗賊位置、各人の公開称号と使用騎士数、発展山札残数、交易回数、ターン、銀行在庫など。盤面上の自分のblocked pipsも観測情報から計算可能。
- **決定的に再計算可能だがv2に専用特徴なし**: 到達可能な開拓地点、都市化による生産増、最長交易路の長さ、称号gap、必要資源と推定待ち時間。ただしPlanner config、候補順位付け、コスト計算をStudent側で再現する必要があり、v2の単純な線形分類で読めるとは限らない。
- **v2で欠落**: `new_development_cards`の購入当ターンflag/内訳。`get_observation()`の構造にはあるが、v2の1667次元encoderへは入らない。対人交渉選択に使うBayesianHandEstimatorの入力となる`resource_events`/交渉履歴・相手手札beliefもv2に入らない。Plannerの選択分岐そのもの、Planner configや一時候補評価もv2の直接入力ではない。

従って新Headへ進む際、Intent stateの多くは公開盤面から再計算可能でも、選択Actionの因果的attributionまでv2だけから再現するとは期待できない。今回はv2を変更していない。

## parity・テスト・次のGate

固定seedで診断なし/ありを比較し、Rule 414、Champion 468、Mixed 355、計1,237 Actionの全列が完全一致。winner、最終VP/rank、4人分の表示、ランダム採番ID/トークンだけ除いた**全GameState**、RNG counterも各profileで完全一致した。ChampionのPPO weight、Actor logits、377 Action Space、configには変更なし。

全backendテストは**258 passed、1,161 subtests passed**（非致命の既存deprecation warning 2件）。初回実行で無関係な浮動小数点厳密比較テストが許容差を約`1.9e-8`超えて一度失敗したが、単体再実行および最終全件実行では成功した。

次は大量収集より前に、`SETTLEMENT`/`CITY`でsource-levelに確定できるnegativeを増やせるか、ROAD_TITLEに道路建設以前から持続する安全なactive根拠があるかを調査する。その際も、証拠がないだけの状態をinactiveへ変換しない。これが解決しない限り、現在の5 Intentを同等のMasked BCE対象として学習しない。ROBBER_RELIEFのみはpositive/negative双方のcoverageが十分で、単独診断Headの候補になり得るが、今回は学習しない。
