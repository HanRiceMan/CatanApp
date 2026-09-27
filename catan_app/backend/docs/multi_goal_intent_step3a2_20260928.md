# Step 3A-2 Multi-Goal Intent意味論監査

## 結論

排他的Softmax 5クラスは採用しない。カタンでは複数の戦略目的が同時に成立し、
保存済み300局から安全に復元できる**下限値だけでも10.61%**のDecisionで2 Goal以上が
同時activeだった。

次の教師形式には **Multi-label + Unknown Mask** を推奨する。単純なMulti-label BCEは、
現在のPlannerで否定根拠を持たないGoalを大量の偽Falseへ変換するため採用しない。

ただし、現行PlannerのDEVELOPMENT / ROAD_TITLE / KNIGHT_TITLEは依然として
「長期的な資源待ち目標」より「現在実行可能な戦術機会」に近い。Multi-label化は
Goal競合を解消するが、rare Goalの持続性問題までは解消しない。学習前に、今回追加した
source-level診断をDecision時点で保存するDataset再収集が必要である。

## 実装範囲

- `app/rl/multi_goal_intent.py`
  - 5 Goalを独立した`active / inactive / unknown`で診断する。
  - GameStateのdeep copy上だけで診断し、Action選択へ接続しない。
  - 異種scoreの正規化や共通Intent strengthは作らない。
- `app/rl/audit_multi_goal_intents.py`
  - 既存300局・13,898 Decisionへ、安全に再構成できる信号だけを後付けする。
  - active率、unknown率、共起、active数、継続長を集計する。
- `tests/test_rl_multi_goal_intent.py`
  - 複数同時active、unknown維持、共起・継続集計、GameState/RNG非変更を確認する。

ChampionのAction、Plannerの選択順、PPO、Observation、377 Action Space、モデル・configは
変更していない。

## 1. Goalごとのsource-level active条件

### SETTLEMENT

active:

- `analyze_expansion_plan()`に選択対象がある。
- 開拓地コマがあり、対象の推定待ち時間が`max_wait_rounds`以内。
- 現行Plannerが実際に開拓計画を扱う拠点数・終盤条件に入る。

診断値としてtarget vertex、追加道路数、推定待ち時間を保持できる。

inactive:

- 開拓地5個かつ都市4個で、都市化による開拓地コマ回収も不可能な場合だけ確定する。

それ以外はunknownとする。有望候補が見つからないことを「開拓意思なし」とは扱わない。

### CITY

active:

- `_best_city()`で即時合法な都市建設がある。
- `construction_trade_goal()`が都市を現在の建設目標として返す。

inactive:

- 都市4個で都市コマを使い切っている場合。

都市候補が存在するだけでは、現在の都市Intentを一意に示さないためunknownを許す。

### DEVELOPMENT

active:

- `assess_development_purchase()`の`eligible=True`。
- その根拠として停滞、盗賊解除、最大騎士力競争を構造化して保持する。

inactive:

- 発展カード山札が空の場合。

購入価値があるが未購入可能な状態を、現行Plannerは持続目標として保持していない。
したがって、現時点ではそこをactiveに拡張せずunknownとする。

### ROAD_TITLE

active:

- 現行の称号gateを満たし、`_longest_road_progress()`が合法な道路Actionを返す。
- その道路が即時勝利なら別Reason Codeを付ける。

inactive:

- 道路15本を使い切った場合。
- 最長交易路保持中かつ`defend_titles=False`で、現行設定が防衛を明示的に行わない場合。

開拓地用道路はROAD_TITLEのactive根拠にしない。ただし、同じ局面で開拓計画と称号用の
別の有効道路候補がある場合、SETTLEMENTとROAD_TITLEは同時activeでよい。

### KNIGHT_TITLE

active:

- `_largest_army_progress()`が騎士使用または購入Actionを返す。
- または`assess_development_purchase()`がeligibleで、reasonに`largest_army_race`を含む。

inactive:

- 最大騎士力保持中かつ`defend_titles=False`の場合。

最大騎士力目的で発展カードを買う局面は、DEVELOPMENTとKNIGHT_TITLEの両方をactiveにする。

## 2. active / inactive / unknownの安全性

positive evidenceがある時だけactive、コマ上限・山札枯渇・明示的なPlanner configによる
rule-outがある時だけinactiveとした。それ以外はunknownである。

特にrare Goalで「そのActionを選ばなかった」ことをinactiveにはしていない。現在の
Teacherには、Goalを狙っていないことを広く証明するnegative source signalがないためである。

## 3. 既存300局への後付け可否

完全な後付けは不可能である。既存DatasetにはObservation、Action、単一Goal、
`available_goal_mask`、推定ターンはあるが、全Decisionにおける以下がない。

- `assess_development_purchase()`のeligible/reasons/build delay
- `_longest_road_progress()`が返した候補
- `_largest_army_progress()`が返した候補
- 各Goalがinactiveであるsource-level根拠

したがって今回の数値は、保存済み情報だけで確定できるactiveの**下限値**である。
Action結果からGoalを推測してunknownを埋めることはしていない。

## 4. Goal別tri-state分布（保守的後付け）

| Goal | active | inactive | unknown |
|---|---:|---:|---:|
| SETTLEMENT | 10,628 (76.47%) | 0 (0.00%) | 3,270 (23.53%) |
| CITY | 3,825 (27.52%) | 162 (1.17%) | 9,911 (71.31%) |
| DEVELOPMENT | 671 (4.83%) | 13 (0.09%) | 13,214 (95.08%) |
| ROAD_TITLE | 282 (2.03%) | 0 (0.00%) | 13,616 (97.97%) |
| KNIGHT_TITLE | 296 (2.13%) | 0 (0.00%) | 13,602 (97.87%) |

SETTLEMENT/CITYは継続計画を部分的に復元できるが、rare Goalはほぼaction-local情報しか
後付けできない。これはrare classが少ないというより、古いschemaに持続診断がないことを示す。

## 5. Goal co-occurrence

件数（対角は各Goalのactive総数）:

|  | SETTLEMENT | CITY | DEVELOPMENT | ROAD_TITLE | KNIGHT_TITLE |
|---|---:|---:|---:|---:|---:|
| SETTLEMENT | 10,628 | 570 | 492 | 68 | 201 |
| CITY | 570 | 3,825 | 181 | 231 | 95 |
| DEVELOPMENT | 492 | 181 | 671 | 0 | 296 |
| ROAD_TITLE | 68 | 231 | 0 | 282 | 0 |
| KNIGHT_TITLE | 201 | 95 | 296 | 0 | 296 |

指定された主な組み合わせ:

- SETTLEMENT + ROAD_TITLE: 68件、全Decisionの0.49%
- SETTLEMENT + CITY: 570件、4.10%
- CITY + DEVELOPMENT: 181件、1.30%
- DEVELOPMENT + KNIGHT_TITLE: 296件、2.13%
- ROAD_TITLE + KNIGHT_TITLE: 0件

KNIGHT_TITLE 296件はすべてDEVELOPMENTもactiveであり、単一ラベルでは必ず片方の意味を
失う。これはMulti-label化の最も明確な根拠である。

## 6. 1 Decisionあたりactive Goal数

| active Goal数 | 件数 | 割合 |
|---|---:|---:|
| 0 | 0 | 0.00% |
| 1 | 12,423 | 89.39% |
| 2 | 1,147 | 8.25% |
| 3以上 | 328 | 2.36% |

2個以上は合計1,475件、10.61%。これは保守的な下限であり、rare Goalの事前診断を保存して
再収集すると増える可能性がある。

## 7. Goal persistence

| Goal | mean | median | max | 1 Decisionのみ | 2以上 | 5以上 |
|---|---:|---:|---:|---:|---:|---:|
| SETTLEMENT | 21.82 | 23 | 72 | 16.84% | 83.16% | 71.66% |
| CITY | 10.12 | 7.5 | 49 | 33.07% | 66.93% | 57.94% |
| DEVELOPMENT | 1.03 | 1 | 2 | 97.24% | 2.76% | 0.00% |
| ROAD_TITLE | 1.21 | 1 | 3 | 80.34% | 19.66% | 0.00% |
| KNIGHT_TITLE | 1.02 | 1 | 2 | 98.28% | 1.72% | 0.00% |

SETTLEMENTとCITYはpersistent、rare 3 Goalは現行保存情報ではほぼaction-localである。

## 8. DEVELOPMENT / KNIGHT_TITLEは購入前からactiveになるか

source-level診断はAction選択後のAction型から逆算せず、購入評価を行う同じ事前状態で
`assess_development_purchase()`を呼ぶ。そのため購入Actionを返す前の状態でactiveを得られる。

ただし、現行Plannerのpositive predicateは`eligible`であり、affordable条件を含む。
「小麦・羊・鉱石をまだ集めている」という数Decision前の持続的DEVELOPMENT intentは存在しない。
KNIGHT_TITLEも`largest_army_race`か即時騎士使用の段階で初めてactiveになる。

したがって「購入Actionより前に診断」は可能だが、「資源待ち期間まで遡るpersistent goal」は
現在のsource-levelロジックからは安全に生成できない。

## 9. ROAD_TITLEは道路建設前からactiveになるか

Action適用前の状態で`_longest_road_progress()`を評価できるため、道路建設前の診断ではある。
しかし同関数は合法かつ現在購入可能な道路候補を要求する。称号を目指して木・レンガを待つ
数Decision前のpersistent intentは現行Plannerにない。

よってROAD_TITLEも「実行直前の戦術機会」であり、長期目標としては未実装である。

## 10. Observation v2からの再現可能性

| Goal | 判定 | 理由 |
|---|---|---|
| SETTLEMENT | v2から決定的に再計算可能 | 盤面、道路、資源、港、生産pips、bank、free roadを保持。Planner閾値は固定configとして与えられる。 |
| CITY | v2から決定的に再計算可能 | 自分の開拓地位置、都市数、資源、生産pipsを保持。 |
| DEVELOPMENT | v2では一部情報不足 | ほぼ再計算できるが、`new_development_cards`の内訳/有無をvectorへ別途encodeしていない。 |
| ROAD_TITLE | v2から決定的に再計算可能 | 全道路・建物・得点・称号保持者・資源を保持。 |
| KNIGHT_TITLE | v2では情報不足 | 騎士総数と使用済みはあるが、新規購入騎士と使用可能騎士を区別する`new_development_cards`がない。 |

Bayesian belief、trade history、activity logは、今回定義した5 Goalのpositive signalには必須でない。
ただし交渉のsource objectiveをStudentへ予測させる別問題では不足要因になり得る。

## 11. 推奨学習形式

**Multi-label + Unknown Mask**を推奨する。

- 出力は5独立logit + Sigmoid。
- active=1、確実なinactive=0。
- unknownはlossから除外する。
- Goalごとにmask後のpositive/negative件数を記録する。
- class weightだけでunknownを負例化してはいけない。

現状のrare Goalはunknown率が95〜98%であるため、Unknown MaskなしBCEはほぼ全状態を
誤ったnegative Teacherとして学習させる。

## 12. Dataset再収集と次Step

新しいDataset再収集は必要である。既存300局はtaxonomy検証と下限分析には使えるが、
Multi-label学習用の完全Teacher Datasetにはできない。

次は学習ではなく、Step 3A-3として以下を行うのが安全である。

1. `MultiGoalIntentReport`をTeacher収集recordへoptional fieldとして保存する。
2. Rule / Champion / Mixed各5局でAction・RNG・GameState parityを確認する。
3. 小規模収集でGoal別known率、positive/negative比、共起、persistenceを再評価する。
4. DEVELOPMENT / KNIGHT_TITLE用に`new_development_cards`情報差をどう扱うか決める。
5. その後に同profile・seat rotationでDatasetを再収集する。
6. 十分なknown positive/negativeが得られてからMasked BCEの学習へ進む。

現時点では学習へ進まない。さらに、DEVELOPMENT / ROAD_TITLE / KNIGHT_TITLEを将来も
持続的なIntentへ拡張できない場合は、SETTLEMENT/CITYと同列の長期Goalではなく、
「persistent construction intent」と「tactical opportunity」の2 Headへ分ける案を再検討する。
