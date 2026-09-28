# Strategic Intent taxonomy再整理

## 結論

次期taxonomyは、Strategic Intentを以下の5種類とする方向が妥当である。

- `SETTLEMENT`
- `CITY`
- `ROAD_TITLE`
- `KNIGHT_TITLE`
- `ROBBER_RELIEF`

`DEVELOPMENT`はStrategic Intentから削除する。`BUY_DEVELOPMENT`はExecutionであり、
`WIN_PUSH`、`VP_CARD`もStrategic Intentには追加しない。

教師形式は5 Intent独立の`active / inactive / unknown`とし、学習時は
Multi-label + Unknown Maskを使う。今回、enum、Dataset schema、学習処理は変更していない。

## 1. 現在のDEVELOPMENTラベルの実態

Step 2A-3の300局・13,898 Decisionでは、発展カード購入は671件あった。

| 現行Goal / 保存Reason | 件数 | 新taxonomyでの候補 |
|---|---:|---|
| DEVELOPMENT + ROBBER_RELEASE | 249 | `ROBBER_RELIEF` + `BUY_DEVELOPMENT` |
| DEVELOPMENT + STALLED | 1 | Intent unknown + `BUY_DEVELOPMENT` + `STALL_BREAK` |
| DEVELOPMENT_PURCHASEのみ | 125 | Intent unknown + `BUY_DEVELOPMENT` |
| KNIGHT_TITLEのみ | 226 | `KNIGHT_TITLE` + `BUY_DEVELOPMENT` |
| KNIGHT_TITLE + ROBBER_RELEASE | 70 | `KNIGHT_TITLE`と`ROBBER_RELIEF`の同時active + `BUY_DEVELOPMENT` |

したがって、現行`DEVELOPMENT` 375件のうち249件はROBBER_RELIEFへ再解釈する候補、
残る126件はStrategic Intentを確定できない。停滞は購入理由ではあるが、「何を実現するか」
ではないためIntentへ昇格させない。

ただし、これは保存Reasonによる候補分類であり、既存Datasetをそのまま確定ラベルへ変換しては
ならない。現在の`macro_goal._goal_for_action()`は、最終Actionが`BuyDevelopmentAction`なら、
Action選択後に再計算した`assess_development_purchase()`のReasonを付ける。そのReasonが実際に
Actionを選んだ分岐だったか、base PPOが独自に購入を選び、同時にReasonも成立していただけかを
区別できない。

新しいsource-level診断では、最低限次を区別する必要がある。

- `execution_source = PLANNER_ADAPTIVE_DEVELOPMENT`
- `execution_source = TITLE_PLANNER`
- `execution_source = BASE_PPO`
- `reason_contributed_to_selection = true / false / unknown`

## 2. KNIGHT_TITLE → BUY_DEVELOPMENT

### 安全に診断できる範囲

以下を同時に満たす場合は安全に診断できる。

1. `assess_development_purchase()`が`largest_army_race`を返す。
2. `eligible=True`である。
3. adaptive development分岐が実際に`BUY_DEVELOPMENT`を返した、または
   `_largest_army_progress()`が購入Actionを返した。

`largest_army_race`は、既知の自分の騎士数と相手の使用済み騎士数から、最大騎士力までのgapが
`title_horizon`以内かを判定している。カード種類を選べない不確実性は残るが、Strategic Intent
として「最大騎士力を狙う」はsource-levelで説明可能である。

### 現状の制約

Champion configでは`buy_for_largest_army=False`である。このため、現在のKNIGHT_TITLE購入の
多くは専用title plannerによる直接購入ではなく、adaptive developmentまたはbase PPOの購入と
同時に`largest_army_race`が成立したものになり得る。

既存296件すべてを因果的に「最大騎士力のために買った」と断定するのは安全でない。次回収集時に
Action選択分岐を保存すれば、安全なTeacher labelにできる。

## 3. ROBBER_RELIEF → BUY_DEVELOPMENT

### active signal

`development_strategy._blocked_production_pips()`は、盗賊タイルの出目pipsと自分の開拓地・都市を
使い、塞がれている生産量を計算する。現行Plannerは4 pips以上で`robber_relief`をReasonへ追加し、
騎士カードの価値を引き上げる。

したがって以下は安全なpositive signalになる。

- blocked production pipsが閾値以上。
- 盗賊が移動すれば自分の重要生産が回復する。

`BUY_DEVELOPMENT`との因果関係まで確定するには、さらに`eligible=True`かつadaptive development
分岐が購入を返したことを要求する。

### PLAY_KNIGHTとの関係

使用可能な騎士を持ち、重要タイルが塞がれていれば、

`ROBBER_RELIEF → PLAY_KNIGHT`

は意味論上自然である。ただし現行PlannerにはROBBER_RELIEF専用の騎士使用分岐がなく、実際の
`PLAY_KNIGHT`はbase PPO由来になり得る。Intent自体はsource-levelでactiveにできるが、そのIntentが
Actionを選んだと断定するにはexecution sourceの記録が必要である。

## 4. VP_CARD_WIN_CHANCE

現行Plannerから安全には診断できない。

`development_strategy.py`ではvictory pointカード価値を固定0.8とし、コメントでも「種類を選べず、
購入理由にはしない」と明示している。9点専用のVP期待Reasonは存在せず、
`endgame_development_fallback`も現行Champion configでは無効である。

既存Datasetには9点での発展購入が25件ある。

- 現行DEVELOPMENT: 11件
- 現行KNIGHT_TITLE: 14件

しかし、9点で買ったという事実だけからVPカード狙いとは判断できない。最大騎士力、盗賊解除、
base PPO判断などがあり得るため、これらへ`VP_CARD_WIN_CHANCE`を後付けしてはいけない。

将来このReasonを使う場合は、少なくとも次をsource-levelで明示的に計算する必要がある。

- 現在9点である。
- 発展カード山札と既知カードからVPカード残存確率を計算できる。
- 確実な開拓地・都市・称号ルートとの機会費用を比較した。
- その比較が購入判断へ実際に寄与した。

これはReasonであり、新しいStrategic Intentにはしない。

## 5. その他の発展購入

以下はStrategic Intentをunknownとする。

- `construction_stall`だけを根拠にした購入。
- base PPOが購入したが、Plannerに明確な目的がない購入。
- 一般的な将来価値、余剰資源処理、なんとなくの期待値による購入。
- 豊作、独占、街道建設のどれを引くか選べず、特定の最終目的を断定できない購入。

この場合も情報は捨てず、ExecutionとReasonを保存する。

```text
Strategic Intent = unknown
Execution        = BUY_DEVELOPMENT
Reason           = STALL_BREAK / GENERAL_EXPECTED_VALUE / ...
```

発展カード使用時も同じである。例えば街道建設カードを開拓計画へ使う根拠がsource-levelにある
場合だけ`SETTLEMENT → PLAY_ROAD_BUILDING`とし、用途不明ならIntentはunknownにする。

## 6. ROAD_TITLEをStrategic Intentへ残すか

残すべきである。

最長交易路の獲得・防衛は、独立した2点と相手への相対優位を実現する目的であり、
`BUILD_ROAD`とは分離できる。開拓地用道路と称号用道路が同じActionになることもあるため、
SETTLEMENTとの同時activeを許すMulti-label表現とも整合する。

ただし現行Plannerの`_longest_road_progress()`は、ほぼ「今置ける道路」を返す戦術候補生成である。
木・レンガを待つ、交換する、数手先の経路を予約するというpersistent ROAD_TITLE stateは保持して
いない。従って現時点の安全なactive範囲は狭く、Action実行直前に偏る。

taxonomyとしては妥当だが、Teacher側で持続Goalとして十分にラベルできるとはまだ言えない。

## 7. 新5 Intentのtri-state定義

### SETTLEMENT

active:

- Plannerが有効な開拓候補を選択し、現在の拠点数・終盤条件・待ち時間閾値で追跡対象にしている。
- その候補のための道路、交換、資源待ち、建設Actionが選択されている。

inactive:

- 開拓地5個かつ都市4個で、都市化による開拓地コマ回収も不可能。

unknown:

- 候補がない、または待ち時間閾値外だが、将来道路や盤面変化で候補が生まれ得る。

### CITY

active:

- 即時都市建設候補がある。
- `construction_trade_goal()`がCITYを返す。
- 都市用交換、資源待ち、終盤最速1点比較でCITYが選ばれている。

inactive:

- 都市4個で都市コマを使い切っている。

unknown:

- 都市化可能な開拓地はあるが、現在のPlannerが都市目標を選んだ根拠がない。

### ROAD_TITLE

active:

- 称号用gateを満たし、`_longest_road_progress()`が道路候補を返す。
- 即時獲得、即時勝利、または`defend_titles=True`で防衛道路が必要。

inactive:

- 道路15本を使用済み。
- 保持中かつ`defend_titles=False`で、現在のPlannerが防衛を明示的に行わない。

unknown:

- gapは現実的だが現在は道路を買えない。
- 将来の道筋はあり得るが、現行Plannerに持続的title planがない。

### KNIGHT_TITLE

active:

- `_largest_army_progress()`が騎士使用・購入を返す。
- またはdevelopment assessmentが`eligible=True`かつ`largest_army_race`を返す。

inactive:

- 保持中かつ`defend_titles=False`。
- 山札が空、使用可能騎士もなく、これ以上騎士数を増やせない。

unknown:

- 最大騎士力までのgapは近いが、購入・使用Actionとのsource-level接続がない。

### ROBBER_RELIEF

active:

- `_blocked_production_pips()`が重要生産閾値以上。
- 盗賊が自分の生産地点を明確に塞いでいる。

inactive:

- 自分の開拓地・都市の生産が盗賊によって全く塞がれていない。

unknown:

- 塞がれているが閾値未満で、現行Plannerが解除を目標として扱うほど重要か断定できない。

このIntentは騎士の有無や購入可能性とは独立してactiveになり得る。Intentは目的、実行可能性は
Execution候補・合法Action Mask側の問題である。

## 8. Multi-Goal構造

5 Intentは独立したdict/bit vectorとして維持できる。1 Decisionで複数activeを許す。

代表例:

- `SETTLEMENT + ROAD_TITLE → BUILD_ROAD`
- `KNIGHT_TITLE + ROBBER_RELIEF → BUY_DEVELOPMENT`
- `KNIGHT_TITLE + ROBBER_RELIEF → PLAY_KNIGHT`
- `SETTLEMENT + CITY`が同時activeで、Execution選択時に機会費用を比較する。

Reasonは各Intentの排他選択には使わない。例えば`LARGEST_ARMY_RACE`と
`ROBBER_BLOCKING_IMPORTANT_TILE`が同時に成立してよい。

## 9. Reason Code案

### 建設・盤面目的

- `SETTLEMENT_TARGET_AVAILABLE`
- `SETTLEMENT_RACE`
- `SETTLEMENT_RESOURCE_SHORTAGE`
- `CITY_PRODUCTION_GAIN`
- `CITY_RESOURCE_SHORTAGE`
- `ROAD_TITLE_OPPORTUNITY`
- `ROAD_TITLE_DEFENSE`
- `ROAD_TITLE_IMMEDIATE_WIN`
- `LARGEST_ARMY_RACE`
- `LARGEST_ARMY_DEFENSE`
- `LARGEST_ARMY_IMMEDIATE_WIN`
- `ROBBER_BLOCKING_IMPORTANT_TILE`

### 発展カード購入価値

- `STALL_BREAK`
- `VP_CARD_WIN_CHANCE`
- `EXPECTED_KNIGHT_VALUE`
- `EXPECTED_RESOURCE_GAIN`
- `ROAD_BUILDING_CARD_SYNERGY`
- `GENERAL_EXPECTED_VALUE`
- `NO_IMMEDIATE_CONSTRUCTION`
- `BUILD_DELAY_ACCEPTABLE`

### 資源・交渉

- `HAND_OVERFLOW`
- `RESOURCE_REBALANCE`
- `TRADE_COMPLETES_GOAL`
- `TRADE_IMPROVES_GOAL_ETA`

Reason Code自体はStrategic Intentではない。特に`STALL_BREAK`、`VP_CARD_WIN_CHANCE`、
`HAND_OVERFLOW`、`RESOURCE_REBALANCE`だけからGoalを作らない。

既存Reason Codeの移行案:

| 既存 | 新しい扱い |
|---|---|
| DEVELOPMENT_PURCHASE | Executionが表すため原則廃止または診断互換用のみ |
| DEVELOPMENT_STALLED | `STALL_BREAK` |
| DEVELOPMENT_ROBBER_RELEASE | `ROBBER_BLOCKING_IMPORTANT_TILE` |
| KNIGHT_TITLE_OPPORTUNITY | `LARGEST_ARMY_RACE` |
| KNIGHT_TITLE_DEFENSE | `LARGEST_ARMY_DEFENSE` |
| ROAD_TITLE_DEFENSE | 維持 |
| TRADE_HAND_OVERFLOW | `HAND_OVERFLOW` |
| TRADE_RESOURCE_REBALANCE | `RESOURCE_REBALANCE` |

## 10. Strategic Intent / Execution / Reason対応表

| Strategic Intent | 主なExecution | source-level条件 | 代表Reason |
|---|---|---|---|
| SETTLEMENT | BUILD_SETTLEMENT | target vertexへ合法建設 | SETTLEMENT_TARGET_AVAILABLE / SETTLEMENT_RACE |
| SETTLEMENT | BUILD_ROAD | active targetのfirst road | SETTLEMENT_TARGET_AVAILABLE |
| SETTLEMENT | TRADE_PLAYER / TRADE_BANK | trade source objectiveがsettlement | SETTLEMENT_RESOURCE_SHORTAGE / TRADE_COMPLETES_GOAL |
| SETTLEMENT | END_TURN | active planの資源待ち | SETTLEMENT_RESOURCE_SHORTAGE |
| SETTLEMENT | PLAY_ROAD_BUILDING / PLAY_YEAR_OF_PLENTY | 使用カードがactive targetの距離・不足を改善 | ROAD_BUILDING_CARD_SYNERGY / EXPECTED_RESOURCE_GAIN |
| CITY | BUILD_CITY | best/legal city candidate | CITY_PRODUCTION_GAIN |
| CITY | TRADE_PLAYER / TRADE_BANK | trade source objectiveがcity | CITY_RESOURCE_SHORTAGE / TRADE_COMPLETES_GOAL |
| CITY | END_TURN | source-level city goalの資源待ち | CITY_RESOURCE_SHORTAGE |
| CITY | PLAY_YEAR_OF_PLENTY | 都市不足資源を取得 | EXPECTED_RESOURCE_GAIN |
| ROAD_TITLE | BUILD_ROAD | longest roadを伸ばすtitle candidate | ROAD_TITLE_OPPORTUNITY / DEFENSE / IMMEDIATE_WIN |
| ROAD_TITLE | PLAY_ROAD_BUILDING | 称号gapを縮めることを検証可能 | ROAD_BUILDING_CARD_SYNERGY |
| KNIGHT_TITLE | BUY_DEVELOPMENT | eligible + largest_army_race | LARGEST_ARMY_RACE / EXPECTED_KNIGHT_VALUE |
| KNIGHT_TITLE | PLAY_KNIGHT | usable knight + title gap/defense | LARGEST_ARMY_RACE / DEFENSE / IMMEDIATE_WIN |
| ROBBER_RELIEF | BUY_DEVELOPMENT | eligible + blocked production閾値 | ROBBER_BLOCKING_IMPORTANT_TILE / EXPECTED_KNIGHT_VALUE |
| ROBBER_RELIEF | PLAY_KNIGHT | usable knight + blocked production閾値 | ROBBER_BLOCKING_IMPORTANT_TILE |
| unknown | BUY_DEVELOPMENT | 明確な最終目的なし | STALL_BREAK / GENERAL_EXPECTED_VALUE / VP_CARD_WIN_CHANCE |

Execution enumは、現在の汎用`PLAY_DEVELOPMENT`だけでなく、少なくとも将来は
`PLAY_KNIGHT / PLAY_ROAD_BUILDING / PLAY_YEAR_OF_PLENTY / PLAY_MONOPOLY`を区別した方が、
Intentとの対応を監査しやすい。ただし今回は実装していない。

## 次のStepへの推奨

まだ学習や大量再収集へ進まない。次はtaxonomy実装前のschema設計として、以下を確定する。

1. 5 Intentのtri-state report schema。
2. card-specific Execution enum。
3. Reason Code enumと旧Reason互換方針。
4. `execution_source`と`reason_contributed_to_selection`の保存形式。
5. 旧Datasetを確定Teacherとして使わず、参考・下限分析に限定する方針。

その後、Action parityを維持した診断専用実装、小規模再収集、label coverage監査の順に進む。
Multi-label Head学習は、新taxonomyのsource-level labelが安定した後に行う。
