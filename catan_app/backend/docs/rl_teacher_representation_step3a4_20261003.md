# Step 3A-4: Teacher表現の設計監査

## 結論

ChampionのActionを5つのStrategic Intentのどれかへ強制分類しない。Teacherを次の三層と、その出所情報に分ける。

1. **Strategic state**: `SETTLEMENT / CITY / ROAD_TITLE / KNIGHT_TITLE / ROBBER_RELIEF`が局面で成立するか。各軸は独立し、`active / inactive / unknown`、根拠、数値文脈を持つ。
2. **Tactical / resource-risk context**: 資源不足、手札過多、即時建設不可、停滞、将来の交換柔軟性などの局面条件。複数同時に成立する。条件が成立することと、それが選択の理由になったことは別。
3. **Execution and attribution**: 実際の377 Action、診断用Action family/card種類、選択経路、実際に選択分岐が使用したIntent/Reason。証拠のないBASE_PPO行動の理由は`unknown`とする。

`Strategic Intent attribution=[]`かつ`TRADE_PLAYER / BUY_DEVELOPMENT / END_TURN`などは正常な教師レコードである。ただし「確かにIntent寄与がない」と「BASE_PPO内の理由が観測できない」は区別する。5 Intentを同質の5本のMasked BCEへ直行せず、決定的計算を補助文脈に、異種の学習可能な値判断をPPOへ渡す構成を推奨する。

今回の監査ではコード、Observation、モデル、Dataset schemaを変更せず、新規収集・学習もしていない。

## コード・既存データから確認した境界

- `expansion_planner.analyze_expansion_plan()`は開拓候補、必要道路、資源予算、概算待ちラウンド、`site_value`/`plan_value`を計算する。`plan_value`は生産・資源補完・港・道路コスト・待ち時間を手製重みで合成した値で、他のGoalの数値と比較可能な効用ではない。
- 都市候補の`_best_city()`は**現在合法かつ購入可能**な都市Actionを返す。将来の全都市化候補を返すAPIではない。`_city_upgrade_value()`は都市化による追加生産の最大値を計算するが、現時点の診断では待機中の都市目標vertexを常に特定できない。
- `_longest_road_progress()`は称号距離だけでなく、保持・防衛設定、現得点、道路の合法性・購入可能性、道路網の延長、相手建物・競合道路、PPO確率を使う。現行ROAD_TITLE activeが道路購入時に偏る理由であり、「称号gapが近い」だけで持続Intentと断定できない。
- 最大騎士力のgapは**二種類**ある。`assess_development_purchase()`は使用済み騎士と手札中の既知騎士を足した「追加で引きたい騎士数」を使う。一方`_largest_army_progress()`は「追加で使う必要がある騎士数」を基準にする。単一の`knight_gap`へ混ぜない。
- `assess_development_purchase()`は建設停滞、盗賊による生産阻害、最大騎士力競争、購入後の建設遅延、発展カード種類の期待値を組み合わせる。ただしBASE_PPOが独自にBUYを選んだ時、その内部の理由は取得できない。VPカードの固定価値は明示的なVP狙いではない。
- `trade_strategy._choose_proactive_trade()`は即時建設不足と余剰資源の将来向け交換を区別する。後者は`RESOURCE_REBALANCE`/`FUTURE_VALUE`、手札8枚以上なら`HAND_OVERFLOW`を記録し得る。相手選定にはBayesian手札推定、同一手番の提案履歴を使う。
- Step 3A-3の24局・1,063 Decisionでは`END_TURN=421`、`TRADE_PLAYER=264`、`BUY_DEVELOPMENT=52`、`BUILD_ROAD=147`。BUYのうち17件はBASE_PPO由来で、source-level購入理由は不明。PlayerTradeのうち101件は確定Intent attributionを持たず、そのうち77件に`HAND_OVERFLOW`を含む将来向け交換診断がある。これは「将来向け理由が実際に選択経路で使われた」例だが、各理由の反実仮想上の必要性までは証明しない。
- 現在のTeacher Datasetは`is_macro_decision=True`の**actionフェーズ**のみを保存する。`PLAY_KNIGHT`は24局Datasetで0件であり、`turn_pre_roll`を含む騎士使用の頻度・理由をこのDatasetから評価してはならない。将来の収集対象フェーズを設計し直す必要がある。

同Datasetの`active / inactive / unknown`は、SETTLEMENT `870/0/193`、CITY `265/5/793`、ROAD_TITLE `20/105/938`、KNIGHT_TITLE `345/30/688`、ROBBER_RELIEF `343/681/39`。前二者は建設計画、道路称号は主にAction-localな機会、騎士称号は競争距離、盗賊解除は盤面圧力を表し、同じbinaryラベルに見えても意味・coverageが異なる。

## Intent / Reason / Executionの対応

| 事象 | Strategic stateとAction attribution | Tactical / Resource Reason | Execution | 証拠の扱い |
|---|---|---|---|---|
| 開拓候補へ伸ばす | SETTLEMENT active、開拓分岐なら寄与true | `SETTLEMENT_TARGET_AVAILABLE`等 | BUILD_ROAD | 目標と選択分岐を別保存 |
| 都市用に交換 | CITY active、trade sourceがCITYなら寄与true | `CITY_RESOURCE_SHORTAGE`, `TRADE_COMPLETES_GOAL` | TRADE_PLAYER/BANK | source objectiveが明確な場合のみ |
| 将来用の余剰交換 | 他のactive Intentは維持、今回の寄与は空でもよい | `RESOURCE_REBALANCE`, `FUTURE_VALUE` | TRADE_PLAYER | `trade_diagnostic.source_objective=None`を尊重 |
| 手札過多の交換 | 同上。CITY等への同時寄与も可 | `HAND_OVERFLOW`。`DISCARD_RISK_REDUCTION`は将来、選択経路で確認できた場合だけ | TRADE_PLAYER | 単に8枚以上という事実と選択寄与を分離 |
| 専用Plannerの発展購入 | KNIGHT_TITLE/ROBBER_RELIEFはsource分岐で確認したもののみ寄与true | `LARGEST_ARMY_RACE`, `ROBBER_BLOCKING_IMPORTANT_TILE`, `STALL_BREAK`等 | BUY_DEVELOPMENT | DEVELOPMENT Goalは作らない |
| BASE_PPOの発展購入 | stateは診断可、購入へのIntent寄与はunknown | 購入理由はunknown。`GENERAL_EXPECTED_VALUE`を後付けしない | BUY_DEVELOPMENT | PPO logitsは説明・反実仮想ではない |
| タイトルPlannerの騎士使用 | KNIGHT_TITLEが分岐を選んだ場合に寄与true | `LARGEST_ARMY_RACE`/防衛 | PLAY_KNIGHT | turn_pre_rollも対象にする必要 |
| BASE_PPOの騎士使用 | stateは診断可、選択寄与はunknown | `GENERAL_KNIGHT_VALUE`を事後推測で付けない | PLAY_KNIGHT | 結果的な称号前進 ≠ 選択目的 |
| 建設不能時の終了 | active Intentは維持。Action寄与が無ければ空またはunknown | `NO_IMMEDIATE_CONSTRUCTION`, `INSUFFICIENT_RESOURCES`は事実。待機理由はsource確認時だけ | END_TURN | HOLD Goalを復活させない |

`NO_IMMEDIATE_CONSTRUCTION`や`HAND_OVERFLOW`はまず**tactical fact**として計算できる。これを`selection_reason`へ昇格させるには、実際のAction選択分岐が使った根拠が必要。`GENERAL_EXPECTED_VALUE`、`GENERAL_KNIGHT_VALUE`、`DISCARD_RISK_REDUCTION`は今のBASE_PPO選択から安全に診断できないため、提案コードであり既存局の確定ラベルではない。

## 5戦略軸のstructured signal候補

下表の「D」はルール・盤面・本人情報からの決定的特徴、「A」は安全に教師化できる補助target候補、「V」はPlannerの手製価値を正解に固定せずPPOで学ばせる比較判断を示す。数値はGoal間で共通score化しない。欠損・到達不能には個別のvalid maskを付け、`99`ラウンド等のsentinelを通常の回帰値として学習させない。

| 軸 | 候補signal、単位・範囲 | 役割 | v2との関係・注意 |
|---|---|---|---|
| SETTLEMENT | 候補有無bool、最良vertex ID（盤面vertex集合）、候補数0〜盤面vertex数、必要道路0〜残り道路本数、必要資源不足5次元の枚数、即時建設bool、概算ETA 0〜12ラウンド＋到達不能mask | D。候補有無/道路数/ETAは必要ならmask付きA | 盤面・手札・港から再計算可。ただし最良vertexはPlannerの順位付けを含み、候補集合と分ける |
| SETTLEMENT | `site_value`/`plan_value`（Planner固有の任意単位）、目標競争や道路先の相手干渉 | 診断値、V | plan_valueの回帰はPlanner模倣であり最終価値ではない。競争の相手手札部分はv2不足 |
| CITY | 都市化可能な自分の開拓地数0〜5、最良候補vertex、追加生産0〜15 pips程度、資源不足5次元、即時都市化bool、概算ETA 0〜12＋mask、都市残コマ0〜4 | 盤面・資源・コマはD、候補/ETAは必要ならmask付きA | `_best_city()`は即時合法Actionに限る。待機中の候補vertexを得る診断は未実装 |
| CITY | 都市化による資源補完価値、開拓地との相対優先 | 診断値、V | `_city_upgrade_value()`は手製重み。Goal間共通scoreにはしない |
| ROAD_TITLE | 現道路長0〜15本、保持者4席＋なし、獲得/防衛に要る追加本数0〜15、構造的に伸ばせるedge数、今購入可能な称号向けedge数、最良edge、即時獲得/勝利bool | 長さ/gap/合法候補はD、機会存在と分類はmask付きA | 道路グラフから再計算可。構造的可能と今の手札で可能を別にする |
| ROAD_TITLE | 将来の到達可能性、相手への妨害、道路コストを払って今追う価値 | DとVを分離 | `ROAD_TITLE` activeは現状action-local。単なるgapで持続Goalを作らない |
| KNIGHT_TITLE | 使用済み騎士0〜14枚、手札の使用可能騎士0〜14、保持者、`plays_needed`と`draws_needed`を別記、山札残0〜25、購入可bool、今使用可bool、即時獲得/勝利bool | 枚数・合法性はD、機会分類はmask付きA | v2は自分のカード種類を持つが`new_development_cards`内訳なし。今使用可はv2単独では不完全 |
| KNIGHT_TITLE | 騎士を引く期待、称号争いを今追う価値 | 期待値計算はD、優先価値はV | 山札の実際の並びや他人の隠しカードはTeacherにも使わせない |
| ROBBER_RELIEF | 阻害生産0〜30程度の重み付きpips、阻害tile、資源別阻害5次元、阻害/全生産比0〜1＋分母0mask、騎士使用可bool、発展購入可bool | pips・比・合法性はD、閾値はAの候補 | 盤面から阻害量は再計算可。騎士使用可は新規カードflagがv2に欠落 |
| ROBBER_RELIEF | 今盗賊を退かすことの優先度 | V | blocked pips高値と即時Action価値は同義ではない |

数値範囲は保存時にそのPlanner実装の`definition/version`で明記する。特に概算ETAは確率的将来の真値ではなく、現在の生産・銀行交換を使う決定的**推定値**。Auxiliary回帰にするとしてもmaskとcensoringを設ける。`target_vertex/edge`は単純なordinal回帰ではなく、候補mask付きの盤面上categorical pointerとして扱う。

## Resource / Risk context

| context | 単位・範囲 | 推奨 | v2 |
|---|---|---|---|
| 手札枚数 | 5資源合計、0〜銀行総量95枚 | rawから決定計算。重複入力の価値は測定してから | 5資源内訳が直接存在 |
| 7での廃棄曝露 | `hand_size>7`、7が出た時の廃棄枚数`floor(hand_size/2)` | D。行動価値としてのリスク許容はV | 手札から再計算可 |
| 資源多様性・資源別生産 | 5資源の有無/生産pips | D。資源の戦略価値はV | 盤面から再計算可 |
| 各建設の不足量・今の購入可 | 5資源×建設種、bool | D/合法mask | v2の手札・盤面・港から概ね再計算可。Action maskとの一致を別途確認 |
| 今実行可能なAction数 | 377 maskのtrue数、Action family別件数 | D。raw mask側で算出し、重複追加は検証後 | v2だけでは新規発展カード等で厳密再計算不能。既存合法maskを利用 |
| 港・銀行交換選択肢 | 資源別交換比2/3/4、銀行在庫、受取可能資源 | D。どの交換が得かはV | 港・銀行・手札は存在 |
| 交渉柔軟性 | 提案可能資源と候補相手の推定保有確率 | ルール/ベイズはD、提案価値・倍率・相手はV | 相手beliefと今手番の提案履歴はv2にない |
| 発展購入可・建設遅延 | bool、概算ラウンド差＋mask | D。購入すべきかはV | 購入直後flagが欠落。ETAは再計算が必要 |
| 停滞 | 最短建設ETA、即時建設不可、前後の停滞期間 | 現局面の条件はD、期間は履歴が必要。停滞打開Actionの価値はV | v2は現盤面のみ。過去の継続期間は不足 |

カタンの「7で手札8枚以上なら半分捨てる」という規則を表す特徴と、「この交換による廃棄リスク減が他の行動より価値が高い」という判断を分ける。`DISCARD_RISK_REDUCTION`を全ての8枚超交渉へ一律の因果ラベルとして付けない。

## Observation v2と可観測性

v2（1667次元）には自分の資源・未使用カードの種類、公開盤面・道路・港・称号・使用騎士数・銀行・発展山札枚数・交易回数がある。盤面到達性、称号gap、生産、資源不足などはこの情報から決定的に再計算できる。逆に、`get_observation()`が自分の`new_development_cards`を保持していても、v2 encoderは出力していない。`resource_events`、相手手札のBayesian belief、現在手番に何を誰に何倍で提案したかの履歴もv2へ入らない。既存v4以降にbelief context、v6にdevelopment contextがあるが、今回v2を置換する提案ではない。

不足が及ぶ先を分ける。

- **Strategic state予測**: 開拓/都市候補、道路称号gap、盗賊阻害の多くはv2または決定的計算で得られる。最大騎士力の「今使用可」には新規カードflagが必要。相手の資源を使う戦略的競争の精度にはbeliefが効く可能性がある。
- **具体的Execution選択**: 対人交渉の相手/倍率/再提案、今騎士を使う/購入する判断にはbelief、提案履歴、`new_development_cards`が直接効く。これらを教師だけが参照したままv2から同じActionを模倣させると観測不足になる。
- **Planner分岐・Action attribution**: `execution_source`と実際に使ったReasonはTeacher provenanceであり、v2から決定的に読める局面事実ではない。Student入力へそのまま入れるとPlanner移行を装った情報漏洩になる。

## 推奨する次Dataset schema（未実装）

現行v4の`strategic_intent_targets/known_mask/states/evidence`、`selected_execution`、`execution_source`、`reason_contributed_to_selection`、旧診断fieldは保持し、後方互換の新versionで以下を追加する。

```text
decision_id / game_id / game_seed / seat / phase / observation_index
observation_version / deterministic_context_version / planner_config_hash

strategic_state[5]: state, known_mask, evidence_source, structured_signals,
                    signal_valid_mask, unit, score_semantics
tactical_facts: hand_size, discard_exposure, immediate_build_mask,
                resource_shortages, stall_eta, trade_options, ...
action_attribution: selected_action_id, selected_execution, execution_source,
                    intent_contributions[5] (true/false/unknown),
                    source_confirmed_reasons[], attribution_status
trade_context: target_player, ratio, objective_if_confirmed,
               future_value/overflow facts and source-confirmed reasons
outcome: winner, final_vp, final_rank
```

`tactical_facts`の成立と`source_confirmed_reasons`の因果主張を同じbitにしない。`attribution_status`は少なくとも`SOURCE_CONFIRMED / SOURCE_EXCLUDES / UNOBSERVED_BASE_PPO`を分ける。空配列に曖昧な二つの意味を押し込まない。選択前stateからのみ特徴を作り、相手の秘匿手札・発展山札順・対局結果を入力へ漏らさない。splitはDecision単位でなくgame/seed単位とする。

## PlannerとPPOの役割、およびPolicy案

ルール、合法377 Action mask、盤面グラフ、資源・銀行在庫、タイトルgap、Bayesian belief、決定的到達可能性はアルゴリズム側へ残す。候補の戦略価値、開拓と都市の機会費用、称号を今追うか、盗賊解除、手札保持、目的のない発展購入、Future Value Trade、廃棄リスクをどこまで許容するかはPPOへ移す。

候補構造は既存GNN/PPO backboneと377 Action logitsを維持し、共有表現へ**薄い異種補助Head**を付けるもの。最初は十分な教師が取れる決定的指標（例えばblocked pips、建設不足量、称号gap）の再現確認と、必要なら個別mask付き補助lossに限る。SETTLEMENT/CITYの現在のnegative欠如やROAD_TITLEのaction-local性を、5本同質BCEで覆い隠さない。補助Headの出力を即Actionへ接続せず、最終Action優先順位はmask付きPPO actorで学ぶ。Teacherの`plan_value`等の手製scoreを全Goal共通優先度として固定しない。

300局再収集前の最小実装候補は、(1) Action選択前のstructured signalとそのvalid maskを再計算可能にする診断API、(2) tactical factとsource-confirmed reasonを分離したtrace、(3) actionフェーズ以外の騎士・発展カード使用も含む収集範囲、(4) v2では欠落する`new_development_cards`と交渉履歴・beliefについての可観測性テスト。これらを設計・parity検証してから、小規模再収集でcoverage、分布、欠損、ラベル矛盾を確認する。まだ300局・学習には進まない。
