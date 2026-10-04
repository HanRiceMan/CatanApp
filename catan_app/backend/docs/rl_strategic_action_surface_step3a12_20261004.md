# Step 3A-12: Strategic Action Decision Surface設計監査

## 結論

CITY_ONLYのBUILD_NOW/DEFER二択Gateは一旦終了し、既存コード・artifactを保存する。Step 3A-11の単発/複数future差をTeacher labelにはしない。Reward、DEFER Executor、Champion、Observation v7、377 Action Space、学習済み重みは変更していない。今回は**即時合法なStrategic Action familyの候補maskをAction前stateから生成するShadow診断**だけを追加した。

初期Gateは `BUILD_SETTLEMENT / BUILD_CITY / BUILD_ROAD / BUY_DEVELOPMENT / BANK_TRADE / END_TURN` の6 optionを推奨する。ROADは現時点では1 familyとし、構造的な拡張/最長路への寄与を別診断する。END_TURNはStrategic IntentでもHOLDというGoalでもなく、現時点で他の支出・交換を行わず手番を終える**実行option**である。候補間の機会費用を将来PPOへ移す一方、合法性・資源・盤面グラフ・候補生成・具体Action選択はアルゴリズム側に残す。

ただし、**この6-family Gateは現時点でChampionの全Actionを置換できない**。Action前stateではGate候補が複数ある1,081 Decisionのうち379件（35.1%）でChampionは保護対象PlayerTradeを選んだ。無理にTRADEを6-familyへ混ぜたり、Championの実Actionを正解教師にしたりしない。Step 3A-13のzero-impact / Shadow実装には進めるが、実委譲やPPO学習へ進む前に、protected PlayerTradeの優先権と各Frozen Executorのfamily-only契約を明示して検証する必要がある。

## 現行PlannerのDecision Tree

出典: [`settlement_planning_agent.py`](../app/rl/settlement_planning_agent.py) の `_select_planned_action()`（約703行）と `_select_action_with_diagnostics()`（約985行）。分岐は逐次的で、最終Actionから逆算した単一Goal比較ではない。

```text
setup専用処理 → base PPO Actionを一度選択
  ├─ initial road / robber move・steal: 専用Plannerが必要なら上書き
  ├─ pre-roll/action: 条件を満たす最大騎士力候補を先に選ぶ場合あり
  └─ action phaseのみ
      ├─ 9点以上の即時建設 / 最速勝利経路
      ├─ 3拠点時の都市 vs 接続済み開拓地の効率比較
      ├─ 早期都市化条件
      ├─ target sites到達後:
      │   ├─ 都市候補・接続済み開拓地予約
      │   ├─ 最長路候補 vs construction_before_titles / wait閾値
      │   ├─ first/second/third city deficit閾値
      │   └─ endgame reserveとbase PPO fallback
      └─ expansion planが有効なら:
          開拓地 → 豊作 → 銀行交換 → 開拓地向け道路
          → base PPOを許すか資源予約してEND_TURN
  ↓
paid road purpose guard → city resource reserve guard
  ↓
proactive PlayerTrade（Planner結果がEND_TURN等なら介入）
  ↓
adaptive development purchase / endgame development fallback
  ↓
最終Action
```

最大騎士力判定が建設前、Proactive PlayerTradeと発展購入が建設計画・guardの**後**にある点が重要。たとえば `assess_development_purchase()` は合法性だけでなく、停滞・盗賊・称号・建設遅延・期待値を使って「買うか」を決めている。これはFrozen BUY Executorへ丸ごと移せない。PlayerTradeは[`trade_strategy.py`](../app/rl/trade_strategy.py)で相手手札のベイズ推定、同一手番の提案履歴、比率・相手選択を使うため保護する。

| 現行判断 | 分類 | 初期Gateでの扱い |
|---|---|---|
| phase、legal 377 mask、コマ/山札/銀行在庫、費用、即時勝利・forced | A: Rule/hard constraint | アルゴリズムに残す |
| settlement到達可能性/不足/ETA、都市化候補、生産pip、称号gap、盗賊被害、港比率 | B: deterministic calculation | v7 Contextと候補maskに残す。Planner scoreは入れない |
| settlement/city vertex、道路edge、銀行交換のgive/get | C: Concrete Action execution | family選択後のFrozen Executor候補。ただし後述の抽出監査が必要 |
| 3拠点の都市 vs 開拓地、称号 vs 建設、道路を今伸ばす価値、都市資源予約、発展購入、銀行交換 vs 保持 | D: Value judgement | Gate/PPOへ段階移行。現行Plannerのif-ruleをhard safetyへ昇格しない |
| Proactive PlayerTradeの有無・相手・比率、trade response/counter | D+C、belief/history依存 | protected controller。初回Gateから除外 |
| robber配置・steal、発展カード使用・free road | phase固有のA/C/D | 初回Gate外。既存ロジックを維持 |

## 候補生成・maskとSurface predicate

[`strategic_action_surface.py`](../app/rl/strategic_action_surface.py) は `get_action_mask()` の即時合法Actionを、6 familyへ写像する。候補はそのfamilyの合法な具体Actionが1つ以上ある場合だけavailable。SETTLEMENT/CITYはコマ・資源・接続・距離制約を既存377 maskから継承する。ROADは合法な有料道路、BUY_DEVELOPMENTは現在購入合法かつ山札等のルールを満たす場合、BANK_TRADEは今可能なgive/getが1つ以上ある場合、END_TURNは今合法な場合。将来建てたいが今は建てられないという文脈はv7のshortage/ETAに置き、BUILD候補を偽にする。資源獲得の現在手段はBANK_TRADE、道路、END_TURN等の候補で表現する。

`candidate_mask[6]` は377 legal maskとは別。順序は上記6種で固定し、unavailableのGate logitはmaskする。`candidate_count >= 2` かつ非END_TURN候補があり、hard exclusionがないAction前stateだけをGate Surfaceとする。候補1つなら`forced_candidate`として記録しGate/Actor lossを発生させない。Championの選択Action、execution source、reason、Intent、Planner priorityは候補生成へ入れない。maskは決定的な制約であり、候補間の価値は示さない。

初期hard exclusionはsetup、robber、discard、trade response/counter等のaction以外のphase、free road、pending trade、合法Action1つ、即時10点建設、即時最長路/最大騎士力勝利。後者は合法道路・騎士から検出する。**硬い優先権の対象は勝てる具体Actionそのもの**であり、旧Plannerを通すだけで即時勝利が必ず選ばれるわけではない。今回のShadowでは勝利Actionを適用せず、全て現行Championを維持した。

## 36局Shadow coverage

Rule / Champion / Mixed各12局、各profile内Seat 1～4を各3局、seedは2920000～2920011 / 2921000～2921011 / 2922000～2922011。収集対象は指定1席のChampionのみ。全36局でShadow診断あり/なしのAction列・最終GameState・viewer views・winner/VP/rank・RNG counterが完全一致。観測データは[`summary.json`](../experiments/strategic_action_step3a12_20261004/summary.json)と[`decisions.jsonl`](../experiments/strategic_action_step3a12_20261004/decisions.jsonl)。これは勝率比較ではなく候補集合の設計監査である。

| 指標 | 件数 |
|---|---:|
| 全action-phase Decision | 1,529 |
| Gate predicate成立 | 1,081（70.7%） |
| Rule / Champion / MixedのGate Decision | 337 / 393 / 351 |
| Seat 1 / 2 / 3 / 4のGate Decision | 279 / 246 / 283 / 273 |
| 候補数1 / 2 / 3以上（全action phase） | 428 / 564 / 537 |
| Gate内の候補数2 / 3 / 4 / 5 / 6 | 563 / 365 / 132 / 19 / 2 |
| hard exclusion: forced / 即時建設勝利 / free road / 即時道路称号勝利 | 414 / 14 / 14 / 6 |

Gate局面における候補available数は次の通り。分母1,081で、END_TURN以外は重複する。

| BUILD_SETTLEMENT | BUILD_CITY | BUILD_ROAD | BUY_DEVELOPMENT | BANK_TRADE | END_TURN |
|---:|---:|---:|---:|---:|---:|
| 92 | 61 | 494 | 478 | 650 | 1,081 |

共起はCITY+SETTLEMENT 2、CITY+DEVELOPMENT 40、SETTLEMENT+ROAD 92、ROAD+DEVELOPMENT 171、CITY+BANK_TRADE 45、ROAD+BANK_TRADE 221。建設両方の比較は依然rareだが、ROAD/DEVELOPMENT/BANK_TRADE/END_TURNを含む複数候補局面は十分ある。単なるCITY Timing二択より広い価値比較面を観測できる。

Gate predicateが成立した局面のChampion実Actionは、BUILD_SETTLEMENT 92 / BUILD_CITY 45 / BUILD_ROAD 177 / BUY_DEVELOPMENT 82 / BANK_TRADE 131 / END_TURN 175 / protected PLAYER_TRADE 379。6-family候補集合で表現できたのは702/1,081（64.9%）。**表現できなかった379件は全てPlayerTrade**で、Rule 94/337、Champion 158/393、Mixed 127/351。6 familyに属するChampion実Actionは全件、対応する候補maskがTrueだった。Championのfamilyは比較診断でありTeacher targetではない。

hard exclusionのうち即時道路称号勝利6件では、現行Championが道路を建てたのは2件、PlayerTradeを選んだのは4件だった。これは既存Plannerを通すだけでは硬い勝利保証にならない具体例。今回はAction変更禁止なので修正していない。将来のhard safety実装では、勝利をもたらす合法Actionを独立に選ぶことを要する。

## ROADの意味とFrozen Executor

道路候補494局面で、Plannerの選んだtarget/scoreに依存しない構造解析を別記録した。3本以内で到達可能な距離ルール上空きvertexへ向かう最初の道路がある局面487、自己最長路を伸ばす道路がある局面490、**両方に該当する同一道路がある局面481**。Championが実際にBUILD_ROADした177件では、拡張edge 169、最長路長を伸ばすedge 126、両方118、どちらでもないedge 0。拡張のみ51、最長路長のみ8で、後者は全てTITLE_PLANNER由来だった。

したがって初回は案Aの**BUILD_ROADを1 family**とする。案Bの`ROAD_FOR_SETTLEMENT / ROAD_FOR_TITLE / OTHER_ROAD`は、同じedgeが2目的を満たすため排他的候補にできず、ROAD_TITLEというIntent教師へ戻る危険がある。最長路が1本伸びる事実は「今称号を狙う価値」の証明ではない。ROADの構造情報はmask/診断・将来の候補特徴として保持し、subfamily化は独立Executorと因果的目的境界が確認できるまで見送る。`OTHER_ROAD`を初期Gate optionにはしないが、合法であることと価値があることを混同して「その他道路を禁止する新rule」も今は作らない。

Frozen Executor候補と抽出時の注意:

| Gate family | 具体Actionへの変換候補 | family選択の価値判断を残さないための条件 |
|---|---|---|
| BUILD_SETTLEMENT | `_best_settlement()` の合法vertex内順位 | 目標拠点数・待機閾値で「建てない」を返さない |
| BUILD_CITY | `_best_city()` の合法vertex内順位 | `_city_action()`全体は購入判断/交換/待機を含むため転用しない |
| BUILD_ROAD | 合法edgeから凍結されたfamily内ranking | `_longest_road_progress()`は称号を追うかのgateを含み、そのまま呼べない。選択targetに依存する`_purposeful_road_candidates()`も切り出し監査が必要 |
| BUY_DEVELOPMENT | 合法なら `BuyDevelopmentAction()` | `assess_development_purchase()`のeligible/期待値判断をExecutorへ残さない |
| BANK_TRADE | 合法give/get候補の凍結ranking | `_helpful_trade_for_budget()`はPlanner選択Goalに依存。予算・相手・履歴を使う価値比較を隠さない |
| END_TURN | 合法なら `EndTurnAction()` | HOLD理由や資源予約if-ruleを再判断しない |

Settlement/Cityのvertex rankingは現行Championから比較的直接再利用できる。ROAD/BANK_TRADEは現行Plannerの「選ぶべきか」と「どれを実行するか」が強く絡み、**現時点でfamily-onlyのFrozen Executor完成とは言えない**。Step 3A-13で合法性・副作用・family外Action不返却・同一state決定性を検査する。Executorが無効なら候補をそのstateで実委譲不可にし、後付けでChampion実Actionへ置換しない。

## 保護対象と将来Gate/critic

PLAYER_TRADEの提案・相手・内容・倍率、再提案、response/counterはベイズ手札推定・resource events・同一手番の履歴に依存する。robber placement/stealとPLAY_KNIGHTを含む発展カード使用も初回Gate外。BUY_DEVELOPMENTだけを6 familyに入れる。これらprotected処理は固定controllerとして扱い、Gate Actor lossに入れない。ただし**protected PlayerTradeを優先するかどうか自体が現行Plannerの価値判断**なので、事前処理で恒久的に隠せば完全なPlanner移行にはならない。35.1%の重なりを評価指標として保存し、後段でbelief/history Contextを加えたtrade migrationを別に設計する。

次工程の実験Policy案は、凍結Champion GNN/Actor latent + v7 Runtime Context 79次元 + 6-bit candidate availability → 小さなStrategic Gate → masked family distribution。Champion logits/重み/377 residualは不変とし、最初は完全zero-impact Shadow。ActorとCriticは別encoderにし、Strategic専用value residualを凍結Champion base valueへ加える案を第一候補とするが、元Criticは旧controllerの分布上で学習済みなので正確なbootstrapとはみなさない。

実委譲時は、**実際にGateが選択したDecisionから次の実委譲Decisionまたはterminalまで**をsemi-MDP transitionとする。間のprotected Planner、opponent、trade response、forced actionは固定環境側。既存[`construction_timing.py`](../app/rl/construction_timing.py)のreward accumulation、`gamma ** learner_decision_duration`、terminal bootstrap=0、非委譲ActionをActor bufferへ入れない原則を再利用できる。ただし現実装は二択とCITY/SETTLEMENT_ONLYに型制約されるため、6-family action/maskとprotected preemptionに合わせた**新しい型**が必要で、そのまま流用しない。

## Step 3A-13への判定

**zero-impact / Shadow基盤へは進める。PPO学習・Planner override削除・実委譲はまだ不可。** Gate maskはstate-basedで安定し、36局のparityと1,081件の候補面が確認できた。一方、protected PlayerTrade 379件の優先権、ROAD/BANK_TRADEのfamily-only Executor、即時道路勝利の具体Action保証が未解決である。

`python -m unittest discover -s tests -q` は303件全件成功。保存された1,529 Decisionは `(profile, game_seed, revision)` が全件一意で、6-bit maskとcandidate_countの不整合は0件。小規模smoke 3局でもparityを確認した。実験データはShadow診断であり、実際のGate Actionを環境へ適用していない。

Step 3A-13では、(1) 6-family Gateのzero-impact初期化とmasked distribution、(2) 全familyのFrozen Executor候補をAction前stateから生成し合法/同family/非介入を検査、(3) PlayerTrade等のprotected arbitrationをAction family教師にせず記録、(4) Shadowあり/なしで全Action/GameState/RNG parity、(5) Executor成功率と実委譲可能Surface数をprofile/seat別に報告する。問題が残ればShadowで止める。学習、BC、Reward変更、価値を固定する新if-ruleは行わない。
