# Step 3B-1 — Victory Plan Context / Candidate Schema

> 履歴記録: 本レポートは当時のexact-gain候補仕様を記録する。現行の候補意味論は [Step 3B-1A](rl_victory_plan_minimal_winning_step3b1a_20261012.md) の「10点以上へ到達するための最小得点構成」（schema v2）に更新済み。以下の分布・例はv1の結果として参照すること。

2026-10-12。L1 Victory Planの**決定論的な状態と候補列挙のみ**を実装した。Plan選択、PPO head/update、Priority、Investment、Observation、Action、Controller、Reward、Champion artifactには触れていない。実験記録は本レポートと同梱のShadow datasetに保存する。

## 1. API・責務

新規 [`victory_plan_context.py`](../app/rl/victory_plan_context.py) の `build_victory_plan_context(game, player_id)` は、現在の本人既知状態と公開盤面だけから `VictoryPlanContext` を返す。全schemaは `@dataclass(frozen=True)`、候補集合はtupleで、versionは `victory_plan_context_v1`。PlannerのGoal/Intent/Reason、選択Action、site/plan value、相手の秘匿手札・VP、未来ダイス、山札の中身・次カードは参照しない。`development_deck` は公開の**残枚数のみ**を使う。`Runtime Context79` はv7 Observation用の正規化された既存値としてそのまま維持し、今回はL1の厳密な整数・物理制約を生の`GameState`とルール関数から算出した。79次元を別名で丸ごと複製していない。重なる概念は現在道路長・称号差・使用済騎士・山札残数等に限定される。

`current_score = reward.actual_score(game, self)` はゲームの `_check_winner` と同じ内訳（Settlement + City×2 + 保有称号×2 + **本人が取得済み**VPカード）。`remaining_points=max(0,10-current_score)`。未来VPカードの期待値は加えない。score 10以上は候補なし・`unresolved=False`。

| Candidate軸 | 将来増分 | 二重計上防止 |
|---|---:|---|
| `settlement_gain_count` | 新設1個につき+1 | 建設後は現在点を再計算 |
| `city_upgrade_count` | 既存/将来SettlementのCity化1回につき**純増+1** | Cityの表示2点を追加点+2とはしない |
| `acquire_longest_road` | 未保有から獲得なら+2 | 既保有時のtrue候補はinvalid |
| `acquire_largest_army` | 未保有から獲得なら+2 | 既保有時のtrue候補はinvalid |

`VictoryPlanCandidate` は `(settlement_gain_count, city_upgrade_count, acquire_longest_road, acquire_largest_army)` をcanonical IDとする。`planned_point_gain` は4軸から計算するpropertyで、手書きの値を保持しない。列挙はこのIDの辞書順で安定しており、**増分がremaining_pointsと厳密一致**する組合せだけを列挙する。score 0～10の理論上最大はscore 0の36 raw tuples（通常開始score 2なら28）。overshoot候補は生成しない。各tupleに `feasibility`, `valid`, `diagnostic_reasons` を持たせる。監査のためIMPOSSIBLEもraw候補列には残すが、将来のPlan categorical入力用`candidate_mask`ではfalseにする。FEASIBLE/UNKNOWNはtrue。validなexact候補が0なら`unresolved=True`とし、架空候補を作らない。これは**Action maskではない**。S=0のPlanがあっても合法なSettlement建設を妨げない。

Context内の状態量は次の最小限の整数/booleanで、単位はそれぞれ点・駒・地点・辺・騎士枚数である。`settlement`: `current_count`, `pieces_available`, `distance_rule_empty_count`, `structurally_reachable_count`, `currently_connected_count`, `max_additional_roads`。`city`: `current_count`, `pieces_available`, `upgradeable_count`。`longest_road`: `currently_owned`, `current_length`, `current_holder`, `public_best_opponent_length`, `acquisition_gap`, `defense_gap`, `structurally_extendable`。`largest_army`: `currently_owned`, `used_knights`, `current_holder`, `public_best_opponent_used_knights`, `plays_needed`, `development_deck_remaining`, `known_usable_knights`, `known_new_knights`。加えてscore、remaining、候補tuple、unresolvedを持つ。Context79の`road.current_length/defense_gap`等と意味が重なる値は厳密なraw integerで再計算し、Context79のPlanner由来候補順位やvalueは読まない。

## 2. 物理的feasibilityの境界

| 判定 | 決定的な根拠と保守性 |
|---|---|
| Settlement | 盤面の建物・距離ルールから空vertexを列挙し、本人の建物/道路網から空きedgeのみで到達する最少追加道路数を0–1 BFSで計算する。自分の残road piece数以内に到達可能な地点数を楽観的上限とする。現在road接続済み地点の非隣接集合を実際に構成できたらFEASIBLEの証人とする。現在未接続でも将来道路で到達できる組合せはUNKNOWN。距離ルール地点数または楽観的構造到達地点数が計画S数に足りない場合だけIMPOSSIBLE。初期配置前の未形成networkはUNKNOWN。 |
| City + piece recycling | `current_city+C<=4`, `C<=current_settlement+S`, `current_settlement+S-C<=5` を同時に確認する。Cは**Planで新設するSettlement**を将来upgradeしてもよく、upgradeで戻るSettlement pieceをSの後続建設に再使用できる。違反はIMPOSSIBLE。資源が今足りないことはIMPOSSIBLEにしない。 |
| Longest Road | 現在の公開holder、本人/相手最長長、獲得に必要な長さ差、防衛差、phaseに依存しない現在の合法延伸edge数を保存。既保有なのに再獲得する候補はIMPOSSIBLE。未保有からの将来経路・相手blockを断言しないため保守的にUNKNOWN。防衛は+2の再獲得ではなく後続L2の情報。 |
| Largest Army | 公開holder、本人/相手使用済騎士、必要使用回数、本人既知の使用可能/新規Knight、公開deck残枚数を保存。保有Knightで必要回数を満たせば物理的FEASIBLE。未来drawが必要ならUNKNOWN。`既使用+本人既知Knight+deck残枚数 < 必要総使用数`を証明できた場合だけIMPOSSIBLE。deckのカード**種類**は読まない。既保有なら再獲得候補はIMPOSSIBLE。 |

`FEASIBLE` は物理的な**構成証人**であり、必要資源が現在手札にあるという意味ではない。`UNKNOWN` は有望という評価ではなく、future road、他者block、Knight draw等が未確定という意味。独立集合の簡易な証人探索が失敗しても不可能とは断言せずUNKNOWNに残す。以上にPlannerの優先順位・価値scoreは入っていない。

## 3. Shadow分布・実局面

固定seedはRule `33000000–33000011`、Champion `33001000–33001011`、Mixed `33002000–33002011`。各profile12局、seat1～4各3局で**36 Shadow games / 2,810 learner pre-action records**。同seedでobserverなしの36局も走らせた。rolling-order段階のscore 0が52件含まれる。action開始後の通常score 2～9のrecordは2,758件である。score 10以上は終局しPlan不要。

候補raw件数分布は `2:220, 5:245, 8:195, 12:225, 16:308, 20:429, 24:521, 28:615, 36:52`。最大は**raw 36 / valid 12**で候補爆発なし。全record累計では `FEASIBLE 815 / UNKNOWN 22,732 / IMPOSSIBLE 27,482`。有効candidateは前2者の計23,547個、IMPOSSIBLEはvalid maskで落とす。`unresolved=0/2,810`だが、exactで有効候補がなくなればAPIはtrueを返す。UNKNOWNが多いのは確率的drawや将来道路をFEASIBLEと断言していないためで、価値の低さを意味しない。

| 現在score | record数 | 有効candidate数 平均 [min,max] |
|---:|---:|---:|
| 0（順番決め） | 52 | 11.00 [11,11] |
| 2 | 615 | 10.84 [4,11] |
| 3 | 521 | 10.83 [4,11] |
| 4 | 429 | 10.12 [4,12] |
| 5 | 308 | 8.65 [4,11] |
| 6 | 225 | 6.36 [3,10] |
| 7 | 195 | 5.33 [2,7] |
| 8 | 245 | 3.36 [3,5] |
| 9 | 220 | 1.68 [1,2] |

| 本人保有称号（Road, Army） | record数 | 有効candidate平均 [min,max] |
|---|---:|---:|
| なし (0,0) | 2,265 | 9.56 [1,12] |
| Armyのみ (0,1) | 130 | 3.82 [2,6] |
| Roadのみ (1,0) | 403 | 3.42 [1,6] |
| 両方 (1,1) | 12 | 1.00 [1,1] |

実局面の抜粋（IDは`[S,C,R,A]`、**選択Planではない**）：

- Rule seed `33000000`、score 2、残り8、称号なし：raw28、有効11。`[2,2,true,true]` は増分 `2+2+2+2=8` の有効候補（future title等のためUNKNOWN）。
- Rule seed `33000002`、score 6、Road保有、残り4：有効5。例`[0,2,false,true]`。Road再取得を含む組合せはvalidではない。
- Rule seed `33000010`、score 6、Army保有、残り4：有効5。例`[0,2,true,false]`。Army再取得を含む組合せはvalidではない。
- Rule seed `33000000`、score 9、Road保有、残り1：有効なのは`[0,1,false,false]`と`[1,0,false,false]`。+2 titleだけの架空exact候補はない。
- Rule seed `33000010`、score 9、両称号保有、残り1：有効なのはCity純増`[0,1,false,false]`の1件。称号はどちらも再計上されない。

完全な候補・reason・score/title/turn/revision/digestは [`shadow_records.json.gz`](rl_victory_plan_context_step3b1_20261012_evidence/shadow_records.json.gz)、集計と全36局のseed/seat/終局結果は [`summary.json`](rl_victory_plan_context_step3b1_20261012_evidence/summary.json) に保存した。

## 4. 得点変化と非干渉の検証

[`test_victory_plan_context_step3b1.py`](../tests/test_victory_plan_context_step3b1.py) で次を固定した。

| シナリオ | 検証 |
|---|---|
| VPカード既知取得 | score+1、remaining−1、candidate再生成。deck順を入れ替えてもContext不変 |
| Plan外Settlement | S=0の有効Planがある状態で実際に合法Settlementを建設でき、score+1、remaining−1、candidate再生成 |
| Settlement→City | 実Actionでscore+1（+2ではない）、remaining−1、Settlement piece返却、City piece消費 |
| Longest Road | 実Road Actionで称号獲得時score+2、remaining−2、有効candidateへ同称号を再加算しない |
| Largest Army | 実Knight使用で称号獲得時score+2、remaining−2、有効candidateへ同称号を再加算しない |
| score 9 | remaining=1、exact候補に+2称号単独を混入しない |
| piece/board/deck上限 | City4個、Settlement piece recycling、距離/到達不能、Knight残枚数を診断。初期配置前はUNKNOWN。score9でS/C双方を物理的に作れない構成では2 exact tupleがともにinvalidとなり`unresolved=True` |
| 非干渉 | 同一状態でContext/ID/feasibilityが完全一致。深いGameState比較とRNG counter比較が前後一致。相手hidden resource/VP、deck順に依存しない |

固定seed Shadowでは、同じ36局をobserverあり/なしで実行し、全Action trace・winner・final VP・rank・policy steps・最終GameState・RNG counterを厳密比較した。Shadow recordにはseed/profile/seat/turn/revision/policy step/score/remaining/titles/candidate list/feasibility数/unresolved/Context digest/状態・RNG parityを保存し、`selected_plan`等は記録しない。Champion weightも前後digest比較する。

結果は**36/36局でAction・winner・final VP・rank・policy steps・最終GameState・RNG完全一致**、2,810回のContext計算でGameState mutation/RNG consumptionとも0。Champion digest `4225902d41a83540eb1f1552826ec623706e778b99ad4ae8ec015f976644392a`、Robber PPO digest `ecde7fa9e9d67370bb76686ecd767d2d532a76d467c0051aaebdf8d4aff702bf` は前後一致。テストは**362 passed、1,172 subtests passed**（既存依存のdeprecated warning 2件のみ）。新規10テストも個別実行で10/10通過。

## 5. Step 3B-2へ

次はこの`VictoryPlanContext`、安定candidate ID、valid mask、物理feasibility、score/remainingを**Shadow専用**のPlan/Priority表現へ入力する。Planのrank付けやPriority値はここでは付けていない。headを追加する場合も初期状態でActionを変えず、候補maskはPlan選択のみに適用し、Plan外の合法Actionは遮断しない。UNKNOWNを残し、称号防衛と再計画/persistenceはL2/L1の次工程で検討する。即時勝利は引き続き既存ImmediateWinResolverの責務であり、この候補生成に接続しない。
