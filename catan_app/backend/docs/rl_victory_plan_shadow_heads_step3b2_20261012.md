# Step 3B-2 — Victory Plan / Strategic Priority Shadow Heads

## 対象と非対象

Step 3B-1Aの「10点以上へ到達するための最小得点構成」を、将来のL1 Plan評価に使える可変候補ネットワークへ接続した。現段階では未学習であり、Planを選択・sampleせず、Action、Controller、Observation、Reward、既存Champion、3A Gate、Investmentには一切接続しない。PlannerのGoal/Intent/Reason、ChampionのAction、旧native family logitsをPlanの教師・priorとして使わない。optimizerも作成しない。

## Headと特徴量

新module [`victory_plan_policy.py`](../app/rl/victory_plan_policy.py) は、`VictoryPlanFeatureEncoder`、共有`VictoryPlanScorer`、`StrategicPriorityHead`、`VictoryPlanShadowOutput`を持つ。Frozen Champion v2 Actor latent 392次元は`detach()`し、決定的Runtime Context79とVictoryPlanContextのglobal facts 30次元を連結する（global入力501次元）。Shadow runnerではChampion policyの`requires_grad`もfalseにする。hidden幅は64、global encoderとcandidate encoderを別々に通し、両者を結合して共有fusion MLPへ渡す。candidateの順序・個数に依存した出力層は置かない。Candidate最大36件を36 slotへゼロpaddingし、canonical IDとの対応を保持する。各候補を同じscorer `64→1`で採点し、valid mask付きsoftmaxで相対分布を作る。候補間Attentionはない。

Global featuresは以下の30次元。括弧内が固定scaleであり、ルール上の上限から設定した値である。scoreは10で飽和（10点以上はPlan不要）。holder IDは連続値でなく`self/opponent/none`の3 one-hotに変換する。

| 群 | feature / scale |
|---|---|
| 得点 | current_score/10, remaining_points/10 |
| Settlement | current_count/5, pieces_available/5, distance_rule_empty_count/54, structurally_reachable_count/54, currently_connected_count/54, max_additional_roads/15 |
| City | current_count/4, pieces_available/4, upgradeable_count/5 |
| Longest Road | owned/1, current_length/15, public_best_opponent_length/15, acquisition_gap/15, defense_gap/15, structurally_extendable/72, holder_self/opponent/none 各/1 |
| Largest Army | owned/1, used_knights/14, public_best_opponent_used_knights/14, plays_needed/14, deck_remaining/25, known_usable_knights/14, known_new_knights/14, holder_self/opponent/none 各/1 |

候補featureは9次元: `S/10, C/10, R, A, planned_point_gain/11, overshoot/1, FEASIBLE, UNKNOWN, IMPOSSIBLE`。feasibilityとovershootは入力情報だけで、手製penalty・bonus・Plan価値scoreには使わない。全featureはfiniteかつ`[0,1]`内とassertする。Runtime Context79は既存v1の正規化をそのまま使用する。

Plan scorerの最終Linearはweight/biasとも0なので、初期raw logitは全候補0。IMPOSSIBLEとpaddingは`masked_fill(-1e8)`後、確率へmaskを再乗算するため確率が厳密に0。FEASIBLEとUNKNOWNは等しく候補として残す。valid候補n件では初期確率`1/n`、entropy`log(n)`となるが、**同価値を学習済みという意味ではなく未学習のneutral初期化**である。`unresolved`または`remaining=0`では有効maskがなく、`plan_available=false`、確率・entropyとも0を返す。架空Planは追加しない。

Priorityは候補ごとの共有featureに`64→4` Linearと独立sigmoidを適用する。順序はSettlement、City、Longest Road、Largest Army。最終Linearはweight/biasとも0、初期値は各0.5。排他的softmaxではないため合計1に拘束されず、複数軸を同時に高くできる。今回はPriorityをActionとしてsampleせず、将来L3 Investment lossからend-to-end更新可能な内部表現として保持する。Planをまだ選ばないため、各valid候補を条件としたPriorityを計算するだけである。Plan Critic・再計画頻度は未決定。

## 36局Shadow検証

Rule / Champion / Mixed各12局、各profile内でSeat 1～4を各3局とし、Step 3B-1Aと同じseed `33000000～011` / `33001000～011` / `33002000～011`を使用した。36局で2,810 pre-action状態を観測し、旧3B-1Aの2,810 recordとseed・revision・score・candidate ID・valid maskを全件照合した。candidate一覧とmaskは**2,810/2,810一致**。valid候補の平均は8.520、最大raw36、最大valid12。観測状態の`unresolved`は0。score10以上のPlan不要処理と全invalidのunresolved処理はunit testで固定した。

| 指標 | 結果 |
|---|---:|
| Plan entropy mean / median / p5 / p95 / max | 2.0421 / 2.3026 / 1.0986 / 2.3979 / 2.4849 |
| valid候補1件によるentropy 0 | 12状態 |
| Priority 4軸 min / mean / max | 全て0.5 / 0.5 / 0.5 |
| invalid・padding候補の非zero確率 | 0件 |
| valid確率和・`log(n)` entropyの検証失敗 | 0件 |
| GameState / RNG mutation | 0 / 0件 |
| 対局parity failure | 0 / 36局 |

例: score 2、称号なしの状態はraw28・valid11で各valid Plan確率`1/11`、entropy `log(11)=2.3979`。score 7、称号なしではraw9・valid7に`R+A`候補を含み、entropy `log(7)=1.9459`。score 9、称号なしではraw4・valid4に単独のR/A候補があり、各確率0.25、entropy `log(4)=1.3863`。score 9で両称号を既に保有する実状態はraw4・valid1で、再取得candidateの確率は厳密0。これらは**未学習の均等出力**を示すだけで、いずれのPlanが強いかは示さない。記録には`selected_plan`を作らず、各候補ID・mask・logit・probability・Priority 4値をそのまま保存した。

連続する本人decision状態2,774組について、eventは重複可で次のとおり。`SCORE_CHANGED=240`、`TITLE_CHANGED=60`（公開holder IDの変化）、`CANDIDATE_SET_CHANGED=240`、`VALID_MASK_CHANGED=241`、candidate IDが不変でmaskのみ変わったものは1、どの構造eventもないものは2,497（90.0%）。得点変化時にはminimal-winning tuple集合が変わり、title・盤面変化は主にfeasibility maskへ現れる。ただしPriorityの価値入力は資源等でも変化するため、「構造eventのみでPlanを再選択すべき」とはまだ断定しない。そもそも今回はPlanを選んでおらず、selected Plan churnは存在しない。

Head parameter数はfeature encoder **41,024**、Plan scorer **65**、Priority head **260**、計 **41,349**。初期化seedは3302（Torch RNGをforkして外部状態を復元）。Head weight digestは収集前後とも`2000b1390fcf8135ba819ba40b1ee1720f0596327c2c658ec1e2f24a29dd1887`。Champion digest `4225902d41a83540eb1f1552826ec623706e778b99ad4ae8ec015f976644392a`、Robber PPO digest `ecde7fa9e9d67370bb76686ecd767d2d532a76d467c0051aaebdf8d4aff702bf`も前後一致。optimizerは生成していない。

1状態あたりの平均 / p95は、Plan Context生成`0.463 / 0.547 ms`、Frozen Champion latent＋Context79取得`18.220 / 24.126 ms`、feature化＋Head forward`1.064 / 1.184 ms`、全体`19.746 / 25.609 ms`。Shadow専用Head自体の増分は小さく、主要負荷は既存representationとRuntime Contextの取得にある。これはAction決定経路へまだ入れていないため、対局throughput変更を示す値ではない。

通常対局とShadow対局の36/36局で、Action trace、winner、final VP、rank、policy steps、最終GameState、RNG、terminal/truncationが完全一致した。Plannerのselected Goal/Intent/ReasonやChampion選択ActionをHead入力・Teacher・診断の正解として使わず、native 6-family logitsもPlan priorへ流用していない。全backend testは**368 passed、1172 subtests passed**（既存のdeprecation warning 2件）。Candidate順を反転し、最終層に非zero重みを仮設定しても、同じIDのraw logit・Priorityが一致することをunit testで確認。異なる候補数の2状態を同一padding batchで処理でき、Priorityの2軸を同時に高くするテストも通過した。

[summary](rl_victory_plan_shadow_heads_step3b2_20261012_evidence/summary.json) と[全candidate出力を含む圧縮shadow records](rl_victory_plan_shadow_heads_step3b2_20261012_evidence/shadow_records.json.gz)を証拠として保存した。対局途中の一時checkpointは実験ディレクトリ内のみで、学習artifactではない。

## Step 3B-3へ

即時建設のルール境界を既存Controllerと照合するのが次の候補。Plan/Priorityを今回の未学習出力のままActionへ渡さない。候補集合・maskのevent頻度を再計画設計の材料にするが、ここでhardな再計画ruleは設けない。
