# Step 3A-11: CITY_ONLY multi-future paired audit

## Scope

Step 3A-10の204局面から代表52局面を選び、同一Action前GameStateを複製して、同じfuture seedを与えたBUILD / DEFERのペアを繰り返す。後続Decisionは双方とも凍結Champion Plannerと固定Opponentが担当する。比較するのは最初のCITY_ONLY判断1回だけであり、Gateは後続で使用しない。PPO 2回目update、BC、Reward、DEFER Executor、Champion、Observation、377 Action Spaceの変更は一切行わない。

対象はChampion DEFER全31局面、profile/seat/得点/site数/city数/手札/DEFER familyに近いChampion BUILD control 16局面、および単発差のBUILD大差2・DEFER大差2・near-tie 1局面。controlは単発 `|ΔQ|>0.5` を除外して選び、outlierへの偏りを避けた。選択後は `selection.json` に固定。ゲーム内の複数局面を含むので、52局面を独立した52試合とは扱わない。

Phase 1は全52局面に8つの新しいfuture seed、Phase 2はPhase 1で不安定と判定した最大12局面のみ追加8 seed。各ペアは両枝に同じfuture seedを渡す。元のGameState、RNG counter、revision、Environment bookkeepingは変えない。分岐Actionによってその後の乱数消費順が違うため、「同じ未来事象を経験した」とは主張しない。元のStep 3A-10単発rolloutを8件へ混入させず、独立の比較対象として保存する。

`continuations.jsonl` はfuture seed×stateの単位で、`state_id` / `game_id` / profile / seat / Champion判断 / DEFER family / Context、割引環境return・VP成分・terminal成分、勝敗、最終VP・順位を保存する。`summary.json` はstate別統計と群別集計。実験artifactは `experiments/construction_city_step3a11_20261004/`。実行CLIは `python -m app.rl.run_construction_city_multifuture_step3a11 --phase phase1`、次に `--phase phase2`、最後に `--phase summary`。収集はcontinuation indexで冪等に再開できる。

## 統計の定義

`ΔQ = Q_BUILD - Q_DEFER`。near-tieは各continuationの `|ΔQ|≤0.01` とstate平均の `|mean ΔQ|≤0.01` を別々に数える。`sign consistency` はnear-tieを除いたペアで `max(BUILD優位数, DEFER優位数) / 非near-tie数`。95% CIはstateごとのfuture seedを再標本化した固定seed percentile bootstrap（2,000反復）で、探索的な幅の診断に限る。CIから教師の正解ラベルは作らない。

Variance decompositionは、adaptiveなPhase 2の選択バイアスを避けるため、全stateで均等に8 seedあるPhase 1だけを使用する。`within = mean(sample Var(ΔQ | state))`、`between_observed = sample Var(state mean ΔQ)`、`between_corrected = max(0, between_observed - within/8)`、`ICC_like = between_corrected / (between_corrected + within)`。これらは代表局面を意図的に選んだ探索的指標で、全CITY_ONLY分布の厳密な推定値ではない。

## 結果

### 収集・再現性

対象52 stateはChampion DEFER 31、対応を取ったChampion BUILD control 16、単発BUILD大差2 / DEFER大差2 / near-tie 1。profileはChampion 19 / Mixed 16 / Rule 17、Seat 1～4は14 / 11 / 15 / 12。Phase 1は全52 state×8=416ペア、Phase 2は不確実な12 state×追加8=96ペア、合計512ペア（BUILD 512本・DEFER 512本）。40 stateが8 future seed、12 stateが16 seedで、全1024枝が合法に終局しtruncation 0。Phase 2を全stateへ一律に実施していない。データの各ペアで元GameState、RNG seed/counter、revisionを含むstate全体、およびEnvironment bookkeepingの非変更を検査した。保存済みの1ペアは元ゲームseedとfuture seedから再実行し、BUILD / DEFERの割引returnと差が完全一致した。

### state別ΔQと分散分解

52個のstate mean `ΔQ` は平均+0.0618、中央値+0.0010、SD 0.1804、p25 +0.0004、p75 +0.0839、最小 -0.5440、最大 +0.6532。実質的な閾値 `0.01` ではBUILD平均優位23 / DEFER平均優位8 / near-zero21。各stateのcontinuation内SDは中央値0.0717（p25≈0、p75 0.5436、最大1.5556）、SE中央値0.0252（p75 0.1405）。bootstrap 95% CIの幅は中央値0.0912（p75 0.5372、最大1.4728）で、20/52は0を跨ぐ。CIが実質的同点帯[-0.01,+0.01]を完全に外れるのは10/52だけ。near-tie以外が存在する32 stateのsign consistencyは中央値0.742（p25 0.571、p75 0.862）。残り20 stateは全continuationがnear-tieで、符号一致率を定義しない。

均等なPhase 1の8 seedだけで計算した分散分解は次のとおり。

| 指標 | 値 |
|---|---:|
| 平均 within-state sample variance | 0.2879 |
| within-state SDの中央値 | 0.0717 |
| state meanの観測between variance | 0.02734 |
| `within/8` によるstate mean測定noise推定 | 0.03599 |
| `max(0, observed between − measurement noise)` | 0 |
| 補正between/within・ICC-like | 0 / 0 |

観測されたstate平均のばらつきは、8 seedで平均を測る際のnoise推定を下回る。これはこの選抜52局面・この粗い分散分解では**全体的な局面間signalを確認できない**という意味であり、すべての局面が同価値という証明ではない。少数の大きな勝敗反転でwithin平均が押し上げられる一方、near-zeroが21 state、実質的同点帯を外れるCIは10 stateある。分布は均一な正規noiseではない。

Phase 2に選ばれた12 stateでは、追加前後で `|mean ΔQ|>0.01` / `<-0.01` / near-zeroの分類が2 stateで変わった。CI幅の中央値は0.794→0.541に縮まったが、12 stateのうち実質的同点帯をCIが外れる例は追加前後とも0。この追加評価でも不安定な局面の期待差は確定しなかった。

### ChampionのDEFERとcontrol

| Champion判断 | state | BUILD平均優位 | DEFER平均優位 | near-zero | 実質的同点帯を外れるCI | mean ΔQ |
|---|---:|---:|---:|---:|---:|---:|
| DEFER | 31 | 11 | 5 | 15 | 8（BUILD方向6 / DEFER方向2） | +0.0575 |
| BUILD | 21（control 16 + probe 5） | 12 | 3 | 6 | 2（各方向1） | +0.0680 |

Champion DEFER群のmean sign consistencyは0.730、CI幅中央値0.0787。BUILD control/probe群は0.739、CI幅中央値0.1377。Champion DEFERに実際のDEFER期待値優位がありそうな例はあるが、5/31のmeanが負、その中で実質的同点帯を外れるCIは2件だけ。Champion判断の模倣を成功条件やBC教師にはしない。

### DEFER Executor family

| DEFER初手 | state | state mean ΔQの平均 / 中央値 | BUILD / DEFER / near-zero | mean sign consistency |
|---|---:|---:|---:|---:|
| END_TURN | 8 | +0.1510 / +0.0995 | 8 / 0 / 0 | 0.688 |
| BUY_DEVELOPMENT | 5 | +0.0094 / 0 | 1 / 2 / 2 | 0.636 |
| BANK_TRADE | 15 | +0.1366 / +0.0808 | 12 / 3 / 0 | 0.768 |
| BUILD_ROAD | 17 | -0.0092 / +0.0005 | 1 / 3 / 13 | 0.839 |
| PLAYER_TRADE | 7 | +0.0092 / +0.0010 | 1 / 0 / 6 | 0.563 |

END_TURNは8 stateすべてのmeanがBUILD方向だが、2 stateだけが実質的同点帯を外れるCIで、他はfuture seedの揺れが大きい。Champion自身がEND_TURNを選んだ例にもBUILD優位があるため、GateのTiming判断が悪い可能性と、DEFER Executorの代替行動が弱い可能性はこの監査だけでは分離できない。8例の選抜群をもってEND_TURNを固定ruleで禁止したりExecutorを変更したりしない。BANK_TRADEでもBUILD側が多いが、同じ注意が必要。BUILD_ROADでは13/17がnear-zeroで、少数にDEFER安定優位がある。各familyはサンプルが少なく、全CITY_ONLY分布の頻度ではない。

### Reward alignmentとContext

state meanの総returnと割引terminal成分を閾値0.01で比較すると、A（両方BUILD）21、B（両方DEFER）8、C（総return BUILD・terminal DEFER）1、D 0、near-zero等22。Cの1 stateはBUILD/DEFERの実勝率差が0。実勝率差と総return差の符号を判定できる15 stateでは一致15 / 逆転0、残り37は同勝率またはnear-zeroで判定不能。少なくとも今回、現行Rewardが実勝率と系統的に逆向きという証拠はなく、Reward変更は見送る。

最終VP・rank・勝率のBUILD−DEFER差をstateごとに平均してから52 stateで等重み平均すると、それぞれ +0.278 VP、−0.091 rank（小さいほど良い）、+0.0349 勝率だった。ペア単位ではBUILDのみ勝利43、DEFERのみ勝利25、両枝の勝敗が同じ444。これも意図的に選んだ52局面の診断値であり、全CITY_ONLY局面でBUILDが優るという推定には使わない。

state mean BUILD優位23 / DEFER優位8 / near-zero21におけるContext平均は、hand size 7.91 / 9.88 / 11.10、discard exposure 48% / 75% / 90%、blocked pips 0.43 / 3.38 / 3.76、city production gain 11.13 / 10.12 / 10.48、score 3.43 / 4.62 / 4.62、site count 2.87 / 3.88 / 3.76。settlement ETA（valid 21/6/20）は1.12 / 0.74 / 0.11、shortage totalは3.10 / 2.83 / 1.35、Road Title gap（valid 22/8/20）は2.86 / 2.50 / 2.35、Knight plays neededは2.39 / 2.88 / 2.67、development affordabilityは52% / 50% / 86%、bank trade optionsは3.30 / 4.50 / 5.52。差の候補はあるが、score・盤面段階・familyが交絡しており、しかもDEFER優位は8 stateだけ。単変量の閾値や新しいPlanner if-ruleは作らない。

### Step 3A-10の単発値との比較

同じ52 stateで単発ΔQとmulti-future mean ΔQを比較すると、Pearson 0.090、Spearman 0.408。双方がnear-zeroでない26 stateの符号一致は18/26（69.2%）。残り26 stateは片方または両方がnear-zeroで、符号比較から外した。元の最大DEFER大差 `city-cf-rule-2661002-2-52` は単発 -2.0135から、16 future seedの平均 +0.6532へ反転した。ただしその16回のSDは1.5556、95% bootstrap CIは[-0.090, +1.294]で、BUILD期待値優位と断定もできない。単発の大差順位1位はmulti-futureでも絶対平均差1位だったが**符号が逆**。別の単発BUILD大差順位4位はmulti-futureの絶対平均差31位まで低下した。単発ΔQはこの監査群で非常にnoisyなproxyであり、正解ラベルとして使えない。

## 判断と次工程

今回の判定は**Case B（noise支配）寄り**。全体の補正between-state varianceは0、21/52が平均near-zero、追加8 seedを配った不安定12 stateでも実質的同点帯を外れるCIは0。少数の安定したstate差やContextとの関係は存在するため「CITY Timingは全局面で無価値」とは言わない。しかし、現状の独立した二択Gateを2回目updateで強く最適化するとfuture RNG noiseと即時VP shapingへ過適合する危険が高い。**CITY_ONLY GateのRL再開は今回は見送る**。Champion BCもしない。

Case CのReward不整合もCase DのExecutor欠陥も未確認なので、RewardConfig / Frozen DEFER Executorは変更しない。SETTLEMENT_ONLYを再導入せず、ChampionとStep 3A-9の実験artifactも維持する。次工程では、Timing単独への追加PPOより、`CITY / SETTLEMENT / ROAD / TRADE / DEVELOPMENT / END_TURN` の機会費用を扱う、より広いStrategic Action decisionへの統合可否を設計監査することを推奨する。その際も決定的Context・合法mask・Bayesian belief・資源計算はアルゴリズム側に残し、価値判断だけをPPOへ渡す。もしCITY_ONLY二択を再検討するなら、今回のstate単位の複数future評価を**教師ラベルにせず**独立評価として使い、別game seedで少数の安定したDEFER優位局面が再現するかを先に確認する。

## 安全性とテスト

`python -m unittest discover -s tests -q` で297件すべて成功。保存された512ペアは `(state_id, future_seed_index)` がすべて一意で、truncation 0、`q_build - q_defer = delta_q = delta_vp + delta_terminal` を全件で確認した。収集時には毎回、原局面のGameState・RNG・Environment bookkeepingが分岐rolloutで変わらないことを検査している。保存済みの1ペアをgame seedとfuture seedから再実行し、両枝returnと差の完全一致を確認した。既存Championや学習済みGateの重みを変更せず、PPO updateも行っていない。
