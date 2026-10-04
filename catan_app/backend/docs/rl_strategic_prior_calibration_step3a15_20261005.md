# Step 3A-15: Strategic Prior Calibration / Exploration Audit

2026-10-05。Strategic Gate→Frozen B Executor→Environment、ImmediateWinResolver、phase/forced controller、protected PlayerTradeの仲裁順はStep 3A-14のまま。PPO/Gate/Criticのoptimizer step、Reward・Executor・Champion artifact変更はない。追加したのは診断用のmasked temperature/mixture計算、Gateの固定prior temperature引数、offline監査、同seed小規模実委譲smokeである。実験データは `experiments/strategic_prior_step3a15_20261005/{offline,summary,games}.json`。

## Offline監査とprior選定

Step 3A-13のAction前Shadow 1,123局面（候補数2: 570、3: 399、4以上: 154）を全温度で再評価した。Step 3A-14の実委譲66局面も同じ方法で監査し、結果は `offline.json` に別集計した。available candidateだけにsoftmaxを適用し、残差はゼロ。全温度・mixtureでmask後確率和は1、unavailable確率は0、native top-1保持率は100%。温度は順位を変えず、探索確率のみ変える。

| prior | entropy平均 (nat) | top平均 | 2位平均 | available最小確率平均 | exp(entropy)平均 | 代替確率平均 | 代替確率 p25 / median / p75 | top≥.999 |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| T=1 | .328 | .855 | .130 | .086 | 1.470 | .145 | .004 / .084 / .237 | 16.3% |
| T=1.25 | .380 | .829 | .150 | .108 | 1.549 | .171 | .013 / .129 / .282 | 8.5% |
| T=1.5 | .428 | .805 | .168 | .128 | 1.620 | .195 | .028 / .170 / .315 | 4.5% |
| T=2 | .514 | .763 | .199 | .159 | 1.750 | .237 | .074 / .233 / .358 | 0% |
| T=3 | .649 | .694 | .247 | .205 | 1.974 | .306 | .184 / .312 / .404 | 0% |
| mixture ε=.05 | .413 | .832 | .143 | .102 | 1.573 | .168 | .038 / .105 / .250 | 0% |
| mixture ε=.10 | .479 | .810 | .157 | .118 | 1.664 | .190 | .071 / .126 / .263 | 0% |
| mixture ε=.20 | .586 | .765 | .185 | .150 | 1.835 | .235 | .138 / .169 / .290 | 0% |

各候補数の `entropy / 代替確率` 平均は以下。代替確率は `1 - P(native top)`。

| prior | 2候補 | 3候補 | 4候補以上 |
|---|---|---|---|
| T=1 | .345 / .138 | .411 / .208 | .051 / .011 |
| T=1.25 | .403 / .173 | .453 / .225 | .108 / .024 |
| T=1.5 | .445 / .202 | .498 / .242 | .184 / .043 |
| T=2 | .503 / .247 | .591 / .277 | .361 / .094 |
| T=3 | .571 / .305 | .747 / .343 | .684 / .211 |
| ε=.05 | .392 / .156 | .510 / .231 | .239 / .049 |
| ε=.10 | .430 / .174 | .585 / .254 | .384 / .086 |
| ε=.20 | .493 / .210 | .708 / .300 | .620 / .161 |

familyがavailableな局面での平均確率（括弧内は局面数）：

| family | T=1 | T=2 | T=3 |
|---|---:|---:|---:|
| BUILD_SETTLEMENT (97) | .991 | .964 | .904 |
| BUILD_CITY (72) | .964 | .924 | .852 |
| BUILD_ROAD (616) | .418 | .365 | .352 |
| BUY_DEVELOPMENT (403) | .821 | .775 | .703 |
| BANK_TRADE (666) | .409 | .370 | .364 |
| END_TURN (1,123) | .086 | .159 | .206 |

採用候補T=2でのavailable時確率の `p10 / median / p90` はSettlement `.934/.992/.996`、City `.883/.982/.996`、Road `.004/.410/.764`、Development `.071/.915/.979`、BankTrade `.006/.383/.719`、EndTurn `.002/.171/.355`。他温度のfamily別median/p10/p90、候補数別代替確率p25/p75、top family件数は `offline.json` に保存した。特にSettlement/CityはT=2でも強いnative preferenceを維持する一方、END_TURNの探索余地が増える。4候補以上ではT=1のほぼ決め打ちをT=2が緩める。T=3も一様分布ではないが、変化が大きいためT=2とともに実委譲で検証した。mixtureは比較診断にとどめる。将来採用するなら混合**分布そのもの**のlog_prob・entropy・PPO ratioを使う必要があり、nativeからsample後の後付けε-randomは不可。初回は実装とon-policy整合が単純な固定T方式を推奨する。

## 同seed実委譲smoke

T=2/T=3を各36局（Rule/Champion/Mixed各12局、席1〜4各9局）、委譲抽選10%、同じgame/delegation/gate sampling seedで比較した。T=1はStep 3A-14の同じ36局の保存データ。到達局面は選択Actionにより変わるので、確率のoffline比較と実際のfamily選択は分けて読む。各episodeのtemperature・外部RNG seedは記録。T=1の保存データと新実装で同seed1局を再実行し、winner、VP、rank、policy steps、RNG counter、family行動数、Gate選択列が一致した。

| 指標 | T=1 | T=2 | T=3 |
|---|---:|---:|---:|
| 完走 / 全局 | 36/36 | 36/36 | 36/36 |
| Gate surface | 1,126 | 1,127 | 1,102 |
| protected PlayerTrade preemption | 380 | 385 | 379 |
| 委譲抽選対象 | 746 | 742 | 723 |
| 実委譲 / StrategicTransition | 66 | 66 | 64 |
| native非top選択 | 6 (9.1%) | 14 (21.2%) | 15 (23.4%) |
| Gate family一致 / 具体Action一致（対Champion診断） | 60.6% / 42.4% | 51.5% / 39.4% | 53.1% / 39.1% |
| Resolver発火 | 16 | 17 | 18 |
| illegal / Executor失敗 / family外 / deadlock / truncation / nonfinite reward | すべて0 | すべて0 | すべて0 |

| 選択family | T=1 | T=2 | T=3 | 非top選択 T=1 / 2 / 3 |
|---|---:|---:|---:|---:|
| BUILD_SETTLEMENT | 9 | 8 | 7 | 0 / 0 / 0 |
| BUILD_CITY | 5 | 4 | 4 | 0 / 0 / 0 |
| BUILD_ROAD | 21 | 14 | 16 | 0 / 0 / 0 |
| BUY_DEVELOPMENT | 18 | 20 | 16 | 0 / 2 / 1 |
| BANK_TRADE | 10 | 14 | 11 | 3 / 6 / 4 |
| END_TURN | 3 | 6 | 10 | 3 / 6 / 10 |

実委譲局面に限ったentropy平均は候補2/3/4+でT=1が `.303/.318/.018`、T=2が `.467/.522/.256`、T=3が `.570/.733/.592`。T=3はEND_TURNを10回選び、T=2の6回より増えた。ただし36局でnon-top件数差はわずか1、別状態へ到達する影響もある。Development、BankTrade、EndTurnの過多・collapseは今回の小規模screenでは認めないが、上位候補以外のSettlement/City/Roadはまだ実選択0件であり、全familyの探索十分性を示すものではない。Champion一致率は温度採択基準でも教師指標でもない。

参考の対局結果（同seedでもGateにより軌跡が異なる。36局で強弱は判定しない）：

| 指標 | T=1 | T=2 | T=3 |
|---|---:|---:|---:|
| 勝数 | 17 | 17 | 18 |
| 平均VP / rank | 7.86 / 1.94 | 7.92 / 1.93 | 7.94 / 1.90 |
| 平均policy steps | 72.7 | 73.0 | 70.4 |
| Action phase 開拓 / 都市 / 道路 | 104 / 64 / 227 | 106 / 62 / 221 | 107 / 61 / 227 |
| 発展購入 / 銀行交換 / EndTurn / protected PlayerTrade | 73 / 134 / 586 / 402 | 73 / 135 / 588 / 406 | 67 / 125 / 566 / 399 |

## Semi-MDP returnと初回RL見積り

実委譲DecisionだけがActor sample。次の実委譲またはterminalでtransitionをcloseし、既存learner policy step数をdurationとして `R_macro=Σ gamma^i r_i`、非terminal bootstrapを `gamma^duration` とするStep 3A-14の検証を全transitionで再実施した。T=1/2/3の196実委譲recordすべてについて、masked distributionの正規化と非候補確率0、選択確率のlogと保存log_prob、対応transitionのlog_probが一致した。Reward、Executor、Criticは更新していない。

| 指標 | T=1 | T=2 | T=3 |
|---|---:|---:|---:|
| macro duration mean / median / p95 / max | 23.1 / 17 / 64 / 85 | 23.3 / 17 / 64 / 92 | 22.5 / 17 / 48 / 92 |
| macro reward平均 / 標本分散 | .0945 / .3016 | .0931 / .3077 | .1270 / .3193 |
| durationとrewardのPearson | .196 | .185 | .140 |
| terminal reward平均 (n=29) | .144 | .146 | .215 |
| nonterminal reward平均 (n=37/37/35) | .056 | .052 | .054 |

T=2のfamily別macro reward平均/件数はSettlement `.159/8`、City `.325/4`、Road `-.012/14`、Development `.123/20`、BankTrade `.017/14`、EndTurn `.173/6`。T=3と各familyの分散も `summary.json` に保存した。terminal transitionの分散が大きい（T=2で `.704`、非terminal `.004`）。familyごとの平均returnを、この少数・異なるstate分布から優劣や教師scoreとして使用しない。

**採用候補は固定T=2**。T=1より非top選択が8件増え、4+候補のoffline代替確率は1.1%→9.4%。T=3はさらに平坦だが、今回の36局では非top選択の上積みが1件にとどまりEND_TURN増加も見える。これはT=3が悪いと断定するものではなく、最初のRLにおける保守的な探索初期値の選択である。T=2の10%時66 transition/36局から、20%委譲なら約3.67 transition/局・256件に約70局、25%なら約4.58件/局・約56局と線形概算する。最初の具体案は20%固定でRule/Champion/Mixed各24局・席1〜4各6局（72局、約264 transition期待）。状態訪問とepisode長が変わるため、実収集ではepisodeを切らず256件を目安に停止する。6-familyのrare candidate/非top sample数を途中で点検し、不足なら局数を増やす判断を別途行う。

次Step 3A-16のPPO比率・KLは、**実際にrolloutを生成した固定T=2旧Policy** `softmax(z_native/2 + residual_old, mask)` に対する更新後同温度・同maskの分布で計算する。T=1 nativeとのKLではない。pre-action observation/context、mask、旧masked family分布、sampled family、旧log_prob、StrategicTransitionを保存し、old/new全6-familyのmasked categorical KLのmean/max、候補数・family別KLを監視する。初回1 updateの暫定警戒線はmean KL `.01`、max KL `.1` とし、超過なら追加updateせず局面別に調査する（この値自体は学習前に確定する）。temperatureをepisode途中で変更しない。Gate residualと独立Strategic Critic以外、Champion GNN/377 Actor/base Critic/Executorを凍結し、実委譲sampleのみで小規模1 updateから開始する案。PPO超過KLやfamily collapse、Full Surface側の異常があれば続行しない。学習設定の確定・学習実行はStep 3A-16で行う。

Step 3A-16へ進むintegration/exploration Gateは通過。ただし「強さ改善」や全family十分な探索の合格ではない。最終コードで全319テスト成功。Champion重みdigestは実験前後で一致し、既存artifactには書き込んでいない。
