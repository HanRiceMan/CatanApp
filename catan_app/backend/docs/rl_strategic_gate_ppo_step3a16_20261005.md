# Step 3A-16: 初回Strategic Gate on-policy PPO update

2026-10-05。目的は6-family Strategic Gateが実委譲returnだけから**1回**更新可能かの検証であり、本格連続学習・Champion置換・勝率改善の証明ではない。`Strategic-T2-preupdate` と `Strategic-T2-postupdate` を別artifactとして保存。Champion `ppo_gnn_board_65k_s03_exp_v003` の重みdigestは実験前後で `4225902d41a83540eb1f1552826ec623706e778b99ad4ae8ec015f976644392a` のまま。学習seedは `3160000` 系列、評価seedは `3260000` 系列で重複なし。委譲抽選seedはgame seed+12345、family sampling seedは+54321、minibatch seedは3169001。GameState RNGは抽選に使用しない。

## Controller・Policy・artifact

仲裁順はImmediateWinResolver → phase/forced controller → protected PlayerTrade → state-based Strategic Surface → 外部RNGによる委譲抽選 → `softmax(native 15-family logits/2 + Gate residual, candidate mask)` → Frozen B Executor → Environment。family順はSettlement / City / Road / Development購入 / BankTrade / EndTurn。Resolver・protected PlayerTrade・非委譲SurfaceはGate Actor sampleではない。ExecutorはStep 3A-14のSettlement `_best_settlement()`、City `_best_city()`、RoadとBankTradeはB_FROZEN_PPO、Development購入とEndTurnは直接Actionのまま。Rewardも勝+1、敗-1、VP差+0.05、gamma .99のまま。T=2は全episode・updateを通じて固定し、PlayerTrade/Robber/card使用への委譲拡張はしていない。

両artifactはGate/Criticのstate_dictとmanifestを持つ。manifestにChampion親ID・digest、Gate schema、Context `runtime_context_v1`/79次元、v2 prefix + v7、family順、Executor/Resolver/protected trade版、温度、報酬、seed、後者にはPPO hyperparameterとrollout範囲も保存。保存→再読込でGate/Critic重み全tensorが一致。Champion artifactには書き込んでいない。

## On-policy rolloutとpreflight

20%委譲・Rule/Champion/Mixed各24局、各profile内席1〜4各6局、計72局をepisode単位で完走し、**299 actual delegated StrategicTransition**を保存。256件を少し超えるが384件未満。候補3以上142件、native非top選択61件で、各family実選択5件以上を満たした。手動のfamily oversamplingはない。Action前v2 Observation 1667、Runtime Context 79、候補mask、native/T=2旧logits・確率・log_prob・entropy、sampled family、具体Action、凍結Champion base value、Strategic Critic valueを各実委譲Decisionへ保存した。

| family | 選択 | available | native非top選択 |
|---|---:|---:|---:|
| BUILD_SETTLEMENT | 39 | 40 | 0 |
| BUILD_CITY | 22 | 24 | 0 |
| BUILD_ROAD | 54 | 163 | 3 |
| BUY_DEVELOPMENT | 81 | 117 | 2 |
| BANK_TRADE | 61 | 160 | 14 |
| END_TURN | 42 | 299 | 42 |

候補数は2候補157、3候補91、4以上51。native top238/299、非top61/299。選択Policyのentropyは平均 `.4713`、中央値 `.5178`、p95 `1.0245`。macro duration（既存learner policy step）は平均13.59、中央値11、p95 37、最大70。macro rewardは平均 `.0537`、標準偏差 `.4466`、範囲 -1.0〜1.148。terminal transition68件はreward平均 `.0834`・標準偏差 `.9305`、非terminal231件は平均 `.0450`・標準偏差 `.0548`。terminal分散が大きい点を維持監視する。transition内を横断したprotected tradeは延べ559、非委譲Surface899、Resolver31。Resolver発火は32件で、そのActionはActor bufferに混ぜていない。illegal/Executor失敗/truncation/nonfinite rewardは0。

semi-MDPの境界は「実際にGateが選んだ局面→次に実際にGateが選ぶ局面、または真の終局」。途中の固定controllerや対戦相手ActionをActor sampleへ含めない。各macro rewardは `R_i=Σ_{j=0}^{k-1} gamma^j r_{i,j}`、discountは `gamma^k`、`k` はlearner policy step数。terminal bootstrapは0。GAEはStep 3A-9と同じ `δ_i=R_i+gamma^{k_i}V_{i+1}-V_i`、`A_i=δ_i+gamma^{k_i}λ A_{i+1}`、`λ=.95` を**連続する実委譲Decision**間で使用。time-limit truncationはterminal扱いせずtraining bufferから除外。全299件で報酬列からの再計算・次実委譲mask/step境界・保存log_prob・Critic valueを検証した。

| update前Advantage | raw | normalized |
|---|---:|---:|
| mean / std | .3864 / .7482 | ≈0 / 1.0000 |
| min / p5 | -.7557 / -.5346 | -1.5264 / -1.2309 |
| median / p95 / max | .5042 / 1.4967 / 1.7859 | .1575 / 1.4841 / 1.8706 |

terminal Advantageはn=68、平均 `.4550`、std `.9339`、中央値 `-.1697`。非terminalはn=231、平均 `.3661`、std `.6827`、中央値 `.5333`。family別raw Advantage平均（n）はSettlement `.643` (39)、City `.633` (22)、Road `.346` (54)、Development `.367` (81)、BankTrade `.207` (61)、EndTurn `.369` (42)。異なるstate distributionから選ばれたfamilyの平均をTeacher scoreや固定優先ruleにしない。NaN/inf、ほぼゼロstd、単一極端outlierはなし。保存旧calibrated logits最大誤差0、確率/価値最大誤差は約1.2×10^-7、log_prob最大誤差2.4×10^-7。optimizer parameter IDはGate/Criticのみで、Championと交差しない。

## ただ1回のupdate

Gate encoder/residual 30,982 parameter、Strategic Critic encoder/residual 30,657 parameter、合計61,639のみ更新した。Champion GNN、377 Actor/15-family logits、base Critic、v7の377残差、Runtime Context、mask、Executor、Resolver、protected controllerは凍結。Actor LR `1e-4`、Critic LR `3e-4`、batch64、2 epoch、clip `.1`、entropy coefficient `.005`、合計10 minibatch。PPO ratioは保存済み**T=2旧Policy**に対する同mask T=2新Policyで計算し、T=1 nativeを基準にしない。再update防止のattempt記録を保存済み。

| update metric | 値 |
|---|---:|
| 判定 | ACCEPTED |
| Actor loss平均 | -.00224 |
| Critic loss平均 | .67260 |
| entropy loss平均 | -.00235 |
| clip fraction平均 | 0 |
| PPO ratio mean / p5 / median / p95 / max | 1.0004 / .9891 / 1.0000 / 1.0232 / 1.0322 |
| old→new masked KL mean / median / p95 / max | .0000408 / .0000339 / .0001214 / .0001634 |
| 全体entropy before→after | .4713 → .4730 |
| 最高family確率 >.95 の固定probe state | 70 → 70 |

候補数別KL平均は2候補 `.0000481`、3候補 `.0000386`、4以上 `.0000219`。同じAction前state・同じmaskで全6 familyのCategorical KLを計算した。計算上の極小負KL（最小 -1.1×10^-7）はfloat32丸め誤差。警戒線mean `.01` / max `.10` より十分低く、採用した。候補2/3/4+のentropyはそれぞれ `.4846→.4878`、`.5373→.5372`、`.3129→.3132` でcollapseなし。固定rollout局面でfamilyのtop-1数はSettlement40→40、City23→23、Road87→86、Development82→82、BankTrade65→66、EndTurn2→2。family平均確率の変化も概ね数千分の一で、特定familyへの急激な集中はない。Critic/Actorを別encoderにしたためCritic lossはActor表現を直接更新しない。

| 固定rollout局面のfamily available時平均確率 | before | after |
|---|---:|---:|
| BUILD_SETTLEMENT | .9597 | .9595 |
| BUILD_CITY | .9136 | .9118 |
| BUILD_ROAD | .3639 | .3603 |
| BUY_DEVELOPMENT | .6698 | .6710 |
| BANK_TRADE | .3566 | .3554 |
| END_TURN | .1470 | .1493 |

## 別seed評価

Safety ChampionはResolver ON・Gate 0%。Pre/PostはResolver ON、protected trade ON、T=2で、20%とFull Surface (100%) を**同じ評価seed**で比較した。A→Bはprior/委譲の差、B→Cだけが今回のupdate差である。各variantはRule/Champion/Mixed各12局、席1〜4各9局。36局の勝数差を強さの確定判断には使用しない。

| variant | 勝数/36 | 平均VP | 平均rank | 平均policy steps | Gate実委譲 | settlement / city / road | dev購入 / bank trade / EndTurn / PlayerTrade |
|---|---:|---:|---:|---:|---:|---|---|
| Safety Champion | 19 | 8.194 | 1.889 | 74.0 | 0 | 106 / 61 / 225 | 51 / 141 / 602 / 447 |
| T=2 Pre 20% | 16 | 8.083 | 2.042 | 71.4 | 138 | 101 / 58 / 217 | 63 / 139 / 590 / 383 |
| T=2 Post 20% | 16 | 8.083 | 2.042 | 71.4 | 138 | 101 / 58 / 217 | 63 / 139 / 590 / 383 |
| T=2 Pre 100% | 14 | 7.556 | 2.125 | 81.9 | 723 | 80 / 33 / 229 | 153 / 136 / 636 / 449 |
| T=2 Post 100% | 14 | 7.528 | 2.125 | 81.2 | 718 | 79 / 33 / 226 | 152 / 135 / 631 / 447 |

全variantが36/36終局、truncation・illegal・Executor失敗・deadlock・非有限rewardは0。更新前後で、20%では36/36局の委譲family/具体Action列と最終結果・action family件数が同一。100%では35/36局が同一で、唯一 seed `3262008` (Mixed) はPreのVP4/66 policy stepsからPostのVP3/41 stepsへ変化し、両者とも順位4・勝者は相手P3。よって更新そのものは今回の評価seed上でほぼ行動へ反映されず、勝率改善を主張できない。

| Gate診断 | Pre 20% | Post 20% | Pre 100% | Post 100% |
|---|---:|---:|---:|---:|
| selected Settlement / City / Road | 23 / 11 / 18 | 23 / 11 / 18 | 74 / 31 / 197 | 73 / 31 / 194 |
| selected Development / BankTrade / EndTurn | 27 / 30 / 29 | 27 / 30 / 29 | 153 / 136 / 132 | 152 / 135 / 133 |
| native非top選択 | 38 | 38 | 178 | 178 |
| entropy候補2 / 3 / 4+ | .501 / .602 / .326 | .504 / .602 / .327 | .518 / .579 / .373 | .521 / .577 / .373 |
| Champion family / concrete Action一致率（参考） | 60.1% / 47.8% | 60.1% / 47.8% | 53.9% / 38.3% | 53.8% / 38.2% |
| Resolver発火 / protected preemption | 15 / 360 | 15 / 360 | 13 / 417 | 13 / 415 |

**重要な分離:** B→C（学習効果）は小さく、大幅な崩壊は観測されなかった。一方、A→Bのうち特に100%委譲は、Safetyから建設合計167→113（-54、約32%減）、発展購入51→153（3倍）、平均VP8.19→7.56、平均policy steps74→81.9であり、初期T=2 priorを全面委譲したときの明確な懸念である。対ChampionではSafety 5/12勝・平均VP7.50に対しPre Full 2/12・6.17、Mixedでは3/12・7.25に対し2/12・6.83。今回の1 updateがこのbaseline問題を解消したわけではない。

Full SurfaceのGate 723委譲局面では、Champion counterfactualがDevelopment購入を選ぶのは44件だったのに対し、Gateは153件購入した。Gate購入153件のChampion counterfactualはRoad56、EndTurn38、Development42、BankTrade15、Settlement1、City1。この不一致はChampion模倣を目標にすべきという意味ではなく、Development過多と建設減少の要因候補を示す診断である。購入が建設資源を減らして後続の都市化を阻害した可能性はあるが、異なる到達state・他Actionとの相互作用があるため、ここでは因果断定しない。family別return平均をTeacher scoreへ変換したり、Dev抑制・建設bonus・Executor ruleを追加したりはしない。

## 判断

**今回の目的である「安全なT=2 priorから、実委譲returnのみでGate residualと専用Criticを1回更新できるか」は成立**。更新は受理され、mask/on-policy log_prob/値・semi-MDP境界・凍結範囲・KL・entropy・対局完走を確認した。ただし、**Step 3A-17で追加Strategic RLを自動継続するGateは保留**。Full Surfaceで建設数が大きく減りDevelopment購入が急増する問題はPre-updateから存在し、Post-updateでも残る。追加updateだけで直ると仮定するには早い。

次は学習を回す前に、Full SurfaceのDevelopment購入・建設減少をAction前Context、native family確率、候補mask、資源消費後の建設可能性、Executor Action、Champion counterfactual、終局寄与に分けて診断することを推奨する。特に「GateがDevelopmentを選んだため後のCity機会を失ったか」と「元の15-family priorが6-way maskへ写ったときの尺度問題か」を同seed軌跡と必要ならpaired continuationで検証する。これは次の価値判断学習範囲・rollout配分の判断材料であり、Champion模倣、Reward変更、手製優先rule、Executor変更を先に行う提案ではない。

実装は `strategic_gate_ppo_step3a16.py`（preflight、semi-MDP GAE、1回のPPO、artifact）、`run_strategic_gate_step3a16.py`（seed分離した段階別実行・集計）、`strategic_gate_rollout_step3a14.py` と `strategic_gate_step3a13.py` の診断拡張。詳細数値は `experiments/strategic_gate_step3a16_20261005/` のrollout、preflight、update結果、5評価summary、比較JSONへ保存した。最終コードで全322テスト成功。
