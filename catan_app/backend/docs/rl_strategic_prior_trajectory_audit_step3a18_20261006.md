# Step 3A-18 — Strategic prior / trajectory shift 監査

2026-10-06。追加PPO update、Reward変更、candidate mask変更、Frozen Executor変更、Champion変更は行っていない。比較するGateは **Step 3A-16のT=2 pre-update（residual=0）** であり、post-update効果を混ぜない。Safety、Full T=2、各priorのscreenはRule / Champion / Mixed各12局、seat 1～4各profile内3局の同一36 game seed（3260000台 / 3261000台 / 3262000台）を使う。外部delegation / family抽選seedはGameState RNGから分離した。

## 監査設計と境界

- 初回分岐は、Action前から各learner boundaryを追跡して、SafetyとFullの**最初に異なる具体Action**を検出した。Champion final ActionでSurfaceを定義していない。Action前Observation v2、Runtime Context79、candidate mask、native logits、T=2確率、Champion/Full Actionを保存。
- 36局の再生はStep 3A-16保存済みSafety/Pre-Fullと、勝者・得点・policy steps・RNG counter・action-family件数・実委譲件数で照合して全件一致。初回差の検出indexも再生した2本の全Action traceの真のfirst mismatchと全件一致した。
- 初回分岐後30 learner policy stepsは両軌跡を別々に記録。Ready / ETA / hand等はAction前の決定的Contextであり、Teacherの価値ラベルではない。
- paired continuationは同じAction前GameStateから2 branchを作り、最初の観測済みSafety/Full Actionを各々適用した後、各controllerを**終局まで継続**する。8 future seedを同一stateの2 branchで共有する。これはStep 3A-17の「一Actionのみ介入、以後両方Champion」とは異なる。GameStateの未来RNGのみ同seedへ置換し、盤面・手札・デッキとreward/policy-step境界はコピーする。`SettlementPlanningAgent`は設定以外の持続内部状態を持たず、履歴はGameState側へ保持される。
- 先験分布のoffline比較はStep 3A-13の同一Shadow 1,123状態、Dev確率はStep 3A-17のDev選択153状態で実施。順位は**frozen native logitsのみ**から作り、Planner Action / Scoreは入力していない。tieは固定family index順。
- Full Surface screenは学習なし。安全層、protected PlayerTrade、6 family、候補mask、Frozen B ExecutorはStep 3A-16と同一。rank / gapは全family共通の変換で、Dev固有補正はない。

元の15-family headは `hierarchical_distribution.py` の `P(family)×P(action|family)` を通じて377 Action分布を作るためのもの（22–23、57–64行）である。`train_strongest_league.py` のfamily蒸留/通常PPOは、このheadを6候補の長期機会費用へ較正する目的ではない。Step 3A-17で6-way mask inflation=0と確認済みなので、問題はmaskではなく**native logit gapをStrategic value gapと同一視した仮定**にある。

## 初回分岐と30-step cascade

36/36局にfirst concrete Action divergenceがあり、初手Full=BUY_DEVELOPMENTは**9局**だけだった。first-divergence family pairは次の通り。

| Safety → Full | 局数 |
|---|---:|
| Road → Road（edge差） | 10 |
| Road → EndTurn | 8 |
| Road → Dev | 7 |
| EndTurn → Dev | 2 |
| EndTurn → BankTrade | 4 |
| その他 | 5 |

既存36局の最終結果差はFull−Safetyで、学習者勝利 **-5局**、VP平均 **-0.639**、順位平均 **+0.236**、policy steps平均 **+7.92**。winnerが異なる局は13/36。ただしこれは各局1本のfutureであり、初手差の因果効果としては扱わない。

| 最初の分岐から30 learner steps、action phase件数 | Safety | Full T=2 |
|---|---:|---:|
| Settlement | 50 | 29 |
| City | 8 | 10 |
| Road | 118 | 101 |
| Development | 19 | 55 |
| BankTrade | 34 | 38 |
| EndTurn | 289 | 276 |
| Protected PlayerTrade | 144 | 136 |

この窓で、初回Devを除く追加Dev購入はFull平均 **1.28/局**、Safety **0.53/局**。全learner boundaryをDecision-weightedでプールしたSettlement即時合法率はSafety **4.6%** / Full **3.0%**、Settlement ETA（validのみ）平均は **1.41 / 1.63**、City ETAは **1.95 / 2.25**、hand sizeは **5.43 / 5.04**。これは異なる到達stateの記述比較であり、初回Dev購入単独の効果ではない。

最初の分岐を分けた場合、30-stepのFull−Safety建設件数差はRoad→Dev **-8**（7局）、Road→EndTurn **-6**（8局）、Road→Road edge差 **-7**（10局）。Dev購入件数差はそれぞれ **+13 / +8 / +8**。同じRoad family内のExecutor地点差でも後続行動が変わるため、prior選択だけへ原因を限定できない。

## Frozen priorのoffline比較

変換は、Native `logit/2`、Rank `-β×rank`、Gap `max(logit−max_available,−C)`、比較用mixture `(1−ε)P_nativeT2+ε Uniform(mask)`。mixtureも診断時は1つのCategorical分布として評価し、後付けepsilon Action overrideではない。

| Prior | entropy | alternative mass | candidate 4+ entropy | Dev選択153状態 P(Dev) | native top保持 |
|---|---:|---:|---:|---:|---:|
| Native T=2 | 0.514 | 0.237 | 0.361 | 0.918 | 100% |
| Rank β=.5 | 0.873 | 0.442 | 1.268 | 0.544 | 100% |
| Rank β=1 | 0.722 | 0.304 | 0.955 | 0.684 | 100% |
| Rank β=1.5 | 0.554 | 0.199 | 0.668 | 0.790 | 100% |
| Gap C=1.5 | 0.730 | 0.301 | 1.150 | 0.724 | 100% |
| Gap C=2 | 0.632 | 0.245 | 0.953 | 0.806 | 100% |
| Gap C=3 | 0.486 | 0.185 | 0.557 | 0.910 | 100% |
| Mix ε=.05 | 0.566 | 0.254 | 0.489 | 0.891 | 100% |
| Mix ε=.10 | 0.610 | 0.272 | 0.596 | 0.865 | 100% |

| Prior | top確率 | 2位確率 | effective候補数 `exp(H)` | Dev状態でP(Dev)>.9 / >.7 / >.5 |
|---|---:|---:|---:|---:|
| Native T=2 | .763 | .199 | 1.750 | 114 / 151 / 151 |
| Rank β=.5 | .558 | .338 | 2.458 | 0 / 0 / 129 |
| Rank β=1 | .696 | .256 | 2.082 | 0 / 64 / 151 |
| Rank β=1.5 | .801 | .179 | 1.745 | 0 / 151 / 151 |
| Gap C=1.5 | .699 | .211 | 2.146 | 0 / 64 / 151 |
| Gap C=2 | .755 | .179 | 1.933 | 0 / 151 / 151 |
| Gap C=3 | .815 | .148 | 1.671 | 129 / 151 / 151 |
| Mix ε=.05 | .746 | .210 | 1.824 | 65 / 151 / 151 |
| Mix ε=.10 | .728 | .220 | 1.893 | 64 / 151 / 151 |

全分布（top/second確率、effective action count、候補数別entropy、family available時の平均/中央値/p10/p90、Dev確率>.9/.7/.5の件数）は `offline_prior.json` に保存。特にRank β=.5やGap C=1.5は強く平坦化し、初回Full screen候補から外した。Native T=2、Rank β=1/1.5、Gap C=2を同seedでscreenした。

| 同一1,123状態、available時の平均family確率 | n | Native T=2 | Rank β=1 | Rank β=1.5 | Gap C=2 |
|---|---:|---:|---:|---:|---:|
| Settlement | 97 | .964 | .652 | .782 | .732 |
| City | 72 | .924 | .642 | .759 | .724 |
| Road | 616 | .365 | .474 | .499 | .449 |
| Development | 403 | .775 | .603 | .684 | .679 |
| BankTrade | 666 | .370 | .425 | .430 | .442 |
| EndTurn | 1,123 | .159 | .174 | .109 | .139 |

Rank/gapはDev確率だけでなく、即時建設が可能な局面でのSettlement/City確率も大きく下げる。したがって「Devを下げるprior」は「建設を増やすprior」と同義ではない。

## 学習なしFull Surface screen

| 36局 | Win | 平均VP | 平均rank | 平均steps | Settlement+City | Dev | EndTurn | BankTrade | 終局 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Safety Champion | 19 | 8.194 | 1.889 | 74.0 | 167 | 51 | 602 | 141 | 36/36 |
| Native T=2 | 14 | 7.556 | 2.125 | 81.9 | 113 | 153 | 636 | 136 | 36/36 |
| Rank β=1 | 14 | 7.194 | 2.292 | 87.6 | 118 | 120 | 704 | 184 | 36/36 |
| Rank β=1.5 | 13 | 7.500 | 2.222 | 84.4 | 116 | 132 | 687 | 194 | 36/36 |
| Gap C=2 | 14 | 7.278 | 2.389 | 84.2 | 116 | 134 | 685 | 188 | 36/36 |

全variantでillegal / Frozen Executor failure / deadlock / truncation / nonfinite reward は0。rank/gapはDev購入を減らしたが、EndTurnやBankTradeを増やし、建設回復は3～5件のみ。36局は設計screenであり、勝率差の統計的優劣を意味しない。candidate構成別選択件数、即時建設との競合、family総件数は `screen_summary.json` に保存した。即時建設candidateを含むGate Decisionでnon-buildが選ばれた件数はNative T=2 **3**、Rank β=1 **57**、Rank β=1.5 **34**、Gap C=2 **34**。平坦化で別の機会損失が生じ得るため、Dev件数だけでpriorを選ばない。

保存した実委譲Decisionの確率と、同じpre-action logit/maskからoffline再計算した確率の最大絶対差は、Rank β=1 `7.9e-8`、β=1.5 `6.9e-8`、Gap C=2 `8.4e-8`。Gate residual=0のまま、意図したprior分布からsampleできている。

| Candidate set | Native T=2 | Rank β=1 | Rank β=1.5 | Gap C=2 |
|---|---:|---:|---:|---:|
| Dev + End | 64/66 | 39/58 | 58/79 | 57/66 |
| Road + Dev + End | 39/43 | 26/41 | 27/34 | 31/37 |
| Dev + Bank + End | 26/28 | 25/40 | 28/35 | 28/35 |
| Road + Dev + Bank + End | 22/25 | 10/24 | 14/18 | 14/25 |

分子はDev選択、分母はそのcandidate setへ到達したDecision数。priorごとに到達stateが違うため、これは同一stateの条件付き効果ではない（同一stateの確率比較は上記offline表）。

## Policy-level paired continuation

初回分岐36状態からprofile各8、計24状態を選択した。内訳はRoad→Dev 7、Road→EndTurn 6、Road→Road edge差5、EndTurn→Dev 2、EndTurn→BankTrade 2、BankTrade→Road 1、BankTrade→BankTrade 1。**BankTrade→Devは元36局のfirst divergenceに存在せず、作っていない。** 各状態8共通future seed、合計192組（384 branch）。すべて合法・真のterminalへ到達し、truncation 0。最初のAction以後、Safety/Fullそれぞれのcontrollerを継続した。

| Full T=2 − Safety | 結果 |
|---|---:|
| discounted environment return 平均 / SD / 中央値 | **-0.1785 / 0.5082 / -0.1069** |
| 24 state meanの平均 / SD | **-0.1785 / 0.2369** |
| Full優位 / Safety優位 / ±.02同点（state mean） | **5 / 18 / 1** |
| 7/8以上で同符号（全てSafety優位） | **5 / 24** |
| 両符号が2 future以上ある状態 | **17 / 24** |
| winner差 / 学習者win | **81/192** / Safety **92** 対 Full **68** |
| final VP差 / rank差の平均 | **-0.724 / +0.307** |
| policy steps差の平均 | **+9.51** |
| 平均within-state sample variance | **0.2310** |
| observed between-state sample variance | **0.0586** |
| within/8控除後のbetween-state variance | **0.0297** |

future乱数・Gate抽選の影響は依然大きいが、補正後のstate間分散は0ではなく、24局面中18局面の平均がSafety寄り。選択したfirst-divergence状態でFull Policy継続が不利な方向という**診断的証拠**はある。ただし状態は全分布からの無作為標本ではなく、各状態8 futureに過ぎない。1状態ごとのΔreturnをTeacher labelにはしない。

| 最初の分岐 | state数 | Full−Safetyの平均return差 |
|---|---:|---:|
| Road→EndTurn | 6 | **-0.398** |
| Road→Dev | 7 | -0.167 |
| Road→Road edge差 | 5 | -0.085 |
| EndTurn→Dev | 2 | +0.102 |
| EndTurn→BankTrade | 2 | -0.147 |
| BankTrade→Road | 1 | -0.249 |
| BankTrade→BankTrade | 1 | +0.033 |

群が小さいため、この表から固定Action ruleを作らない。特にRoad→EndTurnには8/8でSafety優位の状態が2、7/8が1あり、Dev以外の初回分岐でも悪化が現れる。初回Devだから必ず悪いという結果ではない。

Step 3A-17の一Action Dev−代替（以後どちらもChampion）は平均 **-0.0093**、state間補正分散0だった。ただし**代表状態・比較Action・継続controllerが今回と異なる**ので、2つの平均値を同一実験の処置効果として差し引かない。定性的には「単発Devが一律に悪い証拠はないが、Full Strategic Policyを継続すると到達分布と結果が悪化し得る」というH4/compounding-errorの説明と整合する。

## 判定とStep 3A-19

1. **native logit magnitude問題は実在する。** 15-family headは377 Actionの階層分布のために訓練され、6投資先の機会費用を直接表すものではない。順位はrank/gap変換でも100%保存でき、Dev状態のP(Dev)を下げられる。ただし順位だけでも価値較正はできない。
2. **今回のrank/gap priorは採用しない。** Dev購入は減ったが、即時建設stateの建設確率まで下がり、36局Full screenの建設は116～118でNative113から小幅増にとどまる。Safety167との差は解消せず、VP/rankにも明瞭な改善はない。Devを減らすことを目的関数にしていない。
3. **H4 trajectory shiftは支持、単一原因には未分解。** first divergenceでDevは9/36のみ。Road→Roadの10局やRoad→EndTurnの8局でも後続建設が減りDevが増えた。Policy-level 192 pairedではFull平均return -0.179、state meanの18/24がSafety優位。一方、局面内future varianceも大きく、個別stateへの確定判断は避ける。
4. **Context / Reward / candidate maskの変更根拠はまだ不足。** v7の79 Contextが不足している証拠も、現Rewardが勝敗と逆向きという証拠もない。候補maskは合法性を満たす。Executor BのRoad具体edge差は原因候補だが、今回Executorは変えない。
5. **追加PPO updateは保留。** Full Surface崩壊がprior変換だけで解消されず、このままon-policy updateを追加しても原因が混ざる。低delegation curriculumは将来再開時の安全策として必要だが、今回の結果だけで学習を再開しない。

**Step 3A-19案（まだ実装しない）:** (a) 同family Road→Roadのfirst-divergence状態で、Frozen B edgeとChampion family内edgeを同一state・複数futureで比較し、Executor由来の到達差を単独監査する。(b) Road→EndTurn / Road→Devを分け、Safety・低委譲10～20%・Fullの同seed継続で分岐回数とreturnのdose-responseを測る。(c) これらを踏まえて初めて、priorの学習可能な較正、Gateのcredit assignment、Road候補分解の必要性を設計する。Reward bonus、Dev penalty、Dev mask、Champion BCは追加しない。

## 変更・検証

- `app/rl/strategic_prior_step3a18.py`: native順位、共通gap clip、比較用mixtureの副作用なしprior計算と同state offline監査。
- `app/rl/strategic_gate_rollout_step3a14.py`: 既定動作を維持したまま、任意のAction前observer / prior calibrator / state continuation / future RNG / 初手介入、およびdiscounted return記録を追加。optimizerは使わない。
- `app/rl/run_strategic_prior_step3a18.py`: 36局trajectory、paired policy continuation、36局prior screen、checkpointと集計。
- `tests/test_rl_strategic_prior_step3a18.py`: mask、top保持、順位のscale不変性、tie、gap対称性、正確なmixture分布。
- `experiments/strategic_prior_step3a18_20261006/`: `offline_prior.json`、`trajectory_comparison.json.gz`（Action前Observation/Contextと30-step軌跡）、`trajectory_cascade_summary.json.gz`、`policy_pairs.json`、`trajectory_summary.json`、`policy_extra_summary.json`、`screen_summary.json` をGit管理。圧縮前JSONと個別game/pair checkpointは再実行用ローカル生成物としてGit管理しない。

全テスト **336 passed、1172 subtests passed**（既存dependency warning 2件）。Champion weight digest `4225902d41a83540eb1f1552826ec623706e778b99ad4ae8ec015f976644392a` は前Stepと一致。Gate/Critic artifactは読取のみで、PPO parameter updateは0回。
