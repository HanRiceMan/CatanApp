# Step 3A-17 — Strategic Gate の発展購入過多・因果監査

2026-10-05。2回目の PPO update は行っていない。Reward、Strategic candidate mask、Frozen Executor、Champion artifact も変更していない。主対象は Step 3A-16 の **T=2 pre-update、Full Surface** 36局（Rule / Champion / Mixed 各12局、各席3局）。保存済みのAction列、勝敗、VP、順位、policy step、RNG counterと同seedで再生した36局は全て一致した。監査hookはAction適用前のcloneとfrozen logitsだけを読む。

## 結論

発展購入過多の直接の説明は、**元の15-family head自体が、発展を買えるが即時建設はできない状態でDevelopmentを圧倒的に上位へ置くこと**にある。6-way maskによる確率の膨張は、Devを選んだ153状態では **0**。T=2もむしろ平均Dev確率を0.983から0.918へ下げている。元のfamily headは合法377 Actionをfamilyとfamily内候補に分解するためのもの（`hierarchical_distribution.py`）で、Teacher Action蒸留と通常のPPOを経ている（`train_strongest_league.py`）。6つの戦略的投資先の機会費用を直接学習したheadではない。

一方、代表24状態×8共通未来乱数=192 paired continuationでは、Devを一貫して悪いとも良いとも判定できない。平均 `return(DEV)-return(non-DEV)` は **-0.0093**、future単位の標準偏差 **0.597**。局面平均が実質同点（±0.02）は4/24、Dev優位9/24、代替優位11/24。局面内の未来乱数分散が局面平均間の観測分散より大きく、補正後の局面間分散は0だった。**次のPPO updateを自動実行しない。**

## Devを選んだ153 Decisionの分布

| 項目 | 結果 |
|---|---:|
| Rule / Champion / Mixed | 57 / 39 / 57 |
| Dev available / Dev selected（全Full Surface） | 205 / 153、75% |
| 6-way native top-1 / top-2 | 151 / 2 |
| P(Dev) >.9 / .7–.9 / .5–.7 / ≤.5 | 114 / 37 / 0 / 2 |
| Champion counterfactual: Road / End / Dev / Bank / Settlement / City | 56 / 38 / 42 / 15 / 1 / 1 |
| Devと即時Settlement / Cityが同時合法 | 1 / 1 |
| Devコストが即時建設合法性を消した | 2 |
| 既存のnew development cardを持つ状態 | 15 |

Candidate set別の選択時平均P(Dev): `Dev+End` 64件・0.973、`Road+Dev+End` 39件・0.909、`Dev+Bank+End` 26件・0.909、`Road+Dev+Bank+End` 22件・0.863。建設が同時候補の2件は0.079 / 0.039だった。したがって今回の建設減少は「同じDecisionでDevが即時建設を奪う」ケースの多発ではない。

| 確率の分解（Dev選択153状態の平均） | P(Dev) |
|---|---:|
| 元の合法15-family、T=1 | 0.9827 |
| 元の合法15-family、T=2 | 0.9179 |
| 6-family mask、T=1 | 0.9827 |
| 6-family mask、T=2 | 0.9179 |

同じT=2でのmask inflationは平均・最大とも0、比は全件1.0。除外した合法familyの確率massも全件0。元の15-family headの対応logit（そのfamilyがcandidateに存在した局面だけ）は以下の通り。

| family | n | 平均 | SD | 中央値 | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|
| Dev | 153 | 2.659 | 0.150 | 2.646 | 2.458 | 2.858 |
| Road | 63 | -2.628 | 0.276 | -2.624 | -3.005 | -2.268 |
| Bank | 49 | -2.910 | 0.214 | -2.925 | -3.168 | -2.584 |
| End | 153 | -4.466 | 0.483 | -4.506 | -5.092 | -3.809 |
| City | 1 | 7.771 | — | 7.771 | — | — |
| Settlement | 1 | 8.920 | — | 8.920 | — | — |

City/Settlementは各n=1で一般化不可。Devが合法で即時建設がない局面では、この大きなlogit差によりDevを選びやすい。`native15_logits` 全値、legal15 mask、T別確率、removed mass、pre/post資源は `dev_decisions.json` に保存した。

Dev購入は常に sheep/wheat/ore を1枚ずつ消費する。即時建設を失ったのは2件のみだが、購入後の建設遅延は別問題である。153件中、手札は平均7.44枚、38.6%が7破棄対象、Settlement候補は平均7.88、最少追加道路は有効142件で平均0.92本。候補があることと今建てられることを分けた。Devが合法なら候補になる現maskはルール上正しく、**Strategic opportunity として広すぎるか**はこの観測だけでは決められない。

## Paired continuation

153件から確率帯、即時建設競合、Champion Road/End、profileを層別して24状態を選んだ（各profile 8）。`.5–.7`帯は母集団に存在しない。代替は同じpre-action stateでDevを一時的に除いたT=2 Gateの最上位familyを、既存Frozen B Executorで具体Action化した。代替内訳は Road13 / End6 / Bank3 / Settlement1 / City1。両branchは同じfuture seed（8種/状態）で開始し、最初のAction以降は**同じfrozen Champion Planner**へ戻した。したがってこれは「一Actionの局所的介入」であり、Full Surface Gate全体の価値推定ではない。未来乱数はbranchごとに同じseedへ置換し、当初の盤面・手札・カード山は固定した。

| 指標（Dev − 代替） | 結果 |
|---|---:|
| discounted environment return 平均 / SD / 中央値 | -0.0093 / 0.5973 / -0.0159 |
| 24個のstate meanのSD | 0.1322 |
| Dev優位 / 代替優位 / ±0.02同点（state mean） | 9 / 11 / 4 |
| 両符号が少なくとも2 seedずつ出る状態 | 15 / 24 |
| 7/8以上で同符号 | 8 / 24 |
| 概算95% CIが0を除く状態 | 3 / 24 |
| winner変更 / 学習者のwin変更 | 58 / 192、37 / 192 |
| Dev win / 代替 win | 110 / 111 |
| final VP差 / rank差の平均 | +0.016 / +0.003 |
| policy steps差の平均 | +1.42 |

局面内sample varianceの平均は0.3864。局面平均間の観測varianceは0.0175で、局面内variance/8の0.0483より小さい。補正後のbetween-state varianceとICC-likeは0。8 seedだけで個別局面の価値差を教師ラベルにしない。自己勝敗が反転した37組でdiscounted returnの符号が自己勝敗と逆になった例は0であり、現Rewardが系統的に勝敗と逆向きだという証拠はない。

最初の30 learner policy steps以内にSettlementを建てた組はDev branch 171/192、代替173/192、Cityは89/192対96/192。両branchで建設した組の初回建設step差はSettlement平均 **+2.25** (n=181)、City **+3.58** (n=125)。Road建設数差は30 step内で平均-0.125。Devコストによって短期の資源回復と建設タイミングが遅れる**傾向**はあるが、最終returnの安定した悪化は確認できない。

資源回復は「Action前の各資源枚数へ再到達するまで」の診断proxyであり、建設費の充足と同義ではない。30 step以内に回復した組で、Dev branchの中央値は sheep 7、wheat 5.5、ore 8 steps、代替branchは各1 step。代替が元々その枚数を保持するため、この比較をそのまま実用的な建設ETA差とは解釈しない。初回建設stepとresource vectorの生データをpaired artifactに保持した。

## 仮説と次の判断

| 仮説 | 判定 |
|---|---|
| H1: 15→6 maskによるDev確率膨張 | **否定**。153件でremoved mass=0、mask inflation=0。ただし元の15-family prior自体が戦略6択として未較正で、Devを非常に強く選ぶ。 |
| H2: Devが実際に有利 | **未確定**。paired平均ほぼ0、Dev有利・代替有利が混在。Champion disagreementはTeacherではない。 |
| H3: 建設資源の機会損失 | **短期では支持**。直接の即時建設喪失2件、pairedで建設の遅れを観測。ただし終局returnの一貫した悪化ではない。 |
| H4: 到達stateのshift | **有力だが未分離**。Full SurfaceはSafetyより建設167→113、Dev51→153。直接の即時建設競合は2件だけなので後続の状態変化が重要。ただしDev以外のGate行動も変わっており、差分をDevだけへ帰属できない。 |
| H5: candidate maskの誤り | **合法性バグなし**。候補は377 legal maskから生成。戦略的に候補が広すぎる可能性は残るが、新しい禁止ruleの根拠にはしない。 |

Runtime Context上のstate mean Δreturnとの探索的Pearson相関は、Dev確率0.41 (n=24)、score -0.18、hand size -0.22、Settlement ETA 0.14、City ETA 0.08、blocked pips -0.11等。多数の比較と小標本・未来乱数分散を考えると、特定ContextがDev優劣を説明したとは言えない。v7には資源、建設ETA、称号差、盗賊圧力、新規発展カード情報がある。今回の結果はContext不足を証明しないが、相手belief・当手番trade履歴等を不要とも証明しない。これらは保護Executionの判断には引き続き必要である。

**Step 3A-18案:** 追加PPO、Reward改変、Dev禁止mask、Executor改変を保留する。まず同seed Safety/Full軌跡の初回分岐と、その後の建設可否・資源経路・Dev再購入を時系列で比較し、H4をDev介入と他Gate行動へ分解する。次に、元family headを「safe strategic prior」とみなす設計を見直し、合法候補順位と探索確率を保持した複数の*診断用* prior を、学習なし・固定seedで評価する。価値が曖昧なstateは必要ならfuture seedを増やす。候補maskやRewardを先に手直しせず、returnと行動崩壊を別々に比較してから次RLを決める。

## 変更・検証・再現

- `app/rl/strategic_dev_audit_step3a17.py`: 15/6-family確率分解、前後資源・Context、層別選択、paired future、集計。
- `app/rl/run_strategic_dev_audit_step3a17.py`: 保存済み36局のT=2 pre-update再生、parity、24×8 checkpoint、compact artifact、再集計。
- `app/rl/strategic_gate_rollout_step3a14.py`: optional read-only pre-action audit callback。既存既定動作は不変。
- `tests/test_rl_strategic_dev_audit_step3a17.py`: temperature/mask分離、mask整合、profile-balanced選択。
- `experiments/strategic_dev_audit_step3a17_20261005_balanced/`: `dev_decisions.json`、`selected_states.json`、`paired_continuations.json.gz`、`replay_parity.json`、`summary.json`。個別pair checkpointはローカルのみ。

全テスト **330 passed、1163 subtests passed**（既存dependency警告2件）。36局の保存済み行動・結果・RNG再生parity、192 paired branchの合法・終局、Champion weight digest `4225902d41a83540eb1f1552826ec623706e778b99ad4ae8ec015f976644392a`不変、pre-update Gate weights不変を確認した。PPO optimizerは生成・実行していない。
