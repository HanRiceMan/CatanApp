# Step 3B-8 — Plan-Conditioned Action Policy

## 判定

**技術的には成功、Strategic influence は未達。** 可変Plan候補ごとに4独立Priorityとfamily分布を計算し、`P(a|s)=Σ_i p_i q_i(a)` の混合後family確率だけをPPO Actionとした。新規on-policy 265 transitionでguard付き更新を1回実行し、mask・KL・freeze・0% Safety parityは通過した。ただし、固定held-out局面でPlan/Priorityを中立化してもL3分布はほぼ変わらず、72局でPre→PostのVP・勝敗・順位改善もなかった。**「Planを学習できる形式」は確認したが、「学習したPlanが行動を実質的に左右する」は確認していない。** 新Post1は独立artifactとして保持し、委譲率・routeを拡大しない。

## 実装と意味論

- [`plan_conditioned_policy_step3b8.py`](../app/rl/plan_conditioned_policy_step3b8.py): 36-slotの各有効Planを同じ64幅MLPでscoreし、Planごとの7-family確率を生成。各Planに共通のルール上のfamily maskを使用し、IMPOSSIBLE/paddingの寄与は厳密に0、UNKNOWNは有効候補として残す。混合後にmask外確率は0、valid family確率和は1。Planをsample・argmaxせず、Plan外の得点源もmaskしない。
- Plan unavailable時は`available` flagを保持し、候補Planを偽造せず、同じshared actorの明示的fallbackで合法family分布を計算する。比較familyが全て無効ならAction分布も全0で`available=false`。勝手にEndTurnを追加しない。
- 入力はdetachしたFrozen Champion latent、Runtime Context79、VictoryPlan global facts、candidate feature9、候補別Priority4、route one-hot、合法/比較mask、Construction Commit fact。候補への手作り価値補正、Planner/Champion教師、3A native family priorはない。final family layerは平均0・標準偏差0.01の小さいランダム重み、bias 0。特定familyへの初期優遇は設けない。
- `PlanConditionedInvestmentPolicy`は既存L1/L2 Post2重みから開始し、旧L3を無理に転写せず新Actorを初期化。CriticはPost2から開始した。L1/L2を平均特徴へ潰してからL3へ渡す旧Actorとは異なり、新Actorは各候補の`q_i(a)`を先に作る。Critic向けの旧Differentiable Soft Planは維持し、ActorのPlan-conditioned経路とは分離した。
- PPOは**混合後の**`log P(selected_family|state)`のみを使う。内部Planを別Actionとして二重計上しない。単一のselected familyのlossから新L3、L2 Priority、L1 Plan Scorerへ勾配が戻る。Frozen latentへは戻らない。
- [`run_plan_conditioned_step3b8.py`](../app/rl/run_plan_conditioned_step3b8.py): restartableな固定state probe、balanced rollout、preflight、1回限りのupdate、profile別paired evaluation、独立artifact公開。学習はInvestmentの実行可能な2候補以上、10%委譲のみ。Road/BuyDev/BankTrade/EndTurnに限定し、protected PlayerTrade、ImmediateWinResolver、forced phase、Development Usage、Construction/City routeは旧境界のまま。

## 固定held-out 218局面の感度

Step 3B-7で保存済みの**学習に使わない**218 informative stateを使用。217 stateでPlanあり、212 stateで有効Planが2件以上。以下のTVはL3 family分布のTotal Variation Distance。A/B介入は最初と最後の有効Planをone-hot指定した表現能力診断であり、学習済み戦略の証明ではない。

| 比較 | 新Pre 平均TV | 新Post 平均TV |
|---|---:|---:|
| Plan uniform | 0.000000078 | 0.000000054 |
| Priority全0.5 | 0.000000165 | 0.000000951 |
| Plan/Priority両中立 | 0.000000165 | 0.000000952 |
| 有効Plan A vs B（212 state） | 0.000249 | 0.000222 |

Post2→新Preの固定state平均TVは0.00362、Post2→新Postは0.00702。これは**構造置換と更新の合成差**でありPlan影響ではない。新Pre→新PostでPriorityの平均は概ね`[0.5000,0.4992,0.4996,0.5004]`から`[0.4984,0.4995,0.5008,0.5019]`へ変化したが、Plan entropyは2.07433のまま。Plan A/Bの具体例では`(S0,C0,R1,A1)`指定時のDev/Bank/End確率`[.353405,.321793,.324803]`に対し、`(S2,C1,R0,A0)`では`[.353568,.321537,.324895]`。条件反応は存在するが極小である。全例は[`probe_pre.json`](rl_plan_conditioned_step3b8_20261014_evidence/probe_pre.json)と[`probe_post.json`](rl_plan_conditioned_step3b8_20261014_evidence/probe_post.json)に保存。

## 新規on-policy更新

学習seedはRule `35400000–35400047`、Champion `35401000–35401047`、Mixed `35402000–35402047`。144局、各profile 48、各seat 36、実委譲265件。候補数2/3/4は171/78/16件、family選択Road 43・Development 41・BankTrade 53・EndTurn 128。single-choice 1,695局面はActor bufferから除外したが、直前委譲のmacro rewardには含めた。terminal/nonterminal transitionは117/148、macro duration平均23.22 learner policy steps、terminal absolute Advantage massは52.1%。

preflightでは全episode terminal、truncationなし、実委譲だけのsample境界、old masked確率・log-prob、Critic値、Frozen latent、macro reward/discount、保護境界を再生照合した。`optimizer.step()`はこのpreflight後に**一つのguard付きPPO update**としてのみ実行。設定はactor LR 1e-4、critic LR 3e-4、batch 64、2 epoch、clip .1、entropy coefficient .005、γ .99、GAE λ .95。

| PPO診断 | 値 |
|---|---:|
| Actor / Critic parameter | 80,460 / 33,345 |
| Actor / Critic loss平均 | 0.02340 / 0.53123 |
| Entropy loss / clip fraction | -0.004229 / 0 |
| ratio min–max | 0.97345–1.04106 |
| KL mean / median / p95 / max | 0.000133 / 0.0000339 / 0.000334 / 0.000405 |
| L3 entropy before→after | 0.854260→0.854055 |
| L3 / Priority / Plan Scorer gradient norm平均 | 0.2696 / 0.0000514 / 0.0000163 |
| L3 / Priority / Plan Scorer parameter delta | 0.08897 / 0.00868 / 0.00236 |
| Critic return MSE before→after | 0.53659→0.53039 |

KL guard、entropy/family collapse guardは通過。L1/L2への非ゼロ勾配とparameter更新はあるが、その大きさと固定state介入結果から、実効利用はなお不十分。詳細は[`update_result.json`](rl_plan_conditioned_step3b8_20261014_evidence/update_result.json)。

## 新seed held-out比較

評価seedはRule `35500000–35500023`、Champion `35501000–35501023`、Mixed `35502000–35502023`。各variant 72局、profile/seat均衡。delegation/family外部RNG系列は同じ。Safetyは0%、他はInvestment multi-choice 10%。

| Variant | 勝ち | 平均VP | 平均順位 | 平均steps | 実委譲 | Settlement+City | Road | Dev | Bank | EndTurn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Safety | 38 | 8.333 | 1.910 | 78.31 | 0 | 329 | 412 | 185 | 279 | 1298 |
| Post2 | 38 | 8.111 | 1.944 | 77.50 | 119 | 321 | 421 | 170 | 271 | 1278 |
| 新Pre | 38 | 8.111 | 1.944 | 77.50 | 119 | 321 | 421 | 170 | 271 | 1278 |
| 新Post | 38 | 8.111 | 1.944 | 77.50 | 119 | 321 | 421 | 171 | 270 | 1278 |

Post2と新PreのAction traceは72/72局一致。新Preと新Postは71/72局一致し、実委譲familyはBankTrade 1件がDevelopmentへ変化した。新Pre→新Postのpaired ΔVP、Δrank、Δstepsは全て0（profile-stratified bootstrap 95% CIも `[0,0]`）。Safety→新Postのpaired ΔVPは−0.222、95% CI `[-0.486,0]`。**勝率改善はなく、Safetyとの悪化可能性も残る。** 72局では有意な優劣を断定しない。詳細は[`evaluation_summary.json`](rl_plan_conditioned_step3b8_20261014_evidence/evaluation_summary.json)と[`heldout_games.json.gz`](rl_plan_conditioned_step3b8_20261014_evidence/heldout_games.json.gz)。

全variant terminal 100%、illegal・Executor failure・family mismatch・deadlock・truncation・非有限reward・GameState RNG arbitration消費は0。Safety 0%は72/72で既存trace・最終状態・winner・RNGと一致。Champion/Robber digestはそれぞれ`4225902d41a83540eb1f1552826ec623706e778b99ad4ae8ec015f976644392a` / `ecde7fa9e9d67370bb76686ecd767d2d532a76d467c0051aaebdf8d4aff702bf`で前後不変。Post2 parentも不変。Pre/Postは別weights artifactへ保存し、旧artifactを上書きしていない。

## テストと次工程

[`test_plan_conditioned_policy_step3b8.py`](../tests/test_plan_conditioned_policy_step3b8.py)で確率和、strict mask、候補順序不変、Plan A/B・Priority介入、L1/L2 gradient、Frozen latent、Plan unavailable/all-invalid fallbackを確認。全backendはworkspace内pytest basetempで**405 passed、1,172 subtests passed、依存ライブラリ警告2件**。通常temp指定の初回試行はWindowsのpytest一時ディレクトリ権限で2件setup errorになったが、コード失敗ではなく再実行で全通過した。

次のStep 3B-9では追加PPOや委譲拡大の前に、Plan-conditioned枝の感度を定量設計する。具体的には、direct-state経路とPlan/Priority経路のJacobian・特徴量寄与・per-plan `q_i`差を同一stateで比較し、必要なら**family-specific heuristicなし**の条件付け/bottleneck構造を検討する。今回の1 update結果を「長期Victory Plan学習が完成した」とは扱わない。City/Construction、Dev Usage、PlayerTrade、Plan離散選択・persistenceは引き続き未導入。
