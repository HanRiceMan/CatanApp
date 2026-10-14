# Step 3B-11 — Route-Isolated City PPO

## 判定

City専用Residual Adapterと独立Criticにより、3B-9 Post1のInvestment分布を**固定218 stateで学習前後とも完全一致**に保ちながら、新しいCity on-policy transitionで1回のguarded PPO更新を実行した。これは3B-10で見られた共有ActorのInvestment側top-family flip（118/218）を、構造的に除いた技術的結果である。対局上の改善とPlan/Priority条件の実質的利用は、新seed held-out評価で別判定する。

## Actor / Criticと委譲境界

- [`route_isolated_city_step3b11.py`](../app/rl/route_isolated_city_step3b11.py): 3B-9 Post1のL1 Plan Scorer、L2独立Priority Head、Plan–Action Base Actorを完全freeze。Investment routeは親Actor出力をそのまま返す。City routeだけ、候補Planごとに共有小型MLPのResidual `Δφ(s, plan_i, priority_i, family)`をBase logitへ加え、各Planの合法family softmaxをL1確率で混合する。Adapter最終Linearはweight/biasとも0で、City Preは親Actorと確率完全一致。family固有の手作り優先度なし。
- City Adapter trainable parameterは**2,727**。City専用Criticは3B-10と同形の**33,857** parameterを新規初期化し、旧Investment Criticも3B-10 City Criticも継承しない。Actor optimizerには`requires_grad=True`のAdapterだけを渡す。L1/L2/Base、Champion、Robber PPOは対象外。
- [`hierarchical_ppo_step3b6.py`](../app/rl/hierarchical_ppo_step3b6.py): optimizerのtrainable parameter境界を明示し、frozen moduleとAdapterのgradient/parameter deltaを別記録。既存Investment PPOのdefaultは維持。
- [`run_route_isolated_city_step3b11.py`](../app/rl/run_route_isolated_city_step3b11.py): 固定seed shard、balanced block停止、preflight、一度限りのguarded update、独立held-out、固定state ablation、artifact公開。3B-10 Postは比較対象であり、parentや更新対象ではない。

実委譲はImmediateWinなしの通常action phaseでCity合法・Settlement非合法、実行可能な比較familyが2つ以上ある時のみ。City/Road/BuyDev/BankTrade/EndTurnを合法性とFrozen Executor実行可能性でmaskし、trainingでは25%、held-outでは10%で抽選。Investmentは0%。protected PlayerTrade、Development Usage、forced phase、ImmediateWinResolverは従来どおり先行する。Rewardはterminal win/loss `±1`とVP差`×0.05`のみ。City建設ボーナスは設けない。

## 新規on-policy収集と1回の更新

学習seedはRule `36000000–36000139`、Champion `36001000–36001139`、Mixed `36002000–36002139`。420局＝各profile140、各seat105。35個の12-game balanced blockで初めて目標192を超え、**193件**を採用した。事前に生成した後続blockは学習に含めず、古いrolloutも流用していない。選択はCity60、Road11、Dev24、BankTrade35、EndTurn63。候補数2/3/4/5は18/72/80/23。terminal/nonterminal transitionは132/61、macro duration平均21.35 learner step、macro reward平均0.2738、terminal絶対Advantage比71.57%。

Preflightは全episode terminal、truncation0、実委譲境界、stored old probability/log-prob、Critic value、Frozen latent、semi-MDP reward/durationを再生照合。最大old log-prob誤差`2.38e-7`、probability`5.96e-8`、value`0`。Optimizer attempt markerにより同rolloutの再更新は禁止。γ`0.99`、GAE λ`0.95`、actor LR`1e-4`、critic LR`3e-4`、batch64、2 epochs、clip`0.1`、entropy coefficient`0.005`。

| 更新診断 | 値 |
|---|---:|
| guarded update | 1回、ACCEPTED |
| Actor / Critic loss平均 | 0.11641 / 0.67552 |
| PPO ratio平均 / clip fraction | 0.99995 / 0 |
| KL平均 / 最大 | 0.00000657 / 0.00001418 |
| L3 entropy before → after | 1.23669 → 1.23690 |
| Adapter gradient norm平均 / parameter delta | 0.70606 / 0.01252 |
| L1 / L2 / Frozen Base parameter delta | 0 / 0 / 0 |
| City Critic return MSE before → after | 0.75404 → 0.68663 |

KL・entropy・family collapse guardを通過。固定Investment 218 stateではPre/Postとも3B-9 Post1と各family確率がbitwise一致、top flip0。City Preもzero-init Adapterにより親ActorのCity確率と一致した。これはCity PPOの性能ではなく、route分離とfreeze境界の検証である。

## Held-out対局・Plan条件診断

新seed Rule `36100000–36100047`、Champion `36101000–36101047`、Mixed `36102000–36102047`。各条件144局、各profile48・各seat36、同game seedと委譲／family外部RNG系列。3B-10 City Postは同じCity10%条件の比較であり、今回のtraining parentではない。

| 条件 | 勝ち/144 | 平均VP | 平均rank | 平均steps | 実委譲 | Settlement | City | Road | Dev | Bank | EndTurn | City同turn/機会 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Safety | 87 | 8.465 | 1.753 | 74.986 | 0 | 424 | 267 | 856 | 297 | 509 | 2458 | 227/257 (88.33%) |
| 3B-10 City Post | 86 | 8.389 | 1.764 | 74.743 | 30 | 423 | 258 | 858 | 298 | 519 | 2446 | 218/260 (83.85%) |
| Adapter Pre | 86 | 8.389 | 1.764 | 74.771 | 30 | 422 | 259 | 856 | 299 | 518 | 2448 | 218/260 (83.85%) |
| Adapter Post | 86 | 8.389 | 1.764 | 74.771 | 30 | 422 | 259 | 856 | 299 | 518 | 2448 | 218/260 (83.85%) |

Adapter Pre/Postの委譲familyは両方ともCity9・Road1・Dev3・Bank6・EndTurn11で、**Action traceは144/144局で一致**した。paired Adapter Post−PreのVP/rank/steps差はすべて0、95% bootstrap CIも`[0,0]`。Safety→Adapter Postの平均VP差は`-0.07639`、profile層別paired 95% CI `[-0.17361,0]`。rank差`+0.01042`（`[-0.02431,+0.05208]`）、steps差`-0.2153`（`[-0.8611,+0.3056]`）。3B-10 City Post→Adapter PostのVP差もこのseedでは0だが、Action trace・建設総数は同一ではない。**更新による成績改善は確認できず、Safetyに対する優位性もない。** 小規模な30実委譲/144局から勝率差を一般化しない。City建設回数の増減だけを成功条件にもしていない。

固定held-out City実委譲30 stateでは、Pre→Post family分布の平均TV`0.001470`、KL`0.00000575`、argmax flip0。Postの候補Plan間pairwise TVは`0.01212`、有効Plan A/Bを指定したTVは`0.01207`（argmax flip9/30）。これはPlan差への**表現感度**であり、Adapterが正しいPlan価値を学習した証拠ではない。通常Planを一様化したTVは`1.06e-5`、Priorityを全0.5へ中立化したTVは`1.48e-7`、両中立化は`1.06e-5`で、すべてargmax flip0。Priorityを極端に0対1へ変えたTVも`0.000117`。L1 entropyのuniformとの差は`1.30e-6`で、Priority平均はほぼ0.5のまま。L1/L2は今回freezeされているため改善を期待する設計ではないが、City Actionへの自然な戦略的影響がまだ小さいことは確認できた。

固定Investment 218 stateでは、3B-9 Post1／Adapter Pre／Adapter Postの全family確率が**bitwise一致**、top-family flip0、平均TV0。これは3B-10の共有Actor更新で起きたInvestment出力変化を本実験では完全に除いたことを意味する。Investmentは学習・評価とも実委譲0。詳細は[`evaluation_summary.json`](rl_route_isolated_city_step3b11_20261014_evidence/evaluation_summary.json)と[`fixed_state_diagnosis.json`](rl_route_isolated_city_step3b11_20261014_evidence/fixed_state_diagnosis.json)。

## 安全性と次工程

全576 held-out episodeはterminal、truncation0、illegal0、Executor failure0、family mismatch0、deadlock0、nonfinite0、GameState RNG arbitration消費0。protected PlayerTradeとImmediateWinResolverを維持し、0% SafetyのAction trace・最終GameState・winner・RNG parityは144/144。Champion digest `4225902d…644392a`、Robber digest `ecde7fa9…702bf`、3B-9 parent SHA256 `8325e0c9…ba6`は不変。Adapter Pre重みSHA256 `8fb798da…2fe24`、Post `4b02fbe8…08648`。独立artifact、圧縮rollout・held-out個票、SHA indexを[`evidence`](rl_route_isolated_city_step3b11_20261014_evidence/)へ保存。評価shard分割変更後の最終backend全テストも**416 passed、1,172 subtests passed、既存依存関係warning 2件**。

**結論はCase B**：City Adapterは独立して学習でき、Investmentへの副作用は0になったが、この1更新ではCity判断の実Actionも対局性能も改善していない。Postを技術検証artifactとして保持し、City委譲率を拡大・本番採用しない。次Stepでは「route間干渉を消しても改善しない」条件の下で、City固有のsample不足（193 training transition、30 held-out委譲）、terminal寄与の大きさ（絶対Advantageの71.6%）、Criticのheld-out較正、semi-MDP credit assignmentを優先して検証する。Reward追加やCity hard-build ruleを先行導入しない。L1/L2を共同学習へ戻すなら、Investmentの固定state分布保持をguardとして設計する。
