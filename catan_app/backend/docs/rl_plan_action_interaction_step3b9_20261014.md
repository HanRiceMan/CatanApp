# Step 3B-9 — Strategic Plan–Action Interaction

## 判定

**Plan条件経路の実装と限定on-policy更新は成立した。ただし、学習済みPlanが通常の行動選択へ実質的に影響する段階にはまだ達していない。** 新Actorは固定212局面でPlan A/Bを切り替えたときのfamily分布差を旧3B-8 Post1の平均TV `0.000219`から、新Pre `0.012523`、新Post `0.013450`へ増やした。これは表現能力と勾配経路の証拠であり、戦略の正しさではない。通常のPlan分布はほぼuniformのままで、PostのPlan一様化による平均TVは`0.00000590`、Priority中立化は`0.00001186`にとどまる。

新規272件でPPOを**1回だけ**実行し、KL・mask・freeze・family collapse guardを通過した。新Pre→新Postのheld-out平均VP差は`+0.014`、95% CI `[-0.069,+0.097]`で改善は確認できない。一方、新PostはSafetyより平均VP `-0.333`、95% CI `[-0.653,-0.042]`で、この72局では悪化方向のシグナルがある。**委譲率・対象routeはまだ拡大しない。**

## 実装と学習境界

- [`plan_action_interaction_step3b9.py`](../app/rl/plan_action_interaction_step3b9.py): State encoder（64幅、LayerNorm）から基本family logit `b_a(s)`と状態依存query `u_a(s)`を作り、Plan候補特徴9軸＋候補別独立Priority4軸を32幅Plan encoderへ通す。各候補に `ℓ_i(a)=b_a(s)+α〈u_a(s),e_i〉/√32` を適用し、共通の合法・route比較maskでsoftmaxした `q_i(a)`を、L1の`p_i`で混合する。初期`α=0.1`、Postは`0.10035`。family専用の手作り係数や初期優遇はない。
- PPO Actionは**混合後のfamilyのみ**。`log P(a|s)`、`P(a|s)=Σ_i p_i q_i(a|s)`をold/new ratioへ使い、Planをsampleしたものとして二重計上しない。IMPOSSIBLE/paddingの寄与は厳密に0、UNKNOWNは残す。Plan unavailableは明示flagとbase-only fallback、比較familyなしなら`available=false`で全確率0。偽のEndTurnは作らない。
- Actor入力はdetachしたFrozen Champion latent、Runtime Context79、VictoryPlan global facts、route one-hot、legal/comparison mask、Construction Commit fact。旧3A family prior、Planner Intent、Champion Action教師、Plan価値heuristicは使わない。Frozen latentに勾配は戻らない。
- [`run_plan_action_interaction_step3b9.py`](../app/rl/run_plan_action_interaction_step3b9.py): 再開可能な固定state診断、新規balanced rollout、preflight、one-shot guarded update、同seed4条件評価、独立artifact公開。L1/L2は3B-8 Post1をparentとし、**旧L3を転写せず**新L3を初期化した。Criticはparentから開始。Actor 91,437 parameter（L1/L2 41,349、新L3 50,088）、Critic 33,345。
- 実委譲はInvestmentの実行可能familyが2つ以上ある局面だけ、10%。Road/BuyDev/BankTrade/EndTurnのみ。PlayerTradeは保護、ImmediateWinResolverは先行し、Construction・City・Development Usageは既存処理のまま。Frozen B Executorと報酬、semi-MDP/GAE、PPO guardは変更しない。

## 固定stateでのPlan条件感度

3B-8で保存した**学習に使わない**218 stateを使用。Planあり217、複数有効Planあり212。TVはfamily分布のTotal Variation Distance。旧Actorで候補別raw logitは保存されていないため、旧新で直接比較できる指標は候補別family確率とlog確率のばらつきである。

| 指標 | 旧3B-8 Post1 | 新Pre | 新Post |
|---|---:|---:|---:|
| 候補Plan間pairwise TV平均 | 0.000219 | 0.010416 | 0.010702 |
| pairwise TV p95 / max | 0.000382 / 0.000990 | 0.016479 / 0.041381 | 0.016796 / 0.041819 |
| 候補別log確率std平均 | 0.000340 | 0.016479 | 0.017010 |
| 候補別raw logit std平均 | 未記録 | 0.019127 | 0.019403 |
| 有効Plan A/Bをone-hot指定したTV平均 | — | 0.012523 | 0.013450 |
| A/B指定のargmax flip | — | 67/212 | 63/212 |

新Postのfamily別「Plan候補間確率std」平均はRoad `0.006154`、Dev `0.003628`、BankTrade `0.009368`、EndTurn `0.007609`。旧Post1はそれぞれ`0.000109`、`0.000196`、`0.000134`、`0.000142`。Plan A/B介入は可能性のテストであり、実際のPlan選択や合理性の証明ではない。新Preのstate/Plan/query embedding RMSは`1.000/1.000/0.674`、基本/interaction logit stdは`0.0606/0.0596`。条件経路を極端なスケールで押し上げていない。

| 通常出力からの診断用中立化・平均TV | 新Pre | 新Post |
|---|---:|---:|
| Plan一様化 | 0.00000149 | 0.00000590 |
| Priorityを全0.5 | 0.00001561 | 0.00001186 |
| 両方中立化 | 0.00001683 | 0.00001716 |

いずれもargmax flipは0/218。L1 Plan entropyのuniformとの差は旧`8.04×10⁻⁸`、新Pre`8.04×10⁻⁸`、新Post`8.02×10⁻⁷`。**ボトルネックは候補別`q_i`の表現力から、ほぼ一様なL1混合と微小なL2変動へ移った**と解釈できる。ただし1更新・固定stateだけで最終原因は断定しない。詳細は[`diagnosis_pre.json`](rl_plan_action_interaction_step3b9_20261014_evidence/diagnosis_pre.json)と[`diagnosis_post.json`](rl_plan_action_interaction_step3b9_20261014_evidence/diagnosis_post.json)。

## 新規on-policy更新

学習seedはRule `35600000–35600051`、Champion `35601000–35601051`、Mixed `35602000–35602051`。156局（各profile52、各seat39）、実委譲272件。選択familyはRoad58・Dev45・Bank58・End111。候補2/3/4は158/100/14件、single-choiceは1,822局面をActor sampleから除外したがmacro遷移からは除外していない。terminal/nonterminal transitionは126/146、macro duration平均22.53、macro reward平均0.0285、terminal absolute Advantage share53.2%。

Preflightで全episode terminal、truncation0、実委譲境界、stored old mixture確率/log-prob、Critic値、Frozen latent、macro reward/discount、保護境界を再生照合した。最大replay誤差はlog-prob`2.38×10⁻⁷`、probability`5.96×10⁻⁸`、value`1.49×10⁻⁸`。設定はactor LR`1e-4`、critic LR`3e-4`、batch64、2 epoch、clip0.1、entropy coefficient0.005、γ0.99、GAE λ0.95。optimizer attempt markerにより同じrolloutでの2回目を拒否する。

| PPO診断 | 値 |
|---|---:|
| Actor / Critic loss平均 | 0.000295 / 0.543532 |
| Entropy loss / clip fraction | -0.004327 / 0 |
| ratio min–max | 0.90234–1.07839 |
| KL mean / median / p95 / max | 0.000734 / 0.000115 / 0.002307 / 0.003115 |
| L3 entropy before→after | 0.877147→0.875786 |
| L3 / L2 / L1 gradient norm平均 | 0.85717 / 0.001286 / 0.000825 |
| L3 / L2 / L1 parameter delta | 0.07346 / 0.00617 / 0.00471 |
| Critic return MSE before→after | 0.55416→0.54303 |

旧3B-8 Post1の同種gradientはL3`0.26959`、L2`0.0000514`、L1`0.0000163`。新経路のL1/L2信号は大きくなったが、学習したPlanが実行Actionを大きく変えたこととは別である。KL guard・entropy/family collapse guardを通過し、更新はaccepted。詳細は[`preflight.json`](rl_plan_action_interaction_step3b9_20261014_evidence/preflight.json)と[`update_result.json`](rl_plan_action_interaction_step3b9_20261014_evidence/update_result.json)。

## 新seed held-out対局

評価seedはRule `35700000–35700023`、Champion `35701000–35701023`、Mixed `35702000–35702023`。各variant72局、profile/seat均衡。同一game seed・委譲抽選RNG・family sampling RNG。Safetyは0%、他はInvestment multi-choice 10%。

| Variant | 勝ち/72 | 平均VP | 平均順位 | 平均steps | 実委譲 | Settlement+City | Road | Dev | Bank | EndTurn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Safety | 43 | 8.653 | 1.667 | 76.50 | 0 | 355 | 414 | 153 | 285 | 1206 |
| 3B-8 Post1 | 39 | 8.375 | 1.833 | 78.99 | 126 | 349 | 418 | 159 | 302 | 1270 |
| 新Pre | 38 | 8.306 | 1.847 | 78.25 | 126 | 342 | 418 | 159 | 297 | 1263 |
| 新Post | 38 | 8.319 | 1.854 | 78.11 | 125 | 340 | 414 | 161 | 293 | 1267 |

新Postの実委譲選択はRoad19・Dev18・Bank30・End58。新Pre→新Postはpaired ΔVP`+0.014`、95% CI`[-0.069,+0.097]`、Δrank`+0.0069`、Action trace一致69/72。3B-8 Post1→新PostはΔVP`-0.056`、95% CI`[-0.194,+0.056]`で優劣不明。Safety→新PostはΔVP`-0.333`、95% CI`[-0.653,-0.042]`、Δrank`+0.188`。72局のシグナルを一般的な勝率差と断定はしないが、安全な対象拡大の根拠にはならない。詳細は[`evaluation_summary.json`](rl_plan_action_interaction_step3b9_20261014_evidence/evaluation_summary.json)と[`heldout_games.json.gz`](rl_plan_action_interaction_step3b9_20261014_evidence/heldout_games.json.gz)。

全4 variantがterminal72/72、truncation0。違法Action・Executor失敗・family mismatch・非有限reward・GameState RNG arbitration消費はrollout内のassert/例外が発生しなかった。Safety 0%は既存controllerとAction trace・最終GameState・winner・RNGで72/72一致。Champion digest `4225902d41a83540eb1f1552826ec623706e778b99ad4ae8ec015f976644392a`、Robber digest `ecde7fa9e9d67370bb76686ecd767d2d532a76d467c0051aaebdf8d4aff702bf`、旧3B-8 parent SHA `7cd52d4d8c69a682a7c1253a9632f8c4f91a0d05db6b2800856a91ce6359fbd3`は不変。新Pre/Postは独立した[`evidence`](rl_plan_action_interaction_step3b9_20261014_evidence)へ保存し、既存artifactを上書きしていない。

## テストと次工程

[`test_plan_action_interaction_step3b9.py`](../tests/test_plan_action_interaction_step3b9.py)でPlan A/B・Priority介入、厳密mask、確率和、候補順序不変、L1/L2/L3勾配、Frozen latent、Plan unavailable/all-invalid fallbackを確認。全backendは**408 passed、1,172 subtests passed、依存ライブラリ警告2件**。更新は1回、旧重みは不変。

次工程では同じInvestmentだけの反復更新やCity routeの即時委譲は避け、L1がほぼuniformに残る原因と、Safety比のVP低下を少数の再現可能な介入・held-outで切り分ける。そのうえでCity Value ComparisonをShadowで接続し、実委譲範囲拡大は安全性とPlan/Priorityの**通常stateでの**寄与が確認できてから判断する。離散Plan選択・persistence、Development Usage、PlayerTradeは未導入のままとする。
