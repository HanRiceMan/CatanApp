# Step 3B-7 — Informative Investment PPO と Strategic Influence

## 実験境界

3B-6の独立artifact `Hierarchical-Investment-Post1` を親とし、INVESTMENTの実行可能family `BUILD_ROAD / BUY_DEVELOPMENT / BANK_TRADE / END_TURN` が**2候補以上**の時だけ10%抽選で実委譲した。1候補局面はChampionへ戻し、次の実委譲までのsemi-MDP reward区間からは外していない。PlayerTradeはprotected、建設・Development Usage・forced phase・ImmediateWinResolverは既存処理を維持。GameState RNGを抽選に使わず、Champion Actionは教師にしていない。

学習seedはRule `35200000–35200079`、Champion `35201000–35201079`、Mixed `35202000–35202079`。各profile80局、各seat60局、計240局。3B-6のtraining dataは一切再利用していない。

| 収集指標 | 結果 |
|---|---:|
| informative actual delegated transitions | 399 |
| 1候補Investment局面の非委譲 | 2855 |
| multi-choice eligible surface | 4003 |
| Road / BuyDev / BankTrade / EndTurn選択 | 71 / 68 / 77 / 183 |
| family候補数 2 / 3 / 4 | 270 / 99 / 30 |
| macro duration mean / median / max | 21.97 / 17 / 98 |
| macro reward mean / SD | 0.0851 / 0.5997 |
| terminal / nonterminal macro | 185 / 214 |
| protected trade / Resolver / 非委譲Investment cross | 1425 / 93 / 3067 |

全240局がterminal。truncation、illegal Action、Executor failure、family mismatch、nonfinite reward、GameState RNG arbitration consumptionは0。old確率・log probability・Critic値・Plan entropyの再現、およびmacro reward・duration・bootstrap境界をoptimizer作成前に全件検査した。terminal絶対Advantage mass比は`0.5428`で、長い区間とterminal報酬の分散を引き続き監視する。

## 指定1回のPPO更新

3B-6と同じ`gamma=.99, lambda=.95, actor_lr=1e-4, critic_lr=3e-4, batch=64, epochs=2, clip=.1, entropy_coef=.005`を使用。実委譲399件のold Post1確率を分母にratioを計算した。one-shot markerをoptimizer作成前に保存し、同じデータで二度目の更新はできない。

| 指標 | 結果 |
|---|---:|
| Actor / Critic / entropy loss | −0.003481 / 0.577352 / −0.004197 |
| clip fraction | 0 |
| masked old→new KL mean / max | 0.0000727 / 0.0001616 |
| L3 entropy pre→post | 0.845822→0.845782 |
| gradient norm mean L3 / Priority / Plan Scorer | 0.4210 / 1.15e-5 / 1.87e-6 |
| parameter delta L3 / Priority / Plan Scorer | 0.09170 / 0.00901 / 0.00532 |
| family / entropy collapse | なし / なし |
| Critic MSE（training state）pre→post | 0.5884→0.5812 |

L1/L2への勾配とparameter更新は成立したが、L3より数桁小さい。この事実だけで、Plan/Priorityが行動へ実質的に使われているとは判定しない。

## Held-out / Ablation

評価seedはRule `35300000–35300047`、Champion `35301000–35301047`、Mixed `35302000–35302047`。各variant 144局、profile/seat均衡。Safety 0%、Post1 multi-choice 10%、Post2 multi-choice 10%を同じgame/delegation/family seedで比較する。3B-6の旧Post1評価は委譲境界が違うため使用しない。

| Variant（各144局） | 勝利 | 平均VP | 平均rank | 平均steps | 実委譲 | S / C / Road / Dev | Bank / End |
|---|---:|---:|---:|---:|---:|---:|---:|
| Safety 0% | 68 | 7.938 | 2.038 | 76.569 | 0 | 381 / 232 / 801 / 369 | 507 / 2489 |
| Post1 multi-choice 10% | 63 | 7.854 | 2.080 | 77.063 | 218 | 382 / 231 / 798 / 363 | 510 / 2536 |
| Post2 multi-choice 10% | 63 | 7.861 | 2.083 | 77.104 | 218 | 382 / 232 / 798 / 362 | 513 / 2539 |

Post1→Post2のpaired差はVP `+0.00694`（profile-stratified bootstrap 95% CI `[0, +0.02083]`）、rank `+0.00347`（CI `[0, +0.01042]`）、steps `+0.0417`（CI `[0, +0.125]`）。勝利数は同じ63/144で、差はほぼ1局のtrajectoryに限られる。統計的な改善とは解釈しない。Post2の実委譲familyはRoad `33`、BuyDev `38`、BankTrade `56`、EndTurn `91`（Post1は`32/38/56/92`）。

Safety→Post2のpaired VP差は`−0.0764`（95% CI `[−0.2708, +0.1042]`）、rank差`+0.0451`（CI `[−0.0451, +0.1424]`）、steps差`+0.535`（CI `[−1.861, +2.854]`）。Safetyとの大きな行動崩壊は観測されないが、このsampleからSafety非劣性を確定できない。3条件とも144/144 terminal、illegal/Executor failure/family mismatch/deadlock/truncation/nonfinite reward/GameState RNG arbitration consumptionは0。Safety 0%は既存controllerとのAction/State/RNG parity `144/144`。protected PlayerTradeはSafety/Post1/Post2で`1643/1623/1623`件であり、Actor選択肢には含めていない。

固定state ablationはPost1の新held-out軌跡上のpre-action stateへPost2を適用する。Normal、Plan一様化、Priority全0.5、両方中立を、同一state・同一family maskで比較する。実Actionには適用しない。KL、TV、同点を除いたargmax変化、比較可能familyのlogit差とstate別recordを保存する。

固定held-outは218 state、Plan available 217 state。Post2 NormalのPlan entropyと同候補数uniform entropyの平均差は`1.43e-7`、すなわちPlan分布は実質的にuniform。Plan available stateに限った4 Priorityの平均は`[0.500010, 0.499222, 0.499624, 0.500413]`、state間SDは`[0.000165, 0.000083, 0.000124, 0.000161]`で、状態依存性はごく小さい。Plan unavailableの1 stateはvalid flagを維持し、この分布統計から除外した。

| Post2同一state ablation（Normalとの差） | masked KL平均 | TV平均 | untied argmax flip | valid logit差最大 |
|---|---:|---:|---:|---:|
| Plan uniform、候補別Priority保持 | 3.80e-16 | 8.98e-9 | 0/218 | 8.57e-8 |
| Priority neutral 0.5、Plan保持 | 3.73e-15 | 3.46e-8 | 0/218 | 2.90e-7 |
| Both neutral | 5.02e-15 | 4.00e-8 | 0/218 | 3.56e-7 |

KLはfloat64へ変換・再正規化後に測定した。これらは実質的に数値精度レベルであり、L1/L2がL3へ有意な行動影響を与えている証拠はない。全state別の確率・KL・TV・logit差も圧縮evidenceへ保存した。

## 判断

判定は**Case B**。多候補の自然なon-policy収集とguarded PPO更新は成立し、Post2に明確な性能悪化もない。しかしPlanはほぼuniform、Priorityはほぼneutralで、L3の行動分布は双方の中立化に事実上不変。L1/L2への非ゼロ勾配・parameter deltaを「戦略的に使われた」と見なさない。

したがってPost2は技術的にacceptedな独立artifactとして保存する一方、委譲率の拡大とCity Value Comparisonへの実委譲はまだ行わない。Step 3B-8では、L3入力の直接state経路がsoft Plan/Priority経路を迂回しているかを同一stateの感度・feature/weight寄与で監査し、L1/L2情報を実際に使わせる構造を検討する。RewardやChampion教師を直ちに加える理由はない。

親Post1、3B-6 Pre、Champion、Robberの重みは上書きしない。Post2は独立artifactとして保存し、改善を確認できても自動で委譲率を引き上げたりCity Valueへ移したりしない。

詳細なpreflight、PPO metrics、各局結果、state別ablation、Post2重み・manifestは [`rl_hierarchical_ppo_step3b7_20261013_evidence`](rl_hierarchical_ppo_step3b7_20261013_evidence/) に保存する。

backend全体のテストは`402 passed, 1172 subtests passed`（既存依存のdeprecation warning 2件）。Post2 manifestに学習後のPlan/Priority・L3・Critic digestとPost1 parent SHA-256を保存し、evidence 11ファイルのSHA-256を再照合した。
