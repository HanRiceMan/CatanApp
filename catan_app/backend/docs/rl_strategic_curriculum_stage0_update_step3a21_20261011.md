# Step 3A-21 — Stage 0 Low-Delegation Strategic PPO（初回guarded update）

2026-10-11。Step 3A-20でqualifiedとなった10% delegationのみをtraining責任範囲とする。`Strategic-T2-preupdate` のnative family logits/2 + zero residualから開始し、6 Strategic family、ImmediateWinResolver、protected PlayerTrade、Frozen B Executors、Reward、Runtime Context79、candidate maskを固定する。Step 3A-16 postupdate artifactをparentにしない。Gate residualとStrategic Surface Criticだけをon-policy semi-MDP returnで**1回**更新し、Stage 1へは自動昇格しない。

Trainingは新規`32200000`台（Rule）/`32201000`台（Champion）/`32202000`台（Mixed）を使用し、profile×seatの全12組を1 balanced blockとして完走する。実委譲StrategicTransition 320件以上・6 family各10件以上までblock単位で自然収集する。familyを強制抽選・再抽選しない。候補外・protected・Resolver・非委譲DecisionはActor sampleに含めない。macro rewardは既存learner policy step単位の `Σ γ^j r_j`、bootstrapは `γ^duration`、GAEはStep 3A-16定義 `δ_i + γ^duration_i λ A_(i+1)` を維持する。

held-outはtrainingおよびStep 3A-20 qualificationと重複しない予約seed `32100000…32100023` / `32101000…32101023` / `32102000…32102023` を使用する。Safety、Pre10、Post10、Pre20、Post20の各72局を同一game seedで比較する。Pre10のAction前stateを固定probeとし、同じcandidate mask・T=2でPre/Post categorical KLとfamily probability shiftを計測する。Pre/Post差はChampion imitation scoreではなく、1回のon-policy updateの差として解釈する。

Technical acceptanceとbehavioral acceptanceを分ける。Technicalではpreflightのstored distribution/log-prob/value再現、optimizer parameter ID、KL mean≤0.01・max≤0.10、family collapse、artifact save/reload、Champion/parent digest不変を確認する。BehavioralではPost10 held-outの全72局hard safetyと、Safety比の平均ΔVP>−0.30・建設比≥90%・平均policy steps比≤115%をprofile層化paired bootstrap 5,000回の95% CIとともに評価する。20%はStage 1 probeであり、良好でも昇格を実行しない。

## Training rolloutと唯一のupdate

156局、profile各52局・seat各39局（profile×seat各13局）。seed範囲はRule `32200000–32200051`、Champion `32201000–32201051`、Mixed `32202000–32202051`。episodeを切らず13 balanced blockで終了し、実委譲339遷移を収集した。全局terminalで、illegal / Executor failure / family mismatch / deadlock / truncation / nonfinite reward / GameState RNG arbitration消費はすべて0。protected trade 974回、非委譲Surface 1,768回、Resolver 68回をmacro transitionが跨いだが、Actor sampleには含めない。

| Family | selected | available |
|---|---:|---:|
| BUILD_SETTLEMENT | 34 | 36 |
| BUILD_CITY | 30 | 32 |
| BUILD_ROAD | 48 | 146 |
| BUY_DEVELOPMENT | 118 | 151 |
| BANK_TRADE | 73 | 190 |
| END_TURN | 36 | 339 |

native top 282、non-top 57。候補数2/3/4+は183/106/50。entropy平均0.4314。macro duration平均20.42、中央値16、p95 51.1、最大102 learner policy step。macro reward平均0.0913、SD 0.5655。terminal/nonterminal transitionは137/202。terminal absolute Advantage massは0.470。Road B 48件中 expansion+longest 19、expansion only 16、longest only 4、neither 9。neitherも合法なon-policy sampleとして保持した。

事前検証はPASS。保存済みlog-prob誤差最大3.05e−7、calibrated logits誤差0、確率誤差1.19e−7、Critic valueとfrozen base value誤差各1.19e−7。Gate 30,982 parameter、専用Critic 30,657 parameterの計61,639のみtrainableで、Champion parameter IDとの交差は0。raw Advantage平均0.4253、SD 0.7350、min−0.8855、p5−0.5320、中央値0.6945、p95 1.4250、max1.7973。正規化後は平均ほぼ0、SDほぼ1。terminal Advantage平均0.4636・SD0.8512、nonterminal平均0.3993・SD0.6431。family別平均はSettlement 0.453、City 1.000、Road 0.291、Development 0.436、BankTrade 0.405、EndTurn 0.105であり、異なるstate分布の値なのでfamily priorityとして解釈しない。Critic MSEはupdate前0.7211、後0.6000、説明分散は0.0591→0.0601。

許可されたPPO updateを**1回だけ**実行した。2 epochs、12 minibatch、actor loss平均0.02556、critic loss平均0.66846、entropy loss平均−0.002129、clip fraction 0。ratio平均1.00104、SD0.01297、範囲0.96270–1.02591。old/newはいずれも同state・同mask・T=2で計算した。KL平均0.0000909、中央値0.0000239、p95 0.0002966、最大0.0003648でguard内。候補数2/3/4+のKL平均は0.0001367 / 0.0000505 / 0.0000092。entropy平均は0.43143→0.42758、>0.95の局面数98→98、family collapseなし。親のzero residualとChampion重みは不変。独立した `Strategic-Curriculum-Stage0-Post1` artifactを保存・再読込してGate/Critic digest一致を確認した。

held-out Pre10から固定したAction前149 stateでは、Pre→Post KL平均0.0001000、最大0.0003365、entropy平均0.47315→0.46906、top family反転0、最大family確率>0.95のstate数30→30。familyがavailableなstateでの平均確率（Pre→Post）はSettlement 0.96369→0.96407、City 0.94570→0.94616、Road 0.33285→0.33495、Development 0.67857→0.67862、BankTrade 0.36948→0.37508、EndTurn 0.15015→0.14568。これは未学習の同一局面における小さなPolicy shiftを示すが、対局性能の向上を意味しない。

## Held-out 72局 × 5 variant

全variantともRule/Champion/Mixed各24局、各profileのseat1–4各6局。同一game seedを使用し、delegation/family samplingは外部RNG。全360対局がterminalへ到達し、illegal、Executor failure、family mismatch、deadlock、truncation、nonfinite reward、GameState RNG arbitration消費はいずれも0。ImmediateWinResolverとprotected PlayerTradeは全variantで維持した。

| Variant | 勝利/72 | 平均VP | 平均rank | 平均policy steps | Settlement | City | 建設計 | Road | Dev | BankTrade | EndTurn | protected PlayerTrade | 実委譲 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Safety 0% | 42 | 8.486 | 1.785 | 74.944 | 205 | 126 | 331 | 443 | 167 | 272 | 1210 | 775 | 0 |
| Pre10 | 36 | 8.139 | 1.965 | 77.833 | 199 | 110 | 309 | 432 | 200 | 264 | 1263 | 780 | 149 |
| Post10 | 35 | 8.111 | 1.979 | 77.792 | 199 | 110 | 309 | 430 | 200 | 266 | 1263 | 779 | 149 |
| Pre20 | 39 | 8.264 | 1.861 | 78.875 | 196 | 110 | 306 | 415 | 221 | 299 | 1267 | 828 | 314 |
| Post20 | 39 | 8.264 | 1.861 | 78.889 | 196 | 111 | 307 | 416 | 220 | 298 | 1267 | 829 | 314 |

同一seedのpaired game差。CIはprofile層化bootstrap 5,000 resample、差は右variant−左variant、建設/Devは**1局当たり**の差である。

| 比較 | ΔVP [95% CI] | Δrank [95% CI] | Δsteps [95% CI] | Δ建設 [95% CI] | ΔDev [95% CI] |
|---|---:|---:|---:|---:|---:|
| Safety→Post10 | −0.375 [−0.792, +0.028] | +0.194 [+0.014, +0.389] | +2.847 [−0.084, +5.708] | −0.306 [−0.597, −0.028] | +0.458 [+0.042, +0.875] |
| Pre10→Post10 | −0.028 [−0.083, 0] | +0.014 [0, +0.042] | −0.042 [−0.125, 0] | 0 [0, 0] | 0 [0, 0] |
| Safety→Post20 | −0.222 [−0.681, +0.250] | +0.076 [−0.146, +0.306] | +3.944 [+0.125, +7.445] | −0.333 [−0.722, +0.083] | +0.736 [+0.278, +1.167] |
| Pre20→Post20 | 0 [0, 0] | 0 [0, 0] | +0.014 [0, +0.042] | +0.014 [0, +0.042] | −0.014 [−0.042, 0] |

Post10はSafety比で平均VP差−0.375（guardrail >−0.30を不通過）、建設比309/331=0.934（≥0.90通過）、policy steps比1.038（≤1.15通過）。CIもVPが−0.30を跨ぎ、建設・stepsについて安全側へ十分狭まっていないため、behavioral qualificationは不通過。Pre10自体がSafetyよりVP−0.347であり、Post10の悪化を今回の小さなupdateだけに帰属できない。一方、72局のPre→Postはほぼ同挙動で、改善が立証されたわけでもない。Post20のpoint guardrailは通るがCIは広く、Stage 1 readinessは `STAGE1_NOT_READY`。20% trainingや自動昇格は行わない。

Road B monitoring: Pre10/Post10とも実委譲Road23件、neither5件（21.7%）、同じstateでChampionもRoadだった比較ではsame edge9、different edge11。Pre20は54件中neither8件（14.8%）、same21/different23。Post20は55件中neither9件（16.4%）、same21/different24。neitherを禁止・除外せず、合法性・適用不変条件は全件通過した。

## 判定と次工程

**Technical: accepted。Behavioral: `UPDATE_ACCEPTED_STAGE0_NOT_QUALIFIED`。Stage 1: `STAGE1_NOT_READY`。** PPO更新・保存・freeze境界は成立したが、予約済みの新held-out seedではStage 0 VP guardrailを満たさない。Post artifactは診断用として保存し、Stage 0運用または追加PPOのparentへ自動採用しない。次のStep 3A-22では、Step 3A-20 qualification（別72 seed）との結果差をprofile/seatおよび最初の委譲分岐で監査し、独立seedを追加して10% operating envelopeの不確実性を再推定する案が妥当。Pre→Postの微小な変化、long macro durationとterminal分散を分け、Reward/Executor/candidate maskを即変更せず、追加PPO updateもしない。

Champion weight digestはrollout前・update後・held-outで `4225902d41a83540eb1f1552826ec623706e778b99ad4ae8ec015f976644392a` と一致。親 `Strategic-T2-preupdate/weights.pt` SHA-256も前後 `755e0b868999cd0a616b9529742196c8675ab88a0a21e5b4de0c54d0b702b4c2` で一致する。Stage 0 Post1のGate/Critic digestはmanifestに固定した。Stage 1 promotion statusは `LOCKED_PENDING_SEPARATE_DECISION` のまま。

実装は `app/rl/run_strategic_curriculum_step3a21.py`、専用テストは `tests/test_rl_strategic_curriculum_step3a21.py`。全backendテストは **347 passed、1,172 subtests passed**（依存ライブラリのdeprecation warning 2件）。生のepisode・Decision・transition・manifest・metricsは同名のevidence ZIPに保存する。
