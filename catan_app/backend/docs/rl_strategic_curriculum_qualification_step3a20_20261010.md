# Step 3A-20 — Strategic Gate 低delegation Stage 0 qualification

2026-10-10。Step 3A-16の **Strategic-T2-preupdate**（temperature 2、Strategic residual 0）をparentとして固定し、Safety Champion（ImmediateWinResolver ON、0%）と10%・20%実委譲を新規72 game seedで比較する。20%はStage 1候補の参考評価であり、今回Stage 0を20%へ変更しない。PPO/Critic update、Reward、Strategic candidate mask、Runtime Context79、Frozen B Executor、Champion/Gate weightsは変更していない。

## 方法・境界

Rule / Champion / Mixed各24局、各profileでseat 1〜4を6局ずつ配置した。game seedは `32000000…32000023` / `32001000…32001023` / `32002000…32002023`。0/10/20%の3条件は同一game seedを使用し、delegation・family samplingにはそれぞれ `game_seed+12345`・`game_seed+54321` の外部RNGを使用する。GameState RNGはarbitrationで消費しない。すべてのDecisionはImmediateWinResolver → forced/phase controller → protected PlayerTrade → Strategic Surface → delegation抽選 → T=2 Gate → Frozen Executorの順で処理する。

1局単位のcheckpointにOutcome、Action family counts、Gate decision、semi-MDP transition、Road B構造診断を保存する。実委譲Roadだけ、既存の決定的 `_road_purposes()` で expansion / longest / both / neither を分類する。`neither` は当該2基準で進捗がない意味であり、価値がないことや失敗を意味しない。Championが同じAction前状態でRoadを選んだ場合はedge一致を比較するが、Champion edgeを教師にはしない。

全216局についてterminal、truncation、非有限reward、Executor成功、concrete Action family整合、Road ownership/resource消費、GameState RNG非消費、transition境界と割引報酬を検査する。0%はSafety Championであり、Legacy Championとの一致を要求しない（Resolverによる確定勝利Actionを含み得る）。

同一game seedの差 `rate − Safety` を、profile内で24局ずつ復元抽出する層化paired bootstrap 5,000回で評価する。CIはゲーム単位の不確実性であり、異なるrate間の純粋なdelegation因果効果ではない。10%の暫定guardrailは平均ΔVP > −0.30、建設数Safety比≥90%、平均policy steps Safety比≤115%。point estimateと95% CIを併記し、CIが閾値を跨ぐ場合は安全を確定したとは扱わない。

Stage 0 curriculum schemaは、stage_id / delegation_probability / prior / parent_artifact / training_transition_target / qualification_seed_range / evaluation_seed_range / promotion_status / promotion_reasonを保持する。Stage 0は10%・native T=2・pre-update parent・256〜320実委譲transition、Stage 1候補は20%だがLOCKED。自動昇格なし。Stage 0学習後は同じ10%のheld-outで安全性・family/entropy collapse・pre-updateからの悪化を確認し、20% probeで重大な建設崩壊がない場合のみ次stageを検討する。Full Surface性能はStage 0のpromotion条件にしない。

## 72局×3条件の結果

| rate | 終局/局数 | 実委譲 | /局 | Surface | protected preemption | Settlement | City | 建設計 | Road | Dev | Bank | End | PlayerTrade | 勝利 | 平均VP | 平均順位 | 平均step |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Safety 0% | 72/72 | 0 | 0 | 2,121 | 741 | 196 | 124 | 320 | 405 | 139 | 237 | 1,211 | 794 | 33 | 7.903 | 1.979 | 72.40 |
| 10% | 72/72 | 156 | 2.17 | 2,292 | 811 | 206 | 108 | 314 | 423 | 168 | 269 | 1,264 | 870 | 32 | 7.931 | 2.069 | 77.07 |
| 20% | 72/72 | 299 | 4.15 | 2,199 | 749 | 201 | 103 | 304 | 432 | 193 | 262 | 1,241 | 827 | 34 | 7.972 | 2.021 | 76.26 |

各rateでRule / Champion / Mixed **24局ずつ**、各profile×seatは**6局ずつ**。同一game seedで比較し、外部delegation/family RNG seedも各rateで同じ式を使った。Surface・protected件数がrate間で違うのは到達stateが変わるため。10%ではSafety比で建設 **98.1%**、policy steps **106.4%**。20%では建設 **95.0%**、policy steps **105.3%**。両rateともStep 3A-19の50〜100%で見た約20〜32%の建設減少は再現していない。ただしDev購入はSafety 139 → 168 → 193と増加し、今後も監視が必要である。

| 実委譲family | 10%件数（/局） | 20%件数（/局） | 256 transition時の自然予測 |
|---|---:|---:|---:|
| BUILD_SETTLEMENT | 18 (0.250) | 36 (0.500) | 29.5 |
| BUILD_CITY | 16 (0.222) | 23 (0.319) | 26.3 |
| BUILD_ROAD | 30 (0.417) | 58 (0.806) | 49.2 |
| BUY_DEVELOPMENT | 34 (0.472) | 83 (1.153) | 55.8 |
| BANK_TRADE | 34 (0.472) | 54 (0.750) | 55.8 |
| END_TURN | 24 (0.333) | 45 (0.625) | 39.4 |

10%で6 familyすべてに実sampleがある。256件への単純な比率外挿では最少のCityでも約26件であり、人工的なfamily oversamplingなしで初回coverageを見込める。ただしこの外挿は将来の学習Policyでfamily分布が変わらない保証ではない。

## 同一gameのpaired差とguardrail

数値は `rate − Safety` の72 game平均。括弧内はprofile層化paired bootstrap **5,000再標本**の95% CI。

| 差 | 10% | 20%（Stage 1参考） |
|---|---:|---:|
| final VP / game | +0.028 [−0.278, +0.333] | +0.069 [−0.347, +0.472] |
| rank / game（小が良い） | +0.090 [−0.056, +0.236] | +0.042 [−0.125, +0.215] |
| policy steps / game | +4.667 [+2.125, +7.250] | +3.861 [+0.500, +7.084] |
| Settlement+City / game | −0.083 [−0.375, +0.194] | −0.222 [−0.583, +0.139] |
| Development / game | +0.403 [+0.056, +0.764] | +0.750 [+0.333, +1.167] |

10%のpoint guardrailは3項目とも通過した。平均ΔVP **+0.028 > −0.30**、建設比 **0.981 ≥ 0.90**、step比 **1.064 ≤ 1.15**。bootstrap下限もVP **−0.278 > −0.30**、建設差 **−0.375/局 > −0.444/局**（Safety 320件の90%閾値）、step差上限 **+7.25/局 < +10.86/局**（Safety平均の15%）で、今回の暫定閾値に対するCI側も通過した。ただしVP下限と建設下限の余裕はそれぞれ **0.022 VP/局**、**0.069建設/局**しかない。性能改善の証明ではなく、Stage 0の**限定的な運用資格**である。20%のΔVP CIは−0.30を跨ぐため、20%を今回の初回学習率へ昇格させない。

## Road B・hard safety

| rate | 実委譲Road | expansion+longest | expansionのみ | longestのみ | neither（率） | ChampionもRoad | 同edge / 異edge |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10% | 30 | 12 | 8 | 5 | 5 (16.7%) | 22 | 9 / 13 |
| 20% | 58 | 23 | 14 | 10 | 11 (19.0%) | 46 | 20 / 26 |

Road Bの全実委譲でActionは合法、道路所有権・木/レンガ1枚ずつの消費・copy-apply時のGameState RNG非消費が成立した。Road B Executor failureとRoad適用不変条件違反は0。従ってStage 0では暫定維持する。`neither` 5/30・11/58やChampion edge不一致は監視項目であり、Roadを禁止したりChampion edgeへ置換したりしない。

全rateでterminal **72/72**、illegal Action / Executor failure / family mismatch / deadlock / truncation / nonfinite reward / arbitrationによるGameState RNG消費は**各0**。実委譲156/299件とStrategicTransition件数は一致し、途中のprotected PlayerTrade・Resolver・非委譲SurfaceはActor sampleへ混入していない。transitionのmacro duration、次の実委譲Surfaceまたはterminalでのclose、`Σ γ^i r_i` の再計算、terminal bootstrap=0を検証した。macro durationは10%で平均22.29・中央値18.5・p95 53・最大99、20%で平均13.67・中央値9・p95 41・最大81 policy steps。これはGateが実際に再委譲されるまでの既存learner decision step数である。

## Stage 0運用判断と次工程

**Stage 0（10%）は初回Strategic PPOを再開する候補としてqualified。今回PPO/Critic updateやStage 1昇格は行っていない。** Parentは`Strategic-T2-preupdate`、T=2、6 family、Frozen B Executor、Safety layerは固定する。今回の10%実測 **156 / 72 = 2.167 transition/局**から、256件には約**119局**、320件には約**148局**が必要。profile×seatの均衡を維持する12局単位ではそれぞれ**120局 / 156局**を目安にする。前回の36局screenの2.19/局と近い。

Step 3A-21の第一案は、別training seedで10%委譲・256〜320実on-policy transitionをepisode完走単位で収集し、**1 guarded PPO updateのみ**行うこと。Pre Stage 0 vs Post Stage 0を未使用のheld-out seedで同じ10%で比較し、hard safety・family/entropy collapse・建設・VP・stepの重大悪化を確認する。20%は診断probeであり、自動stage promotionしない。100% Full SurfaceをStage 0の合格条件にしない。Dev購入増加とRoad `neither`率は引き続き記録するが、新しいReward、mask、Executor ruleは追加しない。

Championのin-memory weight digest、Gate/Critic digest、parent artifact `weights.pt` SHA-256はいずれも収集前後一致した。全backend test **345 passed、1,172 subtests passed**（既存依存のdeprecation warning 2件のみ）。実行コード、全216局checkpoint、qualification summary、curriculum manifestは証跡ZIPに収める。
