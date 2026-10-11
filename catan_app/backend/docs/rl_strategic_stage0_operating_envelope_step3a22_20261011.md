# Step 3A-22 — Stage 0 10% Operating Envelope / Seed Variance Audit

2026-10-11。対象は未学習 `Strategic-T2-preupdate`、temperature 2、residual 0、Strategic delegation 10%。比較はImmediateWinResolver ONのSafety Champion（delegation 0%）。両者で同一game seedを使い、delegationおよびfamily samplingは外部RNG（各game seed+12345/+54321）に分離した。Reward、79次元Runtime Context、Strategic candidate mask、Frozen Road/BankTrade Executor、protected PlayerTrade、Champion、Gate、Criticは変更しない。optimizerは作成せず、PPO updateは0回。

Set A（`32000000/32001000/32002000`）はStep 3A-20、Set B（`32100000/32101000/32102000`）はStep 3A-21の原記録を再利用。Set C（`32300000/32301000/32302000`）、Set D（`32400000/32401000/32402000`）を新規収集。各setでRule/Champion/Mixed各24局、各profileのseat1–4各6局。合計288同seedペア。各setのCIはprofile層化paired bootstrap 5,000回、pooled CIはset×profileの12層（各24局）を固定したpaired bootstrap 5,000回で推定する。異なるtrajectory間の差であり、実委譲回数との相関やfirst divergence別outcomeは探索的診断で、純粋なAction因果効果ではない。

First divergenceは、保存済みPre10の実委譲Decisionで `concrete_action` と同一Action前stateの `champion_counterfactual_action` が初めて異なる点とする。既存A/B原記録には全Action列そのものを保存していないため、全phaseのtraceを再構成したとの主張はしない。分岐なし分類では最終winner/VP/rank/policy steps/RNGおよびlearner Action family countがSafetyと一致することも確認する。pooled CIは観測した4 seed setを固定層とする条件付き推定であり、未知のseed-set母集団への外挿にはset間SDと各set内SEを併記する。

## 結果

新規C/D各72ペアの全対局がterminalまで完走した。A/B/C/Dの各setはSafety/Pre10とも72局で、profile各24局、各profile×seat各6局。新規C/Dの144ペア・288対局はillegal / Executor failure / family mismatch / deadlock / truncation / nonfinite reward / GameState RNG arbitration消費がすべて0。A/Bも元記録を再検証した。全4 setのhard safetyはPASS。収集・解析の前後でChampion、未学習Gate、Critic、親artifactのdigestが一致し、Step 3A-21 Post1 artifactのSHA-256も元の保存記録と一致する。optimizerは作成していない。

各setの結果。Actionはlearnerのaction-phase family総数で、Tradeはprotected PlayerTrade。`S+C`が建設計。

| Set・variant | 勝/72 | VP | rank | steps | S | C | S+C | Road | Dev | Bank | End | Trade | 実委譲 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A Safety | 33 | 7.903 | 1.979 | 72.40 | 196 | 124 | 320 | 405 | 139 | 237 | 1211 | 794 | 0 |
| A Pre10 | 32 | 7.931 | 2.069 | 77.07 | 206 | 108 | 314 | 423 | 168 | 269 | 1264 | 870 | 156 |
| B Safety | 42 | 8.486 | 1.785 | 74.94 | 205 | 126 | 331 | 443 | 167 | 272 | 1210 | 775 | 0 |
| B Pre10 | 36 | 8.139 | 1.965 | 77.83 | 199 | 110 | 309 | 432 | 200 | 264 | 1263 | 780 | 149 |
| C Safety | 33 | 7.750 | 2.007 | 71.07 | 193 | 116 | 309 | 412 | 147 | 256 | 1163 | 800 | 0 |
| C Pre10 | 35 | 7.694 | 2.014 | 73.75 | 184 | 100 | 284 | 427 | 189 | 234 | 1189 | 818 | 124 |
| D Safety | 34 | 7.833 | 2.021 | 74.54 | 202 | 121 | 323 | 411 | 135 | 289 | 1238 | 803 | 0 |
| D Pre10 | 33 | 7.667 | 2.111 | 77.11 | 191 | 112 | 303 | 399 | 178 | 280 | 1259 | 851 | 141 |

各setのPre10−Safety paired差。角括弧は各set内profile層化5,000 resampleの95% CIで、建設/Devは1局当たり差。

| Set | ΔVP | Δrank | Δsteps | Δ建設 | ΔDev |
|---|---:|---:|---:|---:|---:|
| A | +0.028 [−0.264, +0.333] | +0.090 [−0.063, +0.236] | +4.667 [+2.194, +7.278] | −0.083 [−0.361, +0.194] | +0.403 [+0.042, +0.750] |
| B | −0.347 [−0.764, +0.056] | +0.181 [0, +0.375] | +2.889 [−0.028, +5.792] | −0.306 [−0.597, −0.027] | +0.458 [+0.042, +0.875] |
| C | −0.056 [−0.417, +0.319] | +0.007 [−0.139, +0.153] | +2.681 [−0.139, +5.598] | −0.347 [−0.653, −0.056] | +0.583 [+0.319, +0.847] |
| D | −0.167 [−0.444, +0.111] | +0.090 [−0.042, +0.222] | +2.569 [−0.320, +5.389] | −0.278 [−0.542, −0.014] | +0.597 [+0.333, +0.875] |

288ペアpooled（set×profile 12層固定bootstrap 5,000回）は、ΔVP **−0.1354 [−0.3056, +0.0417]**、Δrank +0.0920 [+0.0139, +0.1684]、Δsteps +3.201 [+1.767, +4.573]、Δ建設 −0.2535 [−0.3958, −0.1076]、ΔDev +0.5104 [+0.3403, +0.6771]。Safetyの建設計1,283に対しPre10は1,210で比0.9431。steps比は1.0437。VP下限の境界性を確認するため別seedの20,000 resampleを2回追加したが、両方とも下限−0.3056だった。これを「安全と証明」と読み替えない。

Set平均ΔVPはA +0.0278、B −0.3472、C −0.0556、D −0.1667。最大−最小は0.375、between-set SD 0.1621（variance 0.02628）。各set内SEはA 0.1619、B 0.2095、C 0.1896、D 0.1440。set間のばらつきは各set内の標本誤差と同程度で、A/Bの差だけを特異なseed効果と断定できない。建設/Devを含む全heterogeneity値は証跡JSONに保存した。

### SubgroupとAction分岐

以下はすべてPre10−Safetyの平均（nはpaired game数）。小さいcellからseat固有ruleは作らない。

| 区分 | n | ΔVP | Δ建設 | ΔDev | Δsteps |
|---|---:|---:|---:|---:|---:|
| Rule | 96 | +0.156 | −0.292 | +0.823 | +5.750 |
| Champion | 96 | −0.333 | −0.313 | +0.458 | +1.719 |
| Mixed | 96 | −0.229 | −0.156 | +0.250 | +2.135 |
| Seat 1 | 72 | −0.083 | −0.361 | +0.542 | +2.708 |
| Seat 2 | 72 | −0.153 | −0.208 | +0.444 | +3.819 |
| Seat 3 | 72 | −0.319 | −0.125 | +0.278 | +1.792 |
| Seat 4 | 72 | +0.014 | −0.319 | +0.778 | +4.486 |

| Profile / seat | n | ΔVP | Δ建設 | ΔDev | Δsteps |
|---|---:|---:|---:|---:|---:|
| Rule/1 | 24 | +0.417 | −0.250 | +0.667 | +3.042 |
| Rule/2 | 24 | −0.250 | −0.500 | +0.833 | +6.292 |
| Rule/3 | 24 | +0.042 | −0.042 | +0.458 | +6.125 |
| Rule/4 | 24 | +0.417 | −0.375 | +1.333 | +7.542 |
| Champion/1 | 24 | −0.583 | −0.417 | +0.583 | +4.958 |
| Champion/2 | 24 | −0.083 | −0.208 | +0.417 | +1.708 |
| Champion/3 | 24 | 0 | +0.125 | +0.292 | +0.083 |
| Champion/4 | 24 | −0.667 | −0.750 | +0.542 | +0.125 |
| Mixed/1 | 24 | −0.083 | −0.417 | +0.375 | +0.125 |
| Mixed/2 | 24 | −0.125 | +0.083 | +0.083 | +3.458 |
| Mixed/3 | 24 | −1.000 | −0.458 | +0.083 | −0.833 |
| Mixed/4 | 24 | +0.292 | +0.167 | +0.458 | +5.792 |

89/288局はConcrete Action分岐なし（A/B/C/D = 19/21/27/22）。この群の最終VP・rank・steps・建設・Dev・RNG差はすべて0。199局は分岐あり、平均ΔVP−0.196、Δ建設−0.367、ΔDev+0.739。first divergenceの主なfamily pairはRoad→Road edge差31、EndTurn→Dev32、EndTurn→BankTrade32、Road→Dev23、Road→EndTurn21、BankTrade→Dev11。例えばRoad edge差31局では平均ΔVP−0.290・Δ建設−0.516、EndTurn→BankTrade32局では−0.625・−0.219、一方EndTurn→Dev32局では+0.125・−0.438。これらは後続trajectoryと他の実委譲も含む群間の記述であり、first Actionそのものの因果returnではない。全pairと各gameのfirst-divergence recordは証跡へ保存した。

1局の実委譲Decision数は0:45局、1–2:144局、3–4:85局、5以上:14局。各群の平均ΔVPは0、−0.118、−0.212、−0.286、Δ建設は0、−0.250、−0.294、−0.857、ΔDevは0、+0.438、+0.812、+1.071。実委譲回数とΔVPのPearsonは−0.053で、bucket平均の順序を因果dose-responseとは呼べない。実委譲family exposureとΔVPのPearsonはDev +0.069、Road −0.072、BankTrade −0.129、EndTurn −0.052。弱い探索的相関であり、Dev単独の悪さやBankTrade禁止ruleの根拠にはならない。

Road BはA/B/C/Dの実委譲30/23/26/25件、合計104件。`neither` は5/5/5/6件、計21/104=20.2%。同一stateのChampion counterfactualもRoadだった77件ではsame edge30、different edge47。Roadは合法・適用不変条件を全件通過し、`neither`を除外・変更していない。

凍結確認: Champion digest `4225902d41a83540eb1f1552826ec623706e778b99ad4ae8ec015f976644392a`、未学習Gate digest `04e1c20536c644eaf4197590f775f6af800f1601eba54b57e44ccf6851782d51`、専用Critic digest `6a46a420444316af5c596986bc78da6dc7f71818a393381fb304e23d0a17c1c3`、親weights SHA-256 `755e0b868999cd0a616b9529742196c8675ab88a0a21e5b4de0c54d0b702b4c2` が前後一致。Post1 weights SHA-256もStep 3A-21記録と同じ `74f43112bff5468712bc7b646cae8195241ae344c2ff0d3b8548048f12b109bc`。Post1を評価にも学習にも使用していない。

### 判定と次のStep

既存guardrailのpoint estimateは、平均ΔVP −0.1354 > −0.30、建設比0.9431 ≥0.90、steps比1.0437 ≤1.15で通過。ただし95% CIではΔVP下限**−0.3056**が閾値−0.30をわずかに下回る。建設CI下限−0.3958件/局は許容下限−0.4455より上、steps CI上限+4.573は許容増加+10.986より下。従って総合Caseは **BORDERLINE / INCONCLUSIVE**。10%をREQUALIFIEDとはしないが、pooled pointが不通過のFAILEDでもない。Stage 0を「安定した学習環境」として再開すること、Stage 1へ昇格することは保留する。5%への再設計は現時点では必須でなく、追加seed後も10%のpoint guardrailを外す場合の次候補とする。

Step 3A-23案: 事前にseedを固定した独立Set E/F（各72ペア）を追加し、同じset×profile層化CIとset間SDを更新する。さらにfirst divergenceおよびprofile別の悪化が再現するか診断する。まだPPO、Critic、Reward、Context、candidate mask、Prior、Executorには手を加えない。Step 3A-21 Post1は技術検証用artifactとして保持し、今回のPolicyにも学習parentにも用いない。

実装は `app/rl/run_strategic_stage0_envelope_step3a22.py`、専用テストは `tests/test_rl_strategic_stage0_envelope_step3a22.py`。詳細なset別・各game・bootstrap・weight digestは同名evidence ZIPへ格納する。A/Bの原対局記録は既存Step 3A-20/21 evidenceに保存済みで、今回のZIPには新規C/Dの原対局と4-set paired analysisを収めた。

全backendテストは **350 passed、1,172 subtests passed**。依存ライブラリのdeprecation warningが2件あるが、失敗は0。
