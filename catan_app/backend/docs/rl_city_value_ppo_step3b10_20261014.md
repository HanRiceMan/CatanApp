# Step 3B-10 — City Value PPO Pilot

## 判定

**City-only局面での実委譲、独立Criticを用いたon-policy semi-MDP PPO更新、held-out再評価は技術的に成立した。一方、1回の更新で対局性能もL1/L2の実質的な利用も改善したとは言えない。** City Pre→Postの144局paired平均VP差は`-0.0069`（profile層別bootstrap 95% CI `[-0.0208, 0]`）。Safetyとの差はPostで`-0.0347` VP（`[-0.0764, 0]`）。City Postは実験artifactとして保存し、本番採用、City委譲率増加、Investmentとの同時委譲は行わない。

## 実装・責務境界

- [`city_value_ppo_step3b10.py`](../app/rl/city_value_ppo_step3b10.py): `CITY_VALUE_COMPARISON`専用の実委譲maskと独立`CityValueCritic`（33,857 parameters）。Criticは明示的なroute one-hot、本人既知状態、detachしたChampion latentとsoft strategic representationを入力する。Champion Critic値を教師にしない。
- [`hierarchical_rollout_step3b6.py`](../app/rl/hierarchical_rollout_step3b6.py): 既存Investment委譲をdefaultのまま保ち、City限定routeを追加。通常action phase・ImmediateWinなし・City合法・Settlement非合法の局面だけ抽選し、protected PlayerTrade、forced phase、Development Usage、Resolverを優先する。実委譲maskは診断用shadow maskと分離し、City/Road/BuyDev/BankTrade/EndTurnのうち合法かつ既存Frozen Executorで実行可能なfamilyのみをsampleする。Investment委譲はCity学習・City評価で`0%`。同turn City機会を読むobserverも追加し、Actionには影響させない。
- [`hierarchical_ppo_step3b6.py`](../app/rl/hierarchical_ppo_step3b6.py): 既存semi-MDP収集、GAE、PPO ratio、KL/collapse guardをroute・allowed family・delegation率でパラメータ化。古いInvestment defaultは維持。Actorの91,437 parameters（L1/L2 41,349、L3 50,088）とCity Criticだけを更新し、Frozen Championとはparameter ID交差0。
- [`run_city_value_ppo_step3b10.py`](../app/rl/run_city_value_ppo_step3b10.py): `prepare → collect → preflight → update → eval-profile → summarize → diagnose → publish`の再現可能なrunner。同じrolloutで2回目のoptimizer attemptを拒否する。3B-9 Post1 ActorをCity Preへ完全コピー（policy digest両方`ac4fbc0a…ff5bfc2`）、既存Investment Criticは継承せず新City Criticをseed `3910`で初期化した。City concrete actionは既存Frozen Executor、報酬はwin/loss `±1`とVP差`×0.05`のみ。Cityへの固定ボーナス、Planner/Champion imitation、Plan samplingはない。

学習はCity `25%`、評価はCity `10%`。`Investment → Champion`を保つ。Road/BankTradeはFrozen B Executor、Cityは既存City Executorを使用する。`PLAYER_TRADE`は引き続きprotectedで、sample maskから除外する。Action choiceを伴わないChampion区間も前の実委譲Actionから次の実委譲またはterminalまでのmacro rewardへ含めるが、PPO Actionとして数えない。真のterminalはbootstrap 0、truncationは学習対象外。γ`0.99`、GAE λ`0.95`、actor LR`1e-4`、critic LR`3e-4`、batch64、2 epochs、clip`0.1`、entropy coefficient`0.005`。

## 新規on-policyデータと1回の更新

学習seed: Rule `35800000–35800107`、Champion `35801000–35801107`、Mixed `35802000–35802107`。324局、各profile108、各seat81。balanced 12-game blockでterminalまで完走し、Cityの**実委譲132 transition**を収集した。最大492局より前に目標128へ到達。過去のInvestment rolloutは再利用していない。

| 学習分布 | 件数 |
|---|---:|
| Build City / Road / BuyDev / BankTrade / EndTurn | 42 / 3 / 24 / 32 / 31 |
| 候補数2 / 3 / 4 / 5 | 12 / 46 / 62 / 12 |
| terminal / nonterminal transition | 98 / 34 |
| macro duration 平均 / median | 18.38 / 12.5 learner steps |
| macro reward 平均 / SD | 0.2644 / 0.7476 |
| terminalの絶対Advantage量比 | 76.88% |

Preflightは全episode terminal、truncation・illegal・Executor failure・nonfinite・protected/Resolver混入0。保存入力の再生誤差はlatent最大`2.38e-6`、old log-prob`2.38e-7`、old probability`5.96e-8`、Critic value`0`。正規化前Advantage平均/SDは`0.3027/0.8394`、terminal `0.3415/0.8536`、nonterminal `0.1909/0.7864`。terminal寄与が依然大きいため、City固有の価値推定が十分学習されたとは見なさない。

| PPO診断 | 値 |
|---|---:|
| actor / critic / entropy loss平均 | -0.17303 / 0.82550 / -0.006086 |
| ratio mean / min / max | 1.01662 / 0.84836 / 1.13349 |
| clip fraction | 0.109375 |
| categorical KL mean / median / p95 / max | 0.004602 / 0.004412 / 0.006397 / 0.006806 |
| L3 entropy before → after | 1.23884 → 1.24119 |
| L3 / L2 / L1 gradient norm平均 | 2.12495 / 0.003154 / 0.001645 |
| L3 / L2 / L1 parameter delta | 0.08073 / 0.005131 / 0.002569 |
| City Critic return MSE before → after | 0.79618 → 0.75736 |

KL・family collapse guardは通過（`>.95`確率stateなし）。Cityがavailableなtraining stateで平均P(City)は`0.2696→0.2955`、EndTurnは`0.3211→0.2815`。これは同じ収集state上の分布変化であり、Cityを選べば良いというルールや対局改善の証拠ではない。L1/L2に非ゼロ勾配とparameter deltaはあるが、L3との比では小さい。

## 新seed held-out 144局×4条件

評価seed: Rule `35900000–35900047`、Champion `35901000–35901047`、Mixed `35902000–35902047`。各条件144局（各profile48、各seat36）、同game seed・外部分離RNG系列。Safety 0%、旧3B-9 Post1はInvestment10%・City Champion、City Pre/PostはInvestment0%・City10%。旧Post1とCity版は委譲routeが異なるため直接の学習効果比較ではない。

| 条件 | 勝ち/144 | 平均VP | 平均順位 | 平均steps | 実委譲 | Settlement | City | Road | Dev | Bank | EndTurn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Safety | 74 | 8.174 | 1.927 | 76.903 | 0 | 426 | 271 | 865 | 290 | 575 | 2498 |
| 3B-9 Post1 Investment10 | 76 | 8.146 | 1.875 | 76.826 | 256 | 420 | 261 | 864 | 299 | 556 | 2507 |
| City Pre10 | 71 | 8.146 | 1.951 | 76.854 | 31 | 424 | 265 | 865 | 291 | 579 | 2508 |
| City Post10 | 71 | 8.139 | 1.958 | 77.007 | 31 | 426 | 264 | 867 | 291 | 587 | 2513 |

City Pre→Postのpaired差はVP`-0.00694`（95% CI `[-0.02083,0]`）、rank`+0.00694`（`[0,+0.02083]`）、steps`+0.1528`（`[-0.00694,+0.34722]`）。Safety→City PostはVP`-0.03472`（`[-0.07639,0]`）、rank`+0.03125`（`[+0.00694,+0.0625]`）。勝率は3条件とも小差で、144局から一般的な勝率優劣を断定しない。City Pre/PostのAction traceは138/144局で同一だった。City Postの委譲familyはCity9、Road1、Dev1、Bank12、End8。City建設率を上げるhard ruleを設けていない。

同turn City機会の診断はSafety`233/264=88.26%`、City Pre`229/274=83.58%`、City Post`228/271=84.13%`。City Postではimmediate City209、Cityを建てず他の得点建設6、Cityなし43。実委譲がそのturn最初のCity機会だった28件ではCity同turn16件（57.1%）。これは選択された少数局面の観測であり、Cityを見送ることが因果的に悪いと決めるものではない。protected PlayerTrade総数はSafety1676、旧Post1 1683、City Pre1665、City Post1671。

## Plan / Priorityの実際の影響とInvestment退行監視

City Preのheld-out実委譲31 stateにPre/Postを同一入力・maskで適用した。Postの有効Plan間pairwise TVは平均`0.01156`、Plan A/Bを指定した場合は`0.01139`でargmax flip`2/31`。**異なるPlanで異なるfamily分布を表現する能力はある。** しかし通常のPost分布に対するPlan一様化TVは`0.00000872`、Priority全0.5化は`0.00001629`、両中立化は`0.00002363`で、いずれもargmax flip`0/31`。Plan entropyのuniformとの差は`8.59e-7`、Priority平均は`[0.49881,0.50045,0.50017,0.50163]`、state間SDは各軸`0.00015–0.00035`。この1更新ではL1/L2を**実際のCity判断へ有意に使う状態には達していない**。

同じActorを旧固定Investment 218 stateへ当てると、City Pre→Postのfamily TV平均は`0.03953`、argmax flip`118/218`。Cityで学習した共有L3がInvestmentの出力も動かす。ただしCity学習・City評価のInvestment実委譲はともに0なので、これは**出力退行リスク**であって対局成績のInvestment退行を実証した値ではない。Investment stateでのPlan一様化TVは`0.00000504`、Priority中立化`0.00001590`。City PostをそのままInvestmentにも委譲しない。

## 安全性・artifact・テスト

全576 held-out episodeはterminal、truncation0、illegal0、Executor failure0、family mismatch0、deadlock0、nonfinite reward0、GameState RNG arbitration消費0。protected PlayerTradeとImmediateWinResolverを維持。0% SafetyのAction trace・最終GameState・winner・RNG parityは144/144。Champion digest `4225902d…644392a`、Robber digest `ecde7fa9…702bf`は不変。3B-9 parent weights SHA256 `8325e0c9…ba6`、City Pre `f4ca0e9a…e34d`、City Post `ef6641a1…09ba6`。parentのActor policy digestとCity Preは完全一致。独立Pre/Post artifact、圧縮rollout・heldout個票、[`preflight.json`](rl_city_value_ppo_step3b10_20261014_evidence/preflight.json)、[`update_result.json`](rl_city_value_ppo_step3b10_20261014_evidence/update_result.json)、[`evaluation_summary.json`](rl_city_value_ppo_step3b10_20261014_evidence/evaluation_summary.json)、[`fixed_state_diagnosis.json`](rl_city_value_ppo_step3b10_20261014_evidence/fixed_state_diagnosis.json)を保存した。

City専用unit 5件と旧3B-6/8/9回帰を通過。最終backend全テストは**413 passed、1,172 subtests passed、2件の既存依存関係warning**（81秒）。既定のOS Tempへのpytestアクセス拒否で最初の全テスト実行にsetup errorが2件出たため、workspace内の新規`--basetemp`で全件を再実行して成功した。テスト失敗やコード変更による例外ではない。

## 次工程（Step 3B-11案）

City委譲は技術pilotとして保管し、**自動拡大しない**。まず共有Actorのroute間干渉を制御する手段（小さいroute専用adapter、または旧Investment固定stateへの分布保持制約など）を設計・比較する。同時に、自然のPlan分布とPriorityがほぼ中立に留まる理由を、L1/L2の識別可能なreturn signalと混合分布でのgradient希釈の両面から検証する。Plan A/Bを手動指定した感度のみを「勝ち筋を学習した」とみなさない。これらの境界を評価してからCity Valueの追加on-policy収集、またはDual Construction/Development Usageへの拡張を判断する。
