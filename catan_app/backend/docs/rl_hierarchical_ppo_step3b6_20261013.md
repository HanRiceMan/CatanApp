# Step 3B-6 — Investment限定 Hierarchical PPO 初回更新

## 結論

L1 Victory Plan Scorer、L2独立Priority Head、L3 Investment Actor、専用Criticの経路で、実委譲されたon-policy returnから**1回のguarded PPO更新**を実行した。Technical guardは通過。Champion/Robberおよび既存3A artifactは変更していない。長期Victory Planの完成やChampion超えを示すものではない。

## 実装と責務境界

- `app/rl/hierarchical_ppo_step3b6.py`: Investment実委譲mask、Actor+独立Critic、semi-MDP GAE、全件preflight、1回のguarded PPO。
- `app/rl/hierarchical_rollout_step3b6.py`: ImmediateWin→forced/Dev usage→protected PlayerTrade→INVESTMENT→10%抽選→L3 sample→既存B Frozen Executor→環境、の実Action rollout。
- `app/rl/run_hierarchical_ppo_step3b6.py`: 独立pre/post artifact、balanced rollout、preflight、one-shot marker、held-out比較、固定state probe、evidence公開。
- `tests/test_hierarchical_ppo_step3b6.py`: mask、全invalid、可変duration GAE、L3→L1/L2勾配とFrozen latent遮断。

実委譲familyは `BUILD_ROAD / BUY_DEVELOPMENT / BANK_TRADE / END_TURN` のみ。Shadowの7-family maskとは別に、実際の377合法ActionとExecutor対象を交差したtraining maskを使用した。PlayerTradeはChampionが選んだ時点でpreemptし、L3 sampleにも入れない。建設3 route、pre-roll/action Development Usage、free road等もChampion/既存処理のまま。Plan/PriorityはAction maskにしていない。

Actorは3B-5と同じseedのL1/L2・L3初期重みから開始し、Criticだけ独立初期化した。L1/L2/L3 Actorの合計は76,108 parameter、Criticは33,345 parameter。Critic入力のsoft Plan/Priorityはdetachし、L1/L2の更新信号はL3 Actor lossからだけ受ける。Champion latentは常時detach。3B-5のsort順digestでL1/L2 `2000b139...`、L3 `a5bd513d...` と一致した。

## On-policy rollout とpreflight

training seedはRule `35000000–35000031`、Champion `35001000–35001031`、Mixed `35002000–35002031`。96局・各profile32局・各profile×seat8局。delegation/family RNGはgame RNGと分離した。

| 指標 | 結果 |
|---|---:|
| actual delegated StrategicTransitions | 269 |
| Road / BuyDev / BankTrade / EndTurn選択 | 22 / 26 / 37 / 184 |
| candidate数 1 / 2 / 3 / 4 | 117 / 87 / 55 / 10 |
| 実質的な2候補以上の比較sample | 152 |
| L3 entropy mean | 0.5003 |
| Plan entropy mean | 2.0619 |
| macro duration mean / median / max | 17.49 / 12 / 86 |
| macro reward mean / SD | 0.0492 / 0.5213 |
| terminal / nonterminal macro | 88 / 181 |
| protected trade / Resolver / 非委譲Investment cross | 692 / 41 / 1582 |

1候補局面117件は全てEndTurnであり、その局面のActor log-probabilityは0で勾配を持たない。Criticとsemi-MDP boundaryには含む。family別return/Advantageを固定価値表やTeacher scoreとして解釈しない。

GAEは3A定義と同じく、`delta = R_macro + gamma^duration V(next_delegated) - V(current)`、`A = delta + gamma^duration lambda A(next delegated)`。`gamma=.99`、`lambda=.95`。terminalの次値は0、truncationは収集対象から除外した。報酬は既存 `win=+1, loss=-1, VP delta=.05` のまま。

preflightで全96局terminal、truncation/illegal/Executor failure/family mismatch/nonfinite reward = 0。保存old分布・log probability・Critic値・Plan entropyの最大再現誤差は0、Frozen Champion latentのみ約`2.9e-6`。raw Advantage mean/SD `0.1516/0.7423`、normalized最大絶対値`1.55`、terminal絶対Advantage mass比`0.4048`。terminal Advantage SD `0.9065`、nonterminal SD `0.6400`。更新前Critic MSE `0.5741`。

## 指定1回のPPO更新

`actor_lr=1e-4, critic_lr=3e-4, batch=64, epochs=2, clip=.1, entropy_coef=.005`。この2 epoch・10 minibatchを**1回のPPO update**とし、再実行不能なattempt markerをoptimizer作成前に保存した。

| 指標 | 結果 |
|---|---:|
| Actor / Critic / entropy loss mean | -0.00117 / 0.54196 / -0.00249 |
| clip fraction | 0 |
| PPO ratio mean / min / max | 1.00050 / 0.98374 / 1.02245 |
| KL mean / median / p95 / max | 0.0000259 / 0.00000083 / 0.0000962 / 0.0001380 |
| L3 entropy mean pre→post | 0.500336→0.500310 |
| family collapse / entropy collapse | なし / なし |
| Critic MSE pre→post（training state） | 0.5741→0.5546 |

初回minibatchではL3最終層がzero-initのため上流勾配0を許容した。その後のgradient norm meanはL3 `0.3164`、Priority `3.29e-6`、Plan Scorer `1.04e-6`。parameter L2差分はPriority `0.00574`、L1はPlan Scorer `0.00377`、共有encoder `0.00122`、L3 `0.06348`、Critic `0.22790`。よって上流への**勾配疎通と更新可能性**は実rollout上で確認した。ただし信号の大きさは小さく、戦略的な意味の学習が完成したとは解釈しない。

training stateでfamilyがavailableな時の平均確率はRoad `0.3841→0.3805`、BuyDev `0.3829→0.3828`、BankTrade `0.3909→0.3965`、EndTurn `0.6741→0.6733`。どれも小変化。EndTurnが高い主因は1候補局面117件であり、この統計だけを「学習したEndTurn嗜好」と解釈しない。

## Held-out評価

評価seedはRule `35100000–35100023`、Champion `35101000–35101023`、Mixed `35102000–35102023`。各variant 72局、各profile24局・seat各6局。同じgame/delegation/family seedでSafety 0%、Pre10、Post10を比較する。Safetyは既存3A Safety ControllerとのAction/State/RNG parityも確認する。

| Variant（各72局） | 勝利 | 平均VP | 平均rank | 平均policy steps | 実委譲 | Road / BuyDev / BankTrade / EndTurn（全Action） |
|---|---:|---:|---:|---:|---:|---:|
| Safety 0% | 36 | 8.278 | 1.840 | 77.125 | 0 | 443 / 167 / 242 / 1270 |
| Pre10 | 35 | 8.236 | 1.854 | 76.667 | 218 | 443 / 167 / 253 / 1277 |
| Post10 | 35 | 8.236 | 1.854 | 76.444 | 217 | 441 / 166 / 254 / 1273 |

Pre10→Post10のpaired差は平均VP `0.000`（95% bootstrap CI `[0, 0]`）、rank `0.000`（同`[0, 0]`）、policy steps `−0.222`（同`[−0.667, 0]`）。72局では勝敗・VP・rankの変化は観測されなかった。Safety→Post10はVP `−0.0417`（CI `[−0.347, +0.250]`）、rank `+0.0139`、steps `−0.681`。建設ActionはSafetyのSettlement/City `205/113`、Pre10 `200/103`、Post10 `200/103`。このsampleから改善を主張できず、Safetyからの大幅な悪化も除外できない。委譲率引上げやCity route実委譲は保留する。

Pre10の実委譲familyはRoad `26`、BuyDev `17`、BankTrade `38`、EndTurn `137`。Post10はそれぞれ`26/17/38/136`。Pre10→Post10で全Action分布はほぼ不変。Plan entropyは `2.1001→2.1047`、L3 entropyは `0.54815→0.54746`。Priority平均は未学習Preの約`[.5,.5,.5,.5]`からPostの`[.50115,.50034,.50095,.50087]`へ微小変化した。

固定held-out 218 pre-action state上のold→new masked KLはmean `2.98e-5`、max `1.29e-4`。同一stateでのL3 entropyは`0.548154→0.548124`、Plan entropyは`2.1000893→2.1000891`。family available時の平均確率はRoad `.3864→.3828`、BuyDev `.3851→.3850`、BankTrade `.3939→.3997`、EndTurn `.6414→.6405`。top-family flip 79件はPre側のzero-init同点argmaxが多数あるため戦略変化として解釈しない。

全variantでterminal `72/72`、illegal/Executor failure/family mismatch/deadlock/truncation/nonfinite reward/GameState RNG arbitration consumptionは全て0。0% Safetyの既存controllerとのAction/State/RNG parityは72局確認。protected PlayerTradeはSafety/Pre/Postで`823/792/790`件処理され、Actor sampleへは入れていない。

## Artifactと安全性

pre/postはChampionとは別の `Hierarchical-Investment-Pre` / `Hierarchical-Investment-Post1` として保存。Champion digest `4225902d41a83540eb1f1552826ec623706e778b99ad4ae8ec015f976644392a`、Robber digest `ecde7fa9e9d67370bb76686ecd767d2d532a76d467c0051aaebdf8d4aff702bf`。pre artifactのSHA-256をpreflightとupdateで照合し、変更なしを確認した。optimizer対象IDは新Actorと新Criticのみ。

実行環境はPython 3.13.15、PyTorch 2.14.1、Stable-Baselines3 2.9.0。これらの依存は既存 `requirements-rl.txt` の範囲に収まる。追加した `pytest` を `requirements-dev.txt` に明記した。

## テスト

backend全体: `399 passed, 1172 subtests passed`（warning 2件はFastAPI/Starletteのdeprecated API）。pytestの一時先はworkspace内の `experiments/pytest_step3b6_tmp` を明示した。

## Step 3B-7に渡す判断材料

初回更新はtechnicalに成立したが、1候補EndTurnが43.5%を占め、2候補以上の比較Actionは152件。held-out Pre10→Post10はVP・rankとも差0で、強さの向上は未確認。Step 3B-7ではまず複数候補Investment stateの自然な頻度、Plan/Priorityへの微小gradient、Safety差の不確実性を踏まえ、追加sampleとguardを設計する。委譲率拡大かCity Value移行かはその後に判断し、今回自動昇格しない。Construction Commit、City Value、Development Usage、PlayerTradeの実委譲はこのStepでは行っていない。

再現用の集計・各局record・pre/post重みとmanifestは [`rl_hierarchical_ppo_step3b6_20261013_evidence`](rl_hierarchical_ppo_step3b6_20261013_evidence/) に保存した。
