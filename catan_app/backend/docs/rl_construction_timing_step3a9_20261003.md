# Step 3A-9: Construction Timing Gate の初回実委譲・1 update

## 範囲と安全境界

学習対象は `SETTLEMENT_ONLY` / `CITY_ONLY` での `BUILD_NOW` / `DEFER` の二択だけ。具体的な建設地点と DEFER 後の行動は Step 3A-8 の凍結 Executor が決める。両方建設可能な局面、他 phase、道路・発展・銀行交換・交渉・盗賊の個別判断は委譲していない。Champion GNN、377 Actor/family logits、基底 Critic、v7 の 377-way context residual、両 Executor は凍結し、Optimizer は `ConstructionTimingGate` と `SurfaceCritic` のパラメータだけを持つ。Champion artifact/config は変更していない。

## 実 Action integration smoke

PPO update 前に、GameState へ実際に Gate Action を適用して確認した。BUILD は開拓地・都市各1例、DEFER は `END_TURN` / `BUY_DEVELOPMENT` / `BANK_TRADE` / `BUILD_ROAD` / `PLAYER_TRADE` 各1例を強制し、全例が合法・完走。PlayerTrade 例は提案から相手応答を経て learner 境界へ復帰し、macro duration 21 learner step、対人提案を4件跨いだ。強制例は `integration_forced` として記録し、on-policy PPO buffer への混入を拒否する。

委譲率0%では Rule / Champion / Mixed × Seat 1～4 の12局で、全 Action 列、winner、VP、rank、policy_steps、ゲーム RNG counter、および新規game UUID/認証tokenを除く最終GameStateが基準と一致。委譲率10%・未更新Gateでは同じ構成の12局すべてが終局し、実委譲5件（BUILD 2 / DEFER 3）。illegal、deadlock、truncation、非finite報酬は0。`macro_duration` は平均25.6、中央値17、最大55。初期Gateは両 subtype とも `P(BUILD)=P(DEFER)=0.5`、Surface Critic residual は0で、凍結Champion valueと一致した。

## Semi-MDP 境界・報酬

`SurfaceTransitionBuilder` は Gate が実際に選択した Surface でのみ開始し、次の**実委譲**Surfaceまたは終局で閉じる。途中の非委譲 eligible Surface は固定Champion判断として通過させ、Gate Actor sample にしない。`macro_duration` は既存 `CatanEnv.policy_steps` と同じ learner Decision step 数であり、相手内部Action数ではない。報酬は `Σ gamma^i r_i`、bootstrap係数は `gamma ** duration`。GAEも可変durationを使用する。人工trajectoryテストで割引報酬、次の実委譲へのbootstrap、非委譲Surface跨ぎを確認した。真のterminalではbootstrap 0。time-limit truncation はterminalと混同せず、初回updateでは該当episode全体を学習から隔離する方針とした（今回の収集では0件）。

## 初回 on-policy ロールアウト

外部 delegation RNG seed を各episodeに固定し、各episode内の委譲率は25%。Rule / Champion / Mixed 各40局、各Seat 30局の計120局で、実委譲143件を取得した。学習seedは評価seedと完全分離し、episode途中では切らずに収集した。

| 指標 | 結果 |
|---|---:|
| SETTLEMENT_ONLY / CITY_ONLY | 86 / 57 |
| BUILD_NOW / DEFER | 74 / 69 |
| macro duration 平均 / 中央値 / p95 / 最大 | 23.92 / 22 / 58.6 / 104 |
| 次の実委譲Surfaceへ到達 / terminal | 64 / 79 |
| 非委譲Surface通過回数 | 183 |
| 対人提案を跨いだ回数 | 549 |
| DEFER直後に同一Surfaceへ戻った回数 | 1 |
| 最大連続DEFER | 3 |
| 手札8枚以上でのDEFER | 36 |
| truncation / 未完走 / 非finite報酬 | 0 / 0 / 0 |

既存RewardConfig（勝利 `+1`、敗北 `-1`、`+0.05 / VP`）により、BUILD直後の報酬平均は `+0.05`、DEFER直後は `-0.0116`。これは明確な BUILD 側 shaping bias なので変更せず別集計した。DEFER直後のVP成分平均は `+0.0029`、BUILDは `+0.05`。macro報酬平均は `+0.1058`、割引済みVP成分合計は `+15.7547`、terminal成分合計は `-0.6207`。GateがDEFERを選んだ後も、凍結Plannerの後続行動でVPが増えるため、即時報酬だけでAction価値を判定しない。

## 初回1 update

Gate `lr=1e-4`、Surface Critic `lr=3e-4`、batch 32、2 epoch、clip 0.1、target KL 0.01、entropy係数0.005、`gamma=0.99`、`gae_lambda=0.95` で**1 updateだけ**実行。Optimizerのパラメータ数は60,867で、ChampionのパラメータIDを含まないこと、および更新後の凍結Champion重み不変を検証した。Planner一致報酬などのTeacher rewardは追加していない。

| 指標 | 結果 |
|---|---:|
| Advantage 正規化前 mean/std/min/max | 0.4696 / 0.7517 / -0.7833 / 1.7868 |
| Advantage 正規化後 mean/std/min/max | 0 / 1 / -1.6668 / 1.7523 |
| Actor loss / Critic loss | -0.00493 / 0.73980 |
| Entropy | 0.69311 |
| KL mean / max | 0.000120 / 0.000156 |
| KL 開拓地のみ / 都市のみ | 0.000115 / 0.000126 |
| 開拓地のみ P(BUILD) 前→後 | 0.5000 → 0.5076 |
| 都市のみ P(BUILD) 前→後 | 0.5000 → 0.5079 |
| P(BUILD)>0.95 / <0.05 の割合 | 0% / 0% |

KL上限内で、BUILD_ALWAYS / DEFER_ALWAYSへのcollapseなし。候補は既存Championとは別のローカル実験artifactとして保存した。1 updateでの改善を意味するものではない。

## 別seed評価

学習とは別のseed範囲で、Champion基準、委譲率25%、100%を同seed・同seatで比較した。Rule / Champion / Mixed 各20局、各variant計60局、全体180局のscreening。全variantで60/60終局し、truncation / illegal / deadlockは0。席は各profileで1～4を各5局。

| 条件 | 対戦相手 | 勝率 | 平均VP | 平均順位 | 平均learner step | 開拓地/都市/道路 合計 |
|---|---|---:|---:|---:|---:|---:|
| Champion基準 | 全60局 | 45.0% | 7.68 | 2.12 | 77.3 | 166 / 98 / 346 |
| 25%委譲 | 全60局 | 41.7% | 7.50 | 2.17 | 75.0 | 156 / 88 / 340 |
| 100%委譲 | 全60局 | 36.7% | 7.07 | 2.40 | 76.7 | 153 / 72 / 339 |
| Champion基準 | Rule 20局 | 80% | 9.40 | 1.30 | 73.9 | 65 / 48 / 119 |
| 25%委譲 | Rule 20局 | 85% | 9.55 | 1.23 | 71.9 | 65 / 41 / 119 |
| 100%委譲 | Rule 20局 | 85% | 9.55 | 1.33 | 78.7 | 63 / 40 / 117 |
| Champion基準 | Champion 20局 | 25% | 6.55 | 2.68 | 82.7 | 49 / 24 / 111 |
| 25%委譲 | Champion 20局 | 20% | 6.30 | 2.80 | 80.2 | 44 / 22 / 111 |
| 100%委譲 | Champion 20局 | 5% | 5.70 | 3.18 | 82.7 | 46 / 17 / 114 |
| Champion基準 | Mixed 20局 | 30% | 7.10 | 2.38 | 75.4 | 52 / 26 / 116 |
| 25%委譲 | Mixed 20局 | 20% | 6.65 | 2.48 | 72.8 | 47 / 25 / 110 |
| 100%委譲 | Mixed 20局 | 20% | 5.95 | 2.70 | 68.6 | 44 / 15 / 108 |

建設数には全variant共通のsetup分も含む。計60局の100%委譲では381 Surfaceのうち BUILD 201 / DEFER 180。DEFERからEND_TURNは116件、手札8枚以上のDEFERは96件、同一Surfaceへ即復帰は23件、最大連続DEFERは10。25%委譲ではBUILD 37 / DEFER 37、DEFERからEND_TURNは26件、最大連続DEFERは3。Full Surfaceの平均learner stepは基準と同程度だが、対戦相手が早く勝った局も含むため「停滞なし」を示すものではない。

同seedのVP差でみると、100%委譲はRuleで基準より上/同/下が5/13/2局だが、Championでは2/8/10局、Mixedでは3/7/10局。20局/profileはscreeningであって有意な強弱判定ではないものの、強い相手で都市建設が減り、Full SurfaceのVP・順位・勝率が一貫して悪化した。Rule戦だけの良い数字で候補を採用しない。

## 検証と次の判断

backend全体の単体テストは286件成功。実対局smoke、0% parity、10%委譲、初回更新の安全Gateは通過した。したがって「BUILD_NOW / DEFERを実際に制御し、on-policy returnでGateだけを1回更新できるか」という**統合・学習基盤の目的は達成**した。一方、更新後のGateはほぼ50:50のままで、強い相手に100%委譲すると都市数と成績が崩れる。**候補は現行AIへ昇格しない。2回目updateも行わない。**

次は学習回数を単純に増やす前に、DEFERの大半がEND_TURNへ行くExecutor経路と、建設を保留する価値を現在のゲーム報酬だけで十分学べるかを分析すべきである。特に、50:50初期Gateの全面委譲がChampionから大きく外れること、`+0.05/VP` がBUILD側へ即時報酬を与える一方で終局リターンが疎であることを検証課題とする。Champion模倣の一致率を成功指標にはせず、再実験する場合も別artifactで保守的な初期委譲・安全Gateを設け、RuleだけでなくChampion/Mixedで評価する。

実験の詳細JSONと候補重みは `catan_app/backend/experiments/construction_timing_step3a9_20261003/` に保存され、既存のrepository方針どおり大容量モデル/実験出力はGit管理しない。コードと本レポートのみGitHubへ反映する。
