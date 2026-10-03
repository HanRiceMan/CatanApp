# Step 3A-8: Construction Timing Gate / Surface-only Rollout基盤

## 結論

最初の移行判断を、`SETTLEMENT_ONLY` / `CITY_ONLY`における**BUILD_NOW vs DEFER**へ限定した。両方即時合法な状態は`DUAL_BUILD_DEFERRED`としてChampionへ残す。地点選択、道路・発展・交易の具体Action、その他phaseは凍結したExecutor / Plannerが担当する。今回Gate・Critic・semi-MDP buffer・conservative delegation APIを実装したが、**delegation probabilityの既定値は0、PPO parameter updateもStudent Actionの実適用も行っていない**。

Rule / Champion / Mixed各12局、席1～4各3局、合計36局の候補Shadowでは185 eligible Decision（開拓地のみ106、都市のみ79）を観測し、BUILD / DEFERの両候補が全件生成でき、**copy上のルール適用でも合法185/185**だった。profile別SurfaceはRule 61、Champion 58、Mixed 66。全Action列・winner・VP・rank・GameState・RNG counterのShadow有無parityは36/36。これは候補生成の成立を示すものであり、DEFERを実際に選んだ後の勝率・安定性を示すものではない。

## 委譲Surfaceとhard exception

`detect_construction_surface()`のAction前GameState + 合法mask判定を継承し、即時合法な建設familyが一方だけのときのみ対象にする。Championの最終ActionやTeacher labelではSurfaceを定義しない。`game.phase != action`、setup、盗賊、交渉応答/逆提案、破棄、無料道路、pending trade、forced single action、既に終了、9点以上の即時建設勝利、即時称号勝利は従来どおり除外。両方即時合法でも今回はPlannerに残す。36局では即時建設勝利18、両建設合法2、即時称号勝利1を除外した。

Gate Actionは`BUILD_NOW=0`と`DEFER=1`。前者は対象familyで今建設、後者は**このDecisionで建設しない**ことを意味する。`DEFER`は`END_TURN`固定ではない。DEFER候補が生成できない局面は委譲不可であり、Championへ戻す。

## Gate、入力、凍結範囲

`ConstructionTimingGate`は凍結Champion v2 Actor latent（実モデル392次元）＋v7 Runtime Context 79次元＋deterministic surface subtype one-hot 2次元を連結し、64 hidden + 2 logitを出す。Champion selected Action、execution source、Reason、Intent、Planner score、outcome等は入力しない。Gate最終層はzero initializationで初期logitは`[0,0]`。このGateは377-way Actor / family logits / context residualには接続しない。

`SurfaceCritic`はActorとは**別encoder**を持ち、凍結Champion Critic valueへのtrainable residualを出す。初期residualは0。`frozen_gate_input()`は`torch.no_grad()`で元GNN/Actor/Criticを実行し、latent/valueをdetachする。将来のoptimizer allowlistは`timing_trainable_parameters(gate, critic)`のみ。Champion GNN、v2 encoder、377 Actor、family logits、Critic base、v7の377用context projection、両Executorは学習しない。

## 凍結Executor

`FrozenConstructionExecutors`のBUILD経路は対象family確定**後にのみ**既存Championの`_best_settlement()`または`_best_city()`を呼び、具体的vertexを得る。Championが実際にBUILDを選んだかどうかは参照しない。地点選択の移行は後工程。

DEFER経路は同一Action前GameStateからのcounterfactual再選択。元のChampion PPOを建設Actionだけfalseにした合法maskで推論し、同じPlanner設定を持つ診断専用subclassで建設選択関数を禁止する。その他の道路、銀行交換、対人交渉、発展購入、手番終了等の経路は既存Plannerを通る。Champion実Actionを後付けで流用していない。候補はGameState copy上のルール適用で合法性を検査し、元GameState / RNGは変えない。PlayerTradeは377 catalog外なのでmaskだけでなくcopy上のルール適用が必要。

| Subtype | 対象 | BUILD候補 | DEFER候補 | 両方有効 | BUILD=Champion実Action |
|---|---:|---:|---:|---:|---:|
| SETTLEMENT_ONLY | 106 | 106 | 106 | 106 | 105 |
| CITY_ONLY | 79 | 79 | 79 | 79 | 65 |

地点一致はExecutorの品質そのものではない。例えばChampionが都市以外を選んだ状態でも、BUILD counterfactualは都市地点を返す。Championが非建設を選んだ状態は15件で、そのうちDEFER候補がChampion実Actionと一致したのは14件。残る1件もDEFER候補は合法であり、建設禁止下でPlannerを再実行した結果が異なった。これをTeacher不一致として再分類しない。

| DEFER Action family | 開拓地のみ | 都市のみ | 合計 |
|---|---:|---:|---:|
| END_TURN | 100 | 14 | 114 |
| BUY_DEVELOPMENT | 4 | 29 | 33 |
| BANK_TRADE | 0 | 23 | 23 |
| BUILD_ROAD | 1 | 10 | 11 |
| PLAYER_TRADE | 1 | 3 | 4 |

この標本でDEFERはかなり高頻度にEND_TURNとなるが、他Actionも候補として生きている。初回RLがDEFERを多用して停滞するリスクはStep 3A-9で監視する。

## Surface-only rolloutの時間単位

既存`CatanEnv.step()`は学習者の**1 Decision**を1 `policy_steps`と数え、その直後の固定Opponent/自動phase進行を同じstep内で行い、到達後に報酬を返す。Championの`gamma=0.99`、`gae_lambda=0.95`、`n_steps=1024`。従ってmacro durationはサイコロ・Opponent・プリミティブAction数ではなく、次の委譲Surface/終局までの**既存の学習者Decision step数**で測る。

`SurfaceTransitionBuilder`は実際にGateがActionを選んだ時だけ開始し、その後の固定Plannerを含む各base-step報酬を`Σ gamma^i r_i`で蓄積する。次Surfaceへのbootstrapは`gamma ** macro_duration`。`SurfaceRolloutBuffer`は`delegated=False`を拒否し、Actor policy lossに非委譲Decisionを混ぜない。transitionにはpre-surface観測・Context・subtype・gate action/logprob・accumulated reward・次Surface/terminal・done・duration・valueと次state用任意fieldを持たせた。Critic lossもこのSurface valueに限定する設計。まだPPO更新処理、GAE、optimizer step、実Action rollout収集は実装していない。

`ConstructionTimingDelegator`は外部RNGによるdelegation probabilityを持ち、初期値0。候補が両方有効で、抽選に通った時だけGate Actionを具体Actionへ変換する。ゲームRNGは抽選へ使用しない。Stage 3A-9でも非委譲ActionをGateのon-policyサンプルと扱わない。

重要な実装境界として、既存`CatanEnv.step()`は377 Action IDを受け取る一方、Frozen DEFER Executorは377外の`ProposeTradeAction`も返しうる。そこで`CatanEnv.step_action(Action)`を追加し、既存stepと**同じ**実Action適用・`policy_steps`・reward boundaryを使うようにした。377内ActionについてID step / Action stepの観測・報酬・done・RNG parityを単体検証済み。377外のPlayerTradeは元GameStateを触らずコピー上でルール検証し、`allow_player_trades=True`の時だけ受け入れる。今回のShadowでPlayerTradeをEND_TURNへ置換していない。実RL前には、現在の学習Environmentが使う交渉補助Agentのaction-phase自動提案がGateのsurfaceを先取りしないよう、交渉応答だけを固定補助として残す設定と、提案→応答を跨ぐmacro報酬のintegration testを追加する必要がある。

## Runtime Context最適化

79 fieldの内容・順序・normalization・versionは変更していない。旧実装はStructured Context生成時に`analyze_expansion_plan()`を実行した後、canonical target取得のため同じGameStateで再度deep copy + expansion解析を行っていた。内部APIで一度目のplanをContext encoderへ返し、重複解析を除去した。Structured Context public reportの形は不変。旧経路と新経路の79 vectorは同一Action前stateで厳密`np.array_equal`を確認する。

最終36局・185 Surfaceで新Context生成は平均11.27 ms、p95 16.31 ms。Step 3A-7の異なる36局標本では平均22.61 ms、p95 34.67 msだった。さらに同一Action前stateのpaired benchmark 36組では旧経路平均16.86 ms / 新経路平均8.70 ms（約48.4%短縮）、79 vectorは全組`np.array_equal`。新Contextの185件平均とpaired 36件平均は分母が違うため直接比較しない。候補検証やPlanner再計算のコストはこのContext値に含めない。

## Step 3A-9の初回設定案とGate

まず固定seedのbranch rolloutでBUILD/DEFER双方の合法性、対人提案→応答、報酬境界を確認する。次に8～12局のdelegation 0/10%安全smoke（0%で完全Champion parity）、続いてRule/Champion/Mixedを混ぜた小規模screeningでdelegation 25%程度から開始。bufferには実際に委譲したDecisionのみ入れる。初期Gateが同確率なのでDEFER過多による手番浪費、連続DEFER、資源保持・破棄、建設数低下を停止指標にする。学習率候補はActor Gate `1e-4`、Surface Critic `3e-4`、`gamma=0.99`固定、`gae_lambda=0.95`を基準にし、少数サンプルでは更新回数を抑える。具体的なbatch/entropy/kl上限はStep 3A-9でon-policy候補数を確認してから確定し、1実験でChampionモデルを上書きしない。

既存PPOの`n_steps=1024`をそのまま「1024 Surface sample」と読み替えない。今回の標本は約5 Surface/局で、25%委譲なら平均1～2 Actor sample/局しか得られない。Step 3A-9ではまず数十～百程度の**実委譲**sampleを貯められる局数・席順・profile配分を見積もり、bootstrap/GAEのmacro durationを確認してから初回updateする。少数対局の勝率だけで採否を決めない。

暫定の初回update案は、上記smokeを通過した後、delegation 25%で席/相手を均等にし、**実委譲128 transition**を目標に収集、batch 32、2 epoch、clip 0.1、target KL 0.01、entropy係数0.005、更新1回に制限する。約5 Surface/局という今回の観測なら100局前後が目安だが、実際の有効候補率とDEFER後分布で再見積もる。比較対局は別seedで行い、この数値を性能保証や固定採用条件とはしない。

移行順はStage 1: 単独建設familyのBUILD_NOW/DEFER、Stage 2: 両方合法のSETTLEMENT/CITY/DEFER、Stage 3: vertex、Stage 4: 道路・発展・銀行交換を含む戦略Action比較、Stage 5: belief/history依存のPlayerTrade/Robber。今回Stage 2以降の学習はしない。

## 検証と残るGate

新規単体テストは10件。backend全体は**281件成功**。36局のAction / GameState / RNG parityは上記のとおり。Champion weight / 377 logitsへの新しい更新経路はなく、Actor/Critic基底とv7 residualは凍結する設計である。元モデル・config・既存artifactは変更していない。

Step 3A-9へは**小規模の統合smokeから進めるが、直ちにPPO updateを始めるべきではない**。特に、`step_action(ProposeTradeAction)`で交渉応答を跨ぐ報酬・duration、DEFER実適用後の連続Surface、delegation 0%の対局完全parityを確認する。これらが通ればGate + Surface Criticのみ初回updateする。Shadowのみの185件をGateのon-policy学習データとして転用しない。
