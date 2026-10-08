# Step 3A-19 — Road Executor 因果監査 / Delegation dose-response

2026-10-08。Step 3A-16の **T=2 pre-update / residual=0** を固定した診断。PPO・Criticのupdate、Reward、Strategic candidate mask、Runtime Context79、Frozen B Executor、Champion、モデル重みは変更していない。Rule / Champion / Mixed各12局、各profileでseat 1〜4を3局ずつ配置した同一36 game seed（3260000台 / 3261000台 / 3262000台）を用いる。delegation RNGとfamily sampling RNGはGameState RNGから分離し、各rateでseedを保存した。

## 比較の境界

Road監査ではFull T=2のAction前局面から、Gate/ChampionがともにBUILD_ROADを選ぶ局面を全抽出する。edge相違局面は **GateがRoad familyを選んだこと** と **B Executorがどのedgeを選んだこと** を分離できる。B edge / Champion edgeを同じpre-action GameStateへそれぞれ1回だけ強制し、以後両branchともSafety Championへ戻す。future game RNG seedは各pairで共有し、8 seed/stateで終局まで追跡する。returnは既存RewardConfigをlearner policy step単位の `gamma=0.99` で割り引いたもの。Champion edgeは教師・正解ではない。Full Strategic継続は別の補助監査とし、Safety継続の結果へ混ぜない。

道路構造の「expansion」は、既存の決定的な `_road_purposes()` に従い、3本以内の開拓候補への最短経路の第一歩を意味する。「longest」はRoad適用で本人の最長路長が増える事実である。どちらもRoadを選ぶ価値スコアではない。合法性、Frozen 377 candidate logit/rank、建設後のlegal road/settlement数、3道路以内に新たに入るempty vertex、最少追加道路数を、元GameStateを変更せずcopy-applyで計算した。代表stateはreturnを見ずに、各profile×4構造分類から2件ずつ、score範囲を分けて計24件選ぶ。

Dose-responseはdelegation 0/10/20/50/100%を同seed36局で比較する。0%はSafety Champion。rate変更後は到達状態が変わるため、rate間の差は追加委譲の純粋な因果効果ではない。`Δ / actual delegated` も記述用の参考値に留める。全rateでImmediateWinResolver、protected PlayerTrade、candidate mask、Frozen Executor、T=2 priorは同一。

## Delegation dose-response

| Delegation | 局数 | 実委譲 | Surface | protected preemption | Settlement | City | 建設計 | Road | Dev | BankTrade | EndTurn | PlayerTrade | 勝利 | 平均VP | 平均順位 | 平均policy steps |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 36 | 0 | 1,168 | 423 | 106 | 61 | 167 | 225 | 51 | 141 | 602 | 447 | 19 | 8.194 | 1.889 | 74.0 |
| 10% | 36 | 79 | 1,184 | 396 | 105 | 68 | 173 | 221 | 66 | 162 | 622 | 415 | 20 | 8.389 | 1.847 | 76.0 |
| 20% | 36 | 138 | 1,081 | 360 | 101 | 58 | 159 | 217 | 63 | 139 | 590 | 383 | 16 | 8.083 | 2.042 | 71.4 |
| 50% | 36 | 373 | 1,074 | 356 | 93 | 40 | 133 | 220 | 100 | 134 | 610 | 377 | 15 | 7.389 | 2.194 | 75.0 |
| 100% | 36 | 723 | 1,140 | 417 | 80 | 33 | 113 | 229 | 153 | 136 | 636 | 449 | 14 | 7.556 | 2.125 | 81.9 |

36/36局×全rateが終局し、illegal / Executor failure / deadlock / truncation / nonfinite rewardは0。Gate surfaceとprotected件数がrate間で異なるのは到達stateが変わるため。0% Safetyと100% T=2の再生はStep 3A-16保存結果と36局ずつ一致し、100%側723件の実委譲Concrete Actionも一致した。

| Delegation | lottery対象 | 実委譲/局 | first concrete divergence局 | うちRoad→Road edge差 | Δ建設/実委譲 | ΔDev/実委譲 | ΔVP/実委譲 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10% | 788 | 2.19 | 24 | 5 (20.8%) | +0.076 | +0.190 | +0.089 |
| 20% | 721 | 3.83 | 27 | 4 (14.8%) | -0.058 | +0.087 | -0.029 |
| 50% | 718 | 10.36 | 36 | 8 (22.2%) | -0.091 | +0.131 | -0.078 |
| 100% | 723 | 20.08 | 36 | 10 (27.8%) | -0.075 | +0.141 | -0.032 |

各rateでprotected PlayerTradeの「candidateあり」と「controllerが選択」は今回同数（0/10/20/50/100%で423/396/360/356/417）だったが、将来の一般法則とはみなさず別fieldとして保存している。first divergenceは同seed Safetyとの**最初の具体Action差**を全Action traceで検証した。10/20%でのRoad→Road edge差は、最初の分岐の約15〜21%を占める。Δ/実委譲は異なる到達stateを混ぜた参考値であり、1委譲の因果効果ではない。

first divergenceの主なpairは、10%でRoad→Road 5、Road→Dev 3、Bank→Dev 3、20%でRoad→Road 4、Road→Dev 4、End→Dev 4、End→Bank 4、50%でRoad→Road 8、Road→Dev 6、End→Bank 6、Road→End 5、100%でRoad→Road 10、Road→End 8、Road→Dev 7、End→Bank 4。全pair・rate別内訳は再現可能な集計JSONに保存している。

建設数は10%ではSafety未満にならず、20%でやや減り、50/100%で大きく減った。Dev購入は高rateほど増える傾向がある。一方、VP・勝利数・episode長は完全な単調列ではない。従って「10〜20%近傍と50〜100%で様子が違う」というdose診断であり、36局screenから精密な性能差やrateの純粋因果効果は主張しない。

## Road edge 構造とpaired continuation

Full T=2の36局で、GateもChampion counterfactualもRoadを選んだ局面は**155件**。同一edgeが62件、相違edgeが**93件**である。相違93件はfirst divergenceだけでなく、Full trajectory上の全該当Decisionから抽出した。profile別の相違はRule 35、Champion 28、Mixed 30件。

| profile | Road/Road | edge相違 | seat 1 / 2 / 3 / 4 の相違件数 |
|---|---:|---:|---|
| Rule | 61 | 35 | 9 / 6 / 9 / 11 |
| Champion | 47 | 28 | 8 / 5 / 4 / 11 |
| Mixed | 47 | 30 | 8 / 11 / 5 / 6 |

| 93 edge相違局面の構造分類 | B Frozen PPO edge | Champion比較edge |
|---|---:|---:|
| expansion + longest | 34 | 47 |
| expansionのみ | 15 | 44 |
| longestのみ | 17 | 2 |
| neither | 27 | 0 |
| 3道路以内に新たに到達可能なempty vertexあり | 27 | 64 |

`neither` は「今回採用した2種類の構造進捗に該当しない」であり、Roadが無価値と証明する分類ではない。Championのedgeは拡張寄りだが、これも正解ラベルではない。B edgeの具体的なreturnへの寄与は以下のpaired continuationで判断する。

全155件ではBの構造分類はboth 82 / expansionのみ29 / longestのみ17 / neither 27、Champion比較edgeは95 / 58 / 2 / 0。相違93件のFrozen candidate logit rankはBが全件1位、Champion比較edgeが平均4.23位（中央値4、範囲2〜10）。logit平均はB **−0.287**、比較edge **−1.498**。これはFrozen 377 Actorの順位であり、長期Road価値の順位とは限らない。Road適用で最長路長が増えた件数はB **51** / 比較edge **49**、最少追加道路数の減少は **42 / 64**、建設後legal road数平均は **7.28 / 7.98**、建設後legal settlement数平均は **0.333 / 0.269**。よってBがすべての構造指標で劣るわけでもない。

代表stateは各profile×`expansion_only` / `longest_only` / `both` / `neither`から2件ずつ、計**24 state**。各stateで8個のfuture game RNG seedを2 branchで共有したSafety継続 **192 paired future（384 branch episode）** を実施。さらに各profile×4構造分類から1件ずつ、計**12 state**でFull Strategic継続 **96 paired future（192 branch episode）** を補助実施した。pairの初手以外は両branchとも同じcontrollerであり、SafetyとFullを混ぜていない。全branchが合法・終局し、truncationは0。

| 返り値はB edge−Champion比較edge | Safety継続（主） | Full継続（補助） |
|---|---:|---:|
| state / paired future | 24 / 192 | 12 / 96 |
| discounted return差 平均 / SD | −0.0321 / 0.6157 | −0.0443 / 0.5255 |
| state平均差のSD | 0.2126 | 0.0911 |
| B優位 / 比較edge優位 / ほぼ同点 state（閾値±0.02） | 7 / 13 / 4 | 2 / 6 / 4 |
| 8 seed中7本以上同方向 | 0 | 0 |
| 正負が各2 seed以上混在 | 17 | 10 |
| winner flip | 51/192 | 30/96 |
| B / 比較edgeの学習者勝利 | 73 / 76 | 24 / 28 |
| final VP差 / rank差 | −0.068 / +0.005 | −0.135 / +0.052 |
| policy steps差 | +2.04 | −2.88 |

Safety継続の局面内future sample variance平均は**0.3816**。8本平均のノイズ量は0.3816/8=**0.04770**で、観測されたstate平均間sample variance **0.04718**を上回る（差し引き推定は0）。Full補助も0.3061/8=**0.03826** > state平均間variance **0.00905**。この規模ではedge別の期待価値差よりfuture RNGの影響が大きく、B edgeの一貫したreturn劣後は立証できない。単発continuationをedge教師ラベルにしない。

初手Roadから30 learner policy steps内の建設数は、Safety継続でBがSettlement **176** + City **96** = **272**、比較edgeが**193 + 106 = 299**。同期間のDevelopment購入は**122 / 97**、Road建設は**588 / 560**。Full補助では建設計は**93 / 93**、Development **133 / 138**、Road **285 / 284**。短期のSafety建設経路には差があるが、終局returnの符号は安定していない。

適用直後に3道路以内へ新たに入るempty vertexを持つ件数は相違93件中B **27** / 比較edge **64**。次のlearner boundaryでのlegal settlement数平均はB **0.333** / 比較edge **0.208**、最少追加道路数は **1.125 / 1.000**、3道路以内のempty vertex数は **7.292 / 7.625**（Safety継続192 pairの各branch）。「遠方の拡張幅」と「直近で建設できる地点」は異なるため、これらの構造値を単一の優先scoreへ合成しない。

## 判断と次工程

Road BはFrozen 377 logitsで常にroad候補の1位を選ぶが、Road/Road不一致の29%（27/93）は今回のexpansion/longest構造指標に該当しない。これは**Executor品質の検証課題**である。ただし、24 state×8 futureのSafety比較ではBがreturn上で安定して劣る証拠はなく、Full補助でも符号はほぼ乱数に埋もれる。従ってStep 3A-19のCase **B（Road edge効果はnoise dominated）** に近く、Champion edgeを模倣するExecutorへの即時交換や「neither」禁止ruleは採用しない。Road familyのAdvantageには引き続きExecutor品質が混入し得る点は診断として保持する。

Doseでは10%の建設・VPはSafety近傍、20%で小幅な建設減少、50/100%で建設大幅減とDev増が見える。VP/順位/episode長は厳密な単調悪化ではないため、Case **C寄りの閾値的なcompounding** と判断する。10〜20%を次の安全な*検証候補範囲*とし、50〜100%をそのまま学習/採用の前提にしない。36局screenから10%と20%の優劣や最適rateは決めない。低delegation curriculumの設計は必要だが、今回実装していない。

Step 3A-20では、まずRoad edge診断を保持したまま、10%を起点とする低delegation operating rangeと評価ゲートを設計するのが妥当。Road Bは低rateでは暫定利用候補だが、構造的な`neither`選択を継続監視する。必要なら **Road familyだけ** の代替Executor候補をShadowで比較し、決定的な合法性/到達性とFrozen候補順位を分離して設計する。ただしChampion edgeのコピー、Road Aの即採用、新heuristicによる一律抑制はしない。候補Executorを実採用する前には、同pre-action state・複数futureで再度検証する。追加PPOは自動再開せず、Road Executorの許容条件と低rate curriculumの監視指標を次工程で確定してから判断する。

Reward変更、Runtime Context追加、Strategic candidate mask変更を要する証拠は今回得ていない。Step 3A-19ではoptimizer stepを一切実行しておらず、Gate/Critic/Champion重みは不変。詳細な全state、pair、rate別first divergenceとsource parityは保存した監査JSONを参照。

## 再現性・不変条件

- 診断runner: `app/rl/run_strategic_road_dose_step3a19.py`。Road選択は既存Frozen B Executorのまま、Road比較用Actionはcopy-applyまたはpaired branchへの最初の1手としてのみ使用。新規Policy ruleやoptimizerはない。
- 評価seedは各profile12局のStep 3A-16既存seed。Road future seedは `319000000 + game_seed × 17 + future_index`（0〜7）、同じpairの2 branchで共有。実GameState RNGと外部delegation/family RNGは分離。
- 0% Safetyと100% T=2の36局ずつについて、保存済みStep 3A-16結果とのwinner / final VP / rank / policy steps / RNG counter / learner Action family / 実委譲件数が一致。100%の**723件の実委譲familyと具体Action**も一致。
- Champion policy `state_dict` digest: `4225902d41a83540eb1f1552826ec623706e778b99ad4ae8ec015f976644392a`（Step 3A-16 manifestと同一）。T=2 pre-update `weights.pt` SHA-256: `755e0b868999cd0a616b9529742196c8675ab88a0a21e5b4de0c54d0b702b4c2`、post-update: `9d757ceca3114d65575da62f8de18abfb6777e76d1f0474c9a2c3dd13f4241fa`。いずれも作業開始時から不変。
- テスト: `pytest -q -p no:cacheprovider tests` は **340 passed、1,172 subtests passed**。既存依存ライブラリからのdeprecation warning 2件のみ。Road診断GameState不変、Road Action表現の厳密なparse、state単位/future単位の集計分離、doseの実委譲分母を新規テストで固定。Controller内のAction前RNG counter検査も全episodeで通過。
- 監査証跡: `docs/rl_strategic_road_dose_evidence_step3a19_20261008.zip` に全Road/Road行、選択24 stateの288 paired future（Safety192 + Full補助96）、dose全rateの36局別記録、集計JSONを収録。個々のcontinued branchは同じfuture seed・同じcontrollerで再現可能。
