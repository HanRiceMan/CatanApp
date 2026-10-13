# Step 3B-5 — Differentiable Strategic Bridge + L3 Shadow Policy

## 結論と実装範囲

L1 Plan Scorer→soft Plan→L2の4独立Priority→Tensor bridge→L3の7-family logitsという**勾配が通る経路**を追加した。実Controllerには接続せず、Action選択・環境適用・Reward・既存重み・PPO optimizerは変更しない。未学習L3のuniform確率は価値判断でも推奨Actionでもない。

実装は [`hierarchical_l3_shadow_step3b5.py`](../app/rl/hierarchical_l3_shadow_step3b5.py)、再現可能な36局runnerは [`run_hierarchical_l3_shadow_step3b5.py`](../app/rl/run_hierarchical_l3_shadow_step3b5.py)、単体テストは [`test_hierarchical_l3_shadow_step3b5.py`](../tests/test_hierarchical_l3_shadow_step3b5.py)。3B-4の診断APIとShadow Controllerは維持した。

## Tensor bridge

`DifferentiableSoftPlan`は`available: BoolTensor[B]`、`weighted_features: Tensor[B,6]`、`expected_priorities: Tensor[B,4]`を持つ。6軸はSettlement増分、City純増、Longest取得、Army取得、planned point gain、overshoot。Plan Headのprobabilityとcandidate-specific Priorityに対し、それぞれcandidate軸の加重和を取る。candidate特徴は3B-2の固定ルール上scale `(10,10,1,1,11,1)`を戻して、3B-4診断APIと同じ生値にする。paddingとIMPOSSIBLEはmaskで厳密に寄与0。unresolved等では`available=false`とし、Tensorは数値演算用に0を返すが、**ゼロPlanを選んだ意味ではない**。

Bridge本体には`.detach()`、`.item()`、`.numpy()`、`float()`を使わない。Frozen Champion actor latentはPlan HeadとL3のencoder入口でdetachされる。L3出力をlossにしたunit testではPlan ScorerとPriority Headの両方へ非ゼロgradientが届き、latent側には届かなかった。zero-initのままでは上流勾配が消えるため、テスト時にだけL3の重みを一時的に非ゼロへ変え、**構造としての疎通**を検証した。optimizer stepは行っていない。

## L3 Shadow Actor

入力はFrozen actor latent、Runtime Context79、VictoryPlanContext global facts30、soft Plan6、expected Priority4、Plan available flag1、route one-hot8、rule上legal family mask7、route別comparison mask7、状態局所的Construction Commit fact1。Plan global factsは3B-2の固定scale、L3へ入るsoft6は`(10,10,1,1,11,1)`で割る。Prioritiesとbool/one-hotは0–1。Planner Goal/Intent/Reason、Champion Action順位、3A native family logitsや手作りPlan価値補正は使わない。

ネットワークは共有`Linear(input,64)→Tanh→Linear(64,7)`。最終Linearはweight/biasとも0初期化し、valid比較familyには初期uniform。Bridgeはparameterを持たない。L3はconcrete road edgeやtrade offerを選ばない。`L3ShadowOutput`はraw/masked logits、probabilities、entropy、available、comparison maskを返す。比較候補外の確率は厳密に0、候補なしではavailable=falseかつ分布0。legal maskは比較外の合法Actionも別途保持し、実環境Action maskへは適用しない。

| route | Shadow比較対象 |
|---|---|
| INVESTMENT | Road / BuyDev / BankTrade / PlayerTrade / EndTurn の合法候補 |
| CITY_VALUE_COMPARISON | Cityと上記Investment候補 |
| DUAL_CONSTRUCTION_COMPARISON | Settlement / Cityのみ。他の合法familyは診断maskに残す |
| CONSTRUCTION_COMMIT | 現在合法なfamilyを診断。次ActionのSettlement強制なし |
| ACTION_DEV_USAGE | Dev Policy側で`NO_USE`を選んだ場合だけfallback通常routeとしてL3へ |
| PRE_ROLL_DEV_USAGE / IMMEDIATE_WIN / FORCED_PHASE | L3 distributionを作らない |

Construction Commitはturnを跨いで持続する制約ではなく、3B-4同様そのstateの分類factである。PlanがCity主体でもSettlementはmaskされず、City priorityが高くてもSettlementは禁止されない。PlayerTradeはルール上の提案可能性だけをShadow familyへ保持し、具体offer生成・実委譲は行わない。Development Usage Policyも未実装であり、`NO_USE` fallbackインターフェースだけ確認した。

## Shadow検証

3B-4の300局をroute/mask照合の参照基準にし、同じseedからRule/Champion/Mixed各12局（各profile/seat3局）の36局を選んだ。各局をobserverなし/ありで再実行し、3B-4保存recordともAction・route・mask・turn/revisionを照合する。Head forwardは各strategic境界で行い、3B-4診断soft値との数値一致、未学習uniform、mask厳密性、GameState/RNG非干渉をassertする。

Rule `34000000–34000011`、Champion `34001000–34001011`、Mixed `34002000–34002011`を使用。各profile12局、各seat3局、全36局。learner boundaryは計2,588件。

| route | Rule | Champion | Mixed | 合計 | L3 distributionあり |
|---|---:|---:|---:|---:|---:|
| IMMEDIATE_WIN | 11 | 3 | 6 | 20 | 0 |
| FORCED_PHASE | 326 | 389 | 297 | 1,012 | 0 |
| PRE_ROLL_DEV_USAGE | 21 | 22 | 18 | 61 | 0 |
| ACTION_DEV_USAGE | 0 | 0 | 0 | 0 | 0 |
| CONSTRUCTION_COMMIT | 35 | 32 | 27 | 94 | 94 |
| CITY_VALUE_COMPARISON | 19 | 21 | 12 | 52 | 52 |
| DUAL_CONSTRUCTION_COMPARISON | 1 | 0 | 1 | 2 | 2 |
| INVESTMENT | 391 | 580 | 376 | 1,347 | 1,347 |

L1/L2/Tensor Bridge forwardは1,556 stateで実行し、Plan availableは1,556/1,556。L3の比較分布は通常routeの1,495 stateでavailable。pre-roll Dev 61件はL3なし。実sample中のaction-phase Dev使用可能stateは0件だが、unit stateで`NO_USE`後のCity route one-hotとL3 distribution復帰を検証した。Plan unavailableも実sampleでは0件で、全候補IMPOSSIBLEの合成batchではvalid flag=false、集約Tensor0、L3側でそれをPlan選択と扱わないことを確認した。

全1,556 forwardで3B-4のsoft Plan/Priority診断値と数値一致し、不一致0。Plan padding/IMPOSSIBLE確率寄与0、候補順序を逆転してもsoft集約とL3出力は一致。L3比較mask外確率0、available時の確率和1、比較候補なしで分布0、zero-initで比較候補内uniformを検証した。例えばRule seed`34000000`のturn1 Investmentでは、Plan soft6は約`(3.0, 2.818, 0.545, 0.545, 8.0, 0.0)`、Priorityは`(0.5,0.5,0.5,0.5)`、比較familyのRoad/PlayerTrade/EndTurnは各約1/3だった。この値は未学習Headの中立分布であり、Actionの推奨ではない。

parameter数は3B-2 Plan/Priority Head **41,349**（既存不変）、新L3 encoder **34,304**、新L3 family head **455**、新L3合計 **34,759**、Tensor Bridge **0**。Frozen latent392と固定feature143を連結した入力535、共有hidden幅64、route別独立networkなし。strategic stateのobserver forwardは平均約**21.1 ms/state**（routing・Plan/Head/L3計算・照合を含むCPU計測）。

36/36局でobserverなしとのAction trace、winner、final VP、rank、policy steps、final GameState、Game RNGが完全一致。3B-4 episode recordとのAction、route、mask、turn/revision照合も全件一致。GameState mutation 0、Game RNG consumption 0、masked probability failure 0。収集前後のdigestはChampion `4225902d41a83540eb1f1552826ec623706e778b99ad4ae8ec015f976644392a`、Robber PPO `ecde7fa9e9d67370bb76686ecd767d2d532a76d467c0051aaebdf8d4aff702bf`、3B-2 Head `2000b1390fcf8135ba819ba40b1ee1720f0596327c2c658ec1e2f24a29dd1887`、新L3 Head `a5bd513d4a10c2a848996906443de05ab46c46ecb8519f621df29e679a742748`で各々前後一致。optimizer未生成。

unit testは **6 passed**。Bridge数値一致・batch/unavailable・候補順序不変・route別mask・Dev fallback・複数Priority同時high・L3→Plan Scorer/Priority Headへのgradient疎通・Champion latent無gradientを含む。backend全体は **395 passed / 1,172 subtests passed**（既存deprecation warning 2件）。[集計JSON](rl_hierarchical_l3_shadow_step3b5_20261013_evidence/summary.json)と36局の圧縮episode recordに各pre-action診断値を保存した。

勾配テストは学習そのものではない。Head重みの前後digest一致とoptimizer未生成を別途確認した。Champion Actionは照合用に保存するが、Teacher targetやagreement評価には使用しない。

## Step 3B-6案

L3を実委譲する前に、routeごとの実Action選択時点をsemi-MDP transition開始とする収集設計を固める。`ACTION_DEV_USAGE`で`NO_USE`からL3へ進む場合、Dev使用判断とL3 Actionのsample/log_probを混同しない。PlayerTradeは具体offerの合法性・保護境界、Construction Commitは同turn Action順序と持続状態の意味論を先に設計する。L3の実行結果に基づくon-policy lossがBridge経由でPlan Scorer/Priority Headへ届くことを検証してから、限定delegationへ進む。今回はPPO update・Action委譲を行わない。
