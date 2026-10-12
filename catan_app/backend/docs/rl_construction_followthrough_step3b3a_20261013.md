# Step 3B-3A — Same-Turn Construction Follow-through Audit

## 判定

**Settlement-only: `SAME_TURN_BUILD_COMMIT_CANDIDATE`。** turn単位で797/802（99.38%）が同一turnにSettlementを建設した。残る5件も同turn中にCityを建設した。ただし14個の非即時Build decisionにはBankTrade/PlayerTradeの先行があり、**次ActionをSettlementへ固定して安全**とはまだ言えない。

**City-only: `VALUE_COMPARISON_REQUIRED`。** 同一turnのCity建設は411/484（84.92%）。73件はCityを建設せず、うち41件は起点から直接EndTurnした。非即時125 decisionのうち31件は後でCityを建設したが、残り94件はそうではない。Cityを「建てられるなら同turnに必ず建てる」とする根拠はない。

**BOTH: 将来の比較候補。** 19 opportunityはSettlement 15、City 4を即建設したが標本は小さく、Champion選択を教師にはしない。Development使用とのaction-phase競合はこのsampleでは0件だった。pre-roll Development判断は別に残す。

## Source、追跡定義、非干渉

Step 3B-3の300局を**再実行せず**再解析した。Rule `34000000–34000099`、Champion `34001000–34001099`、Mixed `34002000–34002099`。各profile100局・各seat25局。profileは対戦相手構成で、対象はlearner Championの通常`action` phaseのみ。ImmediateWin対象はSettlement-only 45、City-only 83、BOTH 4 decisionを除外した。

起点は同一game・turn・build classの**最初の通常action decision**。後続の同class decisionはstate-levelには保持するが、opportunityには再計上しない。対象build、EndTurn、turn終了またはterminalまで追跡する。一時的にbuildabilityが失われたり、Roadでtitleを得たり、別の得点建設が先行したりしても、その後に対象buildが起きれば「later same-turn build」とする。最終分類は互いに排他的で、別途「一時的buildability loss」「他得点先行」「EndTurnで終了」を重複可能な診断flagとして保持する。資源vectorは起点と**build/終了Action直前**を比較する（建設費支払い後の資源を意味しない）。

source recordはlearnerのpre-action boundaryを全て含み、PlayerTrade後にaction phaseへ戻ったかは追える。相手のresponse/counterの**内部Action詳細はsourceにない**のでrevision差から推測していない。各BankTradeのgive/get、直前/直後の資源・建設可否と、PlayerTrade後のlearner復帰は圧縮evidenceに保存した。

Step 3B-3収集時に300/300局でobserver有無のAction列、winner、VP、rank、policy steps、final GameState、RNGが一致し、各forwardのGameState/RNG mutationも0と検証済み。本Stepは保存済みJSONを読み込むだけでGameState、Action、Controller、Reward、PPO weightに触れず、optimizerも作っていない。今回のparityは**既存evidenceから継承した検証**であり、新規game再実行による独立再検証ではない。

source summaryの`mutation_failures=0`、`rng_failures=0`、`parity_failures=0`、`optimizer_created=false`と全300 episodeのterminal/parityを解析時にassertした。既存収集時のweight digestはChampion `4225902d41a83540eb1f1552826ec623706e778b99ad4ae8ec015f976644392a`、Robber PPO `ecde7fa9e9d67370bb76686ecd767d2d532a76d467c0051aaebdf8d4aff702bf`、3B-2 Head `2000b1390fcf8135ba819ba40b1ee1720f0596327c2c658ec1e2f24a29dd1887`（いずれも収集前後一致）。本Stepはweightファイルへの書き込みなし。

## Opportunity単位の結果

state-levelではSettlement-only 816、City-only 543、BOTH 19、計1,378 decision。turn・class重複除去後は計1,305 opportunity（802 / 484 / 19）。

| class | opportunity | 即対象Build | 後で対象Build | 対象同turn合計 | 別得点建設 | buildability消失・対象なし | EndTurn・対象なし | terminalその他 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Settlement-only | 802 | 792 | 5 | **797 (99.38%)** | 5 | 0 | 0 | 0 |
| City-only | 484 | 382 | 29 | **411 (84.92%)** | 11 | 20 | 41 | 1 |
| BOTH | 19 | 19 | 0 | **19 (100%)** | 0 | 0 | 0 | 0 |

Settlement-onlyの別得点5件は**全てCity建設**。City-onlyの別得点11件は**全てSettlement建設**。分類欄の「EndTurn」は直接終了など優先原因であり、他得点・buildability消失の後にEndTurnした例も含めて数えると、対象を建てずEndTurnしたのはSettlement 4件、City 72件。City-onlyの対象未建設73件の残り1件はterminalである。対象建設までの一時的buildability消失はSettlement 6、City 31 opportunityで観測され、消失だけで追跡を打ち切っていない。

| profile | Settlement同turn / opportunity | City同turn / opportunity |
|---|---:|---:|
| Rule | 289/290 (99.66%) | 188/199 (94.47%) |
| Champion | 249/250 (99.60%) | 103/146 (70.55%) |
| Mixed | 259/262 (98.85%) | 120/139 (86.33%) |

これはprofile別ruleを作る材料ではない。City-onlyの差は、少なくとも「ほぼ必ず同turn建設」という一律仮説を弱める。

## 非即時Buildの全件監査

Step 3B-3のSettlement-only非即時14 **decision**（BankTrade 9、PlayerTrade 5）を全件追跡した。9件は同一turnでの重複を含み、初回opportunityではBankTrade 6、PlayerTrade 4。state-level結果は、BankTrade先行でSettlement後建設3・Cityへ変更6、PlayerTrade先行でSettlement後建設3・Cityへ変更2。

| profile / seed / turn / revision | 最初のAction | 同turnの結果 |
|---|---|---|
| Rule 34000013 / 62 / 410, 411 | BankTrade ×2 | Settlement後建設（同一opportunity） |
| Rule 34000056 / 53 / 364 | PlayerTrade | City、PlayerTrade、BankTradeを経てSettlement |
| Rule 34000057 / 74 / 528–531 | BankTrade ×3、PlayerTrade | Settlementは建てずCity（同一opportunity） |
| Champion 34001028 / 81 / 483 | PlayerTrade | Settlement後建設 |
| Champion 34001047 / 72 / 356 | BankTrade | Settlementは建てずCity |
| Champion 34001049 / 70 / 492 | BankTrade | Settlement後建設 |
| Mixed 34002013 / 70 / 395 | PlayerTrade | Settlement後建設 |
| Mixed 34002027 / 68 / 401 | BankTrade | Settlementは建てずCity |
| Mixed 34002027 / 72 / 420 | PlayerTrade | Settlementは建てずCity、後にterminal |
| Mixed 34002081 / 66 / 420 | BankTrade | Settlementは建てずCity |

City-only非即時125 **decision**の全追跡結果は次の通り。Road先行32のうち25件は後にCity。BankTrade先行24のうち20件は必要資源を失い、そのturnのCityに到達しなかった。直接EndTurn 41は全てCityを建てなかった。

| City-onlyの次Action | 件数 | City後建設 | 別得点建設 | EndTurn・Cityなし | buildability消失・Cityなし | terminalその他 |
|---|---:|---:|---:|---:|---:|---:|
| Road | 32 | 25 | 2 | 4 | 0 | 1 |
| BankTrade | 24 | 0 | 3 | 1 | 20 | 0 |
| PlayerTrade | 28 | 6 | 12 | 10 | 0 | 0 |
| EndTurn | 41 | 0 | 0 | 41 | 0 | 0 |
| **合計** | **125** | **31** | **17** | **56** | **20** | **1** |

state-levelでのCity後建設31と、初回opportunity単位の29は、同turnの重複stateを数えるかどうかの差である。Settlement非即時も同様にstate-level後建設6、初回opportunity後建設5。

## Action順序・資源・phase跨ぎ

opportunityで後から対象を建設した34件の先行Action数は、1 Actionが30件、2 Actionが3件、3以上が1件（即Buildは別途Settlement 792、City 382、BOTH 19）。先行Action familyはRoad 25、BankTrade 4、PlayerTrade 10、Build City 1（同じ経路内に複数familyを含み得る）。**最初のAction**で区切るとRoad→対象Build 24、BankTrade→対象Build 2、PlayerTrade→対象Build 8、PlayDevelopment→対象Build 0。したがって「BankTrade/PlayerTrade→Build」は少数だが実在し、Build-first固定でAction順序を変える可能性がある。

opportunity内のBankTrade eventは36件。合法なgive/get tupleを全件保存し、その後同turnの次learner actionで元の対象がbuildableだったのは9件、不可は27件。PlayerTrade eventは51件で、**51/51件で同turnのlearner action phaseへ復帰**した。response/counterの内部詳細は記録がなく評価していない。Road event39件では次learner action時に対象buildable 32、不可5、次boundaryなし2。不可5件には、Road以前のBankTrade等で既に建設可能性を失っていた経路も含まれるため、Roadの因果効果とは読まない。

起点から対象Build直前までの平均資源変化（wood, brick, sheep, wheat, ore）は、City later 29件で **(-0.86, -0.83, 0.00, 0.00, -0.03)**、Settlement later 5件で **(0.00, -1.40, -0.80, +0.20, 0.00)**。City buildabilityを失った20件では終了Action直前まで **(+0.55, -0.20, +0.15, -1.05, -3.30)**。これは選択された経路の記述統計であり、Action価値の差ではない。各opportunityの初期/終点資源とtrade前後の資源は圧縮記録に残した。同turn複数建設を含むopportunityは61件。

## 次工程

Step 3B-4ではControllerを直ちに変更せず、Settlement-onlyについて「同turn中の建設機会commit」と「BankTrade/PlayerTrade等の先行順序」を分ける**Shadow routing**を設計する。City-onlyはBuild/No-BuildをL1 Plan・L2 Priority条件付きの価値比較へ残す。BOTHはSettlement/City比較候補として残す。Development Usageはpre-rollを含む独立境界として設計し、今回0件だったaction-phase競合だけを根拠に消さない。Champion ActionをTeacherにせず、Reward・Policy・Action・Controllerを変更しない。

再現用: [解析コード](../app/rl/analyze_construction_followthrough_step3b3a.py)、[集計JSON](rl_construction_followthrough_step3b3a_20261013_evidence/summary.json)、`opportunities.json.gz`、`state_followthrough.json.gz`。unit testは [test_construction_followthrough_step3b3a.py](../tests/test_construction_followthrough_step3b3a.py)。

検証: Step 3B-3A unit test **5 passed**、backend全体 **382 passed / 1,172 subtests passed**（既存deprecation warning 2件）。Road/title得点先行でもCityを後建設として数えること、一時的なbuildability loss後の復帰、直接EndTurn、同turn重複除去、BOTHの建設familyをテストした。
