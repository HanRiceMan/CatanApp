# Step 3B-3 — Immediate Construction / Development Usage Boundary Audit

## 結論

通常action phaseで確定即時勝利を除くと、Settlementだけ建設可能な局面ではChampionが **802/816件（98.28%）** で建設した。一方、Cityだけ建設可能な局面では **418/543件（76.98%）** に留まり、非建設125件はRoad 32、EndTurn 41、PlayerTrade 28、BankTrade 24だった。よって**Settlement-onlyは決定論的routingの有力候補（CASE A寄り）だが、例外14件の監査とprotected trade境界を残す**。**City-onlyの無条件Buildは支持されず、価値比較を残すべき（CASE B）**。BOTHは19通常局面しかなく、Settlement 15 / City 4という観測をTeacherにはしない。

建設可能局面でのaction-phase Development使用可能性・実使用はともに0件、Buy Developmentも0件だった。ただしpre-rollではKnight等の使用が実際に発生しており、**Development Usage Policyが不要という結論ではない**。pre-rollのRoll対Card使用は建設routingとは別の境界として設計する。今回Controller、Action、Observation、Reward、Executor、Plan/Priority Head、Championを変更せず、optimizerを作成・実行していない。

## 監査方法と固定sample

新seedはRule `34000000–34000099`、Champion `34001000–34001099`、Mixed `34002000–34002099`。各profile100局、seat 1–4を各25局、計300局。profileは**相手controller構成**であり、監査する意思決定は全てlearner Championのもの。各seedでobserverなしのLegacy対局とobserverありの同一対局を再生した。候補生成・Plan Headは行動に渡していない。全300局でAction列、winner、最終VP、rank、policy steps、最終GameState、RNG、terminal/truncationが一致した。各episodeはterminal到達・truncationなし。

対象はlearnerの通常`action` phase。建設可否は資源だけでなく、`legal_actions()`に合法な`BuildSettlementAction` / `BuildCityAction`が1個以上あるかで判定した。`game.free_road_remaining > 0`は同じ`action` phaseでも`free_road`として別計上し、建設比較には入れない。`turn_pre_roll`、robber、discard等も別。`resolve_immediate_win()`で証明できる確定10点Actionがある状態は、通常の「建てる／建てない」分母から除外した。Legacy Championの実Actionは診断値であって正解ラベルではない。

| learner boundary phase | 件数 |
|---|---:|
| 通常action | 13,144 |
| turn_pre_roll | 5,632 |
| free_road | 64 |
| robber_move / robber_steal | 1,260 / 1,253 |
| discarding | 515 |
| rolling_order / ready_to_play | 412 / 75 |

action 13,144件のうちImmediateWin 166件を除いた通常比較は12,978件。分類は次の通り。

| 建設分類 | ImmediateWin含む | 除外後 | Championの実Action |
|---|---:|---:|---|
| NONE | 11,634 | 11,600 | Road 1,603、BuyDev 597、BankTrade 1,022、PlayerTrade 3,389、EndTurn 4,989 |
| SETTLEMENT_ONLY | 861 | 816 | Settlement **802**、BankTrade 9、PlayerTrade 5 |
| CITY_ONLY | 626 | 543 | City **418**、Road 32、BankTrade 24、PlayerTrade 28、EndTurn 41 |
| BOTH | 23 | 19 | Settlement **15**、City **4**、その他0 |

Settlement-onlyのBuild率は95%を超えるが99%・99.5%には届かない。City-onlyは95%に届かない。profile別City-only Build率はRule **191/218=87.6%**、Champion **106/166=63.9%**、Mixed **121/159=76.1%** と差がある。Settlement-onlyはRule 292/299、Champion 249/252、Mixed 261/265。相手構成によってCity判断が変わる可能性はあるが、この監査のみでprofile別hard ruleは作らない。

## Development競合と資源機会

カード使用可能性は、本人の既知カードから今ターン購入分を除き、`used_development_this_turn`、Year of Plentyの銀行在庫・合法target等を反映した**実際の合法Action**で判定した。Monopolyの相手非公開手札真値は参照していない。各stateにはusable card種類・保有旧カード数、資源5-vector、手札枚数、discard exposure、BuyDev合法性、BankTrade合法性、建設可能vertex数を保存した。手製の建設価値scoreは作っていない。

| 建設分類（ImmediateWin除外） | 使用可能Knight | Monopoly | Year of Plenty | Road Building | 実際のPlayDev | 実際のBuyDev |
|---|---:|---:|---:|---:|---:|---:|
| NONE 11,600 | 0 | 0 | 0 | 0 | 0 | 597 |
| SETTLEMENT_ONLY 816 | 0 | 0 | 0 | 0 | 0 | 0 |
| CITY_ONLY 543 | 0 | 0 | 0 | 0 | 0 | 0 |
| BOTH 19 | 0 | 0 | 0 | 0 | 0 | 0 |

pre-rollでは使用可能state-card件数がKnight **341**、Monopoly **45**、Year of Plenty **38**、Road Building **34**。実使用はそれぞれ **340 / 44 / 38 / 32** 件だった。action-phaseで競合が観測されなかったのは、この既存controllerのphase運用に依存する。将来のDevelopment Usage Policyがpre-rollで使用を控えるようになれば、action-phase競合が発生し得る。Road Buildingのpre-roll使用32件に対しfree-road learner boundaryは64件だったが、個々のカードの価値判断を示す数字ではない。

建設可能な通常局面での非建設は計139件。Road 32、BankTrade 33、PlayerTrade 33、EndTurn 41。特にSettlement-onlyの例外14件はBankTrade 9とPlayerTrade 5のみで、Road/EndTurn/BuyDev/PlayDevは0。City-onlyでは非建設125件が複数familyに分かれ、カード競合を先に設計すれば単純Buildへ落とせる、という状況ではない。PlayerTradeの優先は暫定protected境界であり、`PlayerTrade > Build`を戦略ruleとして採用しない。

## ImmediateWin、BOTH、実例

ImmediateWin 166件は、建設分類別にNONE 34、Settlement-only 45、City-only 83、BOTH 4。Resolverのcatalog順で選んだ確定勝利Action familyはSettlement 42、City 80、Road（Longest Road獲得を含む）44だった。これはLegacy Championが必ずそのActionを**実行した**件数ではない。確定勝利は既存Resolver最上位という別責務であり、通常の建設選択率へ混ぜない。

BOTHは全23 state（確定勝利除外後19）。合法Settlement vertexは各state 1個、合法City vertex平均2.57個。3B-2 Plan/Priority Shadowは全23 stateで実行し、valid Plan候補2–12件、City最大生産増分7–13 pip（平均9.78）を記録した。Plan確率は未学習zero-initによるvalid候補一様分布、各Priorityは全て0.5。**この値からSettlement/Cityの優劣は判断していない**。Plan candidateやPriorityをActionへ渡していない。

実局面の代表例（seed / seat / turn / revision、全てlearner pre-action state）:

| 観測 | 実例とChampion Action |
|---|---|
| Settlement-only→Build | `34000000 / 1 / 13 / 101`、score2、`BuildSettlement(vertex=29)` |
| City-only→Build | `34000001 / 2 / 38 / 287`、score8、`BuildCity(vertex=29)` |
| BOTH→Settlement | `34000003 / 4 / 40 / 255`、score3、`BuildSettlement(vertex=37)` |
| BOTH→City | `34000056 / 1 / 53 / 366`、score7、`BuildCity(vertex=7)` |
| Settlement-only→BankTrade | `34000013 / 2 / 62 / 410`、score8、羊→鉱石 |
| City-only→Road | `34000008 / 1 / 49 / 340`、score7、edge3 |
| City-only→EndTurn | `34000029 / 2 / 22 / 154`、score2 |
| City-only→PlayerTrade | `34000052 / 1 / 69 / 483`、score7 |

建設可能→Play Development、建設可能→Buy Developmentの実例はそれぞれ**0件**であり、架空例は作らない。BuyDev自体は建設不可局面で597件観測された。

## 不変性と次工程

Observerは各記録前後でGameStateとGame RNG counterの一致をassertし、違反0件。Champion weight digestは収集前後とも`4225902d41a83540eb1f1552826ec623706e778b99ad4ae8ec015f976644392a`、Robber PPOは`ecde7fa9e9d67370bb76686ecd767d2d532a76d467c0051aaebdf8d4aff702bf`、3B-2 Plan/Priority Headは`2000b1390fcf8135ba819ba40b1ee1720f0596327c2c658ec1e2f24a29dd1887`。本監査用optimizerを生成せず、gradient/updateは0回。PlannerのReason/Intent、Champion ActionをHead教師に使わず、相手hidden resource真値・未来乱数・将来Development drawも入力していない。全backend testは**377 passed、1,172 subtests passed**（既存のdeprecation warning 2件）。

[全件summary](rl_immediate_construction_audit_step3b3_20261012_evidence/summary.json) と[episode別圧縮record](rl_immediate_construction_audit_step3b3_20261012_evidence/episodes/)を保存した。各episode recordにはseed/profile/seat、turn/revision/policy step、build class、合法vertex、カード使用可能性、資源、実Action、確定勝利分離、BOTH時のPlan/Priority Shadowを含む。

Step 3B-4では**実Actionを変えないShadow Controller**として、ImmediateWinとforced/pre-rollを先に分離し、Settlement-onlyの直接Build候補（BankTrade/PlayerTrade例外を別診断）、City-onlyのBuild対Road/Trade/EndTurn比較、BOTHのPlan/Priority条件付き比較、NONEのInvestment候補を表現する。City-onlyを無条件Buildへhard routingしない。Development Usageはpre-rollのRoll対各カード使用を独立してShadow設計する。今回のChampion行動頻度は価値・returnの証明ではないため、routing採否には次工程で反実仮想または限定的実委譲の安全性検証が必要である。
