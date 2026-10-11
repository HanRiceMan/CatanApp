# Step 3B-1A — Minimal Winning Plan Semantics

Victory Planを「ちょうど10点の内訳」から、**10点以上へ到達するための最小得点構成**（minimal winning composition）へ修正した。L1のraw候補列挙だけの変更であり、feasibility、Action、Controller、Observation、Reward、Planner、Champion、PPO head/weightは変えていない。既存v1の結果は[履歴レポート](rl_victory_plan_context_step3b1_20261012.md)として残し、現在のContext schemaを`victory_plan_context_v2`、Shadow schemaを`victory_plan_shadow_step3b1a_v2`とした。

## 1. 候補の意味論

[`victory_plan_context.py`](../app/rl/victory_plan_context.py) は、S=Settlement新設数、C=City純増upgrade数、R=Longest Road新規獲得、A=Largest Army新規獲得について、`gain=S+C+2R+2A`を計算する。`gain>=remaining`で、存在する**各**得点要素を1単位取り除くと`gain<remaining`になるtupleだけをraw候補とする。S/Cは1個、R/Aは称号全体の2点が取り除く単位。すべてのexact候補はそのまま残る。`VictoryPlanContext.overshoot_points(candidate)`は`gain−remaining`を重複状態なしで返す算術診断であり、0/1を優劣scoreにしない。canonical IDは従来どおり`(S,C,R,A)`、辞書順で安定。IMPOSSIBLEはraw列に理由付きで残し、Plan候補valid maskでのみ無効化する。Action maskではない。

| 残り点 | 新しく表現できる最小勝利構成 | 除外する冗長例 |
|---:|---|---|
| 1 | `R`または`A`単独で+2、overshoot=1 | `S+R`等はSを除いても勝てる |
| 3 | `R+A`で+4、overshoot=1 | `S+R+A`と`C+R+A`は+1が余分 |
| 8 | 従来の`S2+C2+R+A=8`を保持 | — |

feasibility判定そのものはv1から変更していない。既に称号を持つ場合の再獲得tupleは従来どおりIMPOSSIBLE。本人既知のVP、piece recycling、未来情報禁止、Planner非依存も同じ。ImmediateWinResolverは「今1 Actionで確定勝利」、本Contextは「将来の最小得点構成」であり、接続していない。

## 2. 回帰テスト

score9でS/Cの+1手段が両方物理的に不可能な状態を構成し、未保有R/Aのraw ID `(0,0,True,False)` / `(0,0,False,True)` がUNKNOWN・validとして残り、`unresolved=False`となることを検証した。旧exact版ではこの状態はunresolvedだった。残り3点で`(0,0,True,True)`を保持し、`(1,0,True,True)` / `(0,1,True,True)`を生成しない。score2・残り8点では旧28 exact tupleと`(2,2,True,True)`を保持する。得点・City純増・title+2・future情報・GameState/RNG非干渉の既存テストもそのまま実行した。

## 3. 同seed 36局Shadow監査

Rule / Champion / Mixed各12局、各profileでSeat 1～4を3局ずつ配置した。seedは順に`33000000～011`、`33001000～011`、`33002000～011`。通常対局とobserver付き対局を同seedで実行し、合計2,810のpre-action状態を得た。旧v1の同じ2,810状態とrecord単位で比較した結果、既存51,029 raw候補はID・gain・valid・feasibility・理由まで全件不変で、新規候補は635件だけだった。内訳はscore 7で195件、score 9で440件。全新規候補のovershootは1点で、UNKNOWN 394件、IMPOSSIBLE 241件。FEASIBLEの新規増加はなかった。feasibility判定ロジックは変更していない。

| score | 状態数 | raw候補/状態 v1→v2 | valid候補/状態 平均 v1→v2 |
|---:|---:|---:|---:|
| 0 | 52 | 36→36 | 11.000→11.000 |
| 1 | 0 | — | — |
| 2 | 615 | 28→28 | 10.841→10.841 |
| 3 | 521 | 24→24 | 10.825→10.825 |
| 4 | 429 | 20→20 | 10.121→10.121 |
| 5 | 308 | 16→16 | 8.649→8.649 |
| 6 | 225 | 12→12 | 6.360→6.360 |
| 7 | 195 | 8→9 | 5.328→5.923 |
| 8 | 245 | 5→5 | 3.355→3.355 |
| 9 | 220 | 2→4 | 1.682→2.945 |

raw候補数の全分布（候補数:状態数）は`4:220, 5:245, 9:195, 12:225, 16:308, 20:429, 24:521, 28:615, 36:52`。最大raw **36**、最大valid **12**で、候補爆発はない。v2の総raw 51,664件のうち、FEASIBLE 815、UNKNOWN 23,126、IMPOSSIBLE 27,723。valid総数は23,941。`unresolved`は旧版・新版とも観測状態では0件だが、回帰テストの「score 9で+1建設のみ不可能、未保有titleはUNKNOWN」状態では旧版のtrueから新版のfalseへ正しく変化する。

score 9の実game例（rule、seed `33000001`）は現在9点・称号なし・残り1点で、`R`単独、`A`単独、`C1`、`S1`の4 raw候補を列挙し、全てvalid（FEASIBLE 1、UNKNOWN 3）。score 7の実gameでは`R+A=4`の候補を追加する。score 9で既所有のLongest/Largestはraw候補として監査可能なままIMPOSSIBLE・valid=falseとし、再獲得得点としては使わない。実gameのtitle ownership別valid件数は[summary](rl_victory_plan_minimal_winning_step3b1a_20261012_evidence/summary.json)に保存した。

observerは2,810/2,810状態でGameState・RNGを変更せず、36/36局でAction列、winner、final VP、rank、policy steps、最終GameState、RNGが通常対局と完全一致した。ChampionとRobber PPOのweight digestはそれぞれ`4225902d41a83540eb1f1552826ec623706e778b99ad4ae8ec015f976644392a`、`ecde7fa9e9d67370bb76686ecd767d2d532a76d467c0051aaebdf8d4aff702bf`で前後不変。既存3A artifactも変更しておらず、optimizerは生成していない。Planner選択・Intent・Reasonや未来dice/card/相手の非公開情報への新規依存はない。

再現用の[shadow records](rl_victory_plan_minimal_winning_step3b1a_20261012_evidence/shadow_records.json.gz)と[summary](rl_victory_plan_minimal_winning_step3b1a_20261012_evidence/summary.json)を保存した。対象unit testは11件成功。backend全体は`363 passed, 1172 subtests passed`（既存のdeprecation warning 2件のみ）。

## 4. Step 3B-2

以後はschema v2の`VictoryPlanContext`、stable candidate IDs、valid mask、3値feasibility、minimal-winning候補をPlan/Priority Shadow Headへ渡す。overshootは算術的特徴として利用可能だが、それだけで候補を順位付けしない。PlanはAction禁止maskではなく、価値判断・選択・Priorityは今回実装していない。
