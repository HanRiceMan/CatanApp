# Step 3A-10: CITY_ONLY の paired counterfactual 監査

## 範囲と実験方法

この工程では学習・Reward変更・Champion変更を行わない。対象は `CITY_ONLY` だけで、`SETTLEMENT_ONLY` と `SETTLEMENT_AND_CITY` は含めない。Action前の同一 GameState（乱数源の seed/counter、Environment の得点・step 境界を含む）を複製し、凍結 Executor の `BUILD_NOW → BuildCityAction` と `DEFER → non-city Action` をそれぞれ1回だけ実適用した。その後はGateを呼ばず、両枝とも従来Champion Plannerと固定対戦相手で終局まで進めた。これは「その1回だけ今都市を建てるか」の比較であり、後続のすべての都市建設を禁止する比較ではない。

主実験は両枝とも元の乱数源状態から開始する。Actionの違いで後続の乱数消費がずれるため、完全に同じ未来乱数を保証する実験ではない。追加のfuture RNG seedを両枝へ同じように渡せる診断APIはあるが、主結果では使っていない。同一game内の複数局面は独立標本とは見なさない。

Rule / Champion / Mixed × Seat 1～4を均等化し、Step 3A-7～9と別の `2631000` 起点・profile/seat別seed bandを使用した。stateの `game_id` / `game_seed` / `pre_policy_steps` を残し、将来の分割はgame単位で可能。生データは `experiments/construction_city_step3a10_20261004/pairs.jsonl`、集計は同ディレクトリの `summary.json` に保存した。再実行は `python -m app.rl.run_construction_city_counterfactual_step3a10 --per-stratum 17`、集計のみ再生成は `--summary-only`。既存 `.venv-rl` のPython本体が壊れている環境では、Codex同梱Pythonと既存venvライブラリを組み合わせる必要がある。

割引単位は従来の `CatanEnv.policy_steps` と同じlearner decision step。`gamma=0.99` で、各枝の環境報酬を `Q = Σ 0.99^t r_t` とし、既存RewardConfigのVP成分とterminal成分も同様に分けて保存した。差はすべて `BUILD − DEFER`。終局の実際の勝敗・得点・順位と割引済みterminal rewardは別物として保存・集計する。特に同じ勝敗でも終局時点が異なれば割引済みterminal差は0とは限らない。

## 収集結果

204 CITY_ONLY局面、98 baseline game。Rule / Champion / Mixedは各68局面、Seat 1～4は各51局面。1 gameから平均2.08局面（最大7）を採取した。Champion実判断はBUILD 173 / DEFER 31。BUILD branchとDEFER branchはいずれも204/204で合法・終局、truncation / 非finite報酬 / 未完走は0。元GameStateは両branch後も不変。保存した最初のstate `city-cf-rule-2641000-1-51` を元seedから再生し、両枝の結果が保存値と完全一致することも確認した。保存された全408枝で、割引環境return = 割引VP成分 + 割引terminal成分が誤差1e-7以内で成立した。

DEFER初手の内訳は `BANK_TRADE 64 / BUY_DEVELOPMENT 63 / END_TURN 37 / BUILD_ROAD 33 / PLAYER_TRADE 7`。PLAYER_TRADEも通常の環境処理で相手応答を通過して終局した。Championが実際にDEFERした31局面の内訳は、道路11・銀行交換8・対人交渉7・手番終了4・発展購入1。

### Returnと最終結果

各値の差はBUILD−DEFER。`Q_current_reward` は既存RewardConfig `win=+1 / loss=-1 / VP差×0.05` と `gamma=.99` による実環境returnで、教師一致報酬などは含めない。

| 指標 | BUILD | DEFER | paired差 |
|---|---:|---:|---:|
| Q_current_reward 平均 | 0.3166 | 0.3426 | -0.0260 |
| 割引VP成分 平均 | 0.1485 | 0.1404 | +0.0081 |
| 割引terminal成分 平均 | 0.1681 | 0.2022 | -0.0341 |
| 勝利数 | 126 | 130 | BUILDのみ勝ち10 / DEFERのみ勝ち14 / 同じ勝敗180 |
| 最終VP平均 | 9.000 | 8.892 | +0.108 |
| 最終順位平均（小さい方が良い） | 1.615 | 1.642 | -0.027 |
| 終局時city平均 | 2.554 | 2.284 | +0.270 |
| 終局時settlement平均 | 2.613 | 2.848 | -0.235 |
| 終局時発展カード取得枚数平均 | 2.049 | 2.343 | -0.294 |
| 終局までの累計learner step平均 | 78.55 | 79.63 | -1.07 |
| 後続の本人discard action平均 | 0.735 | 0.745 | -0.010 |
| 最終Longest Road / Largest Army保持件数 | 71 / 24 | 78 / 27 | -7 / -3 |

`ΔQ_total` は平均 -0.0260、中央値 +0.0010、p25 -0.0278、p75 +0.0505、最小 -2.0135、最大 +1.8296、標準偏差0.5395。厳密な符号ではBUILD優位130 / DEFER優位73 / 同点1だが、`|ΔQ|≤0.01` が49件、`|ΔQ|>0.05` に限ればBUILD優位57 / DEFER優位38である。`ΔVP shaping` の平均 +0.0081、中央値 +0.0019、p25 -0.0006、p75 +0.0154。`Δterminal` の平均 -0.0341、中央値0、p25 -0.0268、p75 +0.0336。最終VP差の平均 +0.108、同点142/204。順位差の平均 -0.027、同点153/204。episode length差の平均 -1.07 step、中央値0、p25 -5、p75 +2。

### Champion判断、profile、DEFER候補

Champion BUILD 173局面ではBUILD側のreturnが高いのは110、DEFER側62、同点1（非同点の64.0%）。Champion DEFER 31局面ではDEFER側11、BUILD側20（DEFER側35.5%）。ただし後者のうち17件は `|ΔQ|≤0.01`。閾値0.01を超える差はBUILD側6 / DEFER側8、0.05を超える差はBUILD側5 / DEFER側3に過ぎない。Champion DEFER時も実勝敗は26件で同じ、BUILDのみ勝利3、DEFERのみ勝利2。よって20対11を「DEFER判断の多くが誤り」とは断定しない。

| profile | 局面 | Champion DEFER | ΔQ平均 / 中央値 | BUILD / DEFER優位（厳密符号） |
|---|---:|---:|---:|---:|
| Rule | 68 | 10 | -0.1078 / +0.0005 | 39 / 29 |
| Champion | 68 | 12 | +0.0690 / +0.0072 | 49 / 19 |
| Mixed | 68 | 9 | -0.0392 / +0.0027 | 42 / 25（同点1） |

| DEFER初手 | 件数 | ΔQ平均 / 中央値 | Δterminal平均 | Δ最終VP平均 | `|ΔQ|>0.01` BUILD / DEFER |
|---|---:|---:|---:|---:|---:|
| END_TURN | 37 | +0.0612 / +0.0166 | +0.0483 | +0.162 | 19 / 9 |
| BUY_DEVELOPMENT | 63 | -0.1340 / +0.0102 | -0.1342 | -0.048 | 32 / 25 |
| BANK_TRADE | 64 | +0.0168 / +0.0160 | -0.0023 | +0.313 | 34 / 23 |
| BUILD_ROAD | 33 | -0.0064 / +0.0005 | -0.0044 | -0.030 | 5 / 8 |
| PLAYER_TRADE | 7 | +0.0015 / +0.0010 | 0 | 0 | 0 / 0 |

END_TURN型DEFERはBUILD優位が27/37（厳密符号）だが、DEFER優位も10件あり、初手が常に不自然な待機とまでは言えない。PLAYER_TRADEは7件すべて最終結果が同じで、微小な割引差だけだった。これらは各Action familyを将来個別に扱う動機にはなるが、今回のデータだけでExecutorを変更しない。

### Reward alignment

割引済みterminal成分の符号で分類すると、A（総return/terminalともBUILD）79、B（ともDEFER）70、C（総returnはBUILD・terminalはDEFER）11、D（総returnはDEFER・terminalはBUILD）0、terminal差ゼロ等44。**Cの11件はすべて実際の勝者が両枝で同じ**であり、terminal報酬を受けるstepの違いによる割引差である。実勝敗の符号で分類するとA10、B14、C0、D0、勝敗同じ・return同点180。今回、`+0.05/VP` が異なる勝敗を誤方向へ選ばせるという証拠は得られず、Reward変更は不要と判断する。ただし一対ずつの試行で因果効果を確定したわけではない。

### Contextとpaired variance

厳密符号でのBUILD優位130件 / DEFER優位73件の平均値は、hand size 9.10 / 9.27、discard exposure 59.2% / 67.1%、blocked pips 2.35 / 3.27、city production gain 10.29 / 10.19、settlement candidate有 71.5% / 67.1%、score 5.43 / 5.70、city count 0.58 / 0.53、settlement count 3.50 / 3.84、site count 4.08 / 4.37、development affordable 52.3% / 61.6%、bank trade options 4.45 / 4.66。settlement ETAはvalid例93 / 49件で平均0.77 / 0.57、settlement shortage totalは同じvalid例で2.86 / 2.57。Road Title acquisition gapはvalid例104 / 58件で3.06 / 2.45、Knight plays neededは2.44 / 2.53。CITY_ONLYなのでcity ETAは両群とも0。差はいずれも小さく、単変量の閾値ruleを導く根拠ではない。Contextが予測に使えるかは、game分割の追加評価が必要。

局面レベルのΔQ標準偏差0.5395は平均差0.0260の約21倍で、98 gameへまとめた各gameの平均ΔQでも標準偏差0.4990。各gameから最大7局面取っているため204件を独立試行と扱わない。追加乱数を共有して再実行できるAPIを設け、元データで最もDEFER有利に見えた `city-cf-rule-2661002-2-52` を試した。元乱数のΔQは -2.0135、別の3つの共有future seedでは +2.0069 / +1.9778 / -2.0014 と符号が反転した。これは一例であり、全体の条件付き期待値を推定するものではないが、**単発rolloutの大差を局面の正解ラベルにできない**ことを直接示す。一方、微小差の1例は3つの追加seedでも同じΔQ≈+0.0005だった。追加seed診断は主204件の統計へ混ぜていない。

## 解釈上の注意と次の判断

ChampionのBUILD/DEFERは正解ラベルではない。paired returnの差は、両枝の後続Plannerと対戦相手を固定したときのone-decision counterfactualであり、Gate学習後の長期的な新方策の価値そのものではない。小さい差、同一gameの相関、後続乱数の分岐を考慮して読む。Reward alignmentは、割引済みterminal成分とのA/B/C/Dと、実勝敗とのA/B/C/Dを別表で扱う。Cを見つけても即座にVP shapingが原因と断定せず、即時+0.05以外の将来VPや終局の差も検討する。

Case Aは未確認。Champion DEFERが高return側なのは11/31だが、微小差が多くContextにも明快な分離はないため、BC warm-startの根拠にならない。Case Bに見える20/31のBUILD優位も17/31のnear-tieと乱数感度を考えると確証ではないが、少なくとも**Champion imitationを次に採用しない**判断はできる。Case Cは異なる実勝敗では0件で、Reward変更は今は不要。Case DはEND_TURNがBUILDより劣る例が多いものの一貫した低価値ではなく、DEFER Executorを今すぐ変更する根拠も不足する。

したがって4択では**環境returnに基づくCITY_ONLY GateのRL継続**を候補とし、2回目updateの前に少数の代表局面（とくにChampion DEFER、END_TURN、勝敗反転outlier）で複数future continuationを行い、符号の安定性を確認する。そこでDEFER Executor固有の低価値が反復して確認されればExecutorを先に修正する。RewardやExecutorを現時点で変更したり、単発counterfactualを正解ラベルとしてBCに使ったりはしない。SETTLEMENT_ONLYと両建設可能局面は引き続き対象外。

## 安全性

本工程でPPOの2回目update、BC、Reward/Executor/Championの変更は行っていない。両枝のActionは既存 `CatanEnv.step_action()` で合法性を検査し、非finite reward、終局未達、元状態の変化があれば収集を停止する。既存Champion artifact、377 Action Space、Observation v7、Planner本体は未変更。
