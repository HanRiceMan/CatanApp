# Macro Goal Step 2A-2: 対戦分布拡張とtaxonomy安定性

## 結論

現行ChampionのActionを変えず、Teacher Datasetの対戦相手をRule / Champion /
Mixedへ拡張した。300局・13,898 Decisionを収集した結果、DEVELOPMENT、
KNIGHT_TITLE、ROAD_TITLEの比率と、BuildRoadの未分類率は対戦条件を変えても
大きく崩れなかった。

一方、Rule戦ではCITYが28.32%、Champion戦では17.14%となり、強い相手ほど
SETTLEMENT待ち・交渉が多い状態分布へ移った。これはtaxonomyの破綻ではなく、
対象Championの勝率と到達局面の差を反映した分布差と判断する。今後の学習・評価は
profileを混ぜ、game単位かつprofile×seatを層化して分割する必要がある。

HOLDは全300局で0件だった。無理な再分類は行っていない。初回Distillationでは
学習クラスから外すか、常時maskするのが妥当である。

## 実装範囲

- `play_teacher_episode()`でRule / Champion / Mixedを選択可能にした。
- 収集対象Championは実際のseat 1〜4を均等にローテーションする。
- game recordへ`opponent_profile`と全seatのagent type、model ID、Planner使用有無を保存する。
- Decision recordへ`opponent_profile`を保存し、`game_id`でgame recordへ結合可能にした。
- profile別のGoal、Execution、None、turn band、reason、勝敗、seat集計を追加した。
- Goal比率の最大差とprofile間Total Variation距離を追加した。
- 既存Champion、PPO/GNN、377 Action Space、Planner設定、モデルartifactは変更していない。
- Macro Head、Distillation、Shadow Policy、DAgger、PPO学習変更は未実装である。

CLI例:

    python -m app.rl.collect_macro_teacher_dataset \
      --experiment-id macro_teacher_step2a2_300_s01_20260928 \
      --opponent-profiles rule champion mixed \
      --games 100 --seed-start 2370001 --profile-seed-stride 1000 \
      --parity-games 5 --parity-seed-start 2380001

Python 3.13で保存された盗賊PPOをそのまま読むため、実収集は既存Python 3.13と
`.venv-rl-313`のsite-packagesを使用した。モデルの変換や再保存はしていない。

## Dataset

- experiment: `macro_teacher_step2a2_300_s01_20260928`
- schema: `macro_teacher_v2`
- Observation: v2、1667次元
- game数: 300（各profile 100）
- Decision数: 13,898
- Observation shape: `(13898, 1667)`
- collection seed:
  - Rule: 2370001〜2370100
  - Champion: 2371001〜2371100
  - Mixed: 2372001〜2372100
- 各profileで対象Championのseat 1〜4を25局ずつ割り当てた。
- 生Datasetは`experiments/`配下のローカルartifactでありgitignore対象である。

### Opponent profile

| profile | 対象Champion以外 |
|---|---|
| Rule | ルールAI × 3 |
| Champion | `ppo_gnn_board_65k_s03_exp_v003 + 現行Planner` × 3 |
| Mixed | 現行Champion × 1、`ppo_robber_value_73k_s01_exp_v001 + 既存Planner` × 1、ルールAI × 1 |

## 対局・Decision概要

| profile | games | Decision | 対象Champion勝利 | 平均VP | 平均順位 |
|---|---:|---:|---:|---:|---:|
| Rule | 100 | 4,372 | 86 | 9.70 | 1.180 |
| Champion | 100 | 5,170 | 23 | 6.97 | 2.525 |
| Mixed | 100 | 4,356 | 34 | 7.54 | 2.060 |

Champion戦でDecisionが多いのは、強い相手との対局が長くなり、対象Championが
勝利局面まで進めない対局も増えたためである。Decision件数をprofile間で同数に
揃えるより、game単位で保持してprofileを層化する方がリーク防止と実分布の保存に適する。

## Objective Goal分布

| Goal | Rule | Champion | Mixed | 全体 |
|---|---:|---:|---:|---:|
| SETTLEMENT | 2,517 (57.57%) | 3,419 (66.13%) | 2,817 (64.67%) | 8,753 (62.98%) |
| CITY | 1,238 (28.32%) | 886 (17.14%) | 855 (19.63%) | 2,979 (21.43%) |
| DEVELOPMENT | 98 (2.24%) | 151 (2.92%) | 126 (2.89%) | 375 (2.70%) |
| ROAD_TITLE | 126 (2.88%) | 85 (1.64%) | 71 (1.63%) | 282 (2.03%) |
| KNIGHT_TITLE | 95 (2.17%) | 106 (2.05%) | 95 (2.18%) | 296 (2.13%) |
| HOLD | 0 | 0 | 0 | 0 |
| None | 298 (6.82%) | 523 (10.12%) | 392 (9.00%) | 1,213 (8.73%) |

rare classは各100局で最低71〜151件確保できた。HOLDだけは全条件で0件であり、
現行ChampionのTeacher taxonomyでは実質未使用である。

## Objective Goal × Execution Mode

### Rule

| Goal | Execution内訳 |
|---|---|
| SETTLEMENT | BUILD_ROAD 504、BUILD_SETTLEMENT 334、END_TURN 1,203、TRADE_BANK 163、TRADE_PLAYER 313 |
| CITY | BUILD_CITY 238、END_TURN 488、TRADE_BANK 112、TRADE_PLAYER 400 |
| DEVELOPMENT | BUY_DEVELOPMENT 98 |
| ROAD_TITLE | BUILD_ROAD 126 |
| KNIGHT_TITLE | BUY_DEVELOPMENT 95 |
| None | BUILD_ROAD 10、TRADE_BANK 60、TRADE_PLAYER 228 |

### Champion

| Goal | Execution内訳 |
|---|---|
| SETTLEMENT | BUILD_ROAD 493、BUILD_SETTLEMENT 257、END_TURN 1,493、TRADE_BANK 314、TRADE_PLAYER 862 |
| CITY | BUILD_CITY 126、END_TURN 330、TRADE_BANK 115、TRADE_PLAYER 315 |
| DEVELOPMENT | BUY_DEVELOPMENT 151 |
| ROAD_TITLE | BUILD_ROAD 85 |
| KNIGHT_TITLE | BUY_DEVELOPMENT 106 |
| None | BUILD_ROAD 9、TRADE_BANK 50、TRADE_PLAYER 464 |

### Mixed

| Goal | Execution内訳 |
|---|---|
| SETTLEMENT | BUILD_ROAD 467、BUILD_SETTLEMENT 273、END_TURN 1,349、TRADE_BANK 215、TRADE_PLAYER 513 |
| CITY | BUILD_CITY 161、END_TURN 314、TRADE_BANK 104、TRADE_PLAYER 276 |
| DEVELOPMENT | BUY_DEVELOPMENT 126 |
| ROAD_TITLE | BUILD_ROAD 71 |
| KNIGHT_TITLE | BUY_DEVELOPMENT 95 |
| None | BUILD_ROAD 6、TRADE_BANK 54、TRADE_PLAYER 332 |

`SETTLEMENT → BUILD_ROAD / TRADE / END_TURN`と`CITY → TRADE / END_TURN`が
全profileで観測され、objectiveとexecutionを分離した設計は維持できている。

## None分析

| profile | None率 | PlayerTrade未分類 | BankTrade未分類 | BuildRoad未分類 |
|---|---:|---:|---:|---:|
| Rule | 6.82% | 228 / 941 (24.23%) | 60 / 335 (17.91%) | 10 / 640 (1.56%) |
| Champion | 10.12% | 464 / 1,641 (28.28%) | 50 / 479 (10.44%) | 9 / 587 (1.53%) |
| Mixed | 9.00% | 332 / 1,121 (29.62%) | 54 / 373 (14.48%) | 6 / 544 (1.10%) |

全None 1,213件の内訳はPlayerTrade 1,024件、BankTrade 164件、BuildRoad 25件。
BuildRoad未分類率は全profileで1.1〜1.6%と安定して低い。強い相手で増えたNoneは
主に余剰資源交渉、将来目的の交換、即時建設に直結しない提案である。

したがって、Noneを新しいGoalへ推測分類するのではなく、将来
`choose_proactive_trade()`が選択時に持っている建設目標・strategic trade理由を
診断情報として返す方が安全である。

## Goal × turn band

| profile / band | SETTLEMENT | CITY | DEVELOPMENT | ROAD_TITLE | KNIGHT_TITLE | None |
|---|---:|---:|---:|---:|---:|---:|
| Rule early | 89.8% | 0.7% | 2.4% | 0.0% | 0.7% | 6.4% |
| Rule mid | 60.0% | 23.5% | 2.6% | 2.9% | 2.6% | 8.4% |
| Rule late | 28.7% | 55.9% | 1.7% | 5.2% | 3.0% | 5.4% |
| Champion early | 84.0% | 1.3% | 3.3% | 0.3% | 0.9% | 10.2% |
| Champion mid | 74.8% | 7.4% | 3.2% | 0.8% | 2.4% | 11.5% |
| Champion late | 44.6% | 38.1% | 2.4% | 3.5% | 2.6% | 8.9% |
| Mixed early | 87.7% | 1.5% | 3.0% | 0.3% | 1.4% | 6.2% |
| Mixed mid | 68.9% | 12.0% | 3.3% | 1.4% | 2.6% | 11.9% |
| Mixed late | 37.6% | 46.3% | 2.3% | 3.2% | 2.4% | 8.1% |

Rule戦はlateでCITYへ移りやすい。Champion戦はlateでもSETTLEMENTが44.6%残る。
強い相手で都市化以前に競争が進むことと、開拓待ち・交渉が長引くことを示す。

## Goal × reason code

主要reasonの件数をprofile順（Rule / Champion / Mixed）で示す。

| Goal | reason | Rule | Champion | Mixed |
|---|---|---:|---:|---:|
| SETTLEMENT | RESOURCE_WAIT / END_TURN | 1,203 | 1,493 | 1,349 |
| SETTLEMENT | ROAD_STEP | 504 | 493 | 467 |
| SETTLEMENT | IMMEDIATE | 334 | 257 | 273 |
| SETTLEMENT | PLAYER_TRADE | 313 | 862 | 513 |
| SETTLEMENT | BANK_TRADE | 163 | 314 | 215 |
| CITY | RESOURCE_WAIT / END_TURN | 488 | 330 | 314 |
| CITY | PLAYER_TRADE | 400 | 315 | 276 |
| CITY | IMMEDIATE / HIGH_PRODUCTION_GAIN | 238 | 126 | 161 |
| CITY | BANK_TRADE | 112 | 115 | 104 |
| DEVELOPMENT | PURCHASE | 98 | 151 | 126 |
| DEVELOPMENT | ROBBER_RELEASE | 65 | 104 | 80 |
| ROAD_TITLE | OPPORTUNITY | 93 | 64 | 60 |
| ROAD_TITLE | DEFENSE | 23 | 18 | 8 |
| KNIGHT_TITLE | OPPORTUNITY / PURCHASE | 95 | 106 | 95 |
| None | GOAL_ACTION_AMBIGUOUS | 298 | 523 | 392 |
| None | PLAYER_TRADE | 228 | 464 | 332 |
| None | BANK_TRADE | 60 | 50 | 54 |

全reason matrixはローカルartifactの`profile_stability.json`に保存した。

## Goal × 勝敗

| profile / outcome | SETTLEMENT | CITY | DEVELOPMENT | ROAD_TITLE | KNIGHT_TITLE | None |
|---|---:|---:|---:|---:|---:|---:|
| Rule winner | 56.9% | 28.9% | 2.3% | 3.2% | 2.1% | 6.7% |
| Rule non-winner | 61.3% | 24.9% | 2.1% | 1.3% | 2.9% | 7.6% |
| Champion winner | 61.3% | 25.3% | 3.3% | 2.3% | 1.9% | 6.0% |
| Champion non-winner | 68.0% | 14.0% | 2.8% | 1.4% | 2.1% | 11.7% |
| Mixed winner | 58.3% | 26.4% | 3.1% | 2.3% | 1.9% | 8.0% |
| Mixed non-winner | 68.4% | 15.6% | 2.7% | 1.2% | 2.3% | 9.6% |

強い相手へのnon-winner局面でCITYが減り、SETTLEMENTとNoneが増える。この差は
対戦profileだけでなく、最終結果とゲーム長にも依存するため、Decision単位の
ランダムsplitは禁止し、game/seed単位で分割する。

## seat偏り

各profileのゲーム数はseatごとに25局で均等。Decision数と主要Goal率は次の通り。

| profile / seat | Decision | SETTLEMENT | CITY | None |
|---|---:|---:|---:|---:|
| Rule 1 | 1,037 | 59.0% | 28.3% | 6.3% |
| Rule 2 | 1,337 | 57.5% | 29.3% | 6.3% |
| Rule 3 | 1,142 | 55.3% | 30.7% | 6.1% |
| Rule 4 | 856 | 59.0% | 23.6% | 9.2% |
| Champion 1 | 1,645 | 67.7% | 15.9% | 10.0% |
| Champion 2 | 1,513 | 66.0% | 20.0% | 8.4% |
| Champion 3 | 1,061 | 61.3% | 20.8% | 10.0% |
| Champion 4 | 951 | 69.1% | 10.6% | 13.1% |
| Mixed 1 | 1,160 | 63.4% | 18.6% | 10.3% |
| Mixed 2 | 921 | 67.5% | 16.6% | 9.3% |
| Mixed 3 | 871 | 64.6% | 21.8% | 6.1% |
| Mixed 4 | 1,404 | 63.9% | 21.1% | 9.5% |

ゲーム割当は均等でも、ゲーム長と勝敗によりDecision数は均等にならない。
Champion seat 4ではCITYが少なくNoneが多い。25局/seatなので断定はしないが、
将来のtrain/validation/testはprofile×seatを層化してgame単位で分割する。

## profile間の分布差

| Goal | 最大割合差 |
|---|---:|
| SETTLEMENT | 8.56 percentage points |
| CITY | 11.18 points |
| DEVELOPMENT | 0.68 points |
| ROAD_TITLE | 1.25 points |
| KNIGHT_TITLE | 0.13 points |
| HOLD | 0.00 points |
| None | 3.30 points |

| profile pair | Total Variation距離 | 最大Goal差 |
|---|---:|---:|
| Champion vs Mixed | 0.0262 | 2.49 points |
| Champion vs Rule | 0.1254 | 11.18 points |
| Mixed vs Rule | 0.0994 | 8.69 points |

ChampionとMixedは近く、Ruleとの主差はSETTLEMENT/CITYの到達段階である。
希少GoalとNoneは大きく崩れていないため、taxonomy自体は複数対戦分布でも
Teacher Datasetとして利用可能と判断する。ただしRule戦だけで学習するのは不十分である。

## Observation v2との情報差

### v2に含まれる情報

- 自分の資源内訳・発展カード・実得点
- 相手の資源総数、発展カード総数、公開得点、建設数、使用済み発展カード
- タイル、盗賊、交差点・道路所有、交差点別生産pips、港
- 銀行資源、発展山札残数、手番・phase、trade_count
- 称号保持者、プレイ済み騎士、free road、使用済み発展フラグ

### Plannerが使うがv2ベクトルにない情報

1. `game.resource_events`から復元するベイズ相手手札分布。
   `choose_proactive_trade()`は相手が欲しい資源を持つ確率で提案相手を選ぶ。
   v4のbelief contextには要約があるが、今回のv2にはない。
2. `game.activity_log`内の当該手番の交渉履歴。
   誰へ1:1を提示したか、誰に断られたか、2:1へ値上げ済みかを使う。
   v2には`trade_count`しかなく、相手・offer・結果の履歴はない。
3. `player.new_development_cards`の区別。
   Observation objectは保持するが、v2ベクトルは通常のdevelopment card総数と
   分離してencodeしていない。発展購入Plannerは同一手番の再購入抑制に使う。

### 明示特徴ではないがv2から再計算可能な情報

- 開拓候補、必要道路、候補vertex/edge、site value、plan value
- 開拓・都市・発展の推定待ち時間と資源不足
- 最長交易路・最大騎士力までの距離
- 盗賊で塞がれた生産pips
- 発展カードの公開情報ベース事前確率
- legal action mask

これらはPlanner内部でのみ明示計算されるが、固定盤面トポロジー、公開盤面、
自分の手札、銀行在庫から決定的に再計算できるため、情報漏洩ではない。ただし
Macro Headに生のv2だけを渡す場合は、同じ計算をネットワークが近似する学習負荷がある。

Plannerの固定configと現行PPO logitsもv2の一部ではないが、全Datasetで固定された
Teacher定数、またはv2と固定モデルから決定的に得られる値であり、状態情報の欠落とは
区別する。

## parity

収集seedと完全分離したseedで各profile 5局を比較した。

| profile | games | 全プレイヤーAction数 | Action列 | winner | score | rank | 最終GameState |
|---|---:|---:|---|---|---|---|---|
| Rule | 5 | 2,308 | 一致 | 一致 | 一致 | 一致 | 一致 |
| Champion | 5 | 2,081 | 一致 | 一致 | 一致 | 一致 | 一致 |
| Mixed | 5 | 2,041 | 一致 | 一致 | 一致 | 一致 | 一致 |
| 合計 | 15 | 6,430 | 完全一致 | 完全一致 | 完全一致 | 完全一致 | 完全一致 |

parity seed:

- Rule: 2380001〜2380005
- Champion: 2381001〜2381005
- Mixed: 2382001〜2382005

## Distillation前の提案

1. Objective taxonomyは変更しない。TRADEをGoalへ昇格せず、objective / execution分離を維持する。
2. HOLDは初回の学習クラスから外すか常時maskする。0件を埋めるための再分類はしない。
3. Distillation前に、交渉選択元から`trade_objective`と構造化reasonを返す診断だけを追加し、
   1,213件のNone、特にPlayerTradeをどこまで安全にラベル化できるか確認する。
4. v2を直ちに置換しない。まず分類済みGoalに限定したgame-level validationで上限を見る。
   交渉局面の誤りが支配的なら、既存v4 belief contextに交渉試行履歴と
   new-development flagを加える案を次段階で検討する。
5. train / validation / testはgame_idまたはseed単位で分割し、profile×seatを層化する。
6. rare classにはclass weightまたはgame-level balanced samplerを使うが、同一gameの
   Decisionを複数splitへ分けない。

以上から、Step 3へ直接進むより、Plannerの行動を変えない小さな診断拡張として
「交渉目的のsource-level記録」を先に行う方が、現在の300局を正しく解釈できる。
