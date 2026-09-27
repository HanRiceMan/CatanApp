# Macro Goal Step 3A: Distillation基盤と独立評価

## 結論

Step 3Aとして、現行Championを凍結したままObjective Goalだけを予測する
Macro Headの学習・保存・独立評価基盤を実装した。Macro logitsは377 Action logitsへ
接続しておらず、Championの対局行動には影響しない。

技術的な学習基盤とgame単位splitは正常に動作したが、300局Datasetによる5 Goal
分類は合格水準へ達しなかった。最終構成の3学習seedで、完全未使用45 gameに対する
Balanced Accuracyは39.7〜43.1%だった。特にDEVELOPMENTとKNIGHT_TITLEの
generalizationが弱い。

したがって、このMacro HeadをStep 3BのShadow評価または実行Actionへ接続しない。
現行Championは変更せず、まずTeacher game数とrare Goalを増やす必要がある。

## 実装範囲

- `macro_goal_policy.py`
  - Macro Goal label定義
  - Step 2A Dataset loader
  - profile×seatを層化したgame単位split
  - 凍結Champion表現の抽出
  - Macro Head
  - Accuracy、Balanced Accuracy、Goal別Precision/Recall/F1、confusion matrix
  - artifact保存・再読込
- `train_macro_goal_distillation.py`
  - Headだけを学習するCLI
  - class imbalance補正
  - validation Balanced Accuracyによるearly stopping
  - profile別・seat別test評価
  - Champion Actorのstate digest・logit parity検証
- `test_rl_macro_goal_policy.py`
  - game leakage防止
  - metricのGoal順
  - artifact再読込
  - feature抽出時のActor state・train/eval mode不変

未実装:

- Shadow Policy / Shadow evaluation
- Macro Goalから377 Actionへの接続
- DAgger / League PPO
- Observation変更
- Champion、Planner、Action Spaceの変更

## 学習対象

学習対象はsource-levelで根拠のある次の5 Goalだけとした。

- SETTLEMENT
- CITY
- DEVELOPMENT
- ROAD_TITLE
- KNIGHT_TITLE

`objective_goal=None` 1,213件はunknown/fallbackとして損失から除外した。
HOLDはDatasetに0件のため学習classへ含めていない。TRADEもObjective Goalへ
追加していない。

## Dataset split

- source: `macro_teacher_step2a3_300_s01_20260928`
- 全game: 300
- 明示Goal Decision: 12,685
- 除外None: 1,213
- Observation: v2、1667次元
- split seed: 20262901

| split | games | Decisions |
|---|---:|---:|
| train | 210 | 8,826 |
| validation | 45 | 1,881 |
| test | 45 | 1,978 |

同一gameは必ず一つのsplitだけに入り、profile×seatごとに分割した。
Decision単位のrandom splitは使用していない。

## 最終Macro Head入力

Observation仕様は変更せず、既存Championが同じObservationから計算した凍結表現を
利用した。

| 入力 | 次元 |
|---|---:|
| Observation v2 | 1,667 |
| GNN vertex/edge/tile pooling | 448 |
| 377 Action候補＋15 Family logits | 392 |
| Critic latent | 128 |
| 合計 | 2,635 |

Plannerは既存PPOが選んだ具体Actionを入口にして判断するため、Action/Family logitsを
含めた。ただしlogitsはObservationから既存Championが生成する値で、新しい秘匿情報や
Planner診断値ではない。

Headは`2635 → 256 → 128 → 5 Goal`のMLP。ChampionのGNN、Actor、Criticは
完全凍結した。

## 段階比較

### 1. 凍結GNN要約＋平方根class weight

- test Accuracy: 76.90%
- test Balanced Accuracy: 39.30%
- DEVELOPMENT recall: 0%
- KNIGHT_TITLE recall: 0%

多数classのSETTLEMENT/CITYを予測するだけで全体Accuracyが高く見えるため、
採用不可とした。

### 2. Observation v2併記＋完全逆頻度weight

- test Accuracy: 71.94%
- test Balanced Accuracy: 38.18%

train Balanced Accuracyは81.08%だったが、testではrare Goalが再現せず、
game固有状態へのoverfitが明確だった。

### 3. Observation＋GNN＋既存Actor logits＋Critic

splitを固定し、学習初期値だけ3 seedで比較した。

| training seed | Accuracy | Balanced Accuracy | SETTLEMENT recall | CITY recall | DEV recall | ROAD_TITLE recall | KNIGHT_TITLE recall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20262901 | 64.26% | 42.38% | 70.10% | 61.72% | 6.78% | 55.81% | 17.46% |
| 20262902 | 52.53% | 39.71% | 54.30% | 56.34% | 15.25% | 48.84% | 23.81% |
| 20262903 | 62.39% | 43.13% | 63.87% | 71.61% | 18.64% | 48.84% | 12.70% |
| 3 seed平均 | 59.72% | 41.74% | 62.76% | 63.22% | 13.56% | 51.16% | 17.99% |

乱数seedを変えてもrare Goalの低い再現率が続いた。単一seedの失敗ではない。

## なぜ全体Accuracyだけでは採用できないか

test 1,978件の内訳は次のとおり。

- SETTLEMENT: 1,348
- CITY: 465
- DEVELOPMENT: 59
- ROAD_TITLE: 43
- KNIGHT_TITLE: 63

SETTLEMENTだけで68.1%を占める。したがって全体Accuracyが70%前後でも、
DEVELOPMENTや称号判断をほぼ学べていない可能性がある。今回は最初から
Balanced AccuracyとGoal別Recallを合格判定に使ったため、この問題を検出できた。

## 原因の所見

### rare classの独立game数不足

testで各rare Goalは43〜63 Decisionしかない。Decision数以上に重要なのは、
同一game内で似た状態が反復されることである。game単位splitをすると、trainで覚えた
rare局面が別gameへ一般化していない。

### Teacher GoalがPlannerのAction経路に依存する

現行Plannerには全Goalを同一尺度で比較する統一scoreがない。今回のGoalは、
Plannerが最終的に選択したActionと分岐経路から安全に診断したhard labelである。
そのためDEVELOPMENT/KNIGHT_TITLE/ROAD_TITLEは「持続的な戦略目標」というより、
特定Actionを実行した局面に集中する。

既存Actor logitsを加えるとROAD_TITLE recallは改善したが、完全なGoal valueではない。
偽のmarginやregretを作らなかったStep 1の判断は維持する。

### Teacherにはsoft targetがない

`scores_comparable=False`のため、Goal間の順位・margin・regretを教師信号へ使えない。
今回はhard labelだけで学習しており、境界局面を弱く重み付けすることもできない。

## Champion parity

全実験で次を確認した。

- Champion Actor state digestが学習前後で完全一致
- 固定64 ObservationのActor logitsがbitwise一致
- Headは別artifactとして保存
- `runtime_connected=false`
- `action_logits_connected=false`

Macro Head学習による既存AIのAction変更はない。

## 次の方針

Step 3Bへ進む前に、Step 3A-2としてTeacher Datasetを増やす。

推奨条件:

- Rule / Champion / Mixedを維持
- profile×seatを均等化
- rare Goalを各splitで最低200〜300 Decision、可能なら1,000件前後確保
- 既存300 gameを捨てず、game_id/seed単位で追加Datasetと結合
- 追加収集後も同じ固定test gameを保持し、testへ再学習データを混ぜない

単にDecisionを複製したoversamplingは、同一gameの暗記を強めるため採用しない。
Teacher marginを作るための異種score正規化も行わない。

Datasetを増やしてもrare Goalが一般化しない場合は、taxonomyを無理に統合せず、
Planner側に持続的な`active_objective_goal`が本当に存在するかを再調査する。
現段階ではMacro HeadをChampionの代替・Shadow候補として扱わない。
