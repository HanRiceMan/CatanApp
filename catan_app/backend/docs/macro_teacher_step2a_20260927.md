# Macro Goal Step 2A: Teacher Dataset収集とtaxonomy診断

## 実施範囲

現行Champion ppo_gnn_board_65k_s03_exp_v003 + SettlementPlanningAgent の
Actionは変更せず、Action直前の既存Observationと
PlannerDecisionReportを収集する仕組みを追加した。

この段階では、Macro Head、Distillation、Shadow Policy、DAgger、
League PPO、377 Action logitsとの接続は実装していない。

## Dataset形式

CLI:

    python -m app.rl.collect_macro_teacher_dataset --experiment-id <ID> --games 200 --seed-start 2330001 --parity-games 100 --parity-seed-start 2331001

出力:

- decisions.jsonl
  - 1行が1回のActionフェーズのMacro Decision。
  - PlannerDecisionReport、seed、game ID、seat、Observation行番号、
    勝者、Champion最終VP・順位を含む。
- observations.npz
  - observationsというfloat32配列。
  - JSONLのobservation_indexと対応する。
- games.jsonl
  - game単位の勝者、全員の最終得点、Champion順位、手数。
- taxonomy_summary.json
  - Goal分布、Goal×Execution、None内訳、Reason、進行帯、勝敗別分布、
    score semantics別統計。
- definition.json
  - schema、モデル、Planner設定、seed、ファイル対応。
- parity_summary.json
  - --parity-games指定時の非介入検証結果。

experiments/はgitignore対象なので、生Datasetはローカルartifactとして扱う。

## 8局smoke結果

- seed: 2330001〜2330008
- Champion席: 1〜4をローテーション
- 対戦相手: ルールAI 3人
- ゲーム数: 8
- Macro Decision数: 361
- Observation shape: (361, 1667)、version v2
- Champion勝利: 4/8
- 平均VP: 9.0
- 平均順位: 1.625

### Objective Goal分布

| Goal | 件数 | 割合 |
|---|---:|---:|
| SETTLEMENT | 217 | 60.11% |
| CITY | 94 | 26.04% |
| None | 27 | 7.48% |
| ROAD_TITLE | 11 | 3.05% |
| DEVELOPMENT | 6 | 1.66% |
| KNIGHT_TITLE | 6 | 1.66% |
| HOLD | 0 | 0.00% |

### 主なObjective × Execution

| Objective | Execution | 件数 |
|---|---|---:|
| SETTLEMENT | END_TURN | 122 |
| SETTLEMENT | BUILD_ROAD | 43 |
| SETTLEMENT | BUILD_SETTLEMENT | 26 |
| SETTLEMENT | TRADE_PLAYER | 23 |
| CITY | END_TURN | 38 |
| CITY | TRADE_PLAYER | 31 |
| CITY | BUILD_CITY | 18 |
| ROAD_TITLE | BUILD_ROAD | 11 |
| DEVELOPMENT | BUY_DEVELOPMENT | 6 |
| KNIGHT_TITLE | BUY_DEVELOPMENT | 6 |

SETTLEMENT → BUILD_ROADが43件あり、目的と実行手段の分離は意図どおり
データに現れた。

### objective_goal=None

- 27/361 = 7.48%
- ProposeTradeAction: 18
- BankTradeAction: 6
- BuildRoadAction: 3
- 全BuildRoad 57件中Noneは3件 = 5.26%
- early: 7、mid: 9、late: 11

未分類の中心は道路ではなく、直近の建設不足を直接埋めない余剰資源交渉だった。
例として、開拓地が既に購入可能な状態で将来用の小麦を求める交渉や、
都市候補しかない状態で羊を得る銀行交換がある。現行Plannerから最終目的を
一意に復元できないため、無理にCITYやDEVELOPMENTへ割り当てずNoneを維持した。

### 主なReason Code

- SETTLEMENT_RESOURCE_WAIT: 122
- END_TURN_RESOURCE_WAIT（SETTLEMENT）: 122
- SETTLEMENT_ROAD_STEP: 43
- SETTLEMENT_IMMEDIATE: 26
- PLAYER_TRADE_REACHABLE:
  - CITY 31
  - SETTLEMENT 23
  - None 18
- CITY_RESOURCE_WAIT: 38
- CITY_IMMEDIATE: 18
- ROAD_TITLE_OPPORTUNITY: 8
- ROAD_TITLE_DEFENSE: 3
- DEVELOPMENT_PURCHASE:
  - DEVELOPMENT 6
  - KNIGHT_TITLE 6
- GOAL_ACTION_AMBIGUOUS: 27

### goal_score_semantics別統計

異なる行同士の統計だけを計算し、semantics間の比較・margin生成はしていない。

| semantics | 件数 | min | max | mean | std |
|---|---:|---:|---:|---:|---:|
| expansion_plan.plan_value | 323 | -0.325 | 13.808 | 5.967 | 3.215 |
| best_city_vertex_production_value | 361 | 4.000 | 13.000 | 10.285 | 2.468 |
| expected_development_card_value | 361 | 0.924 | 2.574 | 1.478 | 0.488 |
| current_progress_ratio_to_title_target | 157 | 0.600 | 1.000 | 0.743 | 0.162 |
| known_knight_progress_ratio_to_title_target | 144 | 0.333 | 1.000 | 0.426 | 0.154 |
| constant_diagnostic_baseline | 361 | 0.000 | 0.000 | 0.000 | 0.000 |

## 100局Action parity

- collection seedと重複しないseed 2331001〜2331100
- 100局
- 全プレイヤー合計49,734 Action
- 全Action列: 完全一致
- winner: 完全一致
- final score: 完全一致
- final rank: 完全一致
- 4プレイヤー視点の最終GameState: 完全一致

## taxonomy上の懸念

1. HOLDが0件であり、現行のGoal復元では実質的に未使用クラスになっている。
   Distillation時に6クラス固定で扱う前に、明示的なHOLD判断をPlannerが持つか、
   学習対象から外すか決める必要がある。
2. DEVELOPMENTとKNIGHT_TITLEは各6件、ROAD_TITLEも11件しかなく、
   8局smokeだけでは希少クラスの妥当性を判断できない。
3. 最大騎士力理由と停滞理由が同時成立する発展購入は、現在
   KNIGHT_TITLEを優先している。Planner自体に単一の目的値がないため、
   これは今後重点確認が必要な境界である。
4. Noneの大半は戦略的余剰交渉である。None率をゼロにするための推測分類は
   行わず、必要ならPlannerの交渉判断から明示的な目的を返す診断情報を追加する。
5. Goal間で比較可能なTeacher valueがないため、margin、second goal、
   teacher regretは引き続き未定義である。

## Step 3前の推奨

まず100〜300局のTeacher Datasetを収集し、希少Goal、HOLD、交渉Noneの分布が
安定するか確認する。クラス不均衡を確認する前にDistillationへ進まない。

大量収集後もHOLDが0件なら、HOLDを学習クラスに残す根拠を再検討する。
交渉Noneが大きく増える場合は、Action後の推定を強めるのではなく、
trade_strategyが選択時に持つ目的を診断レポートへ渡せるようにする。
