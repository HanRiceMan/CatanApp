# 候補共有型PPO：接続と1万step予備実験

## 変更と互換性

`app/rl/candidate_policy.py` に実験用 `CandidateMaskablePolicy` を追加した。Observation v2とAction Space v1（377 ID）を維持し、交差点54、辺72、タイル19の各候補を種類ごとの共有重みで採点する。辺の両端とタイル周囲の交差点情報を固定トポロジーから集約するが、複数段のメッセージパッシングをするGNNではない。通常の非空間Actionは別Head、価値推定は全Observationを使うCritic。既存の合法手マスクはSB3-contribのMaskablePPO処理をそのまま通す。

`TrainConfig.policy_architecture` の既定値は従来通り `mlp`。候補方式は明示的に `candidate` とObservation `v2` を選んだ場合のみ使う。教師あり初期配置ウォームスタートも `pretrain_initial_settlement=True` の明示指定のみ。既存モデル・通常UIの既定動作を変更していない。実験モデルは `experiments/` 配下に保存し、通常のUIモデル一覧に自動追加していない。

## 比較条件

同じ学習seed `20260923`、1万step、Heuristic×3相手、Observation v2、Action Space v1、報酬、PPOパラメータで以下を比較した。教師ありウォームスタートは別途600盤面の教師データを使用するため、純粋に同じ学習データ量の比較ではない。

1. 固定ID型MLP-PPO
2. 候補共有型PPO（事前学習なし）
3. 候補共有型PPO（初期開拓地だけ教師あり事前学習）

未使用seed `40001`〜`40100` でPlayer IDを回し、Heuristic×3とHeuristic×1+Random×2の2条件を評価した。各モデル200対局、計600対局で非法Action・打ち切り0。これは**1学習seed・短い1万stepの予備診断**であり、300k/1Mモデルとの優劣を断定しない。

## 結果

| 方策 | 初期2軒pips | 1軒目の最多ID集中率 | Heuristic×3 勝率/平均VP | Heuristic×1+Random×2 勝率/平均VP |
| --- | ---: | ---: | ---: | ---: |
| 固定ID型MLP | 12.28 | 91% | 0% / 3.96 | 1% / 4.19 |
| 候補共有・事前学習なし | 11.48 | 8% | 0% / 2.16 | 0% / 2.15 |
| 候補共有・初期配置を事前学習 | 20.59 | 9% | 2% / 5.27 | 0% / 5.82 |

事前学習なしの候補共有型は固定IDへの集中を解消したが、木材に極端に偏った。初期資源種類数は平均1.41、木材の生産pipsは10.62、レンガのない対局は87%。**偏りのないID分布だけでは、良い戦略を学んだことにならない。**

ウォームスタートは未使用盤面で教師一致率64.7%。その後のPPO 1万stepでも初期配置は平均20.59 pips、最良合法地点からのpips差0を維持した。事前学習なしとの差は、同じ盤面・席で対応付けた平均VPでHeuristic×3が+3.11（bootstrap 95%区間+2.73〜+3.49）、混合相手が+3.67（+3.25〜+4.08）。ただしウォームスタートには追加の教師データがあるため、改善を候補型構造だけの効果とは解釈しない。

勝率は依然低い。ウォームスタート済み方策はHeuristic×3相手でも通常ターンの新規開拓地が平均0.26軒/試合にとどまり、発展カード購入は平均8.08枚/試合。都市建設の機会があっても手番終了する記録がある。良い初期配置を使えても、**通常ターンの意思決定が次の主な課題**と考えられる。1万stepという短さの影響もあり、根本原因は未確定。

## 次の実験判断

GNNはまだ導入しない。次は初期配置を既存Heuristicへ固定するアブレーションで、候補共有型の通常ターンだけを比較する。その結果で通常ターンの学習信号・行動Headの問題を切り分ける。必要なら通常ターンの教師ありウォームスタートを限定的に検討するが、報酬や対戦相手を同時に変更しない。複数学習seedとより長いstepで再現するまでは新方策をUI既定モデルにしない。

## 実装・保存物・テスト

- 実装：`app/rl/candidate_policy.py`、`app/rl/pretrain_candidate_setup.py`、`app/rl/train.py`、実験ランナー2本
- 実験条件・各対局・モデル：`experiments/candidate_ab_10k_s20260923/`
- 教師あり診断の生産量差の表示計算にあった誤りを修正した。学習処理と対局結果には影響していない。
- バックエンド全95件の `unittest` が成功。FastAPIの依存から非致命的なdeprecation warningが1件出る。

参照：SB3-contribの[MaskablePPO公式仕様](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)、[MaskableActorCriticPolicy実装](https://sb3-contrib.readthedocs.io/en/master/_modules/sb3_contrib/common/maskable/policies.html)。
