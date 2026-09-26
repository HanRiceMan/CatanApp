# 階層候補PPO：3 seed比較結果

## 目的

従来の候補共有PPOでは、道路のように候補数が多い行動種別の確率が複数候補へ分散し、`手番終了`などの単一Actionに最大確率で負ける学習seedがあった。

推論時だけ確率を補正する方式はholdout盤面で一貫しなかったため、学習時から次の確率分解を扱う実験用方策を追加した。

```text
Observation
  ├─ 行動種別head: P(family | state)
  └─ 候補head:     P(action | family, state)

P(action | state)
  = P(family | state) × P(action | family, state)
```

PPOが使うlog probabilityとentropyはこの結合分布から計算し、Action Maskは個別Actionと、そのActionを1つ以上持つ行動種別の両方へ適用する。推論時の後付け補正ではない。

## 実験条件

- 比較: 従来`candidate` 対 `hierarchical_candidate`
- 学習seed: `20260927`、`20260928`、`20260929`
- 学習量: 指定10,000 steps、PPO rollout境界による実行値10,240 steps
- Observation: v2（1,667次元）
- Action Space: v1（377 Action）
- 相手: HeuristicAgent × 3
- 初期配置: 両方式ともHeuristicAgentへ固定
- 評価: 未学習盤面seed `60001`〜`60100`、各モデル100対局
- 学習対象Player ID: 評価時にローテーション
- Reward、PPOパラメータ、盤面、相手は方式間で同一

## 結果

| 学習seed | 方策 | 勝率 | 平均VP | 平均順位 | 有償道路 | 追加開拓地 | 銀行・港交易 | 最長路 | 最大騎士 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20260927 | candidate | 0% | 3.95 | 3.57 | 0.00 | 0.15 | 13.15 | 2% | 0% |
| 20260927 | hierarchical | 6% | 5.35 | 2.96 | 0.00 | 0.15 | 7.85 | 1% | 51% |
| 20260928 | candidate | 10% | 6.13 | 2.90 | 7.26 | 0.31 | 12.52 | 52% | 51% |
| 20260928 | hierarchical | 19% | 6.09 | 2.79 | 7.65 | 0.90 | 12.21 | 56% | 37% |
| 20260929 | candidate | 3% | 4.49 | 3.30 | 0.00 | 0.10 | 15.03 | 0% | 27% |
| 20260929 | hierarchical | 22% | 6.71 | 2.54 | 6.40 | 0.94 | 13.26 | 41% | 53% |
| **3 seed平均** | **candidate** | **4.3%** | **4.86** | **3.26** | **2.42** | **0.19** | **13.57** | **18%** | **26%** |
| **3 seed平均** | **hierarchical** | **15.7%** | **6.05** | **2.76** | **4.68** | **0.66** | **11.11** | **33%** | **47%** |

同じ評価盤面を使った対応比較では、階層方策の勝率差はseed順に `+6`、`+9`、`+19`ポイントだった。95% bootstrap区間はそれぞれ `[+2,+11]`、`[+1,+17]`、`[+12,+27]`ポイントで、3 seedすべて階層方策側が上回った。

平均VP差は `+1.40`、`-0.04`、`+2.22`。seed `20260928`は勝率が上がった一方で平均VP差の区間が0をまたいだため、すべての指標が常に改善したとは扱わない。

全600評価対局で非法Action 0、打ち切り0だった。

## 判断

階層方策は、従来方策で2/3 seedに発生した「有償道路を一度も選ばない」崩壊を1 seedで解消し、もう1 seedでは道路以外の最大騎士戦略で平均VPを改善した。良好だったseedでも道路選択を維持しつつ追加開拓地を増やした。結果として、勝率・平均VP・順位・建設行動の3 seed平均がすべて改善した。

したがって、**次の学習実験では`hierarchical_candidate`を主候補として採用する**。ただし、seed `20260927`では道路0が残っており、10k steps・3 seedだけで安定化完了とは判断しない。既存UIのPPO既定モデルや従来モデルはまだ置き換えない。

## 次の実験

次は変更を増やさず、階層方策だけを複数seedで50k stepsまで延長し、10k/25k/50k checkpointを同じholdout盤面で比較する。

見る項目は次の通り。

1. seed `20260927`の道路0が学習継続で解消するか。
2. 10k時点の勝率15.7%・平均VP6.05が維持または改善するか。
3. entropy低下と行動種別確率を記録し、道路・建築が再び消えないか。
4. 50kで安定した場合だけ、別holdout盤面とRandom/Heuristic混合相手でgeneralizationを確認する。

GNN、Self-play、Reward変更、Opponent curriculumは同時に導入しない。階層化単独の効果と学習継続時の安定性を先に確定する。

## 実装・保存物・検証

- `app/rl/action_families.py`: 全Actionを15種類へ排他的に分類。
- `app/rl/hierarchical_distribution.py`: 行動種別と種別内候補の結合分布。
- `app/rl/candidate_policy.py`: `HierarchicalCandidateMaskablePolicy`。
- `app/rl/train.py`: `hierarchical_candidate`学習・再読込対応。
- `app/rl/compare_hierarchical_candidate.py`: 条件固定の複数seed比較ランナー。
- `tests/test_rl_candidate_policy.py`: 分布正規化、mask、決定的選択、短時間学習を検証。
- `experiments/hierarchical_candidate_10k_s20260927/`
- `experiments/hierarchical_candidate_10k_s20260928_29/`
- バックエンド全99件の`unittest`が成功。FastAPI依存由来の非致命的なdeprecation warningが1件。

