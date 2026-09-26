# CatanApp 強化学習実装と施策判断の流れ

## 1. この取り組みの目的

CatanAppでは、単にPPOを動かすだけではなく、同じゲームエンジン上で次のAgentを交換・比較できることを目標にした。

- Human
- RandomAgent
- HeuristicAgent
- PPOAgent
- 将来のMCTSAgent、Self-play Agent、Multi-Agent RL Agent

そのため、最初からニューラルネットを追加するのではなく、次の境界を先に作った。

```text
GameState
   ↓ get_observation(player_id)
Observation
   ↓ encode
固定長ベクトル
   ↓ Agent
Action
   ↓ apply_action()
GameState'
```

重要なのは、AIがゲーム状態を直接変更しないことである。Human、ルールAI、PPOのすべてが共通の`Action -> apply_action()`を通るため、学習用に別のカタンルールを持たずに済む。

## 2. 強化学習前のリファクタリング

### 実施内容

- ゲーム中の操作を型付きActionとして定義
- `apply_action()`をルール処理の共通入口に設定
- HeuristicAgentとRandomAgentを、状態を変更せずActionだけ返す構造へ変更
- `legal_actions()`と既存の合法手判定を共通利用
- ゲーム単位のseed付き乱数源を用意
- APIのURL・JSON・Next.js UIは維持

### なぜ先に行ったか

PPO専用のルール処理を作ると、通常ゲームと学習環境で合法手や勝敗判定がずれる危険がある。また、将来MCTSや別のRL方式を追加するたびにゲームエンジンを変更することになる。先にAgentとActionの境界を作ることで、以後の実験をゲームルールから分離した。

## 3. Phase 1: Observation・Action ID・Action Mask

### 実施内容

Observation v1は相手の非公開情報を除外し、1343次元の固定長ベクトルにした。

- 自分の資源・発展カード・得点・コマ
- 他プレイヤーの公開情報とカード合計枚数
- 19タイル、54交差点、72辺、9港
- phase、手番、ダイス、盗賊、銀行、保留処理

Action Space v1は377個の固定IDとした。合法ActionだけをTrueにするAction Maskも追加した。

### なぜこの設計か

PPOの出力層は固定サイズが必要である。一方、カタンのプレイヤー間交渉や7の任意破棄を全組合せで表すとAction Spaceが非常に大きくなる。最初のPPOでは交渉を学習対象外とし、通常UIの交渉機能は残したまま、RL環境だけを簡略化した。

相手の資源内訳や非公開勝利点を入れなかったのは、実戦で得られない情報を使って強くなることを防ぐためである。

詳細: [rl_phase_1.md](rl_phase_1.md)

## 4. Phase 2〜3: CatanEnvと評価基盤

### 実施内容

`CatanEnv`を追加し、FastAPIとNext.jsを通さずにゲームエンジンを直接実行できるようにした。

- `reset(seed)`
- `step(action_id)`
- `action_masks()`
- `terminated` / `truncated`
- P1を学習者、P2〜P4を固定Agentとして自動進行
- 任意の席を`external`にして人間操作を挟める構造

Reward v1は次の最小構成にした。

- 勝利: +1
- 敗北: -1
- 勝利点変化: `0.05 × ΔVP`

RandomAgentとHeuristicAgentで大量対局を完走させ、非法Actionが0であることを確認した。

### なぜ報酬を増やさなかったか

「道路を作る」「資源を持つ」などへ細かく報酬を与えると、勝利せずに補助報酬だけを集めるReward hackingが起きる可能性がある。最初は勝利と勝利点だけに絞り、問題が報酬なのか表現なのかを切り分けられるようにした。

詳細: [rl_phase_2.md](rl_phase_2.md)、[rl_phase_3.md](rl_phase_3.md)

## 5. Phase 4〜5: MaskablePPOとUI接続

### 実施内容

- Stable-Baselines3 / sb3-contribのMaskablePPOを導入
- 学習・checkpoint・再開・TensorBoardログを実装
- モデルごとにconfig、metadata、Observation版、Action Space版を保存
- `PPOAgent`を通常のAgentとして実装
- Model Registryと`GET /api/ai-models`を追加
- UIで席ごとにHeuristic / Random / PPOを選択可能にした
- ゲーム途中のHuman / AI切り替えを維持

### なぜ学習環境とUIを分けたか

大量学習でNext.jsやHTTP通信を通す必要はない。一方、学習済みモデルを人が観戦・操作できることは重要である。学習は`CatanEnv -> Game Engine`で高速に行い、通常対局では`FastAPI -> PPOAgent -> Action -> apply_action()`だけを追加した。

詳細: [rl_phase_4.md](rl_phase_4.md)、[rl_phase_5.md](rl_phase_5.md)

## 6. 最初の本学習: 300k / 1M PPO

### 結果

Heuristic 3体を相手にした初期モデルは次の傾向だった。

- RandomAgentより明確に強い
- 300k stepsで約5%の勝率
- 1M stepsでも約5%で頭打ち
- 1MモデルはRandomより平均VPが高いが、Heuristicには届かない
- 非法Actionは0

### なぜすぐ3M・10Mへ進まなかったか

300kから1Mへ増やしても明確に改善しなかったため、単純な学習量不足と決めつけられなかった。長時間学習の前に、何を学べておらず、どこで崩れているかを診断する方針へ変更した。

詳細: [rl_training_2026-09-22.md](rl_training_2026-09-22.md)

## 7. 評価指標と診断の強化

勝率だけでは、5点から7点へ成長しても最後に勝ち切れない場合を見落とす。そのため次を追加した。

- 平均VP、平均順位、2位率
- 道路・開拓地・都市・発展カード・交易回数
- 最長交易路・最大騎士力の取得率
- 初期配置pipsと資源種類数
- Player IDと実際の席順を分けた集計
- Random / Heuristicの混合相手
- 方策確率、entropy、合法Actionごとの選択傾向
- 同一seedの対応比較とbootstrap区間

Action Maskは学習時・評価時とも有効で、Illegal Action Countは0だった。Reward、Observationの正規化、PPOパラメータ、seed再現性も監査した。

### 分かったこと

- PPOは基本的な得点行動を学んでいる
- 相手がRandomからHeuristicへ増えるほど滑らかに成績が下がる
- 単に何も学んでいないわけではない
- 主要な弱点の一つが初期配置
- 初期配置をHeuristicへ置き換えると成績が改善
- それでも中盤の開拓地・都市判断はHeuristicより弱い

詳細: [rl_diagnostics_2026-09-22.md](rl_diagnostics_2026-09-22.md)

## 8. Observation v2を試した理由と不採用理由

### 仮説

1343次元のMLPは、各交差点がどの資源をどれだけ生産するかを盤面接続から自力で推論する必要があった。そこで交差点ごとの資源別生産pipsを明示し、Observation v2を1667次元にした。

### 結果

単純な特徴追加だけでは独立seedで改善を再現できず、特定の交差点IDや資源へ集中するモデルも出た。

### 判断

情報不足だけでなく、固定Action IDごとに別の出力重みを持つMLP構造にも問題があると判断した。Observation v2の情報自体は後の候補共有型で利用するが、「入力を増やしただけの通常MLP」は採用しなかった。

詳細: [rl_observation_v2_experiment_2026-09-22.md](rl_observation_v2_experiment_2026-09-22.md)

## 9. 候補共有型Policyへ変更した理由

### 問題

固定ID方式では、交差点0と交差点1が似た局面でも、それぞれ別の出力パラメータになる。そのため盤面の意味ではなく「ID 48を選ぶ」といった暗記が起きやすかった。

### 施策

Tile、Vertex、Edgeの局所特徴を共有Encoderで処理し、同じ採点Headで各候補を評価するCandidate Policyを追加した。

```text
各Vertexの特徴 ─ shared encoder ─ settlement score
各Edgeの特徴   ─ shared encoder ─ road score
各Tileの特徴   ─ shared encoder ─ robber score
```

### 結果

IDへの固定集中は軽減したが、事前学習なしでは木材へ極端に偏った。初期配置をHeuristicでウォームスタートすると初期pipsは改善したが、通常ターンでは発展カード偏重や追加開拓地不足が残った。

### 判断

共有候補方式は方向性として正しいが、「ID偏りがない」だけでは良い戦略ではない。初期配置をHeuristicへ固定し、通常ターンの方策だけを検証するアブレーションへ進んだ。

詳細: [rl_candidate_policy_10k_2026-09-23.md](rl_candidate_policy_10k_2026-09-23.md)

## 10. Action Family問題と階層方策

### 発見した問題

道路は最大72候補、手番終了は1候補である。候補ごとの最大確率だけを比較すると、道路全体では十分な確率質量があっても、単一の終了Actionに負けることがあった。学習seedによって「道路を一度も建てない」崩壊が発生した。

### 一時的な診断

候補確率を道路・建築・交易などのFamily単位で合計してから選ぶ`family_mass`を試した。弱いseedの道路建設は復活したが、holdoutでは改善が安定せず、推論時だけの補正を最終解にはしなかった。

### 本対応

Policy自身が、

1. 何をするかというAction Family
2. そのFamily内のどの候補か

を階層的に学習する`HierarchicalCandidateMaskablePolicy`を実装した。

### 結果

10k steps、3 seed比較では通常Candidateより階層方策が全seedで勝率を改善し、平均では勝率4.3%から15.7%、平均VP4.86から6.05へ上昇した。道路・追加開拓地も改善したため、次の主方式として採用した。

詳細: [rl_candidate_family_selection_2026-09-23.md](rl_candidate_family_selection_2026-09-23.md)、[rl_hierarchical_candidate_2026-09-23.md](rl_hierarchical_candidate_2026-09-23.md)

## 11. 階層PPOを50kまで伸ばした理由と結果

10kの改善が偶然のseed依存でないかを確認するため、3 seedを10k、25k、50k checkpointで追跡した。

### 結果

- 50kでは3 seedすべてで道路と追加開拓地を維持
- 3 seed平均で50kが最良
- 途中の25kで一時的に崩れるseedもあり、学習性能は単調増加ではない
- 未使用盤面600局で、相手がRandomからHeuristicへ強くなるほど自然に勝率が低下
- Heuristic×3に対する3 seed平均勝率は22%
- 直接baseline比較で最良seedはHeuristicの25%に対して34%、平均VPも上回った
- ただし勝率差の95%区間は0をまたぎ、優越を断定していない

### なぜ50k seed 20260928をUIへ公開したか

同一seed・同一席条件の直接比較で最も良く、非法Actionと打ち切りが0だったためである。完全な学習AIではなく、初期配置をHeuristic、通常ターンをPPOが担当するハイブリッドとして`ppo_hierarchical_50k_v001`をModel Registryへ登録した。

UI公開は強さの証明ではなく、人間が行動を観察して次の問題を発見するための入口でもある。

詳細: [rl_hierarchical_50k_learning_curve_2026-09-23.md](rl_hierarchical_50k_learning_curve_2026-09-23.md)

## 12. 50kモデルの行動監査

20局の方策確率と行動を確認した。

- 勝率30%、平均6.50点
- 有料道路5.80、追加開拓地1.20、都市0.80
- 発展カード5.40、銀行・港交易14.10
- 明白な往復交易ループはなし
- 勝利局では開拓地・都市・発展カードが増えていた

この結果から、方策全体が崩壊しているのではなく、勝利経路は形成されていると判断した。一方、Heuristicより都市が少なく、交易と発展カードが多い傾向を次の改善候補として残した。

## 13. 初期配置をPPO側へ戻す実験

標準モデルは初期配置をHeuristicへ委譲している。最終的には初期配置も学習させるため、通常ターンの50k PPOを固定して段階的に試した。

### 13.1 未学習head

未学習の初期配置headへ任せると勝率0%、平均3.33点、初期pips13.16まで崩れた。初期配置を独立して学ぶ必要性を確認した。

### 13.2 headだけの模倣学習

初期開拓地・道路HeadだけをHeuristicのActionで学習した。

- 勝率15%
- 平均6.35点
- 初期pips18.24
- 開拓地教師一致20.7%、道路44.0%

通常PPOを壊さず大きく回復したが、表現部分を固定しているため精度が不足した。

### 13.3 独立した初期配置専用ネットワーク

通常PPOの共有表現を複製し、初期配置専用コピーだけを学習した。これにより通常ターンの重みを完全に保持した。

- 開拓地教師一致71.3%、道路76.3%
- 初期pipsはHeuristicと同水準
- 最初の100局では勝率22%まで改善
- 別の200局では17.5%、Heuristic委譲は27.5%

100局だけなら良く見えたが、独立200局で差が出たため昇格を見送った。pipsが同じでも、資源バランスや道路方向が異なることを示した。

### 13.4 DAgger

1手目を外した後の未学習状態が原因という仮説を立て、モデル自身の配置経路上でHeuristicへ正解を問い合わせるDAggerを3ラウンド実施した。

結果はHeuristic委譲21%、教師軌跡学習20%、DAgger19%。平均点もDAggerは改善しなかった。

詳しく比較すると、初期pipsは約20.5で同じだが、初期資源種類数はHeuristicの4.07に対して学習モデルは約3.71だった。したがって、主因は訪問状態の分布ずれではなく、同じpips帯の候補から資源多様性や将来価値を区別する学習信号の不足と判断した。

## 14. なぜGNN・Self-play・Multi-Agentへまだ進んでいないか

### GNN

盤面はGraphなのでGNNは有望である。ただし、固定Action ID、Family間の候補数差、初期配置教師信号といった、より単純で検証可能な問題が先に見つかった。複数要因を同時に変えると改善理由が分からなくなるため、候補共有・階層化・ランキング学習を先に検証している。

### Self-play

不安定な方策同士を学習させると対戦環境まで変動し、問題切り分けが難しくなる。まず固定Heuristicに対してSingle-Agent PPOが安定して学習できることを優先した。

### Multi-Agent RL

4人同時学習は非定常性がさらに強い。現在のObservation、Action Mask、評価基準を安定させてからParameter SharingやMAPPOを検討する。

## 15. 現在採用しているもの

- ゲームルールの唯一の正は既存Game Engine
- AgentはActionだけ返し、`apply_action()`で状態更新
- 非公開情報を守るObservation
- 固定Action IDとAction Mask
- Single-Agent CatanEnv
- MaskablePPO
- 候補共有型Encoder
- Action Familyを明示的に学ぶ階層方策
- 初期配置Heuristic + 通常ターン50k PPOのハイブリッドモデル
- 複数seed、holdout、席回転、bootstrapを使う評価
- UIからHeuristic / Random / PPOを選ぶ仕組み

## 16. 現在採用していないもの

- 未学習の全フェーズPPO
- Observationを増やしただけの通常MLP
- 推論時だけFamily確率を補正する方式
- headだけで行う初期配置学習
- 現時点の初期配置専用ネットワーク
- DAgger版初期配置モデル
- GNN、LSTM、Self-play、Multi-Agent RL、MCTS

「採用していない」は永久に使わないという意味ではない。現時点の比較結果だけでは標準モデルを置き換える根拠がない、または原因切り分けの順番がまだ来ていないという意味である。

## 17. 次の施策

次は初期配置のランキング学習を行う。

現在はHeuristicが最終的に選んだ1候補だけを正解としている。次は全合法候補へ教師評価値を付ける。

- 合計pips
- 資源種類数
- 既存生産にない資源の補完性
- 2個の開拓地を組み合わせた価値
- 道路の先にある次の開拓地候補価値

候補の絶対IDを覚えるのではなく、良い候補が悪い候補より高くなるようにランキング学習する。その後、未使用seedでHeuristic委譲との非劣性を確認し、問題なければ初期配置をPPO側へ戻して短時間のPPO微調整を行う。

それでも盤面戦略が頭打ちになる場合に、GNNを次の構造候補として検討する。

```text
Tile / Vertex / Edge Graph
          ↓ GNN
Board Embedding + Player / Progress Features
          ↓
Hierarchical Policy / Value
```

## 18. 現在の到達点

強化学習用の基盤、実学習、評価、保存、UI観戦まで実装済みである。PPOはRandomAgentを大きく上回り、Heuristicへ近い性能を示すseedも得られた。一方、全フェーズを安定してHeuristic以上に操作する完成版ではない。

現在の成果は、単に「勝率の高いモデルが1つできた」ことよりも、次の点にある。

- ゲームルールを壊さず複数AIを比較できる
- 非公開情報と合法手を守って学習できる
- 学習結果をUIで人間が観察できる
- seed、checkpoint、行動内訳から失敗原因を追える
- 改善しなかった施策も結果を残し、次の仮説へつなげられる

今後も、一度に複数要因を変えず、仮説、比較条件、結果、採否理由を記録して進める。
