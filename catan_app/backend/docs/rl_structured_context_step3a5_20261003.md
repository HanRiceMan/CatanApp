# Step 3A-5: Action前Structured Context診断と小規模Teacher Dataset

## 結論

ルール・盤面・本人手札から計算できるStrategic / Resource / Risk Contextを、Action適用前のGameStateの**deep copy**上で計算する診断APIを追加した。既存ChampionのAction、Observation v2（1667次元）、377 Action、PPO重み・Planner設定は変更していない。Rule / Champion / Mixedを各8局（各席2局）収集し、合計24局・2,305診断Decisionを得た。3 profileの固定seed parityは計1,206 Action、勝者・得点・順位・全GameState・RNG counter・PPO重み・固定probeのActor logitsが完全一致した。

構造化signalは64 field、数値の入れ子成分を含めると118 field。各fieldには`value`、別個の`valid` mask、単位、値域、意味、Observation v2に対するprovenanceを持たせる。未定義のETAは`99`を値に入れず`null`と`valid=false`にした。小規模DatasetにvalidなETAの99混入、invalidで非nullの値はなかった。これは**value judgementをPPOへ移す準備**であり、Plannerの手製scoreを共通のGoal優先度へ変換するものではない。

300局再収集とHead/PPO学習は未実施。大量収集は、Student入力に何を実時計算して渡すかと、v2に欠ける新規発展カード・belief・交渉履歴をどう扱うかを確定してからがよい。

## 実装とschema

| ファイル | 内容 |
|---|---|
| `app/rl/structured_context.py` | 純粋診断API、6 sectionのsignalとvalid mask、tactical facts、可観測性メタデータ |
| `app/rl/macro_teacher_dataset.py` | 後方互換の`macro_teacher_v5_structured_context`、非action phase収集、選択理由のsource確認、coverage集計、重み/logits parity |
| `app/rl/strategic_intent.py` | 診断用Executionをカード別・盗賊・交渉応答等へ拡張。Action定義は変更なし |
| `app/rl/collect_macro_teacher_dataset.py` | 新集計表示、parity gate |
| `tests/test_rl_structured_context.py` | 非変更・mask・pre-roll構造道路・カード新規性・schema・席ローテーション・BASE_PPO帰属のテスト |

v4の`objective_goal`、`execution_mode`、trade診断、5 Intentの`active/inactive/unknown`、`execution_source`はそのまま保持した。v5は`strategic_state`、`strategic_context`/`strategic_context_valid_mask`、`resource_risk_context`/valid mask、`tactical_facts`、`intent_contributions`、`source_confirmed_reasons`、`attribution_status`、`trade_context`、`selected_execution`、phase、`observation_reference`、対局結果を追加。`definition.json`は各signalのunit・range・semantics・provenanceを収録する。Student入力にしてはいけないAction attributionや結果は別fieldである。

Datasetのローカル保存先は`experiments/structured_context_step3a5_final_20261003/`（実験データとしてGit対象外）。`structured_context_summary.json`に全signalのprofile別valid/invalidと、数値成分のmin/max/mean/medianを保存した。

## phase・Execution coverage

| Profile | Games | Decision | 席 |
|---|---:|---:|---|
| Rule | 8 | 831 | 1〜4各2 |
| Champion | 8 | 789 | 1〜4各2 |
| Mixed | 8 | 685 | 1〜4各2 |

phase内訳はaction 1,057、turn_pre_roll 439、trade_response 521、trade_counter_offer 53、robber_move 100、robber_steal 99、discarding 36。旧Macro taxonomyと5 Intentの従来指標は**action phaseの1,057件だけ**で計算し、非action phaseを未知Goalの誤ラベルとして混ぜていない。

| Execution | 件数 | Execution | 件数 |
|---|---:|---|---:|
| END_TURN | 390 | TRADE_PLAYER | 285 |
| TRADE_BANK | 83 | BUILD_ROAD | 144 |
| BUILD_SETTLEMENT | 68 | BUILD_CITY | 34 |
| BUY_DEVELOPMENT | 53 | PLAY_KNIGHT | 27 |
| PLAY_ROAD_BUILDING | 4 | PLAY_YEAR_OF_PLENTY | 2 |
| PLAY_MONOPOLY | 4 | ROLL_TURN | 402 |
| MOVE_ROBBER | 100 | STEAL_RESOURCE | 99 |
| RESPOND_TRADE | 521 | COUNTER_TRADE | 53 |
| DISCARD | 36 | | |

Step 3A-3のaction-only Datasetで0件だった`PLAY_KNIGHT`が、pre-roll等を含め27件取れた。カード別Executionは診断マッピングであり、377 Action catalogは不変。

## Structured signal coverageと値域の抜粋

全64 fieldの結果は保存済みsummaryを参照。以下は単位と欠測の性質が異なる代表例。

| Signal | valid / invalid | 実測min–max、平均・中央値 | 解釈 |
|---|---:|---|---|
| settlement candidate / count | 2,305 / 0 | 候補有り率92%程度 | 最大5拠点の現時点候補。Planner選択vertexと分離 |
| settlement ETA | 2,126 / 179 | 0–6.00、1.35・1.11 rounds | Planner選択候補に条件づけた概算。診断専用 |
| city upgradeable count | 2,305 / 0 | 0–5、3.09・3 vertices | 将来候補。今購入可能な候補とは別 |
| city ETA | 2,278 / 27 | 0–8.36、1.57・1.33 rounds | 候補がなければinvalid |
| road title acquisition gap | 2,013 / 292 | 1–6、2.79・3 edges | 保持中はinvalid、代わりにdefense gapがvalid |
| knight draws needed | 2,305 / 0 | 0–5、2.64・3 cards | `plays_needed`と別field |
| blocked production | 2,305 / 0 | 0–20、2.25・0 pips | 盗賊被害。即解除すべき価値ではない |
| hand size | 2,305 / 0 | 0–18、5.84・6 cards | 7時の破棄曝露は別bool |
| bank trade options | 2,305 / 0 | 0–16、1.95・0 pairs | 手札・港・銀行在庫の可能な資源ペア |

Road Contextは「盤面グラフ上で伸ばせるedge」と「現phase・資源で買える称号方向edge」を分けた。pre-rollでも前者は計算する。City Contextは将来都市化できる自分の開拓地と、今合法・支払い可能な都市候補を分け、PPO確率を都市候補のtie-breakに使わない。Knight Contextは使用済み、手札所持、今使用可能、称号に必要な**使用**数、まだ**引く**必要がある枚数を別記する。Robber Contextは阻害pip、全生産比、資源別被害、今解除可能かを分ける。

`current_stall_duration`と相手の`opponent_trade_flexibility`は現GameStateだけでは確定しないため、2,305件すべてinvalid。称号保持者がいない場合のholder IDなどもinvalidであり、0や架空IDを代入しない。`resource_risk`は手札5種、建設4種の資源不足、購入可能性、港比、銀行交換選択肢、固定377 Actionの合法数・family別数、即時建設、7での破棄曝露を保存する。入れ子資源ベクトルも118 numeric leafの分布に含めた。

## Intent state、選択理由、tactical fact

`intent active`は局面の状態であり、`intent_contributions`は選択されたActionに使ったsource-level証拠。たとえばROBBER_RELIEFがactiveでもCITY建設に帰属させない。`hand_size>7`は`HAND_OVERFLOW`/`DISCARD_EXPOSURE`という事実を立てるが、それだけで`DISCARD_RISK_REDUCTION`を選択理由へ追加しない。

| Execution | SOURCE_CONFIRMED | SOURCE_EXCLUDES | UNOBSERVED_BASE_PPO | UNKNOWN |
|---|---:|---:|---:|---:|
| BUY_DEVELOPMENT | 29 | 0 | 24 | 0 |
| BUILD_ROAD | 135 | 1 | 8 | 0 |
| PLAY_KNIGHT | 5 | 0 | 22 | 0 |
| TRADE_PLAYER | 285 | 0 | 0 | 0 |
| END_TURN | 120 | 1 | 269 | 0 |

BASE_PPO由来Actionで`source_confirmed_reasons`が非空の例は**0件**。`SOURCE_EXCLUDES`は既知のIntent寄与falseに限り、Actionに理由が無かったとまでは主張しない。盗賊移動・盗み・交渉応答などの一部は選択分岐の理由を観測できず`UNKNOWN`である。

Intent attribution無しはEND_TURN 270、TRADE_PLAYER 96、BUY_DEVELOPMENT 24、PLAY_KNIGHT 22件。TRADE_PLAYER 96件にはsource診断の`TRADE_RESOURCE_REBALANCE`/`TRADE_FUTURE_VALUE`があり、52件は手札過多も伴う。BUY 24件と騎士22件はBASE_PPO由来なので、`GENERAL_EXPECTED_VALUE`や最大騎士力等を後付けしていない。これらはlabel failureではなく、「今のsourceでは行動理由を観測できない」記録である。

## Observation v2との差

`OBSERVED_IN_V2`または`DETERMINISTIC_FROM_V2`は本人手札、公開盤面、港・銀行、道と称号、資源不足、阻害pip等。`PLANNER_DIAGNOSTIC_ONLY`は最良開拓target、その選択に条件づく道路・不足・ETA、`site_value`/`plan_value`。後者をStudentの正解優先度として与えない。

`MISSING_FROM_V2`は新規発展カードによる「今使用できる騎士」、一部の厳密な合法Action数・購入/解除可否など。v2自体には本人のカード種類があるが`new_development_cards` flagがない。小規模Datasetで新規騎士の使用可否が変わる局面は30 Decision。既存source診断では新規カードにより発展購入分岐を塞ぐ4件、騎士使用分岐を塞ぐ9件を確認した。これらは重複し得るため単純合算しない。

`REQUIRES_HISTORY`は同一手番の交渉再提案履歴や停滞継続時間、`REQUIRES_BELIEF`は相手手札のBayesian推定を使う交換・盗賊判断。resource_eventsが存在し、対象phaseにいる**依存可能性**はbelief 1,830 Decision、交渉履歴480 Decision。ただし「その情報が実際にActionを変えた件数」はこの観測だけでは同定できず、0件とも1,830件とも断定しない。後のablationか選択経路traceが必要である。これらはStrategic Contextの盤面計算より、具体的Execution（誰と何を交換、誰から盗む、同手番で再提案するか）に強く関係する。

## parityとテスト

固定seedのRule 485、Champion 440、Mixed 281 Action（計1,206）について、診断なし/ありで全Action列、winner、final VP/rank、全GameState、RNG counterが一致。共有Championモデルのpolicy `state_dict` hashと固定probeのActor logitsも診断前後で完全一致した。全テストは**266件成功、1,163 subtest成功**（既知の依存ライブラリ警告2件）。学習、Observation、PPO、Action catalog、Champion configは未変更。

## A / B / C分類と次の判断

- **A: Runtimeで直接計算** — ルール上の手札・資源不足・港比・銀行在庫・合法mask、7曝露、道路グラフ・title gap・騎士のplays/draws needed、blocked pips、候補到達道路数と到達可能性。厳密・安価な値をNNに再推定させない。ただしPlanner-selected targetやその`plan_value`はここへ混ぜない。
- **B: Auxiliary targetの候補** — 候補有無、将来都市化の生産増分、ETAのvalid付き回帰、構造的称号到達性など。Runtimeで同じ決定的値を渡せるなら補助学習は主目的にせず、表現学習上の増益をablationで確認する。vertex/edge IDは序数回帰でなく候補mask付きpointer等を使う。
- **C: PPOが学ぶべき価値判断** — 開拓と都市の機会費用、ROAD/KNIGHT_TITLEを今追う価値、盗賊解除の優先度、資源を保持/消費する判断、Future Value Trade、目的が明確でない発展購入、7リスク許容。Plannerの手製閾値や異種scoreを正解として固定しない。

300局を**学習用の確定Teacher Datasetとして再収集するにはまだ早い**。先に(1) Runtime Contextの入力仕様、(2) `new_development_cards`と必要なbelief/交渉履歴を含む新Observationの仕様、(3) source未観測Actionをどのlossに使うか、(4) train/validation/testのgame単位分割を決める。その後、Championを固定し、別artifactで300局規模へ進む。v2をそのまま使っても盤面Contextの実験は可能だが、騎士・交渉のExecution模倣ではTeacherだけが持つ情報を解消する必要がある。Policy候補は既存GNN/PPO actorと377 maskを保ち、決定的Runtime Contextを別入力、必要なら異種のmasked auxiliary headを追加する形。Actionへの重み付け・優先順位はPPO actor側に残す。
