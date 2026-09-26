"""学習済みMaskablePPOを、既存Agentと同じAction選択者として使う。"""

from pathlib import Path
import json

import numpy as np

from app.domain.actions import Action
from app.domain.game import GameActionError, GameState
from app.agents.heuristic import HeuristicAgent
from app.rl.action_space import get_action_mask, id_to_action
from app.rl.model_registry import ModelManifest, ModelRegistry
from app.rl.observation import encode_observation, get_observation


class PPOAgent:
    """モデル本体は読むが、GameStateは変更せずActionだけを返す。"""

    def __init__(self, model_id: str, *, models_root: Path | str | None = None,
                 deterministic: bool = True, device: str = "auto"):
        self.registry = ModelRegistry(models_root)
        self.manifest: ModelManifest = self.registry.load(model_id)
        if self.manifest.algorithm != "MaskablePPO":
            raise ValueError(f"PPOAgentで読み込めないアルゴリズムです: {self.manifest.algorithm}")
        try:
            from sb3_contrib import MaskablePPO
        except ImportError as error:
            raise RuntimeError("PPOAgentにはrequirements-rl.txtの学習環境が必要です。") from error
        artifact = self.registry.models_root / model_id / self.manifest.artifact_filename
        config_path = artifact.parent / "config.json"
        training_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        planning = training_config.get("settlement_planning")
        if planning is not None:
            if (not isinstance(planning, dict)
                    or planning.get("target_sites") not in range(3, 6)
                    or not isinstance(planning.get("max_wait_rounds"), (int, float))
                    or planning["max_wait_rounds"] <= 0
                    or not isinstance(planning.get("max_city_wait_rounds", 6), (int, float))
                    or planning.get("max_city_wait_rounds", 6) <= 0
                    or planning.get("title_horizon", 1) not in range(1, 4)
                    or planning.get("first_city_max_deficit", 0) not in range(0, 6)
                    or planning.get("second_city_max_deficit", 0) not in range(0, 6)
                    or planning.get("second_city_min_score", 0) not in range(0, 10)
                    or planning.get("third_city_max_deficit", 0) not in range(0, 6)
                    or planning.get("third_city_min_score", 0) not in range(0, 10)
                    or not isinstance(planning.get(
                        "post_target_connected_reserve_max_wait", 0), (int, float))
                    or not 0 <= planning.get(
                        "post_target_connected_reserve_max_wait", 0) <= 6
                    or planning.get(
                        "post_target_connected_reserve_min_score", 0) not in range(0, 10)
                    or planning.get(
                        "post_target_connected_reserve_max_score", 9) not in range(0, 10)
                    or planning.get("post_target_connected_reserve_min_score", 0)
                    > planning.get("post_target_connected_reserve_max_score", 9)
                    or planning.get("max_plan_roads", 3) not in range(1, 6)
                    or planning.get("defend_titles_min_score", 0) not in range(0, 10)
                    or planning.get("endgame_point_reserve_score", 0) not in range(0, 10)
                    or not isinstance(planning.get("title_road_build_wait_threshold", 0),
                                      (int, float))
                    or not 0 <= planning.get("title_road_build_wait_threshold", 0) <= 6
                    or planning.get("longest_road_min_score", 0) not in range(0, 10)
                    or planning.get("extended_road_horizon_score", 0) not in range(0, 10)
                    or planning.get("proactive_trade_max_scarcity_deficit", 1)
                    not in range(1, 4)
                    or planning.get("proactive_trade_opponent_score_limit", 8)
                    not in range(6, 10)
                    or planning.get("strategic_trade_opponent_score_limit", 10)
                    not in range(6, 11)
                    or not isinstance(planning.get(
                        "development_stall_wait_rounds", 2.5), (int, float))
                    or not 1 <= planning.get("development_stall_wait_rounds", 2.5) <= 12
                    or not isinstance(planning.get(
                        "development_max_build_delay", 2.0), (int, float))
                    or not 0 <= planning.get("development_max_build_delay", 2.0) <= 6
                    or planning.get("development_tactical_score", 8) not in range(6, 10)
                    or planning.get("expansion_setup_min_pips", 1) not in range(1, 6)
                    or planning.get("setup_min_wheat_pips", 0) not in range(0, 6)
                    or planning.get("setup_min_ore_pips", 0) not in range(0, 6)
                    or not isinstance(planning.get("city_growth_value_weight", 0),
                                      (int, float))
                    or not 0 <= planning.get("city_growth_value_weight", 0) <= 2.0
                    or not isinstance(planning.get(
                        "city_reservation_wait_margin", 1.0), (int, float))
                    or not 0 <= planning.get("city_reservation_wait_margin", 1.0) <= 4
                    or not isinstance(planning.get(
                        "city_reservation_value_ratio", 0.9), (int, float))
                    or not 0.5 <= planning.get("city_reservation_value_ratio", 0.9) <= 1.5
                    or planning.get("early_affordable_city_min_sites", 0) not in (0, 3, 4)
                    or not isinstance(planning.get("three_site_city_value_ratio", 0),
                                      (int, float))
                    or (planning.get("three_site_city_value_ratio", 0) != 0
                        and not 0.5 <= planning["three_site_city_value_ratio"] <= 2.0)
                    or any(type(planning.get(name, False)) is not bool for name in (
                        "prioritize_affordable_city", "prioritize_immediate_win",
                        "prioritize_immediate_titles",
                        "construction_before_titles",
                        "buy_for_largest_army", "expansion_aware_setup",
                        "setup_best_coverage_fallback", "defend_titles",
                        "longest_road_lookahead",
                        "belief_aware_robber",
                        "proactive_player_trade",
                        "strategic_surplus_trade",
                        "purposeful_paid_roads",
                        "protect_city_bank_trade_reserve",
                        "strategic_trade_response",
                        "adaptive_development_purchase",
                        "endgame_development_fallback",
                        "collision_aware_initial_roads",
                    ))):
                raise ValueError("settlement_planning設定が不正です。")
            self.settlement_planning_config = {
                "target_sites": planning["target_sites"],
                "max_wait_rounds": float(planning["max_wait_rounds"]),
                "max_city_wait_rounds": float(planning.get("max_city_wait_rounds", 6)),
                "max_plan_roads": planning.get("max_plan_roads", 3),
                "prioritize_affordable_city": planning.get("prioritize_affordable_city", False),
                "early_affordable_city_min_sites": planning.get(
                    "early_affordable_city_min_sites", 0
                ),
                "three_site_city_value_ratio": planning.get(
                    "three_site_city_value_ratio", 0.0
                ),
                "prioritize_immediate_win": planning.get("prioritize_immediate_win", False),
                "endgame_point_reserve_score": planning.get(
                    "endgame_point_reserve_score", 0
                ),
                "prioritize_immediate_titles": planning.get("prioritize_immediate_titles", False),
                "construction_before_titles": planning.get("construction_before_titles", False),
                "title_road_build_wait_threshold": planning.get(
                    "title_road_build_wait_threshold", 0.0
                ),
                "title_horizon": planning.get("title_horizon", 1),
                "longest_road_min_score": planning.get("longest_road_min_score", 0),
                "extended_road_horizon_score": planning.get(
                    "extended_road_horizon_score", 0
                ),
                "longest_road_lookahead": planning.get("longest_road_lookahead", False),
                "first_city_max_deficit": planning.get("first_city_max_deficit", 0),
                "second_city_max_deficit": planning.get("second_city_max_deficit", 0),
                "second_city_min_score": planning.get("second_city_min_score", 0),
                "third_city_max_deficit": planning.get("third_city_max_deficit", 0),
                "third_city_min_score": planning.get("third_city_min_score", 0),
                "post_target_connected_reserve_max_wait": planning.get(
                    "post_target_connected_reserve_max_wait", 0.0
                ),
                "post_target_connected_reserve_min_score": planning.get(
                    "post_target_connected_reserve_min_score", 0
                ),
                "post_target_connected_reserve_max_score": planning.get(
                    "post_target_connected_reserve_max_score", 9
                ),
                "expansion_aware_setup": planning.get("expansion_aware_setup", False),
                "expansion_setup_min_pips": planning.get("expansion_setup_min_pips", 1),
                "setup_min_wheat_pips": planning.get("setup_min_wheat_pips", 0),
                "setup_min_ore_pips": planning.get("setup_min_ore_pips", 0),
                "city_growth_value_weight": planning.get(
                    "city_growth_value_weight", 0.0
                ),
                "setup_best_coverage_fallback": planning.get(
                    "setup_best_coverage_fallback", False
                ),
                "collision_aware_initial_roads": planning.get(
                    "collision_aware_initial_roads", False
                ),
                "buy_for_largest_army": planning.get("buy_for_largest_army", False),
                "belief_aware_robber": planning.get("belief_aware_robber", False),
                "proactive_player_trade": planning.get("proactive_player_trade", False),
                "proactive_trade_max_scarcity_deficit": planning.get(
                    "proactive_trade_max_scarcity_deficit", 1
                ),
                "strategic_surplus_trade": planning.get(
                    "strategic_surplus_trade", False
                ),
                "purposeful_paid_roads": planning.get(
                    "purposeful_paid_roads", False
                ),
                "protect_city_bank_trade_reserve": planning.get(
                    "protect_city_bank_trade_reserve", False
                ),
                "city_reservation_wait_margin": float(planning.get(
                    "city_reservation_wait_margin", 1.0
                )),
                "city_reservation_value_ratio": float(planning.get(
                    "city_reservation_value_ratio", 0.9
                )),
                "proactive_trade_opponent_score_limit": planning.get(
                    "proactive_trade_opponent_score_limit", 8
                ),
                "strategic_trade_response": planning.get("strategic_trade_response", False),
                "strategic_trade_opponent_score_limit": planning.get(
                    "strategic_trade_opponent_score_limit", 10
                ),
                "adaptive_development_purchase": planning.get(
                    "adaptive_development_purchase", False
                ),
                "development_stall_wait_rounds": float(planning.get(
                    "development_stall_wait_rounds", 2.5
                )),
                "development_max_build_delay": float(planning.get(
                    "development_max_build_delay", 2.0
                )),
                "development_tactical_score": planning.get(
                    "development_tactical_score", 8
                ),
                "endgame_development_fallback": planning.get(
                    "endgame_development_fallback", False
                ),
                "defend_titles": planning.get("defend_titles", False),
                "defend_titles_min_score": planning.get("defend_titles_min_score", 0),
            }
        else:
            self.settlement_planning_config = None
        self.heuristic_initial_placement = training_config.get("heuristic_initial_placement", False) is True
        self.model = MaskablePPO.load(str(artifact), device=device)
        self.initial_setup_policy = None
        if self.manifest.initial_setup is not None:
            import torch

            from app.rl.gnn_setup_policy import GraphInitialSetupPolicy

            setup_artifact = artifact.parent / self.manifest.initial_setup["artifact_filename"]
            setup_policy = GraphInitialSetupPolicy(self.model.policy)
            setup_policy.load_state_dict(
                torch.load(setup_artifact, map_location="cpu", weights_only=True)
            )
            setup_policy.eval()
            self.initial_setup_policy = setup_policy
            self.heuristic_initial_placement = False
        self.deterministic = deterministic

    def select_action(self, game: GameState, player_id: int) -> Action:
        if self.initial_setup_policy is not None and game.phase == "setup_ready":
            observation = encode_observation(get_observation(
                game, player_id, version=self.manifest.observation_version
            ))
            action_mask = get_action_mask(game, player_id)
            return id_to_action(
                self.initial_setup_policy.select_action_id(observation, action_mask)
            )
        if self.heuristic_initial_placement and game.phase == "setup_ready":
            return HeuristicAgent().select_action(game, player_id)
        # PPO v1ではプレイヤー間交渉を学習していない。通常対局で人間/他AIから
        # 提案された場合は、既存Heuristicがその応答・逆交渉だけを選ぶ。
        if game.phase in {"trade_response", "trade_counter_offer"}:
            return HeuristicAgent().select_action(game, player_id)
        observation = encode_observation(get_observation(game, player_id,
                                                        version=self.manifest.observation_version))
        action_mask = get_action_mask(game, player_id)
        if not action_mask.any():
            raise GameActionError("PPOAgentが操作できる合法Actionがありません。")
        action_id, _ = self.model.predict(observation, deterministic=self.deterministic,
                                          action_masks=action_mask)
        selected_id = int(np.asarray(action_id).item())
        if selected_id < 0 or selected_id >= len(action_mask) or not action_mask[selected_id]:
            raise GameActionError("PPOAgentが非法Actionを返しました。")
        return id_to_action(selected_id)
