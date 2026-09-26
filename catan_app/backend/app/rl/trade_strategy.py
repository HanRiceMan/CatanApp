"""公開情報と建設目標を使う、保守的なプレイヤー間交渉戦略。"""

from __future__ import annotations

from app.domain.actions import (ProposeCounterTradeAction, ProposeTradeAction,
                                RespondToTradeAction, TradeOffer)
from app.domain.game import (BUILD_COSTS, RESOURCES, GameState, _can_afford,
                             legal_settlement_ids, player_for)

from .expansion_planner import (analyze_expansion_plan, estimated_build_wait,
                                player_production, settlement_target_cost)
from .hand_belief import BayesianHandEstimator
from .reward import actual_score
from .victory_race import public_score


def construction_trade_goal(game: GameState, player_id: int, *, target_sites: int,
                            max_plan_roads: int, max_wait_rounds: float,
                            max_city_wait_rounds: float) -> tuple[str, dict[str, int]] | None:
    """現行計画に沿う、次の建設目標と必要資源を返す。"""
    player = player_for(game, player_id)
    site_count = player.settlements + player.cities
    plan = None
    # あと1点なら目標拠点数ではなく、都市と開拓地の完成待ち時間を比較する。
    # 開拓地5個を使用中なら開拓地は候補外。同じ待ち時間なら、生産増加と
    # 開拓地コマ回収を同時に得られる都市を選ぶ。
    if actual_score(game, player_id) >= 9 and player.settlements < 5:
        plan = analyze_expansion_plan(
            game, player_id, max_additional_roads=max_plan_roads,
        )
        target = plan.selected_target
        city_wait = (estimated_build_wait(game, player_id, BUILD_COSTS["city"])
                     if player.settlements > 0 and player.cities < 4 else 99.0)
        if (target is not None and target.wait_rounds <= max_wait_rounds
                and target.wait_rounds < city_wait):
            return "settlement", settlement_target_cost(
                target, free_roads=game.free_road_remaining
            )
    if site_count < target_sites and player.settlements < 5:
        plan = plan or analyze_expansion_plan(
            game, player_id, max_additional_roads=max_plan_roads,
        )
        target = plan.selected_target
        if target is not None and target.wait_rounds <= max_wait_rounds:
            return "settlement", settlement_target_cost(
                target, free_roads=game.free_road_remaining
            )
    # 生産できない資源こそ交渉で補う意味があるため、都市の自然入手待ち時間では
    # 除外しない。実際の提案は「あと1枚」の場合に限定される。
    if player.settlements > 0 and player.cities < 4:
        return "city", dict(BUILD_COSTS["city"])
    return None


def _deficits(resources: dict[str, int], required: dict[str, int]) -> dict[str, int]:
    return {
        resource: max(0, required.get(resource, 0) - resources[resource])
        for resource in RESOURCES
    }


def _attempted_targets(game: GameState, player_id: int) -> set[int]:
    targets = set()
    for entry in game.activity_log:
        trade = entry.get("trade")
        if (entry.get("turn_number") == game.turn_number and trade
                and trade.get("proposer_id") == player_id
                and not trade.get("is_counter", False)):
            targets.add(int(trade["recipient_id"]))
    return targets


def _proposal_attempts(game: GameState, player_id: int,
                       wanted: str) -> list[tuple[int, int]]:
    """この手番に同じ資源を求めた (相手, 最大提示倍率) を返す。"""
    attempts = []
    for entry in game.activity_log:
        trade = entry.get("trade")
        if (entry.get("kind") != "trade" or entry.get("turn_number") != game.turn_number
                or not trade or trade.get("proposer_id") != player_id
                or trade.get("is_counter", False)):
            continue
        ratios = []
        for offer in trade.get("offers", []):
            wanted_count = offer["want"].get(wanted, 0)
            if wanted_count:
                ratios.append(sum(offer["give"].values()) // wanted_count)
        if ratios:
            attempts.append((int(trade["recipient_id"]), max(ratios)))
    return attempts


def _strategic_trade_budget(game: GameState, player_id: int, *, target_sites: int,
                            max_plan_roads: int, max_wait_rounds: float,
                            max_city_wait_rounds: float,
                            stall_wait_rounds: float,
                            burst_warning_cards: int) -> tuple[dict[str, int], str] | None:
    """停滞時に、直近建設と将来の都市の両方を見た予約量・希少資源を返す。"""
    player = player_for(game, player_id)
    hand_size = sum(player.resources.values())
    immediate_build = bool(
        (player.settlements < 5
         and _can_afford(player, BUILD_COSTS["settlement"])
         and legal_settlement_ids(game, player_id, initial=False))
        or (player.settlements > 0 and player.cities < 4
            and _can_afford(player, BUILD_COSTS["city"]))
    )
    if immediate_build:
        return None
    plan = analyze_expansion_plan(
        game, player_id, max_additional_roads=max_plan_roads,
    )
    if (hand_size < burst_warning_cards
            and plan.build_wait_rounds < stall_wait_rounds):
        return None

    active = construction_trade_goal(
        game, player_id, target_sites=target_sites,
        max_plan_roads=max_plan_roads, max_wait_rounds=max_wait_rounds,
        max_city_wait_rounds=max_city_wait_rounds,
    )
    reserve = dict.fromkeys(RESOURCES, 0)
    if active is not None:
        _, active_required = active
        reserve.update(active_required)
    # 開拓地を目指している最中でも、入手困難な鉱石・小麦を将来の都市用に
    # 先回りして集められるよう、同じ資源は直近目標との大きい方を予約する。
    if (actual_score(game, player_id) < 9
            and player.settlements > 0 and player.cities < 4):
        for resource, count in BUILD_COSTS["city"].items():
            reserve[resource] = max(reserve[resource], count)

    production = player_production(game, player_id)
    missing = [resource for resource in RESOURCES
               if player.resources[resource] < reserve[resource]]
    surplus_exists = any(
        player.resources[resource] > reserve[resource] for resource in RESOURCES
    )
    if not missing or not surplus_exists:
        return None
    wanted = max(missing, key=lambda resource: (
        production[resource] <= 1,
        reserve[resource] - player.resources[resource],
        -production[resource],
        resource == "ore",
        -RESOURCES.index(resource),
    ))
    return reserve, wanted


def choose_proactive_trade(game: GameState, player_id: int, *, target_sites: int,
                           max_plan_roads: int, max_wait_rounds: float,
                           max_city_wait_rounds: float,
                           max_proposals: int = 3,
                           max_scarcity_deficit: int = 1,
                           minimum_has_probability: float = 0.45,
                           opponent_score_limit: int = 8,
                           allow_leader_at_score: int = 9,
                           strategic_surplus_trade: bool = False,
                           surplus_trade_stall_wait_rounds: float = 4.0,
                           burst_warning_cards: int = 7,
                           burst_danger_cards: int = 8) -> ProposeTradeAction | None:
    """建設不足または停滞中の余剰を、相手の保有推定に沿って希少資源へ替える。"""
    if game.phase != "action" or game.current_player_id != player_id:
        return None
    proposal_limit = 5 if strategic_surplus_trade else max_proposals
    if game.free_road_remaining or game.pending_trade or game.trade_count >= proposal_limit:
        return None
    goal = construction_trade_goal(
        game, player_id, target_sites=target_sites,
        max_plan_roads=max_plan_roads, max_wait_rounds=max_wait_rounds,
        max_city_wait_rounds=max_city_wait_rounds,
    )
    player = player_for(game, player_id)
    hand_size = sum(player.resources.values())
    standard_trade = False
    if goal is not None:
        goal_name, required = goal
        deficits = _deficits(player.resources, required)
        total_deficit = sum(deficits.values())
        missing_resources = [resource for resource, count in deficits.items() if count]
        standard_trade = total_deficit == 1 or (
            goal_name == "city"
            and 1 < total_deficit <= max_scarcity_deficit
            and len(missing_resources) == 1
            and player_production(game, player_id)[missing_resources[0]] <= 1
        )
    if standard_trade:
        wanted = missing_resources[0]
        strategic = False
    elif strategic_surplus_trade:
        budget = _strategic_trade_budget(
            game, player_id, target_sites=target_sites,
            max_plan_roads=max_plan_roads, max_wait_rounds=max_wait_rounds,
            max_city_wait_rounds=max_city_wait_rounds,
            stall_wait_rounds=surplus_trade_stall_wait_rounds,
            burst_warning_cards=burst_warning_cards,
        )
        if budget is None:
            return None
        required, wanted = budget
        strategic = True
    else:
        return None

    surplus = [
        resource for resource in RESOURCES
        if resource != wanted
        and player.resources[resource] > required.get(resource, 0)
    ]
    if not surplus:
        return None

    beliefs = BayesianHandEstimator().estimate(game, player_id)
    attempts = _proposal_attempts(game, player_id, wanted)
    attempted = _attempted_targets(game, player_id)
    own_score = public_score(game, player_id)
    one_to_one_targets = {target for target, ratio in attempts if ratio == 1}
    two_to_one_targets = {target for target, ratio in attempts if ratio >= 2}
    # 8枚以上は1回断られたら手札圧縮を優先して2:1へ進む。それ未満は
    # 未交渉の相手へ1:1を一巡してから値上げする。
    escalate = bool(strategic and (
        (hand_size >= burst_danger_cards and one_to_one_targets)
        or len(one_to_one_targets) >= 3
    ))
    ratio = 2 if escalate else 1
    unavailable_targets = two_to_one_targets if ratio == 2 else (
        one_to_one_targets if strategic else attempted
    )
    probability_floor = minimum_has_probability
    if strategic:
        probability_floor = 0.10 if hand_size >= burst_danger_cards else 0.20
    def eligible_targets(excluded: set[int]):
        result = []
        for other in game.players:
            if other.id == player_id or other.id in excluded:
                continue
            score = public_score(game, other.id)
            if score >= opponent_score_limit and own_score < allow_leader_at_score:
                continue
            probability = beliefs[other.id].probability_has(wanted)
            if probability < probability_floor:
                continue
            result.append((probability, -score,
                           beliefs[other.id].expected_total, -other.id, other.id))
        return result

    candidates = eligible_targets(unavailable_targets)
    if not candidates and strategic and ratio == 1 and one_to_one_targets:
        ratio = 2
        candidates = eligible_targets(two_to_one_targets)
    if not candidates:
        return None
    target_id = max(candidates)[-1]
    target_production = player_production(game, target_id)
    # 自分の余剰が多く、相手自身も生産しやすい（相手を助けにくい）資源から出す。
    surplus.sort(key=lambda resource: (
        player.resources[resource] - required.get(resource, 0) >= ratio,
        player.resources[resource] - required.get(resource, 0),
        target_production[resource],
        player.resources[resource], -RESOURCES.index(resource),
    ), reverse=True)
    offers = tuple(TradeOffer({resource: ratio}, {wanted: 1})
                   for resource in surplus
                   if player.resources[resource] - required.get(resource, 0) >= ratio)[:3]
    if not offers:
        return None
    return ProposeTradeAction(target_id, offers)


def choose_trade_response(game: GameState, player_id: int, *, target_sites: int,
                          max_plan_roads: int, max_wait_rounds: float,
                          max_city_wait_rounds: float,
                          opponent_score_limit: int = 10) -> RespondToTradeAction:
    """自分の建設不足を純減させる候補だけ承諾する。"""
    trade = game.pending_trade
    if game.phase != "trade_response" or trade is None:
        raise ValueError("応答できる交渉がありません。")
    if (public_score(game, int(trade["proposer_id"])) >= opponent_score_limit
            and public_score(game, player_id) < 9):
        return RespondToTradeAction("decline")
    goal = construction_trade_goal(
        game, player_id, target_sites=target_sites,
        max_plan_roads=max_plan_roads, max_wait_rounds=max_wait_rounds,
        max_city_wait_rounds=max_city_wait_rounds,
    )
    if goal is not None:
        _, required = goal
        player = player_for(game, player_id)
        before = sum(_deficits(player.resources, required).values())
        acceptable = []
        for index, offer in enumerate(trade["offers"]):
            if not _can_afford(player, offer["want"]):
                continue
            after_resources = {
                resource: (player.resources[resource]
                           + offer["give"].get(resource, 0)
                           - offer["want"].get(resource, 0))
                for resource in RESOURCES
            }
            after = sum(_deficits(after_resources, required).values())
            if after < before:
                acceptable.append((before - after, -sum(offer["want"].values()), -index))
        if acceptable:
            index = -max(acceptable)[-1]
            return RespondToTradeAction("accept", index)
    if not trade["is_counter"] and choose_counter_trade(
        game, player_id, target_sites=target_sites,
        max_plan_roads=max_plan_roads, max_wait_rounds=max_wait_rounds,
        max_city_wait_rounds=max_city_wait_rounds,
    ) is not None:
        return RespondToTradeAction("counter")
    return RespondToTradeAction("decline")


def choose_counter_trade(game: GameState, player_id: int, *, target_sites: int,
                         max_plan_roads: int, max_wait_rounds: float,
                         max_city_wait_rounds: float) -> ProposeCounterTradeAction | None:
    """元提案者が不足資源を持ちそうなら、余剰資源との1:1逆提案を作る。"""
    trade = game.pending_trade
    if trade is None or trade.get("is_counter"):
        return None
    goal = construction_trade_goal(
        game, player_id, target_sites=target_sites,
        max_plan_roads=max_plan_roads, max_wait_rounds=max_wait_rounds,
        max_city_wait_rounds=max_city_wait_rounds,
    )
    if goal is None:
        return None
    _, required = goal
    player = player_for(game, player_id)
    deficits = _deficits(player.resources, required)
    wanted = [resource for resource in RESOURCES if deficits[resource]]
    if not wanted:
        return None
    requested = {
        resource for offer in trade["offers"] for resource in RESOURCES
        if offer["want"].get(resource, 0)
        and player.resources[resource] > required.get(resource, 0)
    }
    if not requested:
        return None
    proposer_id = int(trade["proposer_id"])
    belief = BayesianHandEstimator().estimate(game, player_id)[proposer_id]
    wanted = [resource for resource in wanted
              if belief.probability_has(resource) >= 0.45]
    if not wanted:
        return None
    gives = sorted(requested, key=lambda resource: (
        player.resources[resource] - required.get(resource, 0),
        -RESOURCES.index(resource),
    ), reverse=True)
    wanted.sort(key=lambda resource: (
        belief.probability_has(resource), deficits[resource],
        -RESOURCES.index(resource),
    ), reverse=True)
    offers = tuple(
        TradeOffer({give: 1}, {want: 1})
        for want in wanted[:2] for give in gives[:2] if give != want
    )[:3]
    return ProposeCounterTradeAction(offers) if offers else None
