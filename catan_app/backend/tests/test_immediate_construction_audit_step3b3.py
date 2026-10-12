"""Read-only construction and Development legality boundary checks."""

from types import SimpleNamespace

import pytest

from app.domain.actions import (BuildCityAction, BuildSettlementAction,
                                BuyDevelopmentAction, EndTurnAction,
                                UseDevelopmentAction)
from app.domain.game import RESOURCES
from app.rl.run_immediate_construction_audit_step3b3 import (
    CARD_ORDER, _action_family, classify_action_phase, summarize)


def _game(*, cards=(), new=(), used=False, phase="action", free_roads=0):
    player = SimpleNamespace(
        development_cards=list(cards), new_development_cards=list(new),
        resources={name: 0 for name in RESOURCES})
    return SimpleNamespace(phase=phase, free_road_remaining=free_roads,
                           current_player_id=1, used_development_this_turn=used,
                           players=[player])


@pytest.mark.parametrize(("actions", "expected"), [
    ([EndTurnAction()], "NONE"),
    ([BuildSettlementAction(3), EndTurnAction()], "SETTLEMENT_ONLY"),
    ([BuildCityAction(4), EndTurnAction()], "CITY_ONLY"),
    ([BuildSettlementAction(3), BuildCityAction(4), EndTurnAction()], "BOTH"),
])
def test_build_class_uses_legal_concrete_actions(actions, expected):
    game = _game()
    before = repr(game)
    result = classify_action_phase(game, 1, actions[-1], legal=actions)
    assert result["build_class"] == expected
    assert result["settlement_vertices"] == [a.vertex_id for a in actions
                                               if isinstance(a, BuildSettlementAction)]
    assert result["city_vertices"] == [a.vertex_id for a in actions
                                         if isinstance(a, BuildCityAction)]
    assert repr(game) == before


def test_development_usable_requires_old_card_and_legal_target():
    game = _game(cards=("knight", "monopoly", "year_of_plenty", "road_building"),
                 new=("knight",))
    legal = [EndTurnAction(), UseDevelopmentAction("monopoly", ("ore",)),
             UseDevelopmentAction("road_building")]
    result = classify_action_phase(game, 1, legal[0], legal=legal)
    assert result["usable_development_counts"]["knight"] == 0
    assert result["usable_development"]["knight"] is False
    assert result["usable_development"]["monopoly"] is True
    assert result["usable_development"]["year_of_plenty"] is False
    assert result["usable_development"]["road_building"] is True
    assert set(result["usable_development"]) == set(CARD_ORDER)


def test_illegal_new_card_and_used_card_are_rejected():
    game = _game(cards=("knight",), new=("knight",))
    legal = [EndTurnAction(), UseDevelopmentAction("knight")]
    with pytest.raises(AssertionError, match="Newly bought"):
        classify_action_phase(game, 1, legal[0], legal=legal)
    game = _game(cards=("knight",), used=True)
    with pytest.raises(AssertionError, match="already used"):
        classify_action_phase(game, 1, legal[0], legal=legal)


def test_free_road_and_pre_roll_not_normal_construction():
    assert classify_action_phase(_game(free_roads=1), 1, EndTurnAction(),
                                 legal=[EndTurnAction()]) is None
    assert classify_action_phase(_game(phase="turn_pre_roll"), 1, EndTurnAction(),
                                 legal=[EndTurnAction()]) is None


def test_action_family_separates_buy_and_play_development():
    assert _action_family(BuyDevelopmentAction()) == "BUY_DEVELOPMENT"
    assert _action_family(UseDevelopmentAction("knight")) == "PLAY_DEVELOPMENT"


def test_immediate_wins_are_excluded_from_regular_behavior():
    base = {"phase": "action", "build_class": "SETTLEMENT_ONLY",
            "action_family": "BUILD_SETTLEMENT", "played_card": None,
            "usable_development": dict.fromkeys(CARD_ORDER, False),
            "profile": "rule", "immediate_win": False}
    records = [dict(base), dict(base, immediate_win=True,
                                  immediate_win_family="BUILD_SETTLEMENT")]
    result = summarize(records, [], {})
    assert result["normal_action_states"] == 2
    assert result["immediate_win_excluded"] == 1
    assert result["class_counts_all"]["SETTLEMENT_ONLY"] == 2
    assert result["class_counts"]["SETTLEMENT_ONLY"] == 1
    assert result["behavior"]["SETTLEMENT_ONLY"] == {"BUILD_SETTLEMENT": 1}
