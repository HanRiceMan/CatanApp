from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from app.rl.macro_goal_policy import (
    MACRO_ACTOR_LOGIT_SIZE,
    MACRO_CRITIC_FEATURE_SIZE,
    MACRO_FEATURE_SIZE,
    MACRO_GRAPH_SUMMARY_SIZE,
    MACRO_RAW_OBSERVATION_SIZE,
    MacroGoalHead,
    TRAINABLE_MACRO_GOALS,
    classification_metrics,
    extract_macro_features,
    load_macro_head,
    make_macro_game_split,
    save_macro_head,
    split_example_indices,
)


def _records() -> list[dict]:
    records = []
    for profile in ("rule", "champion", "mixed"):
        for seat in range(1, 5):
            for game_number in range(5):
                game_id = f"{profile}-{seat}-{game_number}"
                for decision in range(2):
                    records.append({
                        "game_id": game_id,
                        "opponent_profile": profile,
                        "seat": seat,
                        "decision": decision,
                    })
    return records


def test_macro_game_split_is_grouped_disjoint_and_deterministic() -> None:
    records = _records()
    first = make_macro_game_split(records, seed=17, validation_fraction=0.2,
                                  test_fraction=0.2)
    second = make_macro_game_split(records, seed=17, validation_fraction=0.2,
                                   test_fraction=0.2)
    assert first == second
    groups = [set(first.train_game_ids), set(first.validation_game_ids),
              set(first.test_game_ids)]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    indices = split_example_indices(records, first)
    assert sum(len(values) for values in indices.values()) == len(records)
    for name, values in indices.items():
        game_ids = {records[index]["game_id"] for index in values}
        assert game_ids == set(getattr(first, f"{name}_game_ids"))


def test_classification_metrics_uses_macro_goal_order() -> None:
    truth = np.asarray([0, 0, 1, 2, 3, 4], dtype=np.int64)
    predicted = np.asarray([0, 1, 1, 2, 4, 4], dtype=np.int64)
    metrics = classification_metrics(truth, predicted)
    assert metrics["confusion_labels"] == list(TRAINABLE_MACRO_GOALS)
    assert metrics["accuracy"] == 4 / 6
    assert metrics["per_goal"]["ROAD_TITLE"]["recall"] == 0.0
    assert metrics["per_goal"]["KNIGHT_TITLE"]["precision"] == 0.5


def test_macro_head_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(9)
    head = MacroGoalHead()
    features = torch.randn(4, MACRO_FEATURE_SIZE)
    expected = head.eval()(features).detach()
    from app.rl.macro_goal_policy import MacroGameSplit
    save_macro_head(
        tmp_path / "artifact", head,
        metadata={"source_model_id": "test", "runtime_connected": False},
        split=MacroGameSplit(("a",), ("b",), ("c",)),
        metrics={"test": True}, history=[{"epoch": 1}],
    )
    loaded = load_macro_head(tmp_path / "artifact")
    assert torch.equal(expected, loaded(features).detach())
    metadata = json.loads(
        (tmp_path / "artifact" / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["runtime_connected"] is False
    assert metadata["goal_labels"] == list(TRAINABLE_MACRO_GOALS)


class _DummyActor(torch.nn.Module):
    observation_size = MACRO_RAW_OBSERVATION_SIZE

    def __init__(self) -> None:
        super().__init__()
        self.family_head = torch.nn.Linear(self.observation_size, 15)
        self.critic = torch.nn.Linear(self.observation_size, MACRO_CRITIC_FEATURE_SIZE)

    def _encoded_spatial_states(self, observations: torch.Tensor):
        batch = len(observations)
        device = observations.device
        return (
            torch.zeros(batch, 54, 64, device=device),
            torch.zeros(batch, 72, 192, device=device),
            torch.zeros(batch, 19, 128, device=device),
        )

    def forward_actor(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.zeros(len(observations), MACRO_ACTOR_LOGIT_SIZE,
                           device=observations.device)


def test_extract_macro_features_keeps_actor_state_and_mode() -> None:
    actor = _DummyActor().train()
    state_before = {name: value.detach().clone()
                    for name, value in actor.state_dict().items()}
    observations = np.zeros((3, MACRO_RAW_OBSERVATION_SIZE), dtype=np.float32)
    features = extract_macro_features(actor, observations, batch_size=2)
    assert features.shape == (3, MACRO_FEATURE_SIZE)
    assert MACRO_FEATURE_SIZE == (
        MACRO_RAW_OBSERVATION_SIZE + MACRO_GRAPH_SUMMARY_SIZE
        + MACRO_ACTOR_LOGIT_SIZE + MACRO_CRITIC_FEATURE_SIZE
    )
    assert actor.training is True
    assert all(torch.equal(state_before[name], value)
               for name, value in actor.state_dict().items())
