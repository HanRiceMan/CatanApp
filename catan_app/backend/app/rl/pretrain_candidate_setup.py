"""候補共有型PPOの初期開拓地Headを教師ありでウォームスタートする。"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from torch import nn

from .action_space import action_to_id
from app.domain.actions import PlaceInitialRoadAction, PlaceInitialSettlementAction
from .candidate_policy import CandidateMaskablePolicy
from .compare import PolicyCompatibleHeuristicAgent
from .env import CatanEnv, CatanEnvConfig
from .setup_supervised import collect_examples


INITIAL_START = action_to_id(PlaceInitialSettlementAction(0))
INITIAL_ROAD_START = action_to_id(PlaceInitialRoadAction(0))


def _logits(policy: CandidateMaskablePolicy, examples: dict[str, np.ndarray]) -> torch.Tensor:
    device = next(policy.parameters()).device
    observation = torch.from_numpy(examples["observations"]).to(device)
    mask = torch.from_numpy(examples["masks"]).to(device)
    logits = policy.mlp_extractor.forward_actor(observation)[:, INITIAL_START:INITIAL_START + 54]
    return logits.masked_fill(~mask, -1e9)


def _metrics(policy: CandidateMaskablePolicy, examples: dict[str, np.ndarray]) -> dict:
    policy.eval()
    with torch.no_grad():
        selected = _logits(policy, examples).argmax(dim=1).cpu().numpy()
    labels = examples["labels"]
    pips = examples["pips"]
    best = np.where(examples["masks"], pips, -1).max(axis=1)
    selected_pips = pips[np.arange(len(selected)), selected]
    return {"examples": len(selected), "teacher_agreement": float(np.mean(selected == labels)),
            "mean_selected_pips": float(np.mean(selected_pips)),
            "mean_pips_regret": float(np.mean(best - selected_pips)),
            "first_vertex_most_common_share": float(np.bincount(
                selected[examples["placements"] == 1], minlength=54).max() /
                np.count_nonzero(examples["placements"] == 1))}


def pretrain_initial_settlement(policy: CandidateMaskablePolicy, *, train_boards: int = 600,
                                validation_boards: int = 150, test_boards: int = 150,
                                seed_start: int = 31001, training_seed: int = 42,
                                max_epochs: int = 40) -> dict:
    """交差点Headだけを合法候補内の教師Actionで学習する。PPO報酬は使わない。"""
    if min(train_boards, validation_boards, test_boards, max_epochs) <= 0:
        raise ValueError("盤面数とepochは正の整数にしてください。")
    torch.manual_seed(training_seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    ranges = {
        "train": range(seed_start, seed_start + train_boards),
        "validation": range(seed_start + train_boards, seed_start + train_boards + validation_boards),
        "test": range(seed_start + train_boards + validation_boards,
                      seed_start + train_boards + validation_boards + test_boards),
    }
    datasets = {name: collect_examples(seeds) for name, seeds in ranges.items()}
    initial = _metrics(policy, datasets["test"])
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    rng = np.random.default_rng(training_seed)
    best_loss = float("inf")
    best_state = None
    history = []
    stale = 0
    for epoch in range(1, max_epochs + 1):
        policy.train()
        for indices in np.array_split(rng.permutation(len(datasets["train"]["labels"])),
                                      int(np.ceil(len(datasets["train"]["labels"]) / 128))):
            batch = {key: value[indices] for key, value in datasets["train"].items()}
            labels = torch.from_numpy(batch["labels"]).to(next(policy.parameters()).device)
            loss = nn.functional.cross_entropy(_logits(policy, batch), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        policy.eval()
        with torch.no_grad():
            validation_loss = float(nn.functional.cross_entropy(
                _logits(policy, datasets["validation"]),
                torch.from_numpy(datasets["validation"]["labels"]).to(next(policy.parameters()).device)))
        history.append({"epoch": epoch, "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best_state = deepcopy(policy.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= 8:
            break
    assert best_state is not None
    policy.load_state_dict(best_state)
    policy.eval()
    return {"seed_ranges": {name: [seeds.start, seeds.stop - 1] for name, seeds in ranges.items()},
            "training_seed": training_seed, "best_epoch": min(history, key=lambda row: row["validation_loss"])["epoch"],
            "initial_test": initial, "after_test": _metrics(policy, datasets["test"]),
            "history": history}


def _collect_full_setup_examples(seeds: range) -> dict[str, dict[str, np.ndarray]]:
    """学習者の初期開拓地2件・初期道路2件をHeuristic教師から収集する。"""
    rows = {"settlement": {"observations": [], "masks": [], "labels": []},
            "road": {"observations": [], "masks": [], "labels": []}}
    teacher = PolicyCompatibleHeuristicAgent()
    for index, seed in enumerate(seeds):
        player_id = index % 4 + 1
        env = CatanEnv(CatanEnvConfig(learning_player_id=player_id,
                                     policy_compatible_opponents=True,
                                     observation_version="v2"))
        try:
            observation, info = env.reset(seed=seed)
            counts = {"settlement": 0, "road": 0}
            for _ in range(24):
                game = env.game
                if game is None:
                    raise AssertionError("盤面がありません。")
                action = teacher.select_action(game, player_id)
                action_id = action_to_id(action)
                if not info["action_mask"][action_id]:
                    raise AssertionError(f"教師の非法Action: seed={seed}")
                if isinstance(action, PlaceInitialSettlementAction):
                    kind, start, count, label = "settlement", INITIAL_START, 54, action.vertex_id
                elif isinstance(action, PlaceInitialRoadAction):
                    kind, start, count, label = "road", INITIAL_ROAD_START, 72, action.edge_id
                else:
                    kind = ""
                if kind:
                    mask = info["action_mask"][start:start + count].copy()
                    if not mask[label]:
                        raise AssertionError(f"教師の初期{kind}が合法候補にありません。")
                    rows[kind]["observations"].append(observation.copy())
                    rows[kind]["masks"].append(mask)
                    rows[kind]["labels"].append(label)
                    counts[kind] += 1
                observation, _, terminated, truncated, info = env.step(action_id)
                if counts == {"settlement": 2, "road": 2}:
                    break
                if terminated or truncated:
                    raise AssertionError("初期配置の途中で対局が終了しました。")
            if counts != {"settlement": 2, "road": 2}:
                raise AssertionError(f"seed={seed}: 初期配置4操作を収集できませんでした。")
        finally:
            env.close()
    return {kind: {
        "observations": np.stack(values["observations"]).astype(np.float32),
        "masks": np.stack(values["masks"]).astype(np.bool_),
        "labels": np.asarray(values["labels"], dtype=np.int64),
    } for kind, values in rows.items()}


def _setup_logits(policy: CandidateMaskablePolicy, examples: dict[str, np.ndarray],
                  *, start: int, count: int) -> torch.Tensor:
    device = next(policy.parameters()).device
    observations = torch.from_numpy(examples["observations"]).to(device)
    masks = torch.from_numpy(examples["masks"]).to(device)
    logits = policy.mlp_extractor.forward_actor(observations)[:, start:start + count]
    return logits.masked_fill(~masks, -1e9)


def _setup_metrics(policy: CandidateMaskablePolicy,
                   datasets: dict[str, dict[str, np.ndarray]]) -> dict:
    policy.eval()
    result = {}
    for kind, start, count in (("settlement", INITIAL_START, 54),
                               ("road", INITIAL_ROAD_START, 72)):
        examples = datasets[kind]
        with torch.no_grad():
            selected = _setup_logits(policy, examples, start=start, count=count).argmax(dim=1).cpu().numpy()
        result[kind] = {
            "examples": len(selected),
            "teacher_agreement": float(np.mean(selected == examples["labels"])),
            "distinct_selected": int(len(np.unique(selected))),
        }
    return result


def pretrain_initial_setup_heads(policy: CandidateMaskablePolicy, *, train_boards: int = 600,
                                 validation_boards: int = 150, test_boards: int = 150,
                                 seed_start: int = 67_001, training_seed: int = 42,
                                 max_epochs: int = 40) -> dict:
    """共有表現を固定し、初期開拓地・道路headだけをHeuristic模倣で学習する。"""
    if min(train_boards, validation_boards, test_boards, max_epochs) <= 0:
        raise ValueError("盤面数とepochは正の整数にしてください。")
    torch.manual_seed(training_seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    ranges = {
        "train": range(seed_start, seed_start + train_boards),
        "validation": range(seed_start + train_boards, seed_start + train_boards + validation_boards),
        "test": range(seed_start + train_boards + validation_boards,
                      seed_start + train_boards + validation_boards + test_boards),
    }
    datasets = {name: _collect_full_setup_examples(seeds) for name, seeds in ranges.items()}
    before = _setup_metrics(policy, datasets["test"])
    original_grad = {name: parameter.requires_grad for name, parameter in policy.named_parameters()}
    trainable = ("mlp_extractor.initial_settlement_head", "mlp_extractor.initial_road_head")
    for name, parameter in policy.named_parameters():
        parameter.requires_grad_(name.startswith(trainable))
    parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    if not parameters:
        raise TypeError("候補方策の初期配置headが見つかりません。")
    optimizer = torch.optim.Adam(parameters, lr=3e-4)
    rng = np.random.default_rng(training_seed)
    best_loss = float("inf")
    best_state = None
    history = []
    stale = 0
    try:
        for epoch in range(1, max_epochs + 1):
            policy.train()
            losses = []
            for kind, start, count in (("settlement", INITIAL_START, 54),
                                       ("road", INITIAL_ROAD_START, 72)):
                examples = datasets["train"][kind]
                for indices in np.array_split(rng.permutation(len(examples["labels"])),
                                              int(np.ceil(len(examples["labels"]) / 128))):
                    batch = {key: value[indices] for key, value in examples.items()}
                    labels = torch.from_numpy(batch["labels"]).to(next(policy.parameters()).device)
                    loss = nn.functional.cross_entropy(
                        _setup_logits(policy, batch, start=start, count=count), labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach()))
            policy.eval()
            validation_losses = []
            with torch.no_grad():
                for kind, start, count in (("settlement", INITIAL_START, 54),
                                           ("road", INITIAL_ROAD_START, 72)):
                    examples = datasets["validation"][kind]
                    labels = torch.from_numpy(examples["labels"]).to(next(policy.parameters()).device)
                    validation_losses.append(float(nn.functional.cross_entropy(
                        _setup_logits(policy, examples, start=start, count=count), labels)))
            validation_loss = float(np.mean(validation_losses))
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                            "validation_loss": validation_loss})
            if validation_loss < best_loss - 1e-4:
                best_loss = validation_loss
                best_state = {name: parameter.detach().clone()
                              for name, parameter in policy.state_dict().items()
                              if name.startswith(trainable)}
                stale = 0
            else:
                stale += 1
            if stale >= 8:
                break
        assert best_state is not None
        current = policy.state_dict()
        current.update(best_state)
        policy.load_state_dict(current)
    finally:
        for name, parameter in policy.named_parameters():
            parameter.requires_grad_(original_grad[name])
    policy.eval()
    return {
        "seed_ranges": {name: [seeds.start, seeds.stop - 1] for name, seeds in ranges.items()},
        "training_seed": training_seed,
        "trainable_parameters": list(trainable),
        "best_epoch": min(history, key=lambda row: row["validation_loss"])["epoch"],
        "before_test": before,
        "after_test": _setup_metrics(policy, datasets["test"]),
        "history": history,
    }
