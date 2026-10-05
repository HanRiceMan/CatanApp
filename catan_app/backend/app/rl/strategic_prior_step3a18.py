"""Frozen, family-symmetric prior transforms for Step 3A-18 diagnostics."""

from __future__ import annotations

from collections import defaultdict
from math import exp, log
from typing import Any

import numpy as np

from .strategic_action_surface import FAMILIES


PRIOR_SPECS = ("native_t2", "rank_b0.5", "rank_b1.0", "rank_b1.5",
               "gap_c1.5", "gap_c2.0", "gap_c3.0", "mix_e0.05", "mix_e0.10")


def _arrays(native: list[float] | np.ndarray, mask: tuple[bool, ...]):
    raw = np.asarray(native, dtype=np.float64)
    available = np.asarray(mask, dtype=bool)
    if raw.shape != (len(FAMILIES),) or available.shape != raw.shape or not available.any():
        raise ValueError("Expected six native logits and nonempty candidate mask")
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite native family logits")
    return raw, available


def prior_scores(native: list[float] | np.ndarray,
                 mask: tuple[bool, ...], spec: str) -> np.ndarray:
    """Return six unmasked scores; the Gate applies the unchanged legal mask."""
    raw, available = _arrays(native, mask)
    if spec == "native_t2":
        return raw / 2.0
    if spec.startswith("rank_b"):
        beta = float(spec.removeprefix("rank_b"))
        if beta <= 0:
            raise ValueError("Rank beta must be positive")
        # Stable ties: lower fixed family index comes first, never Planner action.
        order = sorted(np.flatnonzero(available), key=lambda index: (-raw[index], index))
        ranks = np.zeros(len(FAMILIES), dtype=np.float64)
        for rank, index in enumerate(order):
            ranks[index] = -beta * rank
        return ranks
    if spec.startswith("gap_c"):
        cap = float(spec.removeprefix("gap_c"))
        if cap <= 0:
            raise ValueError("Gap cap must be positive")
        return np.maximum(raw - raw[available].max(), -cap)
    if spec.startswith("mix_e"):
        epsilon = float(spec.removeprefix("mix_e"))
        if not 0 < epsilon < 1:
            raise ValueError("Mixture epsilon must lie in (0,1)")
        native_p = prior_probabilities(raw, tuple(available), "native_t2")
        probabilities = (1 - epsilon) * native_p
        probabilities[available] += epsilon / int(available.sum())
        # log(p) is a single Categorical distribution, not an epsilon override.
        return np.log(np.maximum(probabilities, 1e-38))
    raise ValueError(f"Unknown prior specification: {spec}")


def prior_probabilities(native: list[float] | np.ndarray,
                        mask: tuple[bool, ...], spec: str) -> np.ndarray:
    _, available = _arrays(native, mask)
    scores = prior_scores(native, mask, spec)
    selected = scores[available] - scores[available].max()
    weights = np.exp(selected)
    result = np.zeros(len(FAMILIES), dtype=np.float64)
    result[available] = weights / weights.sum()
    return result


def _stats(values) -> dict[str, Any]:
    data = np.asarray(list(values), dtype=np.float64)
    if data.size == 0:
        return {"n": 0}
    return {"n": int(data.size), "mean": float(data.mean()),
            "median": float(np.median(data)), "p10": float(np.percentile(data, 10)),
            "p90": float(np.percentile(data, 90)), "min": float(data.min()),
            "max": float(data.max())}


def offline_prior_audit(shadow_rows: list[dict[str, Any]],
                        dev_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare priors on identical Step 3A-13 states and Step 3A-17 Dev states."""
    result: dict[str, Any] = {"shadow_states": len(shadow_rows),
                              "dev_selected_states": len(dev_rows), "priors": {}}
    for spec in PRIOR_SPECS:
        entropies = []
        top_prob = []
        second_prob = []
        alternative_mass = []
        effective_count = []
        by_count: dict[str, list[float]] = defaultdict(list)
        family_available: dict[str, list[float]] = defaultdict(list)
        retention = 0
        for row in shadow_rows:
            mask = tuple(row["candidate_mask"])
            native = row["prior"]["raw_scores"]["native_family_logits"]
            probabilities = prior_probabilities(native, mask, spec)
            native_top = int(np.argmax(prior_probabilities(native, mask, "native_t2")))
            current_top = int(np.argmax(probabilities))
            retention += current_top == native_top
            available = sorted(probabilities[np.asarray(mask)], reverse=True)
            entropy = -sum(float(p) * log(float(p)) for p in probabilities if p > 0)
            entropies.append(entropy)
            top_prob.append(available[0])
            second_prob.append(available[1] if len(available) > 1 else 0.0)
            alternative_mass.append(1 - available[0])
            effective_count.append(exp(entropy))
            by_count["4+" if len(available) >= 4 else str(len(available))].append(entropy)
            for index, present in enumerate(mask):
                if present:
                    family_available[FAMILIES[index]].append(float(probabilities[index]))
        dev_p = [float(prior_probabilities(
            row["native_strategic6_logits"], tuple(row["candidate_mask"]), spec)[3])
                 for row in dev_rows]
        result["priors"][spec] = {
            "entropy": _stats(entropies), "top_probability": _stats(top_prob),
            "second_probability": _stats(second_prob),
            "alternative_mass": _stats(alternative_mass),
            "effective_action_count": _stats(effective_count),
            "entropy_by_candidate_count": {key: _stats(values)
                                            for key, values in by_count.items()},
            "family_probability_when_available": {name: _stats(values)
                                                   for name, values in family_available.items()},
            "native_top_retention": {"count": retention, "total": len(shadow_rows),
                                     "rate": retention / len(shadow_rows)},
            "dev_selected_state_probability": {
                **_stats(dev_p), ">0.9": sum(p > .9 for p in dev_p),
                ">0.7": sum(p > .7 for p in dev_p),
                ">0.5": sum(p > .5 for p in dev_p)},
        }
    return result
