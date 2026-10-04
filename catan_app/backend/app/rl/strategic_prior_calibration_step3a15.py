"""Offline audit of masked frozen family priors; no policy update or action."""

from __future__ import annotations

from collections import Counter
import json
from math import exp, isfinite, log
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from .strategic_action_surface import FAMILIES


TEMPERATURES = (1.0, 1.25, 1.5, 2.0, 3.0)
MIXTURES = (0.05, 0.10, 0.20)


def masked_temperature_probabilities(
    native_logits: Iterable[float], available: Iterable[bool], temperature: float,
) -> tuple[float, ...]:
    """Only the existing native family ranking is softened, never reordered."""
    values, mask = tuple(native_logits), tuple(bool(item) for item in available)
    if (len(values) != len(FAMILIES) or len(mask) != len(FAMILIES)
            or not any(mask) or not isfinite(temperature) or temperature <= 0
            or any(not isfinite(value) for value in values)):
        raise ValueError("Invalid family logits, mask, or temperature")
    maximum = max(values[i] / temperature for i, enabled in enumerate(mask) if enabled)
    weights = tuple(exp(values[i] / temperature - maximum) if enabled else 0.0
                    for i, enabled in enumerate(mask))
    denominator = sum(weights)
    return tuple(weight / denominator for weight in weights)


def masked_mixture_probabilities(
    native_logits: Iterable[float], available: Iterable[bool], epsilon: float,
) -> tuple[float, ...]:
    if not isfinite(epsilon) or not 0 <= epsilon < 1:
        raise ValueError("Invalid mixture epsilon")
    mask = tuple(bool(item) for item in available)
    native = masked_temperature_probabilities(native_logits, mask, 1.0)
    count = sum(mask)
    return tuple((1 - epsilon) * native[i] + epsilon / count if enabled else 0.0
                 for i, enabled in enumerate(mask))


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    low, high = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[low] + (position - low) * (ordered[high] - ordered[low])


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    return {"n": len(values), "mean": mean(values) if values else None,
            "median": median(values) if values else None,
            "p10": _percentile(values, .10), "p25": _percentile(values, .25),
            "p75": _percentile(values, .75), "p90": _percentile(values, .90),
            "min": min(values) if values else None,
            "max": max(values) if values else None}


def _top(values: tuple[float, ...]) -> int:
    return max(range(len(values)), key=lambda index: (values[index], -index))


def _metrics(probabilities: tuple[float, ...], mask: tuple[bool, ...]) -> dict[str, Any]:
    available = sorted((probabilities[index] for index, enabled in enumerate(mask)
                        if enabled), reverse=True)
    entropy = -sum(value * log(value) for value in available if value > 0)
    return {"entropy": entropy, "top_probability": available[0],
            "second_probability": available[1] if len(available) > 1 else None,
            "minimum_available_probability": available[-1],
            "effective_family_count": exp(entropy),
            "alternative_mass": 1 - available[0],
            "top_index": _top(probabilities),
            "candidate_count": len(available)}


def audit_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute T and mixture distributions from action-preceding Shadow states."""
    if not records:
        raise ValueError("No Shadow states")
    variants = {f"T={temperature:g}": ("temperature", temperature)
                for temperature in TEMPERATURES}
    variants.update({f"eps={epsilon:g}": ("mixture", epsilon)
                     for epsilon in MIXTURES})
    result: dict[str, Any] = {}
    for name, (kind, parameter) in variants.items():
        samples: list[dict[str, Any]] = []
        availability: dict[str, list[float]] = {family: [] for family in FAMILIES}
        top_counts: Counter[str] = Counter()
        changed_top = peaked = 0
        for record in records:
            mask = tuple(bool(value) for value in record["candidate_mask"])
            native_logits = tuple(record["native_family_logits"] if
                                  "native_family_logits" in record else record[
                                      "prior"]["raw_scores"]["native_family_logits"])
            native = masked_temperature_probabilities(native_logits, mask, 1.0)
            probabilities = (masked_temperature_probabilities(native_logits, mask, parameter)
                             if kind == "temperature" else
                             masked_mixture_probabilities(native_logits, mask, parameter))
            if abs(sum(probabilities) - 1) > 1e-10 or any(
                    probability != 0 for probability, enabled in zip(
                        probabilities, mask, strict=True) if not enabled):
                raise AssertionError("Masked prior lost normalization")
            metrics = _metrics(probabilities, mask)
            samples.append(metrics)
            changed_top += metrics["top_index"] != _top(native)
            peaked += metrics["top_probability"] >= .999
            top_counts[FAMILIES[metrics["top_index"]]] += 1
            for index, family in enumerate(FAMILIES):
                if mask[index]:
                    availability[family].append(probabilities[index])
        buckets = {
            bucket: [item for item in samples
                     if (str(item["candidate_count"]) if item["candidate_count"] <= 3
                         else "4+") == bucket]
            for bucket in ("2", "3", "4+")}
        result[name] = {
            "kind": kind, "parameter": parameter, "states": len(samples),
            "entropy": _distribution([item["entropy"] for item in samples]),
            "top_probability": _distribution([
                item["top_probability"] for item in samples]),
            "second_probability": _distribution([
                item["second_probability"] for item in samples
                if item["second_probability"] is not None]),
            "minimum_available_probability": _distribution([
                item["minimum_available_probability"] for item in samples]),
            "effective_family_count": _distribution([
                item["effective_family_count"] for item in samples]),
            "alternative_mass": _distribution([
                item["alternative_mass"] for item in samples]),
            "top1_retention": 1 - changed_top / len(samples),
            "fraction_top_probability_at_least_0999": peaked / len(samples),
            "top_family_counts": dict(top_counts),
            "by_candidate_count": {
                bucket: {
                    "states": len(items),
                    "entropy": _distribution([item["entropy"] for item in items]),
                    "alternative_mass": _distribution([
                        item["alternative_mass"] for item in items]),
                } for bucket, items in buckets.items()},
            "family_probability_when_available": {
                family: _distribution(values)
                for family, values in availability.items()},
        }
    return result


def load_step13(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_step14(path: Path) -> list[dict[str, Any]]:
    games = json.loads(path.read_text(encoding="utf-8"))
    return [decision for game in games["stochastic"]
            for decision in game["strategic"]["decisions"]]
