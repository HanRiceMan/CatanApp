"""Step 3A-11: repeat CITY_ONLY paired rollouts without training a policy."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from math import sqrt
from statistics import mean, median, stdev, variance
from typing import Any

import numpy as np


NEAR_TIE = 0.01
CONTEXT_FIELDS = (
    "risk.hand_size", "risk.discard_exposure", "robber.blocked_pips",
    "city.best_production_gain", "settlement.eta",
    "settlement_shortage_total", "road.acquisition_gap",
    "knight.plays_needed", "score", "site_count",
    "knight.development_affordable", "risk.bank_trade_options",
)


def future_seed(state_id: str, index: int) -> int:
    """Stable, disjoint diagnostic stream; BUILD and DEFER share this seed."""
    if index not in range(16):
        raise ValueError("Future seed index must be 0..15")
    token = sha256(state_id.encode("utf-8")).digest()
    return 31_000_000 + int.from_bytes(token[:8], "big") % 100_000_000 + index * 1_000_003


def _distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    a, b = left["context"], right["context"]
    return (3.0 * (left["profile"] != right["profile"])
            + 1.5 * (left["seat"] != right["seat"])
            + 2.0 * (left["defer_action_family"] != right["defer_action_family"])
            + 0.5 * abs(a["score"] - b["score"])
            + 0.7 * abs(a["site_count"] - b["site_count"])
            + 0.6 * abs(a["city_count"] - b["city_count"])
            + 0.2 * abs(a["risk.hand_size"] - b["risk.hand_size"]))


def select_states(records: list[dict[str, Any]], *, controls: int = 16) -> dict[str, Any]:
    """Every Champion DEFER, matched non-outlier BUILD, plus five audit probes."""
    defer = sorted((r for r in records if not r["champion_build"]),
                   key=lambda r: r["state_id"])
    build = [r for r in records if r["champion_build"]]
    if len(defer) != 31 or controls not in range(15, 26):
        raise ValueError("Unexpected Step 3A-10 source distribution")
    selected = {r["state_id"]: "CHAMPION_DEFER" for r in defer}
    available = {r["state_id"]: r for r in build
                 if abs(r["delta"]["total"]) <= 0.5}
    # Balance the scarce control sample by seat/profile before nearest-neighbor
    # matching. Hashes only break ties; no rollout outcome guides this choice.
    targets = list(defer)
    target_seats: Counter[int] = Counter()
    target_profiles: Counter[str] = Counter()
    matches: list[dict[str, Any]] = []
    for _ in range(controls):
        target = min(targets,
                     key=lambda r: (target_seats[r["seat"]],
                                    target_profiles[r["profile"]],
                                    sha256(r["state_id"].encode()).digest()))
        targets.remove(target)
        target_seats[target["seat"]] += 1
        target_profiles[target["profile"]] += 1
        control = min(available.values(),
                      key=lambda r: (_distance(target, r), r["state_id"]))
        matches.append({"defer_state_id": target["state_id"],
                        "build_state_id": control["state_id"],
                        "distance": _distance(target, control)})
        selected[control["state_id"]] = "MATCHED_CHAMPION_BUILD"
        del available[control["state_id"]]
    remaining = [r for r in build if r["state_id"] not in selected]
    positive = sorted(remaining, key=lambda r: -r["delta"]["total"])[:2]
    negative = sorted((r for r in remaining if r not in positive),
                      key=lambda r: r["delta"]["total"])[:2]
    used = {r["state_id"] for r in positive + negative}
    near = min((r for r in remaining if r["state_id"] not in used
                and abs(r["delta"]["total"]) <= NEAR_TIE),
               key=lambda r: (abs(r["delta"]["total"]), r["state_id"]))
    for group, label in ((positive, "SINGLE_BUILD_OUTLIER"),
                         (negative, "SINGLE_DEFER_OUTLIER"),
                         ([near], "SINGLE_NEAR_TIE")):
        selected.update({r["state_id"]: label for r in group})
    return {"state_categories": selected, "control_matches": matches,
            "source_count": len(records), "selected_count": len(selected),
            "selection_definition": "all 31 Champion DEFER + 16 matched, non-outlier BUILD + 2/2/1 probes"}


def diagnostic_ci(values: list[float], state_id: str,
                  *, resamples: int = 2000) -> tuple[float, float]:
    """Fixed-seed percentile bootstrap CI, diagnostic only (not a label)."""
    array = np.asarray(values, dtype=np.float64)
    if len(array) < 2:
        return (float("nan"), float("nan"))
    token = sha256((state_id + "-ci").encode()).digest()
    rng = np.random.default_rng(int.from_bytes(token[:8], "big"))
    indices = rng.integers(0, len(array), size=(resamples, len(array)))
    means = array[indices].mean(axis=1)
    return (float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def _mean_or_none(values) -> float | None:
    collected = list(values)
    return float(mean(collected)) if collected else None


def state_stats(records: list[dict[str, Any]], state_id: str) -> dict[str, Any]:
    values = [float(r["delta_q"]) for r in records]
    if not values:
        raise ValueError("State has no continuation")
    array = np.asarray(values)
    positive = int((array > NEAR_TIE).sum())
    negative = int((array < -NEAR_TIE).sum())
    ties = len(values) - positive - negative
    non_ties = positive + negative
    low, high = diagnostic_ci(values, state_id)
    return {
        "n": len(values), "mean_delta_q": float(mean(values)),
        "median_delta_q": float(median(values)),
        "sd_delta_q": float(stdev(values)) if len(values) > 1 else None,
        "se_delta_q": float(stdev(values) / sqrt(len(values))) if len(values) > 1 else None,
        "min_delta_q": float(array.min()), "max_delta_q": float(array.max()),
        "p25_delta_q": float(np.percentile(array, 25)),
        "p75_delta_q": float(np.percentile(array, 75)),
        "build_better": positive, "defer_better": negative, "near_tie": ties,
        "sign_consistency_non_ties": (max(positive, negative) / non_ties
                                      if non_ties else None),
        "mean_near_zero": abs(mean(values)) <= NEAR_TIE,
        "ci95_bootstrap_low": low, "ci95_bootstrap_high": high,
        "ci95_width": high - low,
        "ci95_crosses_zero": low <= 0 <= high,
        "ci95_excludes_practical_tie_band": low > NEAR_TIE or high < -NEAR_TIE,
        "build_win_rate": mean(float(r["build_win"]) for r in records),
        "defer_win_rate": mean(float(r["defer_win"]) for r in records),
        "final_vp_delta_mean": mean(r["build_final_vp"] - r["defer_final_vp"]
                                     for r in records),
        "mean_delta_terminal": mean(r["delta_terminal"] for r in records),
        "mean_delta_vp": mean(r["delta_vp"] for r in records),
    }


def choose_phase2(stats: dict[str, dict[str, Any]],
                  source: dict[str, dict[str, Any]], *, limit: int = 12) -> list[str]:
    """Only high-uncertainty states; ensure family coverage before filling."""
    candidates = [(sid, row) for sid, row in stats.items()
                  if row["n"] == 8 and row["se_delta_q"] is not None
                  and row["se_delta_q"] >= 0.02
                  and (row["ci95_crosses_zero"]
                       or row["sign_consistency_non_ties"] is None
                       or row["sign_consistency_non_ties"] < 0.75)]
    def priority(item):
        sid, row = item
        return (row["se_delta_q"] / (abs(row["mean_delta_q"]) + 0.02)
                + (0.5 if not source[sid]["champion_build"] else 0.0))
    candidates.sort(key=lambda item: (-priority(item), item[0]))
    chosen: list[str] = []
    families = sorted({source[sid]["defer_action_family"] for sid, _ in candidates})
    for family in families:
        match = next((sid for sid, _ in candidates
                      if source[sid]["defer_action_family"] == family), None)
        if match is not None and match not in chosen:
            chosen.append(match)
    for sid, _ in candidates:
        if len(chosen) >= limit:
            break
        if sid not in chosen:
            chosen.append(sid)
    return chosen[:limit]


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2 + 1
        start = stop
    return ranks


def _corr(left: list[float], right: list[float]) -> tuple[float | None, float | None]:
    a, b = np.asarray(left), np.asarray(right)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None, None
    return (float(np.corrcoef(a, b)[0, 1]),
            float(np.corrcoef(_rank(a), _rank(b))[0, 1]))


def _describe(values) -> dict[str, float | int | None]:
    collected = [float(value) for value in values if value is not None]
    if not collected:
        return {"count": 0}
    return {"count": len(collected), "mean": mean(collected),
            "median": median(collected),
            "p25": float(np.percentile(collected, 25)),
            "p75": float(np.percentile(collected, 75)),
            "min": min(collected), "max": max(collected)}


def summarize(records: list[dict[str, Any]],
              source_rows: list[dict[str, Any]],
              selected: dict[str, str]) -> dict[str, Any]:
    source = {r["state_id"]: r for r in source_rows}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["state_id"]].append(record)
    if set(grouped) != set(selected):
        raise ValueError("Incomplete selected-state dataset")
    if any(len({r["future_seed_index"] for r in rows}) != len(rows)
           for rows in grouped.values()):
        raise ValueError("Duplicate continuation index")
    stats = {sid: state_stats(rows, sid) for sid, rows in grouped.items()}
    phase1 = [state_stats(
        [r for r in rows if r["future_seed_index"] < 8], sid)
        for sid, rows in grouped.items()]
    within = [row["sd_delta_q"] ** 2 for row in phase1]
    phase1_means = [row["mean_delta_q"] for row in phase1]
    within_mean = mean(within)
    between_observed = variance(phase1_means)
    measurement_noise = within_mean / 8
    between_corrected = max(0.0, between_observed - measurement_noise)
    ids = sorted(selected)
    single = [source[sid]["delta"]["total"] for sid in ids]
    multi = [stats[sid]["mean_delta_q"] for sid in ids]
    pearson, spearman = _corr(single, multi)
    sign_agree = sum((s > NEAR_TIE) == (m > NEAR_TIE)
                     for s, m in zip(single, multi)
                     if abs(s) > NEAR_TIE and abs(m) > NEAR_TIE)
    sign_compared = sum(abs(s) > NEAR_TIE and abs(m) > NEAR_TIE
                        for s, m in zip(single, multi))
    category_summary = {}
    for name, predicate in (("CHAMPION_DEFER", lambda sid: not source[sid]["champion_build"]),
                            ("CHAMPION_BUILD", lambda sid: source[sid]["champion_build"])):
        subset = [stats[sid] for sid in ids if predicate(sid)]
        category_summary[name] = {
            "states": len(subset),
            "mean_delta_q": mean(r["mean_delta_q"] for r in subset),
            "build_mean_positive": sum(r["mean_delta_q"] > NEAR_TIE for r in subset),
            "defer_mean_negative": sum(r["mean_delta_q"] < -NEAR_TIE for r in subset),
            "mean_near_zero": sum(r["mean_near_zero"] for r in subset),
            "ci_excludes_zero": sum(not r["ci95_crosses_zero"] for r in subset),
            "ci_excludes_practical_tie_band": sum(
                r["ci95_excludes_practical_tie_band"] for r in subset),
            "sign_consistency_mean": _mean_or_none(
                r["sign_consistency_non_ties"] for r in subset
                if r["sign_consistency_non_ties"] is not None),
            "ci_width_median": median(r["ci95_width"] for r in subset),
        }
    families = {}
    for family in sorted({source[sid]["defer_action_family"] for sid in ids}):
        subset = [stats[sid] for sid in ids if source[sid]["defer_action_family"] == family]
        families[family] = {
            "states": len(subset),
            "mean_of_state_mean_delta_q": mean(r["mean_delta_q"] for r in subset),
            "median_of_state_mean_delta_q": median(r["mean_delta_q"] for r in subset),
            "build_mean_positive": sum(r["mean_delta_q"] > NEAR_TIE for r in subset),
            "defer_mean_negative": sum(r["mean_delta_q"] < -NEAR_TIE for r in subset),
            "near_zero": sum(r["mean_near_zero"] for r in subset),
            "mean_sign_consistency": _mean_or_none(
                r["sign_consistency_non_ties"] for r in subset
                if r["sign_consistency_non_ties"] is not None),
        }
    alignment = Counter()
    for row in stats.values():
        total = row["mean_delta_q"]
        terminal = row["mean_delta_terminal"]
        win_delta = row["build_win_rate"] - row["defer_win_rate"]
        if abs(total) <= NEAR_TIE or abs(terminal) <= NEAR_TIE:
            alignment["reward_vs_discounted_terminal_near_zero"] += 1
        elif total > 0 and terminal > 0:
            alignment["A_both_build"] += 1
        elif total < 0 and terminal < 0:
            alignment["B_both_defer"] += 1
        elif total > 0:
            alignment["C_reward_build_terminal_defer"] += 1
        else:
            alignment["D_reward_defer_terminal_build"] += 1
        if abs(total) > NEAR_TIE and abs(win_delta) > 1e-9:
            alignment["reward_vs_win_rate_opposite" if total * win_delta < 0
                      else "reward_vs_win_rate_same"] += 1
        else:
            alignment["reward_vs_win_rate_unresolved"] += 1
    context_groups = {}
    for name, predicate in (("BUILD", lambda value: value > NEAR_TIE),
                            ("DEFER", lambda value: value < -NEAR_TIE),
                            ("NEAR_ZERO", lambda value: abs(value) <= NEAR_TIE)):
        group_ids = [sid for sid in ids if predicate(stats[sid]["mean_delta_q"])]
        context_groups[name] = {
            "states": len(group_ids),
            "fields": {field: {"count": len(values),
                               "mean": mean(values) if values else None,
                               "median": median(values) if values else None}
                       for field in CONTEXT_FIELDS
                       for values in [[float(source[sid]["context"][field])
                                       for sid in group_ids
                                       if source[sid]["context"].get(field) is not None]]}}
    outliers = sorted(ids, key=lambda sid: abs(source[sid]["delta"]["total"]),
                      reverse=True)[:10]
    return {
        "states": len(ids), "continuations": len(records),
        "future_seed_count_distribution": dict(Counter(r["n"] for r in stats.values())),
        "profile": dict(Counter(source[sid]["profile"] for sid in ids)),
        "seat": dict(Counter(source[sid]["seat"] for sid in ids)),
        "selection_category": dict(Counter(selected.values())),
        "state_stats": stats,
        "phase1_variance": {
            "within_mean_sample_variance": within_mean,
            "within_median_sd": median(sqrt(v) for v in within),
            "between_observed_state_mean_variance": between_observed,
            "expected_mean_measurement_noise": measurement_noise,
            "between_noise_corrected_estimate": between_corrected,
            "corrected_between_to_within_ratio": between_corrected / within_mean
            if within_mean else None,
            "icc_like": between_corrected / (between_corrected + within_mean)
            if between_corrected + within_mean else None,
            "definition": "Balanced first 8 seeds only; max(0, Var(state means) - mean(within variance)/8) / (corrected between + mean within variance). Exploratory, selected states are not random."},
        "state_mean_distribution": {
            "mean": mean(multi), "median": median(multi),
            "sd": stdev(multi), "p25": float(np.percentile(multi, 25)),
            "p75": float(np.percentile(multi, 75)),
            "min": min(multi), "max": max(multi),
            "build_positive": sum(m > NEAR_TIE for m in multi),
            "defer_negative": sum(m < -NEAR_TIE for m in multi),
            "near_zero": sum(abs(m) <= NEAR_TIE for m in multi),
            "sd_median": median(r["sd_delta_q"] for r in stats.values()),
            "se_median": median(r["se_delta_q"] for r in stats.values()),
            "ci_width_median": median(r["ci95_width"] for r in stats.values()),
            "ci_crosses_zero": sum(r["ci95_crosses_zero"] for r in stats.values()),
            "ci_excludes_practical_tie_band": sum(
                r["ci95_excludes_practical_tie_band"] for r in stats.values()),
            "sign_consistency_median": _describe(
                r["sign_consistency_non_ties"] for r in stats.values())["median"]
            if any(r["sign_consistency_non_ties"] is not None
                   for r in stats.values()) else None,
        },
        "state_sd_distribution": _describe(r["sd_delta_q"] for r in stats.values()),
        "state_se_distribution": _describe(r["se_delta_q"] for r in stats.values()),
        "ci_width_distribution": _describe(r["ci95_width"] for r in stats.values()),
        "sign_consistency_distribution": _describe(
            r["sign_consistency_non_ties"] for r in stats.values()),
        "champion_groups": category_summary, "defer_families": families,
        "reward_alignment": dict(alignment), "context_groups": context_groups,
        "single_vs_multi": {
            "pearson": pearson, "spearman": spearman,
            "sign_agree_non_near_tie": sign_agree,
            "sign_compared_non_near_tie": sign_compared,
            "top_single_abs_outliers": [{"state_id": sid,
                "single_delta_q": source[sid]["delta"]["total"],
                "multi_mean_delta_q": stats[sid]["mean_delta_q"],
                "single_abs_rank": rank,
                "multi_abs_rank": sorted(ids, key=lambda x: abs(stats[x]["mean_delta_q"]),
                                         reverse=True).index(sid) + 1}
                for rank, sid in enumerate(outliers, 1)],
        },
    }
