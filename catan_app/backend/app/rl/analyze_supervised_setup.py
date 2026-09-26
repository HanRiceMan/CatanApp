"""同じ盤面・席条件で行った初期配置方式の対局差を対応付きで集計する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def paired_effect(candidate: dict, baseline: dict, field: str, *, seed: int = 42) -> dict:
    left, right = candidate["episodes"], baseline["episodes"]
    if len(left) != len(right) or not left:
        raise ValueError("同数の対局が必要です。")
    if any((a["seed"], a["learner_id"], a["learner_seat"]) !=
           (b["seed"], b["learner_id"], b["learner_seat"]) for a, b in zip(left, right)):
        raise ValueError("盤面seed・プレイヤー・席が対応していません。")
    differences = np.asarray([float(a[field]) - float(b[field]) for a, b in zip(left, right)])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(10_000, len(differences)))
    samples = differences[indices].mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"games": len(differences), "candidate_minus_heuristic": float(differences.mean()),
            "bootstrap_95_percent_interval": [float(low), float(high)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--heuristic-count", type=int, choices=range(4), required=True)
    args = parser.parse_args()
    if Path(args.experiment_id).name != args.experiment_id or args.experiment_id in {".", ".."}:
        parser.error("experiment-idはフォルダ名だけで指定してください。")
    output = Path(__file__).resolve().parents[2] / "experiments" / args.experiment_id
    suffix = f"H{args.heuristic_count}_R{3-args.heuristic_count}"
    baseline = json.loads((output / f"heuristic_{suffix}.json").read_text(encoding="utf-8"))
    result = {}
    for name in ("supervised_mlp", "supervised_shared", "greedy_pips"):
        candidate = json.loads((output / f"{name}_{suffix}.json").read_text(encoding="utf-8"))
        result[name] = {
            "win_rate_difference": paired_effect(candidate, baseline, "learner_won"),
            "average_score_difference": paired_effect(candidate, baseline, "learner_score"),
        }
    path = output / "paired_comparison.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
