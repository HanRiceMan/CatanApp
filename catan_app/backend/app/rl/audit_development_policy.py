"""発展購入残差が階層方策の決定を変える強さを、収集済み局面で監査する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO

from .action_families import ACTION_FAMILY_INDEX
from .action_space import ACTION_SPACE_SIZE
from .hierarchical_distribution import ACTION_FAMILIES, ACTION_FAMILY_IDS


def audit(source, candidate, data, strengths):
    observations = data["observations"]
    masks = data["masks"].bool()
    reserve = data["reserve"].bool()
    family_ids = torch.as_tensor(ACTION_FAMILY_IDS)
    family_legal = torch.stack([
        masks[:, family_ids == family].any(dim=1)
        for family in range(len(ACTION_FAMILIES))
    ], dim=1)
    negative = torch.tensor(-1e8)
    development = ACTION_FAMILY_INDEX["development_purchase"]
    other_family = torch.arange(len(ACTION_FAMILIES)) != development

    with torch.no_grad():
        source_logits = source.policy.mlp_extractor.forward_actor(observations)
        source_family = source_logits[:, ACTION_SPACE_SIZE:]
        legal_source = torch.where(family_legal, source_family, negative)
        source_choice = legal_source.argmax(dim=1)
        actor = candidate.policy.mlp_extractor
        saved_strength = float(actor.development_strength)
        if saved_strength <= 0:
            raise ValueError("development_strengthは正である必要があります。")
        # 保存checkpointの強さによらず、strength=1相当へ戻して比較する。
        correction = actor.development_correction(observations) / saved_strength
        other_best = torch.where(
            family_legal & other_family, source_family, negative
        ).max(dim=1).values
        margin = source_family[:, development] - other_best

    selected_reserve = reserve & (source_choice == development)
    result = {
        "states": len(observations),
        "reserve_states": int(reserve.sum()),
        "source_development_on_reserve": int(selected_reserve.sum()),
        "candidate_saved_strength": saved_strength,
        "correction_on_reserve": {
            "mean": float(correction[reserve].mean()),
            "minimum": float(correction[reserve].min()),
            "maximum": float(correction[reserve].max()),
        },
        "source_development_margin_on_reserve": {
            "mean": float(margin[selected_reserve].mean()),
            "minimum": float(margin[selected_reserve].min()),
            "maximum": float(margin[selected_reserve].max()),
        },
        "strengths": [],
    }
    for strength in strengths:
        adjusted = legal_source.clone()
        adjusted[:, development] += correction * strength
        choice = adjusted.argmax(dim=1)
        result["strengths"].append({
            "strength": strength,
            "development_on_reserve": int(((choice == development) & reserve).sum()),
            "changed_all": int((choice != source_choice).sum()),
            "changed_on_reserve": int(((choice != source_choice) & reserve).sum()),
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.5, 1, 2, 3, 4, 6])
    args = parser.parse_args()
    source = MaskablePPO.load(str(args.source), device="cpu")
    candidate = MaskablePPO.load(str(args.candidate), device="cpu")
    data = torch.load(args.data, weights_only=True)
    print(json.dumps(audit(source, candidate, data, args.strengths),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
