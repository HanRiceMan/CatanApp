"""開拓地の新設・都市化・建設時期を区別する評価指標。称号は合否条件にしない。"""

from statistics import fmean


def construction_summary(report: dict) -> dict:
    episodes = report["episodes"]
    if not episodes:
        raise ValueError("評価局がありません。")
    result = {
        "new_settlements_excluding_setup": fmean(
            e["learner_actions"].get("build_settlement", 0) for e in episodes
        ),
        "city_upgrades": fmean(e["learner_actions"].get("build_city", 0) for e in episodes),
        "final_sites": fmean(
            e["learner_final_pieces"]["settlements"] + e["learner_final_pieces"]["cities"]
            for e in episodes
        ),
        "final_building_points": fmean(
            e["learner_final_pieces"]["settlements"] + 2 * e["learner_final_pieces"]["cities"]
            for e in episodes
        ),
    }
    for name in ("fourth_site", "first_city", "second_city"):
        reached = [e["strategy"][f"{name}_own_turn"] for e in episodes
                   if e["strategy"].get(f"{name}_reached", 0)]
        result[name] = {
            "reached_games": len(reached),
            "reach_rate": len(reached) / len(episodes),
            "own_turn_if_reached": fmean(reached) if reached else None,
            "by_turn20_rate": sum(turn <= 20 for turn in reached) / len(episodes),
        }
    for turn in (10, 20):
        prefix = f"construction_turn{turn}"
        observed = [e["strategy"] for e in episodes if e["strategy"].get(f"{prefix}_observed", 0)]
        # 早期終了局を0として扱わず、観測対象局数を必ず併記する。
        result[f"turn{turn}_observed_games"] = len(observed)
        for metric in ("sites", "cities", "building_points", "production_pips"):
            result[f"turn{turn}_{metric}_if_observed"] = (
                fmean(row[f"{prefix}_{metric}"] for row in observed) if observed else None
            )
    return result
