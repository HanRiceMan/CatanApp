"""互換用の再エクスポート。

AI実装は ``app.agents`` に移した。既存の import を壊さないため、この経路は維持する。
"""

from app.agents.runner import ai_actor, get_agent_for_player, play_ai_step, run_ai_until_pause, set_agent_for_player

__all__ = ["ai_actor", "get_agent_for_player", "play_ai_step", "run_ai_until_pause", "set_agent_for_player"]
