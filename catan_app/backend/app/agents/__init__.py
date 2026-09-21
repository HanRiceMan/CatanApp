"""差し替え可能なゲームエージェント群。"""

from .base import Agent
from .heuristic import HeuristicAgent
from .random_agent import RandomAgent

__all__ = ["Agent", "HeuristicAgent", "RandomAgent"]
