from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import cast
import gymnasium as gym
from minigrid.minigrid_env import MiniGridEnv

from . import doorkey_events as doorev


# ─────────────────────────────────────────────
# Enum che rappresenta le fasi sequenziali del task DoorKey.
# ─────────────────────────────────────────────
class Stage(Enum):
    FIND_KEY = "find_key"
    OPEN_DOOR = "open_door"
    REACH_GOAL = "reach_goal"
    ERROR = "error"


# ─────────────────────────────────────────────
# Snapshot degli eventi booleani rilevanti in un dato timestep.
# ─────────────────────────────────────────────
@dataclass
class EventSnapshot:
    has_key: bool
    door_is_open: bool
    goal_reached: bool


class DoorKeyViewSystem(gym.Wrapper):
    def __init__(self, env: gym.Env):
        self.current_stage: Stage | None = None
        self.curr_events: EventSnapshot | None = None

        self.key_pos: tuple[int, int] | None = None
        self.door_pos: tuple[int, int] | None = None
        self.goal_pos: tuple[int, int] | None = None

        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.key_pos = self._find_stage_goal_position("key")
        self.door_pos = self._find_stage_goal_position("door")
        self.goal_pos = self._find_stage_goal_position("goal")

        self.curr_events = self._extract_events()
        self.curr_stage = self._infer_stage(self.curr_events)

        return obs, info

    def step(self, action):
        if self.curr_events is None or self.curr_stage is None:
            raise RuntimeError(
                "Wrapper state not initialized. Call reset() before step()."
            )

        obs, env_reward, terminated, truncated, info = self.env.step(action)
        self.curr_events = self._extract_events()
        self.curr_stage = self._infer_stage(self.curr_events)

        return obs, env_reward, terminated, truncated, info

    def _get_base_env(self) -> MiniGridEnv:
        return cast(MiniGridEnv, self.env.unwrapped)

    def _extract_events(self) -> EventSnapshot:
        return EventSnapshot(
            has_key=doorev.has_key(self),
            door_is_open=doorev.door_is_open(self),
            goal_reached=doorev.goal_reached(self),
        )

    def _infer_stage(self, events: EventSnapshot) -> Stage:
        if not events.has_key:
            return Stage.FIND_KEY
        elif events.has_key and not events.door_is_open:
            return Stage.OPEN_DOOR
        elif events.has_key and events.door_is_open and not events.goal_reached:
            return Stage.REACH_GOAL
        else:
            return Stage.ERROR

    def _find_stage_goal_position(self, goal) -> tuple[int, int]:
        base_env = self._get_base_env()
        grid = base_env.grid
        for x in range(grid.width):
            for y in range(grid.height):
                obj = grid.get(x, y)
                if obj is not None and obj.type == goal:
                    return (x, y)
        raise RuntimeError(goal + " not found in the grid")
