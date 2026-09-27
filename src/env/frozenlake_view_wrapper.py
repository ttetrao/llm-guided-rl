from __future__ import annotations
import gymnasium as gym


class FrozenLakeViewSystem(gym.Wrapper):
    """Wrapper che espone lo stato corrente come intero (FrozenLake slippery).
    Non altera reward/dynamics: mantiene solo `_last_s` allineato a obs.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._last_s: int | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_s = int(obs) if isinstance(obs, int) else int(getattr(self.env.unwrapped, "s", 0))
        return obs, info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        self._last_s = int(obs) if isinstance(obs, int) else int(getattr(self.env.unwrapped, "s", 0))
        return obs, rew, term, trunc, info
