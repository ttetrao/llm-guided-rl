from __future__ import annotations
import gymnasium as gym
from typing import Any

LEGEND = """
+----+----+----+----+
| S  | F  | F  | F  |
+----+----+----+----+
| F  | F  | F  | F  |
+----+----+----+----+
| F  | H  | F  | F  |
+----+----+----+----+
| F  | F  | F  | G  |
+----+----+----+----+

Legend:
S = Start  F = Frozen (empty)  H = Hole  G = Goal
A(U/D/R/L) = Agent facing up/down/right/left at that cell
"""

# FrozenLake 8x8 default (4x4 compatibile) desc for legend rendering
DIR_SYM = ["L", "D", "R", "U"]  # gym mapping 0:left 1:down 2:right 3:up -> show arrow
# for view we use A(R)/A(L)/A(U)/A(D)

class FrozenLakeViewSystem(gym.Wrapper):
    """Wrapper che aggiunge current_view() ASCII per FrozenLake slippery.
    Non altera reward/dynamics, solo helper per prompt.
    Identico a DoorKeyViewSystem ma senza Stage.
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
        # propagate p if available
        if "prob" not in info and isinstance(info, dict):
            # gymnasium FrozenLake puts prob in info["prob"]
            pass
        return obs, rew, term, trunc, info

    def current_view(self) -> str:
        base = self.env.unwrapped
        desc = getattr(base, "desc", None)
        nrow = int(getattr(base, "nrow", 4))
        ncol = int(getattr(base, "ncol", 4))
        s = int(getattr(base, "s", self._last_s if self._last_s is not None else 0))
        r, c = divmod(s, ncol)

        # build ascii grid inner (no outer walls as FrozenLake has no walls)
        # use same style as docs: +----+ per cell
        lines: list[str] = []
        border = "+" + "----+" * ncol
        for rr in range(nrow):
            lines.append(border)
            row: list[str] = []
            for cc in range(ncol):
                if rr == r and cc == c:
                    # agent direction unknown in FrozenLake obs, default A
                    # try to keep last action dir? fallback A
                    row.append(" A  ")
                else:
                    # tile char
                    try:
                        ch = desc[rr, cc].decode() if hasattr(desc[rr, cc], "decode") else str(desc[rr][cc])
                    except Exception:
                        ch = "F"
                    if ch == "S":
                        ch = " S "
                    elif ch == "F":
                        ch = " F "
                    elif ch == "H":
                        ch = " H "
                    elif ch == "G":
                        ch = " G "
                    else:
                        ch = f" {ch} "
                    # pad to 4 inc borders: mappiamo " S " -> " S  "
                    if len(ch) == 3:
                        ch = ch + " "
                    row.append(ch)
            lines.append("|" + "|".join(row) + "|")
        lines.append(border)
        return "\n".join(lines)

    def render_map_for_state(self, s: int) -> str:
        """Renderizza mappa per stato arbitrario s (row*ncol+col) senza mutare env."""
        base = self.env.unwrapped
        desc = getattr(base, "desc", None)
        nrow = int(getattr(base, "nrow", 4))
        ncol = int(getattr(base, "ncol", 4))
        r, c = divmod(int(s), ncol)
        lines: list[str] = []
        border = "+" + "----+" * ncol
        for rr in range(nrow):
            lines.append(border)
            row: list[str] = []
            for cc in range(ncol):
                if rr == r and cc == c:
                    row.append(" A  ")
                else:
                    try:
                        ch = desc[rr, cc].decode() if hasattr(desc[rr, cc], "decode") else str(desc[rr][cc])
                    except Exception:
                        ch = "F"
                    if ch == "S":
                        ch = " S "
                    elif ch == "F":
                        ch = " F "
                    elif ch == "H":
                        ch = " H "
                    elif ch == "G":
                        ch = " G "
                    else:
                        ch = f" {ch} "
                    if len(ch) == 3:
                        ch = ch + " "
                    row.append(ch)
            lines.append("|" + "|".join(row) + "|")
        lines.append(border)
        return "\n".join(lines)
