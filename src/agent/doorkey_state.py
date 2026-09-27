#!/usr/bin/env python3
"""Utilità condivise q-learning DoorKey: encode + costanti + resolve_input.

Modulo neutro (niente torch): importato dagli agenti qtable.
"""
import sys
from pathlib import Path

from env.view_wrapper import Stage

_THIS = Path(__file__).resolve()
_SRC = _THIS.parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from paths import LLM_DIR  # noqa: E402

# chiavi stringa (righe JSON LLM), usate dagli agenti qtable
ACTION_IDX = {"left": 0, "right": 1, "forward": 2, "pickup": 3, "drop": 4, "toggle": 5}
STAGE_IDX = {"find_key": 0, "open_door": 1, "reach_goal": 2}
STAGE_TARGET = {"find_key": "key_pos", "open_door": "door_pos", "reach_goal": "goal_pos"}


def resolve_input(name_or_path):
    """Risolve il JSON di risultati LLM (default: unico llm_results*.json in LLM_DIR)."""
    p = Path(name_or_path) if name_or_path else None
    if p is None:
        cands = sorted(LLM_DIR.glob("llm_results*.json"))
        assert cands, f"nessun llm_results*.json in {LLM_DIR}"
        if len(cands) == 1:
            return cands[0]
        if not sys.stdin.isatty():
            print(f"--input omesso: uso {cands[0].name}")
            return cands[0]
        print("Seleziona il file input:")
        for i, c in enumerate(cands):
            print(f"  [{i}] {c.name}")
        return cands[int(input("numero: ").strip())]
    if not p.is_absolute():
        for q in (p, LLM_DIR / p, LLM_DIR / p.name):
            if q.exists():
                return q
        assert False, f"file non trovato: {p}"
    assert p.exists(), f"file non trovato: {p}"
    return p


def encode(env):
    stage = env.get_wrapper_attr("curr_stage")
    base = env.unwrapped
    ax, ay = base.agent_pos
    if stage == Stage.FIND_KEY:
        t = env.get_wrapper_attr("key_pos")
        stage_idx = 0
    elif stage == Stage.OPEN_DOOR:
        t = env.get_wrapper_attr("door_pos")
        stage_idx = 1
    else:
        door_pos = env.get_wrapper_attr("door_pos")
        if ax <= door_pos[0]:
            t = door_pos
            stage_idx = 2
        else:
            t = env.get_wrapper_attr("goal_pos")
            stage_idx = 3
    return (int(t[0] - ax), int(t[1] - ay), int(base.agent_dir), int(stage_idx))


