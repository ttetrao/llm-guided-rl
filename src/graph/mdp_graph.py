#!/usr/bin/env python3
"""
MDP completo deterministico per MiniGrid-DoorKey (6/8/16).

- Estrae il grafo da env gymnasium con seed (seed nel nome file)
- Per ogni nodo: info MDP + mappa ASCII + x,y espliciti
- Salva pickle (veloce) + json (leggibile), oppure carica se esiste
- Accesso indicizzato O(1) via dict state->id

Ispirato a bak/scripts/state_export2.py per transizioni/stage/value.
Legenda mappa in env/view_wrapper.py:151

Uso:
    python -m graph.mdp_graph --seed 42 --size 8
    python -m graph.mdp_graph --seed 42 --size 8 --force
    from graph import load_or_build
    mdp = load_or_build(seed=42, size=8)  # dict con nodes/index/adj
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup per esecuzione sia come modulo che come script
# ---------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
# src/graph/mdp_graph.py -> src/
_SRC = _THIS.parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import gymnasium as gym
from env.view_wrapper import Stage, DoorKeyViewSystem

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------
DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # R, D, L, U  -> 0,1,2,3
ACTIONS = [0, 1, 2, 3, 4, 5]  # left, right, forward, pickup, drop, toggle (done 6 escluso)
ACTIONS_ALL = [0, 1, 2, 3, 4, 5, 6]  # tutte le azioni MiniGrid (incl. done)
ACTION_NAMES = {0: "left", 1: "right", 2: "forward", 3: "pickup", 4: "drop", 5: "toggle", 6: "done"}
GAMMA = 0.99
DIR_SYM = ["R", "D", "L", "U"]  # env/view_wrapper.py:161


# ---------------------------------------------------------------------------
# Helpers griglia / stage
# ---------------------------------------------------------------------------
def _find_obj(grid, obj_type):
    for x in range(grid.width):
        for y in range(grid.height):
            o = grid.get(x, y)
            if o is not None and o.type == obj_type:
                return (x, y, o)
    return None


def _grid_info(env) -> dict:
    base = env.unwrapped
    grid = base.grid
    w, h = grid.width, grid.height
    kw = _find_obj(grid, "key")
    dw = _find_obj(grid, "door")
    gw = _find_obj(grid, "goal")
    if kw is None or dw is None or gw is None:
        raise RuntimeError("key/door/goal non trovati nella griglia")
    walls = set()
    for x in range(w):
        for y in range(h):
            cell = grid.get(x, y)
            if cell is not None and cell.type == "wall":
                walls.add((x, y))
    return {
        "w": int(w),
        "h": int(h),
        "key_pos": (int(kw[0]), int(kw[1])),
        "door_pos": (int(dw[0]), int(dw[1])),
        "goal_pos": (int(gw[0]), int(gw[1])),
        "walls": {(int(x), int(y)) for x, y in walls},
        "start_pos": (int(base.agent_pos[0]), int(base.agent_pos[1])),
        "start_dir": int(base.agent_dir),
    }


def _infer_stage_from_state(state, gi) -> Stage:
    x, y, d, has_key, door_open = state
    gx, gy = gi["goal_pos"]
    if (x, y) == (gx, gy):
        return Stage.REACH_GOAL
    if not has_key:
        return Stage.FIND_KEY
    if has_key and not door_open:
        return Stage.OPEN_DOOR
    if has_key and door_open:
        return Stage.REACH_GOAL
    return Stage.ERROR


def _get_next_state(state, action, gi):
    """Transizione deterministica, self-loop su azione invalida (come state_export2)."""
    x, y, d, has_key, door_open = state
    kx, ky = gi["key_pos"]
    dx, dy = gi["door_pos"]
    w, h = gi["w"], gi["h"]
    walls = gi["walls"]
    # terminale assorbente: resta fermo
    gx, gy = gi["goal_pos"]
    if (x, y) == (gx, gy):
        return state

    if action == 0:  # left
        return (x, y, (d - 1) % 4, has_key, door_open)
    if action == 1:  # right
        return (x, y, (d + 1) % 4, has_key, door_open)
    if action == 2:  # forward
        fx, fy = x + DIRS[d][0], y + DIRS[d][1]
        if not (0 <= fx < w and 0 <= fy < h):
            return state
        if (fx, fy) in walls:
            return state
        if (fx, fy) == (dx, dy) and not door_open:
            return state
        if (fx, fy) == (kx, ky) and not has_key:
            return state
        return (fx, fy, d, has_key, door_open)
    if action == 3:  # pickup
        fx, fy = x + DIRS[d][0], y + DIRS[d][1]
        if (fx, fy) == (kx, ky) and not has_key:
            return (x, y, d, True, door_open)
        return state
    if action == 4:  # drop — ponytail: self-loop (drop mai ottimo, V* invariata)
        return state
    if action == 5:  # toggle
        fx, fy = x + DIRS[d][0], y + DIRS[d][1]
        if (fx, fy) == (dx, dy):
            if has_key and not door_open:
                return (x, y, d, has_key, True)
            if door_open:
                return (x, y, d, has_key, False)
        return state
    if action == 6:  # done — ponytail: self-loop
        return state
    return state


def _value_iteration(gi, gamma=GAMMA, theta=1e-6, actions=None):
    """V* deterministico (come state_export2.py:312)."""
    if actions is None:
        actions = ACTIONS_ALL
    w, h = gi["w"], gi["h"]
    gx, gy = gi["goal_pos"]
    kx, ky = gi["key_pos"]
    dx, dy = gi["door_pos"]
    walls = gi["walls"]
    valid_cells = [(x, y) for x in range(w) for y in range(h) if (x, y) not in walls]
    states = []
    for x, y in valid_cells:
        for d in range(4):
            for has_key in (False, True):
                for door_open in (False, True):
                    if not has_key and (x, y) == (kx, ky):
                        continue
                    if not door_open and (x, y) == (dx, dy):
                        continue
                    states.append((x, y, d, has_key, door_open))
    V = {s: 0.0 for s in states}
    while True:
        delta = 0.0
        for s in states:
            x, y = s[0], s[1]
            if (x, y) == (gx, gy):
                continue  # V=0 terminale
            v_old = V[s]
            max_q = -float("inf")
            for a in actions:
                ns = _get_next_state(s, a, gi)
                r = 1.0 if (ns[0], ns[1]) == (gx, gy) and (x, y) != (gx, gy) else 0.0
                q = r + gamma * V.get(ns, 0.0)
                if q > max_q:
                    max_q = q
            V[s] = max_q if max_q != -float("inf") else 0.0
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta:
            break
    return V


# ---------------------------------------------------------------------------
# Render mappa per stato arbitrario (senza mutare env)
# replica env/view_wrapper.py:151 ma parametrica su state
# ---------------------------------------------------------------------------
def render_map(state, gi) -> str:
    """
    Ritorna la mappa ASCII per lo stato dato.
    Legenda (env/view_wrapper.py + richiesta):
      A(U/D/R/L)=agent, L(U/D/R/L)=agent con chiave,
      K=key, G=goal, ▇=wall, D(L)=door locked / D(C) alias, D(O)=door open
    Nota: view_wrapper usa D(C) per locked, richiesta usa D(L);
          qui si usa D(C) per compatibilità con view_wrapper, ma D(L) è alias
          (entrambi indicano locked). Per aderire strettamente alla legenda
          della richiesta, sostituire \"D(C)\" con \"D(L)\" sotto.
    """
    x, y, d, has_key, door_open = state
    ax, ay = x, y
    w, h = gi["w"], gi["h"]
    walls = gi["walls"]
    kx, ky = gi["key_pos"]
    dx, dy = gi["door_pos"]
    gx, gy = gi["goal_pos"]

    prefix = "L" if has_key else "A"
    agent_sym = f"{prefix}({DIR_SYM[d]})"

    x_start, x_end = 1, w - 1
    y_start, y_end = 1, h - 1
    inner_w = x_end - x_start
    border = "+" + "----+" * inner_w

    lines: list[str] = []
    for yy in range(y_start, y_end):
        lines.append(border)
        row: list[str] = []
        for xx in range(x_start, x_end):
            if (xx, yy) == (ax, ay):
                row.append(agent_sym)
            else:
                if (xx, yy) in walls:
                    sym = " ▇  "
                elif (xx, yy) == (dx, dy):
                    sym = "D(O)" if door_open else "D(C)"
                elif (xx, yy) == (kx, ky) and not has_key:
                    sym = " K  "
                elif (xx, yy) == (kx, ky) and has_key:
                    sym = "    "
                elif (xx, yy) == (gx, gy):
                    sym = " G  "
                else:
                    sym = "    "
                row.append(sym)
        lines.append("|" + "|".join(row) + "|")
    lines.append(border)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build MDP
# ---------------------------------------------------------------------------
def _make_env(size: int):
    base = gym.make(f"MiniGrid-DoorKey-{size}x{size}-v0")
    return DoorKeyViewSystem(base)


def build_mdp(seed: int = 1337, size: int = 8, gamma: float = GAMMA) -> dict:
    """Costruisce il grafo MDP completo per il seed/size dati."""
    env = _make_env(size)
    env.reset(seed=seed)
    gi = _grid_info(env)
    w, h = gi["w"], gi["h"]
    gx, gy = gi["goal_pos"]
    kx, ky = gi["key_pos"]
    dx, dy = gi["door_pos"]
    walls = gi["walls"]

    valid_cells = [(x, y) for x in range(w) for y in range(h) if (x, y) not in walls]

    states: list[tuple] = []
    for x, y in valid_cells:
        for d in range(4):
            for has_key in (False, True):
                for door_open in (False, True):
                    if not has_key and (x, y) == (kx, ky):
                        continue
                    if not door_open and (x, y) == (dx, dy):
                        continue
                    states.append((x, y, d, has_key, door_open))

    # indice O(1) state -> id
    index: dict[tuple, int] = {s: i for i, s in enumerate(states)}
    # V* per ogni stato (ponytail: calcolato una volta, non per nodo)
    V = _value_iteration(gi, gamma=gamma, actions=ACTIONS_ALL)
    nodes: list[dict] = []

    for sid, s in enumerate(states):
        x, y, d, has_key, door_open = s
        stage = _infer_stage_from_state(s, gi)
        is_terminal = (x, y) == (gx, gy)
        m = render_map(s, gi)
        v = float(V.get(s, 0.0))

        transitions: dict[int, dict] = {}
        for a in ACTIONS_ALL:
            ns = _get_next_state(s, a, gi)
            nid = index[ns]
            # reward 1 solo entrando nel goal (come state_export2)
            reward = 1.0 if (ns[0], ns[1]) == (gx, gy) and (x, y) != (gx, gy) else 0.0
            done = (ns[0], ns[1]) == (gx, gy)
            transitions[a] = {
                "next_id": nid,
                "next_state": ns,
                "reward": reward,
                "done": done,
                "action_name": ACTION_NAMES[a],
            }

        nodes.append(
            {
                "id": sid,
                "state": s,
                "x": x,
                "y": y,
                "dir": d,
                "has_key": has_key,
                "door_open": door_open,
                "stage": stage.value,
                "is_terminal": is_terminal,
                "map": m,
                "v_value": round(v, 6),
                "transitions": transitions,
            }
        )

    # adj per convenienza
    adj = {n["id"]: [t["next_id"] for t in n["transitions"].values()] for n in nodes}

    env.close()

    return {
        "seed": seed,
        "size": size,
        "gamma": gamma,
        "grid_info": {
            "w": w,
            "h": h,
            "key_pos": gi["key_pos"],
            "door_pos": gi["door_pos"],
            "goal_pos": gi["goal_pos"],
            "start_pos": gi["start_pos"],
            "start_dir": gi["start_dir"],
            "walls": walls,
        },
        "nodes": nodes,
        "index": index,
        "adj": adj,
        "action_names": ACTION_NAMES,
    }


# ---------------------------------------------------------------------------
# Persistenza duale pickle + json
# ---------------------------------------------------------------------------
def get_paths(seed: int = 1337, size: int = 8, out_dir: Path | str | None = None) -> tuple[Path, Path]:
    if out_dir is None:
        out_dir = Path(__file__).parent / "data"
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"mdp_{size}x{size}_seed{seed}"
    return out_dir / f"{base}.pkl", out_dir / f"{base}.json"


def _to_jsonable(mdp: dict) -> dict:
    """Converte set/tuple/numpy in strutture JSON-serializzabili."""
    gi = mdp["grid_info"]
    def _i(v):  # ponytail: cast numpy int64 -> int per json
        try:
            return int(v)
        except Exception:
            return v
    # copia grid_info con walls come lista ordinata
    gi_j = {
        "w": _i(gi["w"]),
        "h": _i(gi["h"]),
        "key_pos": [_i(v) for v in gi["key_pos"]],
        "door_pos": [_i(v) for v in gi["door_pos"]],
        "goal_pos": [_i(v) for v in gi["goal_pos"]],
        "start_pos": [_i(v) for v in gi["start_pos"]],
        "start_dir": _i(gi["start_dir"]),
        "walls": sorted([[_i(a), _i(b)] for a, b in gi["walls"]]),
    }
    nodes_j = []
    for n in mdp["nodes"]:
        transitions_j = {}
        for a, t in n["transitions"].items():
            transitions_j[str(_i(a))] = {
                "next_id": _i(t["next_id"]),
                "next_state": [_i(v) if isinstance(v, (bool,)) else (_i(v) if not isinstance(v, bool) else v) for v in t["next_state"]],
                "reward": float(t["reward"]),
                "done": bool(t["done"]),
                "action_name": t["action_name"],
            }
            # fix bool handling: bool is subclass of int
            transitions_j[str(_i(a))]["next_state"] = [bool(v) if i>=3 else _i(v) for i, v in enumerate(t["next_state"])]  # x,y,dir,has_key,door_open
        nodes_j.append(
            {
                "id": _i(n["id"]),
                "state": [ _i(n["state"][0]), _i(n["state"][1]), _i(n["state"][2]), bool(n["state"][3]), bool(n["state"][4]) ],
                "x": _i(n["x"]),
                "y": _i(n["y"]),
                "dir": _i(n["dir"]),
                "has_key": bool(n["has_key"]),
                "door_open": bool(n["door_open"]),
                "stage": n["stage"],
                "is_terminal": bool(n["is_terminal"]),
                "map": n["map"],
                "v_value": float(n.get("v_value", 0.0)),
                "transitions": transitions_j,
            }
        )
    # index: key come stringa "x,y,dir,has_key,door_open"
    index_j = {",".join(map(str, [int(k[0]), int(k[1]), int(k[2]), bool(k[3]), bool(k[4])])): int(v) for k, v in mdp["index"].items()}
    adj_j = {str(int(k)): [int(x) for x in v] for k, v in mdp["adj"].items()}
    return {
        "seed": int(mdp["seed"]),
        "size": int(mdp["size"]),
        "gamma": float(mdp["gamma"]),
        "grid_info": gi_j,
        "nodes": nodes_j,
        "index": index_j,
        "adj": adj_j,
        "action_names": {str(int(k)): v for k, v in mdp["action_names"].items()},
    }


def save_mdp(mdp: dict, pkl_path: Path, json_path: Path):
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(mdp, f, protocol=pickle.HIGHEST_PROTOCOL)
    # json human-readable
    j = _to_jsonable(mdp)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(j, f, indent=2, ensure_ascii=False)


def load_mdp(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def load_or_build(seed: int = 1337, size: int = 8, out_dir: Path | str | None = None, force: bool = False, gamma: float = GAMMA) -> dict:
    """
    Se esiste il pkl per (seed,size) e non force, carica; altrimenti costruisce e salva pkl+json.
    Ritorna il dict MDP con accesso indicizzato.
    """
    pkl_path, json_path = get_paths(seed, size, out_dir)
    if pkl_path.exists() and not force:
        print(f"[cache] carico {pkl_path}")
        return load_mdp(pkl_path)
    print(f"[build] MDP {size}x{size} seed={seed} -> {pkl_path}")
    mdp = build_mdp(seed, size, gamma)
    save_mdp(mdp, pkl_path, json_path)
    print(f"  salvato pkl: {pkl_path} ({pkl_path.stat().st_size/1024:.1f} KB)")
    print(f"  salvato json: {json_path} ({json_path.stat().st_size/1024:.1f} KB)")
    print(f"  nodi: {len(mdp['nodes'])}  walls: {len(mdp['grid_info']['walls'])}")
    return mdp


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MDP DoorKey deterministico — build/cache con indice + mappa per nodo")
    parser.add_argument("--seed", type=int, default=1337, help="seed ambiente (nel nome file)")
    parser.add_argument("--size", type=int, default=8, choices=[6, 8, 16], help="dimensione mappa (default 8)")
    parser.add_argument("--gamma", type=float, default=GAMMA, help="gamma incluso nel dump per completezza")
    parser.add_argument("--out", type=str, default=None, help="cartella output (default graph/data)")
    parser.add_argument("--force", action="store_true", help="rigenera anche se esiste")
    parser.add_argument("--show", type=int, default=2, help="quante mappe di esempio stampare")
    args = parser.parse_args()

    mdp = load_or_build(seed=args.seed, size=args.size, out_dir=args.out, force=args.force, gamma=args.gamma)

    # demo accesso indicizzato
    print(f"\nMDP seed={mdp['seed']} size={mdp['size']}x{mdp['size']} gamma={mdp['gamma']}")
    print(f"grid_info: key={mdp['grid_info']['key_pos']} door={mdp['grid_info']['door_pos']} goal={mdp['grid_info']['goal_pos']} start={mdp['grid_info']['start_pos']} dir={mdp['grid_info']['start_dir']}")
    print(f"nodi totali: {len(mdp['nodes'])}")
    # esempio: trova nodo start
    start_state = (mdp["grid_info"]["start_pos"][0], mdp["grid_info"]["start_pos"][1], mdp["grid_info"]["start_dir"], False, False)
    idx = mdp["index"].get(start_state)
    if idx is not None:
        n = mdp["nodes"][idx]
        print(f"\n[esempio] start state {start_state} -> id {idx} stage={n['stage']} v={n.get('v_value',0)}")
        print(n["map"])
        # transizioni
        print("transitions:")
        for a in ACTIONS_ALL:
            t = n["transitions"][a]
            print(f"  {a}({ACTION_NAMES[a]}): -> id {t['next_id']} state {t['next_state']} r={t['reward']} done={t['done']}")
    # stampa qualche mappa
    if args.show > 0:
        print(f"\n--- {args.show} mappe casuali ---")
        import random
        random.seed(0)
        for _ in range(min(args.show, len(mdp["nodes"]))):
            n = random.choice(mdp["nodes"])
            print(f"\n# id={n['id']} state={n['state']} stage={n['stage']} x={n['x']} y={n['y']}")
            print(n["map"])

    # ponytail: self-check minimo (un branch, una transizione)
    assert len(mdp["nodes"]) == len(mdp["index"]), "index size mismatch"
    assert all("map" in n and "x" in n and "y" in n for n in mdp["nodes"]), "nodo senza map/x/y"
    assert all("v_value" in n for n in mdp["nodes"]), "nodo senza v_value"
    print("\n[self-check] OK")


if __name__ == "__main__":
    main()
