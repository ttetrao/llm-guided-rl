#!/usr/bin/env python3
"""
MDP stocastico per FrozenLake slippery (is_slippery=True).

- Slippery: P(intended)=success_rate (1/3), altrimenti perpendicolari  (1-success_rate)/2 ciascuna
- Estrae P da gymnasium FrozenLake-v1 env.P[s][a] -> [(prob, ns, reward, done)]
- Value iteration stocastica: V*(s)=max_a sum_p p*[r+ gamma V*(s')]
- Mappa ASCII per nodo con A su s corrente, S/H/G/F statici (come view_wrapper)
- Salva pickle + json, cache per map_name/desc

Uso:
    python -m graph.frozenlake_mdp_graph --map 8x8 --seed 1337
    python -m graph.frozenlake_mdp_graph --map 8x8 --seed 1337 --force
    python -m graph.frozenlake_mdp_graph --map 8x8 --seed 42  # mappa random seedata (se desc custom)
"""
from __future__ import annotations
import argparse, json, pickle, sys
from pathlib import Path
_THIS = Path(__file__).resolve()
_SRC = _THIS.parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import gymnasium as gym
from env.frozenlake_view_wrapper import FrozenLakeViewSystem

GAMMA = 0.99
ACTION_NAMES = {0: "left", 1: "down", 2: "right", 3: "up"}
ACTIONS = [0, 1, 2, 3]

def _make_env(map_name="8x8", is_slippery=True, desc=None, seed=1337):
    kw = {"is_slippery": is_slippery}
    if desc is not None:
        kw["desc"] = desc
        kw["map_name"] = None
    else:
        if map_name in ("4x4", "8x8"):
            kw["map_name"] = map_name
        else:
            # mappa random seedata (come DoorKey seed -> diversa griglia)
            # generate_random_map usa global RNG: seedalo con seed per riproducibilità
            import random, numpy as np
            from gymnasium.envs.toy_text.frozen_lake import generate_random_map
            random.seed(seed); np.random.seed(seed)
            size = 8 if "8" in str(map_name) else 4
            desc = generate_random_map(size=size)
            kw["desc"] = desc
            kw["map_name"] = None
    env = gym.make("FrozenLake-v1", **kw)
    return env

def render_map(s: int, nrow: int, ncol: int, desc) -> str:
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

def value_iteration(P, nS: int, gamma=GAMMA, theta=1e-8):
    V = {s: 0.0 for s in range(nS)}
    while True:
        delta = 0.0
        for s in range(nS):
            # if terminal, V stays 0 (hole/goal absorbing) - but keep Bellman max
            # P[s][a] empty? then skip
            q_max = -1e9
            has_action = False
            for a in ACTIONS:
                if a not in P[s]:
                    continue
                has_action = True
                q = 0.0
                for prob, ns, r, done in P[s][a]:
                    v_next = 0.0 if done else V[ns]
                    q += prob * (float(r) + gamma * v_next)
                if q > q_max:
                    q_max = q
            if not has_action:
                v_new = 0.0
            else:
                # if all done states have no outgoing, q_max may stay -inf -> 0
                v_new = q_max if q_max != -1e9 else 0.0
                # terminal holes/goals already have P leading to self done, but keep max
                # for hole/goal states, environment transitions are absorbing self-loop 0 reward -> V=0
            delta = max(delta, abs(V[s] - v_new))
            V[s] = v_new
        if delta < theta:
            break
    # also compute Q* for convenience
    Q = {}
    for s in range(nS):
        for a in ACTIONS:
            q = 0.0
            for prob, ns, r, done in P[s].get(a, []):
                v_next = 0.0 if done else V[ns]
                q += prob * (float(r) + gamma * v_next)
            Q[(s, a)] = float(q)
    return V, Q

def build_mdp(seed=1337, map_name="8x8", is_slippery=True, desc=None, gamma=GAMMA):
    env = _make_env(map_name=map_name, is_slippery=is_slippery, desc=desc, seed=seed)
    # seed garantisce stessa mappa se rifatto con stesso seed (random map) e P stocastico slip preservato
    # P resta stocastico (3 outcome p=1/3), non determinismo conservato
    # need unwrapped to get desc/nrow/ncol and P
    base = env.unwrapped
    nS = int(base.observation_space.n)
    nA = int(base.action_space.n)
    nrow = int(getattr(base, "nrow"))
    ncol = int(getattr(base, "ncol"))
    desc_arr = getattr(base, "desc")
    P = getattr(base, "P")  # dict s -> dict a -> list[(prob, ns, reward, done)]
    V, Q = value_iteration(P, nS, gamma=gamma)

    # grid_info for rendering
    # find special tiles
    walls = set()  # frozenlake has no walls
    start_pos = (0, 0)
    goal_pos = (nrow-1, ncol-1)
    holes: list[tuple[int,int]] = []
    for rr in range(nrow):
        for cc in range(ncol):
            try:
                ch = desc_arr[rr, cc].decode() if hasattr(desc_arr[rr, cc], "decode") else str(desc_arr[rr][cc])
            except Exception:
                ch = "F"
            if ch == "S":
                start_pos = (rr, cc)
            elif ch == "G":
                goal_pos = (rr, cc)
            elif ch == "H":
                holes.append((rr, cc))

    nodes: list[dict] = []
    for s in range(nS):
        r, c = divmod(s, ncol)
        try:
            ch = desc_arr[r, c].decode() if hasattr(desc_arr[r, c], "decode") else str(desc_arr[r][c])
        except Exception:
            ch = "F"
        is_hole = (ch == "H")
        is_goal = (ch == "G")
        is_terminal = bool(is_hole or is_goal)
        m = render_map(s, nrow, ncol, desc_arr)
        # transitions per action as list
        trans: dict[int, list[dict]] = {}
        for a in ACTIONS:
            lst = []
            for prob, ns, rew, done in P[s].get(a, []):
                lst.append({"prob": float(prob), "next_id": int(ns), "reward": float(rew), "done": bool(done)})
            trans[a] = lst
        nodes.append({
            "id": int(s),
            "state": int(s),
            "r": int(r), "c": int(c),
            "ch": ch,
            "is_hole": bool(is_hole),
            "is_goal": bool(is_goal),
            "is_terminal": bool(is_terminal),
            "map": m,
            "v_value": round(float(V[s]), 6),
            "transitions": trans,
        })

    env.close()
    return {
        "seed": int(seed),
        "map_name": map_name,
        "is_slippery": bool(is_slippery),
        "gamma": float(gamma),
        "nrow": int(nrow), "ncol": int(ncol),
        "nS": int(nS), "nA": int(nA),
        "start_pos": tuple(start_pos),
        "goal_pos": tuple(goal_pos),
        "holes": list(holes),
        "desc": [ "".join(desc_arr[rr, cc].decode() if hasattr(desc_arr[rr, cc], "decode") else str(desc_arr[rr][cc]) for cc in range(ncol)) for rr in range(nrow) ],
        "nodes": nodes,
        "P": P,  # keep for quick access (not json serializable fully, but pickle ok)
        "V": V,
        "Q": Q,
        "action_names": ACTION_NAMES,
        "is_slippery_desc": f"is_slippery={is_slippery} success_rate=1/3" if is_slippery else "deterministic",
    }

def get_paths(map_name="8x8", is_slippery=True, seed=1337, out_dir: Path | str | None = None):
    if out_dir is None:
        out_dir = Path(__file__).parent / "data"
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slip = "slippery" if is_slippery else "deterministic"
    base = f"frozenlake_mdp_{map_name}_{slip}_seed{seed}"
    return out_dir / f"{base}.pkl", out_dir / f"{base}.json"

def _to_jsonable(mdp: dict) -> dict:
    nodes_j = []
    for n in mdp["nodes"]:
        trans_j = {}
        for a, lst in n["transitions"].items():
            trans_j[str(int(a))] = [{"prob": float(x["prob"]), "next_id": int(x["next_id"]), "reward": float(x["reward"]), "done": bool(x["done"])} for x in lst]
        nodes_j.append({
            "id": int(n["id"]), "state": int(n["state"]), "r": int(n["r"]), "c": int(n["c"]),
            "ch": n["ch"], "is_hole": bool(n["is_hole"]), "is_goal": bool(n["is_goal"]),
            "is_terminal": bool(n["is_terminal"]), "map": n["map"], "v_value": float(n["v_value"]),
            "transitions": trans_j,
        })
    return {
        "seed": int(mdp.get("seed", 1337)),
        "map_name": mdp["map_name"], "is_slippery": bool(mdp["is_slippery"]), "gamma": float(mdp["gamma"]),
        "nrow": int(mdp["nrow"]), "ncol": int(mdp["ncol"]), "nS": int(mdp["nS"]), "nA": int(mdp["nA"]),
        "start_pos": list(mdp["start_pos"]), "goal_pos": list(mdp["goal_pos"]),
        "holes": [list(x) for x in mdp["holes"]], "desc": mdp["desc"],
        "nodes": nodes_j,
        "action_names": {str(int(k)): v for k, v in mdp["action_names"].items()},
    }

def save_mdp(mdp: dict, pkl_path: Path, json_path: Path):
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(mdp, f, protocol=pickle.HIGHEST_PROTOCOL)
    j = _to_jsonable(mdp)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(j, f, indent=2, ensure_ascii=False)

def load_mdp(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def load_or_build(seed=1337, map_name="8x8", is_slippery=True, desc=None, out_dir: Path | str | None = None, force=False, gamma=GAMMA) -> dict:
    pkl_path, json_path = get_paths(map_name, is_slippery, seed, out_dir)
    # fallback vecchio file senza seed per retrocompat
    if not pkl_path.exists() and not force:
        old_pkl, _ = get_paths(map_name, is_slippery, seed=1337, out_dir=out_dir)
        # check legacy without seed
        legacy = Path(__file__).parent / "data" / f"frozenlake_mdp_{map_name}_{'slippery' if is_slippery else 'deterministic'}.pkl"
        if legacy.exists() and seed == 1337:
            print(f"[cache legacy] carico {legacy}")
            return load_mdp(legacy)
    if pkl_path.exists() and not force:
        print(f"[cache] carico {pkl_path}")
        return load_mdp(pkl_path)
    print(f"[build] FrozenLake MDP {map_name} slippery={is_slippery} seed={seed} -> {pkl_path}")
    mdp = build_mdp(seed=seed, map_name=map_name, is_slippery=is_slippery, desc=desc, gamma=gamma)
    save_mdp(mdp, pkl_path, json_path)
    print(f"  salvato pkl: {pkl_path} ({pkl_path.stat().st_size/1024:.1f} KB)")
    print(f"  salvato json: {json_path} ({json_path.stat().st_size/1024:.1f} KB)")
    print(f"  nodi: {len(mdp['nodes'])} holes:{len(mdp['holes'])} V(start)={mdp['nodes'][0]['v_value']}")
    return mdp

def main():
    p = argparse.ArgumentParser(description="MDP FrozenLake slippery - build/cache (seed per riproducibilità, slip preservato)")
    p.add_argument("--map", type=str, default="8x8", choices=["4x4","8x8"])
    p.add_argument("--seed", type=int, default=1337, help="seed per mappa random e cache (default 1337, stessa mappa se rifatto)")
    p.add_argument("--deterministic", action="store_true", help="is_slippery=False")
    p.add_argument("--force", action="store_true")
    p.add_argument("--show", type=int, default=2)
    args = p.parse_args()
    mdp = load_or_build(seed=args.seed, map_name=args.map, is_slippery=not args.deterministic, force=args.force)
    print(f"\nMDP {mdp['map_name']} slippery={mdp['is_slippery']} nS={mdp['nS']} V0={mdp['nodes'][0]['v_value']}")
    for n in mdp["nodes"][: args.show]:
        print(f"\n# id={n['id']} r={n['r']} c={n['c']} ch={n['ch']} V={n['v_value']}")
        print(n["map"])
        for a in ACTIONS:
            print(f"  {a}({ACTION_NAMES[a]}): {n['transitions'][a]}")
    assert len(mdp["nodes"]) == mdp["nS"]
    assert all("map" in n for n in mdp["nodes"])
    print("\n[self-check] OK")

if __name__ == "__main__":
    main()
