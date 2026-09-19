#!/usr/bin/env python3
"""
Mappa ASCII con V-true / V-llm + tipo casella, per DoorKey e FrozenLake.

- DoorKey: per cella (x,y) mostra Vtrue/Vllm = max su tutti gli stati
  (dir, has_key, door_open) nella cella; Vllm(s) = max_a Q_llm(s,a) dalle righe LLM.
  (ponytail: una mappa col max invece di 16 mappe per slice; slice completa se serve davvero)
- FrozenLake: 1 stato = 1 cella, nessun max necessario.
- I grafi MDP vengono riusati dalla cache o rigenerati se assenti (--force per forzare).

Uso:
    python -m thesis.graph.show_values_map --env doorkey --seed 1337
    python -m thesis.graph.show_values_map --env frozenlake --map 8x8 --seed 1337
    python -m thesis.graph.show_values_map --env doorkey --llm thesis/graph/data/altro.json --force
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_SRC = _THIS.parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

W = 9  # larghezza cella ("0.99/0.99" ci sta esatta)


def _load_llm_rows(path: str | None, pattern: str, seed: int) -> list[dict]:
    """Legge il JSON LLM (o tutti i candidati, incluso il multiseed) e ritorna le righe del seed."""
    paths = [path] if path is not None else sorted(glob.glob(pattern))
    if not paths:
        print(f"[warn] nessun file LLM per pattern {pattern}: solo V-true ('--' al posto di V-llm)")
        return []
    out: list[dict] = []
    for p in paths:
        print(f"[llm] {p}")
        data = json.load(open(p, encoding="utf-8"))
        rows = data.get("rows", data) if isinstance(data, dict) else data
        out.extend(r for r in rows if int(r.get("seed", seed)) == seed)
    print(f"  righe: {len(out)} (seed={seed})")
    return out


def _vllm_by_state(rows: list[dict], key: str) -> dict[int, float]:
    """Vllm(s) = max_a Q_llm(s,a) dalle righe (una riga per azione)."""
    d: dict[int, float] = {}
    for r in rows:
        s = int(r[key])
        v = float(r["v_llm"])
        if s not in d or v > d[s]:
            d[s] = v
    return d


def _render(cells: list[list[tuple[str, str]]]) -> str:
    """cells[yy][xx] = (tipo, 'Vt/Vl'); ogni cella 2 righe di testo larghe W."""
    n = len(cells[0])
    border = "+" + ("-" * W + "+") * n
    lines = []
    for row in cells:
        lines.append(border)
        lines.append("|" + "|".join(f"{t:^{W}}" for t, _ in row) + "|")
        lines.append("|" + "|".join(f"{v:^{W}}" for _, v in row) + "|")
    lines.append(border)
    return "\n".join(lines)


def _fmt(vt: float | None, vl: float | None) -> str:
    if vt is None:
        return "#" * W
    return f"{vt:.2f}/{vl:.2f}" if vl is not None else f"{vt:.2f}/--"


def show_doorkey(seed: int, size: int, llm: str | None, force: bool):
    from thesis.graph.mdp_graph import load_or_build

    mdp = load_or_build(seed=seed, size=size, force=force)
    gi = mdp["grid_info"]
    w, h = gi["w"], gi["h"]
    walls = {(int(x), int(y)) for x, y in gi["walls"]}
    pat = str(Path(__file__).parent / "data" / "llm_results_*.json")
    vllm = _vllm_by_state(_load_llm_rows(llm, pat, seed), "node_id")

    cells: list[list[tuple[str, str]]] = []
    cov, tot, err, nerr = 0, 0, 0.0, 0
    for yy in range(1, h - 1):
        row = []
        for xx in range(1, w - 1):
            if (xx, yy) in walls:
                row.append(("###", "#" * W))
                continue
            if (xx, yy) == tuple(gi["door_pos"]):
                t = "D"
            elif (xx, yy) == tuple(gi["key_pos"]):
                t = "K"
            elif (xx, yy) == tuple(gi["goal_pos"]):
                t = "G"
            elif (xx, yy) == tuple(gi["start_pos"]):
                t = "S"
            else:
                t = "."
            st = [n for n in mdp["nodes"] if n["x"] == xx and n["y"] == yy]
            vt = max(float(n["v_value"]) for n in st) if st else None
            lv = [vllm[i] for i in (n["id"] for n in st) if i in vllm]
            vl = max(lv) if lv else None
            if vt is not None:
                tot += 1
                if vl is not None:
                    cov += 1
                    # MAE su V: |max_a Q_llm - V*| per gli stati coperti (ponytail: media, non bootstrap)
                    for n in st:
                        if n["id"] in vllm:
                            err += abs(vllm[n["id"]] - float(n["v_value"]))
                            nerr += 1
            row.append((t, _fmt(vt, vl)))
        cells.append(row)
    print(f"\nDoorKey {size}x{size} seed={seed}  key={gi['key_pos']} door={gi['door_pos']} goal={gi['goal_pos']}")
    print("riga1=tipo (. S K D G #muro)  riga2=Vtrue/Vllm (max su dir/key/door; --=mai queryato)")
    print(_render(cells))
    print(f"celle con Vllm: {cov}/{tot}" + (f"  MAE(V) su stati coperti: {err / nerr:.4f}" if nerr else ""))
    assert tot > 0 and all(0.0 <= float(n["v_value"]) <= 1.0 for n in mdp["nodes"]), "V* fuori [0,1]"
    print("[self-check] OK")


def show_frozenlake(map_name: str, seed: int, llm: str | None, force: bool):
    from thesis.graph.frozenlake_mdp_graph import load_or_build

    mdp = load_or_build(seed=seed, map_name=map_name, force=force)
    nrow, ncol = mdp["nrow"], mdp["ncol"]
    pat = str(Path(__file__).parent / "data" / "frozenlake_llm_*.json")
    vllm = _vllm_by_state(_load_llm_rows(llm, pat, seed), "state")

    cells: list[list[tuple[str, str]]] = []
    cov, tot, err, nerr = 0, 0, 0.0, 0
    for s, n in enumerate(mdp["nodes"]):
        vt = float(n["v_value"])
        vl = vllm.get(s)
        t = {"S": "S", "H": "H", "G": "G"}.get(n["ch"], ".")
        tot += 1
        if vl is not None:
            cov += 1
            err += abs(vl - vt)
            nerr += 1
        r, c = divmod(s, ncol)
        if c == 0:
            cells.append([])
        cells[r].append((t, _fmt(vt, vl)))
    print(f"\nFrozenLake {map_name} seed={seed} slippery={mdp['is_slippery']}")
    print("riga1=tipo (. S H G)  riga2=Vtrue/Vllm (--=mai queryato)")
    print(_render(cells))
    print(f"celle con Vllm: {cov}/{tot}" + (f"  MAE(V) su stati coperti: {err / nerr:.4f}" if nerr else ""))
    assert tot == mdp["nS"] and all(0.0 <= float(n["v_value"]) <= 1.0 for n in mdp["nodes"]), "V* fuori [0,1]"
    print("[self-check] OK")


def main():
    p = argparse.ArgumentParser(description="Mappa ASCII V-true/V-llm + tipo casella")
    p.add_argument("--env", choices=["doorkey", "frozenlake"], required=True)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--size", type=int, default=8, help="solo doorkey")
    p.add_argument("--map", type=str, default="8x8", dest="map_name", help="solo frozenlake")
    p.add_argument("--llm", type=str, default=None, help="JSON LLM (default: auto in graph/data)")
    p.add_argument("--force", action="store_true", help="rigenera il grafo MDP anche se in cache")
    a = p.parse_args()
    if a.env == "doorkey":
        show_doorkey(a.seed, a.size, a.llm, a.force)
    else:
        show_frozenlake(a.map_name, a.seed, a.llm, a.force)


if __name__ == "__main__":
    main()
