#!/usr/bin/env python3
"""
Stati DoorKey per query LLM: bottleneck strutturali + checkpoint di un run pilota puro.

Sostituisce l'approccio a bucket di qlearning_states.py (iniziale/intermedio/avanzato):

- Bottleneck strutturali (da grafo, niente training): cella chiave, cella porta
  (esiste solo con door_open=True: stare sulla porta chiusa e' muro), celle
  adiacenti frontali a porta/chiave (coprono door_open=0), predecessori di
  pickup (max 4 angolazioni chiave) e di goal (reward=1, in angolo max 2-3).
  I terminali sul goal sono esclusi (assorbenti, V*=0, nessuna decisione).
- Checkpoint pilota: un run Q-learning tabulare "puro" (no LLM) sul grafo
  diretto come qlearning_states.py; quando la success rate a finestra raggiunge
  ciascuna soglia (~0.3/0.5/0.7/0.8/1.0) congela gli stati visitati. Per soglie
  >= --diversify-from (default 0.8) diversifica con random walk di k passi
  (Florensa et al., 2017) + stati degli episodi falliti alla finestra, per non
  collassare sul percorso ottimo. Soglie basse (0.3-0.7): visitati diretti.
- Budget --total: se la somma dei pool supera N, bottleneck tenuto intero
  (decine di nodi) e resto ripartito tra checkpoint in proporzione alle
  dimensioni (largest remainder, seeded). Se somma <= N, tutto.

Output json con stessa struttura di qlearning_states (dict "buckets"), quindi
query_gemma.py lo legge cambiando solo BUCKETS_WANTED nelle nuove etichette
(senza "_" per compatibilita' col parsing del code seed_nid_bucket):
  ["bottleneck", "ckpt-0.3", "ckpt-0.5", "ckpt-0.7", "ckpt-0.8", "ckpt-1"]

Uso:
    python -m graph.doorkey_states --seed 1337 --total 100
    python -m graph.doorkey_states --seed 1337 --total 0   # nessun tetto
    python -m graph.doorkey_states --seeds 1337 42 99 --total 100  # N seed espliciti
    python -m graph.doorkey_states --num-seeds 5 --total 100  # N seed random (rng: --seed)
    python -m graph.doorkey_states --num-seeds 5 --seed 7 --total 100  # altro set riproducibile
    python -m graph.doorkey_states --seed 42 --thresholds 0.3 0.5 0.8 --k-walk 5 --force
    from graph.doorkey_states import load_or_extract
    data = load_or_extract(seed=1337, total=100)
    from graph.doorkey_states import load_or_extract_multi
    multi = load_or_extract_multi(seeds=[1337, 42], total=100)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import deque
from pathlib import Path

_THIS = Path(__file__).resolve()
_SRC = _THIS.parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
from graph.mdp_graph import DIRS, load_or_build, get_paths as get_mdp_paths
from graph.qlearning_states import QLearningAgent
from paths import CACHE_DIR

DEFAULT_SEED = 1337
DEFAULT_SIZE = 8
GAMMA = 0.99
DEFAULT_THRESHOLDS = [0.3, 0.5, 0.7, 0.8, 1.0]
DIVERSIFY_FROM = 0.8  # soglie >= questa: random walk + falliti (0.7 resta diretto)
DEFAULT_K_WALK = 3
BOTTLENECK_LABEL = "bottleneck"


def _ckpt_label(t: float) -> str:
    # ponytail: niente "_" (query_gemma fa code.rsplit("_",2))
    return f"ckpt-{t:g}"


def get_states_paths(seed: int = DEFAULT_SEED, size: int = DEFAULT_SIZE,
                      out_dir: Path | str | None = None) -> Path:
    if out_dir is None:
        out_dir = CACHE_DIR
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"doorkey_states_{size}x{size}_seed{seed}.json"


def extract_bottleneck(mdp: dict) -> dict[str, list[int]]:
    """Bottleneck strutturali come node_id. Tutte combo esistenti nel grafo."""
    gi = mdp["grid_info"]
    kx, ky = gi["key_pos"]
    dx, dy = gi["door_pos"]
    nodes = mdp["nodes"]
    out: dict[str, list[int]] = {}
    out["on_key"] = [n["id"] for n in nodes if (n["x"], n["y"]) == (kx, ky)]
    out["on_door"] = [n["id"] for n in nodes if (n["x"], n["y"]) == (dx, dy)]
    # adiacenti frontali alla porta chiusa (copre door_open=0 x has_key 0/1)
    adj: set[int] = set()
    for d, (ox, oy) in enumerate(DIRS):
        ax, ay = dx - ox, dy - oy
        for n in nodes:
            if (n["x"], n["y"]) == (ax, ay) and n["dir"] == d and not n["door_open"]:
                adj.add(n["id"])
    out["adj_door"] = sorted(adj)
    # toggle che APRE (chiuso -> aperto): cambio fase open_door -> reach_goal
    out["door_toggle"] = [n["id"] for n in nodes
                          if not n["door_open"]
                          and nodes[n["transitions"][5]["next_id"]]["door_open"]]
    # pickup che cambia stato: angolazioni di presa chiave (max 4)
    out["key_take"] = [n["id"] for n in nodes if n["transitions"][3]["next_id"] != n["id"]]
    # ingressi al goal (reward=1): angolazioni di arrivo (angolo -> max 2-3)
    out["goal_entry"] = [n["id"] for n in nodes
                         if not n["is_terminal"]
                         and any(t["reward"] > 0 for t in n["transitions"].values())]
    return out


def _random_walk(mdp: dict, start_ids: set[int], k: int, rng: random.Random) -> set[int]:
    """k passi uniformi sul grafo da ogni start (Florensa et al., 2017)."""
    nodes = mdp["nodes"]
    out = set(start_ids)
    for sid in start_ids:
        cur = sid
        for _ in range(k):
            acts = list(nodes[cur]["transitions"].keys())
            cur = nodes[cur]["transitions"][rng.choice(acts)]["next_id"]
            out.add(cur)
    return out


def _greedy_sr(agent: QLearningAgent, mdp: dict, start_state: tuple,
               n_ep: int = 20, max_steps: int = 1000) -> float:
    nodes, index = mdp["nodes"], mdp["index"]
    succ = 0
    for _ in range(n_ep):
        s, done, steps = start_state, False, 0
        while not done and steps < max_steps:
            a = agent.act(s, greedy=True)
            t = nodes[index[s]]["transitions"].get(a, nodes[index[s]]["transitions"][0])
            s, done = t["next_state"], bool(t["done"])
            steps += 1
        succ += done
    return succ / n_ep if n_ep else 0.0


def run_pilot(mdp: dict, seed: int = DEFAULT_SEED, alpha: float = 0.3,
              gamma: float = GAMMA, epsilon_decay: float = 0.998,
              epsilon_min: float = 0.05, max_episodes: int = 10000,
              window: int = 100, eval_every: int = 50,
              thresholds: list[float] | None = None,
              diversify_from: float = DIVERSIFY_FROM,
              k_walk: int = DEFAULT_K_WALK) -> dict:
    """Un run puro su grafo; ritorna pool per soglia + metadati checkpoint."""
    thresholds = sorted(set(thresholds or DEFAULT_THRESHOLDS))
    random.seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)
    nodes, index = mdp["nodes"], mdp["index"]
    gi = mdp["grid_info"]
    start_state = (gi["start_pos"][0], gi["start_pos"][1], gi["start_dir"], False, False)
    agent = QLearningAgent(n_actions=len(mdp["action_names"]), alpha=alpha,
                           gamma=gamma, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min)
    max_steps = 1000
    sr_window: deque = deque(maxlen=window)
    hist: deque = deque(maxlen=window)  # (visited_ids:set, success:bool)
    pools: dict[str, set[int]] = {}
    ckpt_ep, ckpt_sr, ckpt_reached = {}, {}, {}
    pending = list(thresholds)
    total_episodes, final_greedy_sr = 0, 0.0

    def capture(T: float, reached: bool):
        visited = set().union(*[v for v, _ in hist]) if hist else set()
        if T + 1e-9 < diversify_from:
            pool = set(visited)  # diretto: policy ancora esplorativa
        else:
            failed = set().union(*[v for v, s in hist if not s]) if hist else set()
            pool = _random_walk(mdp, visited, k_walk, rng) | failed
        pools[_ckpt_label(T)] = pool
        ckpt_ep[_ckpt_label(T)] = total_episodes
        ckpt_sr[_ckpt_label(T)] = float(sum(sr_window)) / len(sr_window) if sr_window else 0.0
        ckpt_reached[_ckpt_label(T)] = reached

    for ep in range(max_episodes):
        s, done, steps = start_state, False, 0
        visited = [index[s]]
        while not done and steps < max_steps:
            a = agent.act(s)
            t = nodes[index[s]]["transitions"].get(a, nodes[index[s]]["transitions"][0])
            ns, r, done = t["next_state"], float(t["reward"]), bool(t["done"])
            visited.append(index[ns])
            agent.update(s, a, r, ns, done)
            s, steps = ns, steps + 1
        sr_window.append(1 if done else 0)
        hist.append((set(visited), bool(done)))
        sr = float(sum(sr_window)) / len(sr_window)
        agent.decay_epsilon()
        total_episodes = ep + 1
        for T in [t for t in pending if sr + 1e-9 >= t]:
            capture(T, True)
        pending = [t for t in pending if _ckpt_label(t) not in pools]
        if (ep + 1) % eval_every == 0:
            final_greedy_sr = _greedy_sr(agent, mdp, start_state)
            print(f"Ep {ep+1:4d} sr_win={sr:.3f} greedy={final_greedy_sr:.3f} "
                  f"eps={agent.epsilon:.3f} ckpt={sorted(pools)}")
        if not pending:
            break
    for T in pending:  # soglia mai raggiunta: snapshot finale, marcato non raggiunto
        capture(T, False)
        print(f"Warn: soglia {T} mai raggiunta (sr finale "
              f"{float(sum(sr_window))/len(sr_window):.3f}), uso ultimi {window} ep")
    final_greedy_sr = _greedy_sr(agent, mdp, start_state)
    return {"pools": {k: sorted(v) for k, v in pools.items()},
            "ckpt_episodes": ckpt_ep, "ckpt_sr": ckpt_sr, "ckpt_reached": ckpt_reached,
            "total_episodes": total_episodes,
            "final_sr_window": float(sum(sr_window)) / len(sr_window) if sr_window else 0.0,
            "final_greedy_sr": float(final_greedy_sr),
            "hparams": {"alpha": alpha, "gamma": gamma, "epsilon_decay": epsilon_decay,
                        "epsilon_min": epsilon_min, "n_actions": len(mdp["action_names"])}}


def _proportional(pools: dict[str, list[int]], total: int, rng: random.Random):
    tot = sum(len(v) for v in pools.values())
    if tot <= total:
        return {k: list(v) for k, v in pools.items()}
    base, frac = {}, {}
    for k, v in pools.items():
        q = total * len(v) / tot
        base[k], frac[k] = int(q), q - int(q)
    for k in sorted(frac, key=lambda k: (-frac[k], k))[:total - sum(base.values())]:
        base[k] += 1
    return {k: sorted(rng.sample(v, min(base[k], len(v)))) if base[k] < len(v) else list(v)
            for k, v in pools.items()}


def allocate(pools: dict[str, list[int]], total: int | None,
             rng: random.Random, protected: tuple[str, ...] = (BOTTLENECK_LABEL,)):
    """Tetto totale N: protetti interi, resto proporzionale (largest remainder)."""
    pools = {k: sorted(set(v)) for k, v in pools.items()}
    if not total or total <= 0 or sum(len(v) for v in pools.values()) <= total:
        return pools
    prot = [k for k in protected if k in pools]
    if sum(len(pools[k]) for k in prot) >= total:  # N minuscolo: proporzionale su tutto
        return _proportional(pools, total, rng)
    out = {k: list(pools[k]) for k in prot}
    rest = {k: v for k, v in pools.items() if k not in prot}
    out.update(_proportional(rest, total - sum(len(v) for v in out.values()), rng))
    return out


def _to_jsonable(data: dict) -> dict:
    return {
        "seed": int(data["seed"]), "size": int(data["size"]), "gamma": float(data["gamma"]),
        "method": data.get("method", "bottleneck+ckpt-pilot"),
        "hparams": data["hparams"], "total_episodes": int(data["total_episodes"]),
        "final_sr_window": float(data["final_sr_window"]),
        "final_greedy_sr": float(data["final_greedy_sr"]),
        "thresholds": [float(t) for t in data["thresholds"]],
        "diversify_from": float(data["diversify_from"]), "k_walk": int(data["k_walk"]),
        "bucket_labels": data["bucket_labels"],
        "buckets": {k: [int(x) for x in v] for k, v in data["buckets"].items()},
        "bucket_counts": {k: len(v) for k, v in data["buckets"].items()},
        "pools_all_counts": {k: int(v) for k, v in data["pools_all_counts"].items()},
        "bottleneck_counts": {k: int(v) for k, v in data["bottleneck_counts"].items()},
        "bottleneck_ids": [int(x) for x in data["bottleneck_ids"]],
        "ckpt_episodes": {k: (None if v is None else int(v)) for k, v in data["ckpt_episodes"].items()},
        "ckpt_sr": {k: float(v) for k, v in data["ckpt_sr"].items()},
        "ckpt_reached": {k: bool(v) for k, v in data["ckpt_reached"].items()},
        "total_requested": (None if data.get("total_requested") is None
                            else int(data["total_requested"])),
        "graph_ref": data["graph_ref"],
        "visited_unique_total": int(data["visited_unique_total"]),
    }


def save_states(data: dict, json_path: Path):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(data), f, indent=2, ensure_ascii=False)


def load_states(json_path: Path) -> dict:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def extract(seed: int = DEFAULT_SEED, size: int = DEFAULT_SIZE, gamma: float = GAMMA,
            total: int | None = 100, thresholds: list[float] | None = None,
            diversify_from: float = DIVERSIFY_FROM, k_walk: int = DEFAULT_K_WALK,
            out_dir: Path | str | None = None, **pilot_kwargs) -> dict:
    """Bottleneck + pilota + budget; ritorna dict con buckets compatibili."""
    thresholds = sorted(set(thresholds or DEFAULT_THRESHOLDS))
    mdp = load_or_build(seed=seed, size=size, gamma=gamma)
    bn = extract_bottleneck(mdp)
    bottleneck_ids = sorted(set().union(*bn.values())) if bn else []
    pilot = run_pilot(mdp, seed=seed, gamma=gamma, thresholds=thresholds,
                      diversify_from=diversify_from, k_walk=k_walk, **pilot_kwargs)
    pools_all = {BOTTLENECK_LABEL: bottleneck_ids}
    pools_all.update(pilot["pools"])
    buckets = allocate(pools_all, total, random.Random(seed))
    labels = [BOTTLENECK_LABEL] + [_ckpt_label(t) for t in thresholds]
    json_ref = get_mdp_paths(seed, size, out_dir if out_dir else CACHE_DIR)
    return {
        "seed": seed, "size": size, "gamma": gamma, "method": "bottleneck+ckpt-pilot",
        "hparams": pilot["hparams"], "total_episodes": pilot["total_episodes"],
        "final_sr_window": pilot["final_sr_window"],
        "final_greedy_sr": pilot["final_greedy_sr"],
        "thresholds": thresholds, "diversify_from": diversify_from, "k_walk": k_walk,
        "bucket_labels": labels, "buckets": buckets,
        "pools_all_counts": {k: len(v) for k, v in pools_all.items()},
        "bottleneck_counts": {k: len(v) for k, v in bn.items()},
        "bottleneck_ids": bottleneck_ids,
        "ckpt_episodes": pilot["ckpt_episodes"], "ckpt_sr": pilot["ckpt_sr"],
        "ckpt_reached": pilot["ckpt_reached"], "total_requested": total,
        "graph_ref": str(json_ref),
        "visited_unique_total": len(set().union(*buckets.values())) if buckets else 0,
    }


def load_or_extract(seed: int = DEFAULT_SEED, size: int = DEFAULT_SIZE,
                     out_dir: Path | str | None = None, force: bool = False,
                     **kwargs) -> dict:
    json_path = get_states_paths(seed, size, out_dir)
    if json_path.exists() and not force:
        print(f"[cache] carico {json_path}")
        return load_states(json_path)
    print(f"[extract] bottleneck+ckpt seed={seed} size={size} -> {json_path}")
    data = extract(seed=seed, size=size, out_dir=out_dir, **kwargs)
    save_states(data, json_path)
    print(f"  salvato json: {json_path} ({json_path.stat().st_size/1024:.1f} KB)")
    print(f"  buckets: " + ", ".join(f"{k}={len(v)}" for k, v in data["buckets"].items()))
    return data


def get_states_multi_paths(seeds: list[int], size: int = DEFAULT_SIZE,
                           out_dir: Path | str | None = None) -> Path:
    if out_dir is None:
        out_dir = CACHE_DIR
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "-".join(str(int(s)) for s in sorted(set(seeds)))
    return out_dir / f"doorkey_states_{size}x{size}_seeds{tag}.json"


def _to_jsonable_multi(data: dict) -> dict:
    # ponytail: "seeds" prima chiave -> in cima al JSON
    return {
        "seeds": [int(s) for s in data["seeds"]],
        "size": int(data["size"]),
        "total_per_seed": (None if data.get("total_per_seed") is None
                           else int(data["total_per_seed"])),
        "per_seed": {str(s): _to_jsonable(v) for s, v in data["per_seed"].items()},
    }


def save_states_multi(data: dict, json_path: Path):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable_multi(data), f, indent=2, ensure_ascii=False)


def load_states_multi(json_path: Path) -> dict:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    if "per_seed" in data:  # normalizza chiavi str (da JSON) -> int
        data["per_seed"] = {int(k): v for k, v in data["per_seed"].items()}
        data["seeds"] = [int(s) for s in data["seeds"]]
    return data


def load_or_extract_multi(seeds: list[int], size: int = DEFAULT_SIZE,
                          out_dir: Path | str | None = None, force: bool = False,
                          **kwargs) -> dict:
    """Estrae (o riusa cache singola) per ogni seed; --total resta per-seed."""
    seeds = sorted(set(int(s) for s in seeds))
    assert seeds, "servono >=1 seed"
    json_path = get_states_multi_paths(seeds, size, out_dir)
    if json_path.exists() and not force:
        print(f"[cache] carico {json_path}")
        return load_states_multi(json_path)
    per_seed = {s: load_or_extract(seed=s, size=size, out_dir=out_dir,
                                  force=force, **kwargs) for s in seeds}
    # ponytail: seeds per prima -> in cima al file
    data = {"seeds": seeds, "size": size,
            "total_per_seed": kwargs.get("total", 100),
            "per_seed": per_seed}
    save_states_multi(data, json_path)
    print(f"  salvato json: {json_path} ({json_path.stat().st_size/1024:.1f} KB)")
    for s, d in per_seed.items():
        print(f"  seed {s}: " + ", ".join(f"{k}={len(v)}" for k, v in d["buckets"].items()))
    return data


def main():
    p = argparse.ArgumentParser(description="Stati DoorKey: bottleneck + checkpoint pilota (budget totale proporzionale)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                     help="seed singolo (default 1337); con --num-seeds e' il rng-seed del sorteggio")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="N seed espliciti (es. --seeds 1337 42 99), prevale su --num-seeds/--seed")
    p.add_argument("--num-seeds", type=int, default=None, dest="num_seeds",
                    help="N seed pescati random in [0,100000) con rng=args.seed (es. --num-seeds 5)")
    p.add_argument("--size", type=int, default=DEFAULT_SIZE, choices=[6, 8, 16])
    p.add_argument("--total", "--n-stati", dest="total", type=int, default=100,
                   help="tetto totale stati (default 100; <=0 = nessun tetto)")
    p.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS),
                   help="soglie SR checkpoint (default 0.3 0.5 0.7 0.8 1.0)")
    p.add_argument("--diversify-from", type=float, default=DIVERSIFY_FROM, dest="diversify_from",
                   help="soglie >= questa usano random walk + falliti (default 0.8)")
    p.add_argument("--k-walk", type=int, default=DEFAULT_K_WALK, dest="k_walk",
                   help="passi random walk per diversificare (default 3)")
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--gamma", type=float, default=GAMMA)
    p.add_argument("--epsilon-decay", type=float, default=0.998, dest="epsilon_decay")
    p.add_argument("--epsilon-min", type=float, default=0.05, dest="epsilon_min")
    p.add_argument("--max-episodes", type=int, default=10000, dest="max_episodes")
    p.add_argument("--window", type=int, default=100, help="finestra SR (default 100)")
    p.add_argument("--eval-every", type=int, default=50, dest="eval_every")
    p.add_argument("--out", type=str, default=None, help="cartella output (default src/output/cache)")
    p.add_argument("--force", action="store_true", help="rigenera anche se esiste")
    p.add_argument("--show", type=int, default=0, help="stampa N mappe campione")
    args = p.parse_args()

    if args.seeds is not None and args.num_seeds is not None:
        p.error("--seeds e --num-seeds sono mutuamente esclusivi")
    multi = args.seeds is not None or args.num_seeds is not None
    if args.seeds is not None:
        seeds = sorted(set(args.seeds))
    elif args.num_seeds is not None:
        if args.num_seeds <= 0:
            p.error("--num-seeds deve essere >= 1")
        seeds = sorted(random.Random(args.seed).sample(range(100000), args.num_seeds))
    else:
        seeds = [args.seed]

    common = dict(total=args.total, thresholds=args.thresholds,
                  diversify_from=args.diversify_from, k_walk=args.k_walk,
                  alpha=args.alpha, gamma=args.gamma,
                  epsilon_decay=args.epsilon_decay, epsilon_min=args.epsilon_min,
                  max_episodes=args.max_episodes, window=args.window,
                  eval_every=args.eval_every)
    if multi:
        data = load_or_extract_multi(seeds, size=args.size, out_dir=args.out,
                                     force=args.force, **common)
        print(f"\nseeds={data['seeds']} size={data['size']} total_per_seed={args.total}")
        for s in data["seeds"]:
            d = data["per_seed"][s]
            print(f"seed {s}: ep={d['total_episodes']} "
                  f"sr_win={d['final_sr_window']:.3f} greedy={d['final_greedy_sr']:.3f} "
                  f"buckets=" + ", ".join(f"{k}={len(v)}" for k, v in d["buckets"].items()))
        for s in data["seeds"]:  # self-check leggero per seed (range + budget)
            d = data["per_seed"][s]
            mdp = load_or_build(seed=s, size=args.size)
            max_id = len(mdp["nodes"]) - 1
            for lst in d["buckets"].values():
                assert all(0 <= i <= max_id for i in lst), f"seed {s}: node_id fuori range"
            tot = args.total
            if tot and tot > 0:
                assert sum(len(v) for v in d["buckets"].values()) <= tot, f"seed {s}: budget sforato"
        print("[self-check] OK")
        return

    data = load_or_extract(seed=args.seed, size=args.size, out_dir=args.out, force=args.force,
                           **common)
    print(f"\nseed={data['seed']} size={data['size']} ep={data['total_episodes']} "
          f"sr_win={data['final_sr_window']:.3f} greedy={data['final_greedy_sr']:.3f}")
    print(f"bottleneck: {data['bottleneck_counts']} (union={len(data['bottleneck_ids'])})")
    for k in data["bucket_labels"]:
        ep = data["ckpt_episodes"].get(k, "-")
        reached = data["ckpt_reached"].get(k, "-")
        print(f"  {k}: {len(data['buckets'][k])} stati "
              f"(pool={data['pools_all_counts'].get(k)} ep={ep} reached={reached})")
    if args.show > 0:
        mdp = load_or_build(seed=args.seed, size=args.size)
        for nid in data["bottleneck_ids"][:args.show]:
            n = mdp["nodes"][nid]
            print(f"\n# id={nid} state={n['state']} stage={n['stage']}")
            print(n["map"])

    # self-check
    mdp = load_or_build(seed=args.seed, size=args.size)
    max_id = len(mdp["nodes"]) - 1
    for lst in data["buckets"].values():
        assert all(0 <= i <= max_id for i in lst), "node_id fuori range"
    # angolazioni spaziali (le combo x door_open/has_key sono volute): max 4 prese, goal in angolo
    mdp_nodes = {n["id"]: n for n in mdp["nodes"]}
    key_angles = {(mdp_nodes[i]["x"], mdp_nodes[i]["y"], mdp_nodes[i]["dir"])
                  for i in data["buckets"][BOTTLENECK_LABEL]
                  if mdp_nodes[i]["transitions"][3]["next_id"] != i}
    assert len(key_angles) <= 4, f"prese chiave da {len(key_angles)} angolazioni?"
    goal_cells = {(mdp_nodes[i]["x"], mdp_nodes[i]["y"]) for i in data["buckets"][BOTTLENECK_LABEL]
                  if any(t["reward"] > 0 for t in mdp_nodes[i]["transitions"].values())}
    assert len(goal_cells) <= 4, f"ingressi goal da {len(goal_cells)} celle?"
    tot = args.total
    if tot and tot > 0:
        assert sum(len(v) for v in data["buckets"].values()) <= tot, "budget sforato"
        bn_full = set(data["bottleneck_ids"])
        bn_got = set(data["buckets"][BOTTLENECK_LABEL])
        assert bn_got <= bn_full, "bottleneck con estranei"
        if len(bn_full) <= tot:
            assert bn_got == bn_full, "bottleneck protetto ma tagliato"
    print("[self-check] OK")


if __name__ == "__main__":
    main()
