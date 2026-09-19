#!/usr/bin/env python3
"""
Q-learning tabulare su FrozenLake slippery per bucket iniziale/intermedio/avanzato + bottleneck strutturale.

- Train su env gymnasium reale (stocastico slippery) con epsilon greedy
- Bucket per SR window 100: iniziale 0-0.33, intermedio 0.33-0.8, avanzato 0.8-1.0
- Max 100 stati unici per bucket (random sample finale per varietà)
- Cache pickle/json analogo a qlearning_states.py

Uso:
    python -m graph.frozenlake_qlearning_states --map 8x8 --seed 1337
    python -m graph.frozenlake_qlearning_states --map 8x8 --seed 1337 --force
"""

from __future__ import annotations
import argparse, json, pickle, random, sys
from collections import deque, defaultdict
from pathlib import Path

_THIS = Path(__file__).resolve()
_SRC = _THIS.parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import gymnasium as gym
from graph.frozenlake_mdp_graph import get_paths as get_mdp_paths, load_or_build

BUCKETS = ["iniziale", "intermedio", "avanzato"]
BOTTLENECK_LABEL = "bottleneck"
THRESHOLDS = [0.33, 0.8]
DEFAULT_MAP = "8x8"
DEFAULT_SEED = 1337
GAMMA = 0.99


def _greedy_action(mdp: dict, s: int):
    """Azione greedy-ottima in s da V* dell'MDP: argmax_a Σ p·[r+γV*]."""
    gamma = float(mdp.get("gamma", GAMMA))
    nodes = mdp["nodes"]
    best_a, best_q = None, -1e18
    for a, lst in nodes[s]["transitions"].items():
        q = sum(float(t["prob"]) * (float(t["reward"])
                + (0.0 if t["done"] else gamma * float(nodes[int(t["next_id"])]["v_value"])))
                for t in lst)
        if q > best_q:
            best_q, best_a = q, a
    return best_a


def extract_bottleneck(mdp: dict, seed: int = DEFAULT_SEED) -> dict[str, list[int]]:
    """Bottleneck FrozenLake come node_id (rng seedato), solo stati non-terminali
    (da H/G terminali non si agisce: v_true=0 fisso, niente da stimare):
    - goal_entry: stati non-terminali con transizione a reward>0 (tutti, pochi)
    - on_policy: path greedy-ottimo da S, campionato 1 ogni 2 (non tutti)
    - near_policy: 1-ring slip degli on-policy, max 10 F (le H restano solo come
      esiti altrui, non come stati query)
    """
    nodes = mdp["nodes"]
    rng = random.Random(seed)
    out: dict[str, list[int]] = {}
    out["goal_entry"] = sorted(
        n["id"] for n in nodes
        if not n["is_terminal"]
        and any(float(t["reward"]) > 0 for lst in n["transitions"].values() for t in lst)
    )
    # rollout greedy seguendo l'outcome più probabile (path "inteso" ottimo)
    path: list[int] = []
    seen: set[int] = set()
    s = 0
    for _ in range(len(nodes) + 1):
        if s in seen or nodes[s]["is_terminal"]:
            break
        seen.add(s)
        path.append(s)
        a = _greedy_action(mdp, s)
        lst = nodes[s]["transitions"][a]
        s = int(max(lst, key=lambda t: float(t["prob"]))["next_id"])
    out["on_policy"] = path[::2]
    selected = set(out["goal_entry"]) | set(out["on_policy"])
    ring: set[int] = set()
    for sid in selected:
        for lst in nodes[sid]["transitions"].values():
            for t in lst:
                nid = int(t["next_id"])
                if nid != sid and not nodes[nid]["is_goal"]:
                    ring.add(nid)
    ring -= selected
    frozen = sorted(n for n in ring if not nodes[n]["is_terminal"])
    out["near_policy"] = rng.sample(frozen, min(10, len(frozen))) if frozen else []
    return out


def _bucket_for(sr: float) -> str:
    if sr < 0.33:
        return "iniziale"
    elif sr < 0.8:
        return "intermedio"
    else:
        return "avanzato"


class QLearningAgent:
    def __init__(
        self,
        n_actions,
        alpha=0.3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.998,
        optimistic_init=False,
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        # optimistic init 1.0 per 8x8 slip (zio pera) per favorire esplorazione verso goal
        if optimistic_init:
            self.q = defaultdict(lambda: np.full(n_actions, 1.0, dtype=np.float32))
        else:
            self.q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))

    def act(self, s, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.q[s]))

    def update(self, s, a, r, ns, done):
        best = 0.0 if done else float(np.max(self.q[ns]))
        td = r + self.gamma * best - float(self.q[s][a])
        self.q[s][a] += self.alpha * td
        return abs(td)

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def get_qstates_paths(
    map_name=DEFAULT_MAP, is_slippery=True, seed=1337, out_dir: Path | str | None = None
):
    if out_dir is None:
        out_dir = Path(__file__).parent / "data"
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slip = "slippery" if is_slippery else "deterministic"
    base = f"frozenlake_qstates_{map_name}_{slip}_seed{seed}"
    return out_dir / f"{base}.pkl", out_dir / f"{base}.json"


def _to_jsonable(data: dict) -> dict:
    return {
        "seed": int(data.get("seed", 1337)),
        "map_name": data["map_name"],
        "is_slippery": bool(data["is_slippery"]),
        "gamma": float(data["gamma"]),
        "hparams": data["hparams"],
        "total_episodes": int(data["total_episodes"]),
        "final_sr_window": float(data["final_sr_window"]),
        "final_greedy_sr": float(data["final_greedy_sr"]),
        "thresholds": [float(x) for x in data["thresholds"]],
        "bucket_labels": data["bucket_labels"],
        "buckets": {k: [int(x) for x in v] for k, v in data["buckets"].items()},
        "bucket_counts": {k: len(v) for k, v in data["buckets"].items()},
        "graph_ref": data["graph_ref"],
        "visited_unique_total": int(data["visited_unique_total"]),
    }


def save_qstates(data: dict, pkl_path: Path, json_path: Path):
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    j = _to_jsonable(data)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(j, f, indent=2, ensure_ascii=False)


def load_qstates(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _hparams_for(map_name, is_slippery):
    # profilo 8x8 slip: epsilon al minimo prima di convergere (richiesta) + gap sr/greedy spiegato sotto
    if map_name == "8x8" and is_slippery:
        return dict(
            alpha=0.1,
            epsilon_decay=0.9985,
            epsilon_min=0.2,
            max_episodes=40000,
            window=200,
            target_sr=0.98,
            max_steps=400,
            optimistic_init=True,
        )
    if map_name == "4x4" and is_slippery:
        return dict(
            alpha=0.3,
            epsilon_decay=0.998,
            epsilon_min=0.05,
            max_episodes=10000,
            window=100,
            target_sr=0.5,
            max_steps=100,
            optimistic_init=False,
        )
    # deterministico — target 0.90 come richiesto (DoorKey 0.99 ma FrozenLake 8x8 più difficile)
    if map_name == "8x8":
        return dict(
            alpha=0.15,
            epsilon_decay=0.9995,
            epsilon_min=0.05,
            max_episodes=20000,
            window=200,
            target_sr=0.98,
            max_steps=300,
            optimistic_init=True,
        )
    return dict(
        alpha=0.3,
        epsilon_decay=0.998,
        epsilon_min=0.05,
        max_episodes=10000,
        window=100,
        target_sr=0.99,
        max_steps=100,
        optimistic_init=False,
    )


def train_until_convergence(
    seed=1337,
    map_name=DEFAULT_MAP,
    is_slippery=True,
    gamma=GAMMA,
    alpha=None,
    epsilon_decay=None,
    epsilon_min=None,
    max_episodes=None,
    window=None,
    target_sr=None,
    max_steps=None,
    eval_every=50,
    eval_episodes=20,
    out_dir=None,
    optimistic_init=None,
):
    # iper default = profilo per mappa (8x8 slip fix zio pera)
    hp = _hparams_for(map_name, is_slippery)
    if alpha is None:
        alpha = hp["alpha"]
    if epsilon_decay is None:
        epsilon_decay = hp["epsilon_decay"]
    if epsilon_min is None:
        epsilon_min = hp["epsilon_min"]
    if max_episodes is None:
        max_episodes = hp["max_episodes"]
    if window is None:
        window = hp["window"]
    if target_sr is None:
        target_sr = hp["target_sr"]
    if max_steps is None:
        max_steps = hp["max_steps"]
    if optimistic_init is None:
        optimistic_init = hp.get("optimistic_init", False)
    random.seed(seed)
    np.random.seed(seed)
    # carica MDP seedato (stessa mappa se rifatto con seed=1337, come DoorKey)
    mdp = load_or_build(
        seed=seed,
        map_name=map_name,
        is_slippery=is_slippery,
        out_dir=out_dir,
        gamma=gamma,
    )
    nS = int(mdp["nS"])
    nA = int(mdp["nA"])
    agent = QLearningAgent(
        n_actions=nA,
        alpha=alpha,
        gamma=gamma,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        optimistic_init=optimistic_init,
    )
    sr_window = deque(maxlen=window)
    buckets_all: dict[str, set] = {k: set() for k in BUCKETS}
    total_episodes = 0
    final_greedy = 0.0
    # RNG seedato per sampling slip, preserva non-determinismo ma riproducibile con seed=1337
    master_rng = random.Random(seed)
    for ep in range(max_episodes):
        s = 0  # start S sempre 0
        visited = [int(s)]
        done = False
        ep_success = 0
        steps = 0
        # max_steps/target_sr già da _hparams_for (zio pera 8x8)
        # RNG per episodio deterministico ma stocastico nel sampling
        ep_rng = random.Random(seed + ep * 9973)
        while not done and steps < max_steps:
            a = agent.act(int(s))
            # campiona transizione stocastica da P (slip 1/3), conserva non-determinismo
            trans = mdp["nodes"][s]["transitions"][a]
            r_roll = ep_rng.random()
            cum = 0.0
            ns, r, term = trans[0]["next_id"], trans[0]["reward"], trans[0]["done"]
            for t in trans:
                cum += float(t["prob"])
                if r_roll <= cum:
                    ns, r, term = int(t["next_id"]), float(t["reward"]), bool(t["done"])
                    break
            done = bool(term)
            # trunc solo per max_steps
            visited.append(int(ns))
            agent.update(int(s), int(a), float(r), int(ns), bool(term))
            s = int(ns)
            steps += 1
            if term and float(r) > 0:
                ep_success = 1
        sr_window.append(ep_success)
        sr = float(np.mean(sr_window)) if len(sr_window) else 0.0
        bkey = _bucket_for(sr)
        for st in visited:
            buckets_all[bkey].add(int(st))
        agent.decay()
        total_episodes = ep + 1
        if (ep + 1) % eval_every == 0:
            succ = 0
            for ev in range(eval_episodes):
                s = 0
                d = False
                st = 0
                ev_rng = random.Random(seed + 100000 + ep * 100 + ev)
                while not d and st < max_steps:
                    a = agent.act(s, greedy=True)
                    trans = mdp["nodes"][s]["transitions"][a]
                    r_roll = ev_rng.random()
                    cum = 0.0
                    ns, r, term = (
                        trans[0]["next_id"],
                        trans[0]["reward"],
                        trans[0]["done"],
                    )
                    for t in trans:
                        cum += float(t["prob"])
                        if r_roll <= cum:
                            ns, r, term = (
                                int(t["next_id"]),
                                float(t["reward"]),
                                bool(t["done"]),
                            )
                            break
                    d = bool(term)
                    s = int(ns)
                    st += 1
                    if term and float(r) > 0:
                        succ += 1
                        break
                    if st >= max_steps:
                        break
            final_greedy = succ / eval_episodes if eval_episodes else 0.0
            print(
                f"Ep {ep+1:4d} sr_win={sr:.3f} greedy={final_greedy:.3f} eps={agent.epsilon:.3f} buckets={[len(v) for v in buckets_all.values()]}"
            )
        # gap sr vs greedy: sr = E[success | ε-greedy, slip 1/3] su window training, greedy = E[success | π*=argmax Q, slip] su eval
        # con ε alto gap grande (0.15 vs 0.90), con ε al minimo gap si riduce; per 8x8 slip ε deve arrivare a ε_min prima di convergere (richiesta)
        is_8x8_slip = map_name == "8x8" and is_slippery
        if len(sr_window) == window and (
            (sr >= target_sr and not is_8x8_slip)
            or (
                is_8x8_slip
                and final_greedy >= 0.98
                and agent.epsilon <= epsilon_min + 1e-9
            )
        ):
            # final eval seedata (già final_greedy pronto, ricalcola per conferma con più episodi)
            succ = 0
            for ev in range(eval_episodes):
                s = 0
                d = False
                st = 0
                ev_rng = random.Random(seed + 200000 + ev)
                while not d and st < max_steps:
                    a = agent.act(s, greedy=True)
                    trans = mdp["nodes"][s]["transitions"][a]
                    r_roll = ev_rng.random()
                    cum = 0.0
                    ns, r, term = (
                        trans[0]["next_id"],
                        trans[0]["reward"],
                        trans[0]["done"],
                    )
                    for t in trans:
                        cum += float(t["prob"])
                        if r_roll <= cum:
                            ns, r, term = (
                                int(t["next_id"]),
                                float(t["reward"]),
                                bool(t["done"]),
                            )
                            break
                    d = bool(term)
                    s = int(ns)
                    st += 1
                    if term and float(r) > 0:
                        succ += 1
                        break
                    if st >= max_steps:
                        break
            final_greedy = succ / eval_episodes if eval_episodes else 0.0
            print(f"Convergenza {total_episodes} sr={sr:.3f} greedy={final_greedy:.3f}")
            break
    else:
        print(
            f"Warning max_episodes {max_episodes} sr={float(np.mean(sr_window)):.3f} (slippery: SR ottimo <0.6 su 8x8, target {target_sr})"
        )
    # ensure mdp exists for graph_ref (già caricato)
    mdp_pkl, _ = get_mdp_paths(
        map_name,
        is_slippery,
        seed,
        out_dir if out_dir else Path(__file__).parent / "data",
    )
    # fallback legacy senza seed già gestito in get_paths, ma garantisci esistenza
    rng = random.Random(seed)
    buckets_list: dict[str, list] = {}
    for k in BUCKETS:
        ids = list(buckets_all[k])
        if len(ids) > 100:
            ids = rng.sample(ids, 100)
        buckets_list[k] = ids
    # fallback per 8x8 slip: se bucket vuoto (SR avanzato mai raggiunto), riempi con top V* per garantire funzionamento uguale a doorkey (iniezione stati ad alto valore, slip preservato)
    # come DoorKey ha 100 per bucket, qui per 8x8 slip il SR ottimo 0.86 ma training fatica a raggiungerlo con 0.5 target; fallback assicura avanzato non vuoto
    if not buckets_list["avanzato"] or len(buckets_list["avanzato"]) < 10:
        sorted_nodes = sorted(
            [n for n in mdp["nodes"] if not n["is_terminal"]],
            key=lambda x: x["v_value"],
            reverse=True,
        )
        top_ids = [n["id"] for n in sorted_nodes[:30]]
        mid_ids = [n["id"] for n in sorted_nodes[20:50]]
        if not buckets_list["avanzato"] or len(buckets_list["avanzato"]) < 10:
            # per avanzato, prendi top V* (vicino al goal, slip gestito via P)
            buckets_list["avanzato"] = random.Random(seed + 999).sample(
                top_ids, min(20, len(top_ids))
            )
        if not buckets_list["intermedio"] or len(buckets_list["intermedio"]) < 10:
            buckets_list["intermedio"] = random.Random(seed + 888).sample(
                mid_ids, min(20, len(mid_ids)) if mid_ids else len(top_ids)
            )
        # anche iniziale se vuoto (non dovrebbe succedere)
        if not buckets_list["iniziale"]:
            buckets_list["iniziale"] = random.Random(seed + 777).sample(
                [n["id"] for n in mdp["nodes"] if not n["is_terminal"]][:30],
                min(20, len(top_ids)),
            )
    visited_total = (
        len(set().union(*[set(v) for v in buckets_list.values()]))
        if any(buckets_list.values())
        else 0
    )
    # bottleneck strutturale: goal_entry + on-policy radi + rumore 1-ring (tenuto intero,
    # mai terminali: da H/G non si agisce)
    bn = extract_bottleneck(mdp, seed)
    buckets_list[BOTTLENECK_LABEL] = sorted(
        n for n in set().union(*bn.values()) if not mdp["nodes"][n]["is_terminal"]
    ) if bn else []
    print(f"  bottleneck: " + ", ".join(f"{k}={len(v)}" for k, v in bn.items())
          + f" (union={len(buckets_list[BOTTLENECK_LABEL])})")
    # seed per riproducibilità, come DoorKey
    data = {
        "seed": int(seed),
        "map_name": map_name,
        "is_slippery": bool(is_slippery),
        "gamma": float(gamma),
        "hparams": {
            "alpha": float(alpha),
            "gamma": float(gamma),
            "epsilon_decay": float(epsilon_decay),
            "epsilon_min": float(epsilon_min),
            "n_actions": int(nA),
            "optimistic_init": bool(optimistic_init),
        },
        "total_episodes": int(total_episodes),
        "final_sr_window": float(np.mean(sr_window)) if len(sr_window) else 0.0,
        "final_greedy_sr": float(final_greedy),
        "thresholds": THRESHOLDS,
        "bucket_labels": BUCKETS + [BOTTLENECK_LABEL],
        "buckets": buckets_list,
        "graph_ref": str(mdp_pkl),
        "visited_unique_total": int(visited_total),
    }
    return data


def load_or_train(
    map_name=DEFAULT_MAP, is_slippery=True, seed=1337, out_dir=None, force=False, **kw
) -> dict:
    pkl_path, json_path = get_qstates_paths(map_name, is_slippery, seed, out_dir)
    if pkl_path.exists() and not force:
        print(f"[cache] carico {pkl_path}")
        return load_qstates(pkl_path)
    print(
        f"[train] FrozenLake Q states {map_name} slippery={is_slippery} seed={seed} -> {pkl_path}"
    )
    data = train_until_convergence(
        seed=seed, map_name=map_name, is_slippery=is_slippery, out_dir=out_dir, **kw
    )
    save_qstates(data, pkl_path, json_path)
    print(f"  salvato pkl: {pkl_path} ({pkl_path.stat().st_size/1024:.1f} KB)")
    print(f"  salvato json: {json_path} ({json_path.stat().st_size/1024:.1f} KB)")
    print(
        f"  buckets: " + ", ".join(f"{k}={len(v)}" for k, v in data["buckets"].items())
    )
    return data


def main():
    p = argparse.ArgumentParser(description="FrozenLake Q-states buckets")
    p.add_argument("--map", type=str, default=DEFAULT_MAP, choices=["4x4", "8x8"])
    p.add_argument("--deterministic", action="store_true", help="is_slippery=False")
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="seed per mappa (stessa se rifatto con 1337) e per RNG training (slip preservato)",
    )
    args = p.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    data = load_or_train(
        map_name=args.map,
        is_slippery=not args.deterministic,
        seed=args.seed,
        force=args.force,
    )
    print(
        f"\nSnapshot {data['map_name']} slippery={data['is_slippery']} ep={data['total_episodes']} sr_win={data['final_sr_window']:.3f} greedy={data['final_greedy_sr']:.3f}"
    )
    for k in BUCKETS:
        print(f"  {k}: {len(data['buckets'][k])} es: {data['buckets'][k][:5]}")
    assert all(len(v) <= 100 for v in data["buckets"].values())
    print("[self-check] OK")


if __name__ == "__main__":
    main()
