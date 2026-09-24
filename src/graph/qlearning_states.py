#!/usr/bin/env python3
"""
Q-learning tabulare su seed singolo (default 1337) per DoorKey 8x8.

- Train fino a success rate >=0.99 (finestra mobile 100)
- Per ogni scaglione "iniziale" / "intermedio" / "avanzato" (0-0.33 / 0.33-0.8 / 0.8-1.0)
  salva max 100 stati visitati (node_id del grafo mdp_graph), campionati
  in modo randomico per varietà e indipendenti dallo storico, max 1 volta per bucket
- Riferimento ai nodi generati in mdp_graph (mdp["index"][state] -> node_id)
- Training solo se file snapshot non esiste (cache), altrimenti carica
- Tutte le 7 azioni, hyper migliori per convergenza rapida su singolo MDP
- Valutazione greedy ogni 50 episodi (log, non per bucketing)

Uso:
    python -m graph.qlearning_states --seed 1337 --size 8
    python -m graph.qlearning_states --seed 1337 --size 8 --force
    from graph.qlearning_states import load_or_train
    data = load_or_train(seed=1337, size=8)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict, deque
from pathlib import Path

_THIS = Path(__file__).resolve()
_SRC = _THIS.parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
from graph.mdp_graph import ACTIONS_ALL, load_or_build, get_paths as get_mdp_paths
from paths import CACHE_DIR

# ---------------------------------------------------------------------------
# Costanti / bucket per fase di apprendimento (nomi adatti a contesto universitario)
# ---------------------------------------------------------------------------
BUCKETS = ["iniziale", "intermedio", "avanzato"]  # 0-0.33, 0.33-0.8, 0.8-1.0
THRESHOLDS = [0.33, 0.8]  # avanzato è >=0.8 fino a 1.0
DEFAULT_SEED = 1337
DEFAULT_SIZE = 8
GAMMA = 0.99

def _bucket_for(sr: float) -> str:
    if sr < 0.33:
        return "iniziale"
    elif sr < 0.8:
        return "intermedio"
    else:
        return "avanzato"


def _stage_idx(state: tuple, gi: dict) -> int:
    """0=find_key, 1=open_door, 2=reach_goal usa gi per goal, altrimenti has_key/door_open"""
    x, y = state[0], state[1]
    has_key, door_open = bool(state[3]), bool(state[4])
    gx, gy = gi["goal_pos"]
    if (x, y) == (gx, gy):
        return 2
    if not has_key:
        return 0
    if has_key and not door_open:
        return 1
    return 2


class QLearningAgent:
    def __init__(self, n_actions, alpha=0.3, gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.998):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))

    def act(self, state, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        # argmax deterministico, tie -> primo
        return int(np.argmax(self.q[state]))

    def update(self, s, a, r, s_next, done):
        best_next = 0.0 if done else float(np.max(self.q[s_next]))
        td = r + self.gamma * best_next - float(self.q[s][a])
        self.q[s][a] += self.alpha * td
        return abs(td)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def get_qstates_paths(seed: int = DEFAULT_SEED, size: int = DEFAULT_SIZE, out_dir: Path | str | None = None) -> Path:
    if out_dir is None:
        out_dir = CACHE_DIR
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"qlearning_states_{size}x{size}_seed{seed}.json"


def _to_jsonable_qstates(data: dict) -> dict:
    # buckets: OrderedSet/list of node_ids -> list
    buckets_j = {k: [int(x) for x in v] for k, v in data["buckets"].items()}
    out = {
        "seed": int(data["seed"]),
        "size": int(data["size"]),
        "gamma": float(data["gamma"]),
        "hparams": data["hparams"],
        "total_episodes": int(data["total_episodes"]),
        "final_sr_window": float(data["final_sr_window"]),
        "final_greedy_sr": float(data["final_greedy_sr"]),
        "thresholds": [float(x) for x in data["thresholds"]],
        "bucket_labels": data["bucket_labels"],
        "buckets": buckets_j,
        "bucket_counts": {k: len(v) for k, v in buckets_j.items()},
        "graph_ref": data["graph_ref"],
        "visited_unique_total": int(data["visited_unique_total"]),
    }
    # opzionali per tracciabilità garanzia (solo se presenti)
    if "limit_per_bucket" in data:
        out["limit_per_bucket"] = int(data["limit_per_bucket"])
    if "significant_only" in data:
        out["significant_only"] = bool(data["significant_only"])
    if "buckets_all_counts" in data:
        out["buckets_all_counts"] = {k: int(v) for k, v in data["buckets_all_counts"].items()}
    if "critical_counts" in data:
        out["critical_counts"] = {k: int(v) for k, v in data["critical_counts"].items()}
    if "critical_ids" in data:
        out["critical_ids"] = {k: [int(x) for x in v] for k, v in data["critical_ids"].items()}
    return out


def save_qstates(data: dict, json_path: Path):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable_qstates(data), f, indent=2, ensure_ascii=False)


def load_qstates(json_path: Path) -> dict:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def train_until_convergence(
    seed: int = DEFAULT_SEED,
    size: int = DEFAULT_SIZE,
    gamma: float = GAMMA,
    alpha: float = 0.3,
    epsilon_decay: float = 0.998,
    epsilon_min: float = 0.05,
    max_episodes: int = 10000,
    window: int = 100,
    eval_every: int = 50,
    eval_episodes: int = 20,
    out_dir: Path | str | None = None,
    limit_per_bucket: int = 100,
    significant_only: bool = False,
):
    """
    Train tabulare su seed fissato fino a SR finestra >=0.99.
    Ritorna dict con buckets iniziale/intermedio/avanzato (max limit_per_bucket node_id ciascuno).
    Con significant_only=True salva solo i critici cambio-stage calpestati (fino a limit_per_bucket).
    """
    # carica grafo per index mapping
    mdp = load_or_build(seed=seed, size=size, gamma=gamma)
    index = mdp["index"]
    n_actions = len(ACTIONS_ALL)  # 7 azioni, evita gym env
    # seed deterministico per varietà e training
    random.seed(seed)
    np.random.seed(seed)
    agent = QLearningAgent(n_actions=n_actions, alpha=alpha, gamma=gamma, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min)

    sr_window = deque(maxlen=window)
    # per varietà randomica: colleziona TUTTI gli unici per bucket, poi campiona 100 a fine
    buckets_all: dict[str, set] = {k: set() for k in BUCKETS}
    # critical = stati post-transizione stage effettivamente calpestati (garantiti se visitati)
    critical_all: dict[str, set] = {k: set() for k in BUCKETS}

    total_episodes = 0
    final_greedy_sr = 0.0

    # training su grafo diretto per velocità (evita env.step gym)
    # usa mdp transitions, non env, ma mantiene stessa semantica deterministica
    start_state = (mdp["grid_info"]["start_pos"][0], mdp["grid_info"]["start_pos"][1], mdp["grid_info"]["start_dir"], False, False)
    # mappa action index -> n_actions (7) ; ACTIONS_ALL già allineato a env.action_space.n

    for ep in range(max_episodes):
        s = start_state
        visited_this_ep: list[tuple] = [s]
        # traccia transizioni effettivamente calpestate per detect stage-change
        critical_this_ep: set[int] = set()
        done = False
        ep_success = 0
        steps = 0
        max_steps = 1000  # come env._max_episode_steps per 8x8
        while not done and steps < max_steps:
            a = agent.act(s)
            # transizione via grafo (deterministica, veloce) evita gym
            sid = index[s]
            trans = mdp["nodes"][sid]["transitions"][a] if a in mdp["nodes"][sid]["transitions"] else mdp["nodes"][sid]["transitions"][0]
            ns = trans["next_state"]
            r = float(trans["reward"])
            done = bool(trans["done"])
            # detect stage-change solo se edge effettivamente percorso
            # stage su s/ns reali, non teorici; include anche terminale (reach_goal) dove stage non cambia (2->2) ma done=True
            try:
                is_stage_jump = _stage_idx(ns, mdp["grid_info"]) > _stage_idx(s, mdp["grid_info"])
                is_terminal = bool(done)  # done==True <=> ns sul goal
                if is_stage_jump or is_terminal:
                    nid_ns = index.get(ns)
                    if nid_ns is not None:
                        critical_this_ep.add(nid_ns)
            except Exception:
                pass
            visited_this_ep.append(ns)
            agent.update(s, a, r, ns, done)
            s = ns
            steps += 1
            if done:
                ep_success = 1
                break
        # se non done per max_steps, ep_success resta 0 (trunc)
        sr_window.append(ep_success)
        sr = float(np.mean(sr_window)) if len(sr_window) else 0.0

        # assegna stati visitati al bucket corrente — accumula tutti per varietà
        bkey = _bucket_for(sr)
        bset = buckets_all[bkey]
        for st in visited_this_ep:
            nid = index.get(st)
            if nid is not None:
                bset.add(nid)
        # garantiti solo se calpestati in questo episodio/bucket
        cset = critical_all[bkey]
        for nid in critical_this_ep:
            # nid già garantito calpestato perché in visited_this_ep
            cset.add(nid)

        agent.decay_epsilon()
        total_episodes = ep + 1

        # valutazione greedy ogni eval_every (su grafo, non env)
        if (ep + 1) % eval_every == 0:
            # greedy su grafo
            succ = 0
            for _ in range(eval_episodes):
                gs = start_state
                gdone = False
                gsteps = 0
                while not gdone and gsteps < max_steps:
                    ga = agent.act(gs, greedy=True)
                    gsid = index[gs]
                    gtrans = mdp["nodes"][gsid]["transitions"][ga] if ga in mdp["nodes"][gsid]["transitions"] else mdp["nodes"][gsid]["transitions"][0]
                    gs = gtrans["next_state"]
                    gdone = bool(gtrans["done"])
                    gsteps += 1
                    if gdone:
                        succ += 1
                        break
            final_greedy_sr = succ / eval_episodes if eval_episodes else 0.0
            print(f"Ep {ep+1:4d} sr_win={sr:.3f} greedy={final_greedy_sr:.3f} eps={agent.epsilon:.3f} buckets_all={[len(v) for v in buckets_all.values()]}")

        if len(sr_window) == window and sr >= 0.99:
            # greedy finale
            succ = 0
            for _ in range(eval_episodes):
                gs = start_state
                gdone = False
                gsteps = 0
                while not gdone and gsteps < max_steps:
                    ga = agent.act(gs, greedy=True)
                    gsid = index[gs]
                    gtrans = mdp["nodes"][gsid]["transitions"][ga] if ga in mdp["nodes"][gsid]["transitions"] else mdp["nodes"][gsid]["transitions"][0]
                    gs = gtrans["next_state"]
                    gdone = bool(gtrans["done"])
                    gsteps += 1
                    if gdone:
                        succ += 1
                        break
            final_greedy_sr = succ / eval_episodes if eval_episodes else 0.0
            print(f"Convergenza raggiunta a ep {total_episodes} sr={sr:.3f} greedy={final_greedy_sr:.3f}")
            break
    else:
        print(f"Warning: max_episodes {max_episodes} raggiunto con sr={float(np.mean(sr_window)):.3f}")

    # campionamento con garanzia critical (solo calpestati) — limit_per_bucket per bucket
    rng = random.Random(seed)
    buckets_list: dict[str, list] = {}
    critical_counts: dict[str, int] = {}
    N = int(limit_per_bucket) if limit_per_bucket and limit_per_bucket > 0 else 100
    for k in BUCKETS:
        # critical ⊆ visited per costruzione, nessun teorico
        crit = list(critical_all[k] & buckets_all[k])
        rest = list(buckets_all[k] - set(crit))
        rng.shuffle(crit)
        rng.shuffle(rest)
        critical_counts[k] = len(crit)
        if significant_only:
            # solo critici, fino a N (minimo per coprire tutti i cambio-stage)
            if len(crit) <= N:
                sampled = crit
            else:
                sampled = rng.sample(crit, N)
        else:
            if len(buckets_all[k]) <= N:
                # meno di N visitati: prendi tutti (crit già dentro, nessun padding teorico)
                sampled = crit + rest
                seen = set()
                uniq = []
                for nid in sampled:
                    if nid not in seen:
                        seen.add(nid)
                        uniq.append(nid)
                sampled = uniq
            elif len(crit) >= N:
                # più crit di N (es. N=6 ma crit=8): campiona N crit
                sampled = rng.sample(crit, N)
            else:
                need = N - len(crit)
                sampled_rest = rng.sample(rest, min(need, len(rest))) if rest else []
                sampled = crit + sampled_rest
        buckets_list[k] = sampled
    # log garanzia
    for k in BUCKETS:
        print(f"[bucket {k}] visitati={len(buckets_all[k])} critical_calpestati={critical_counts[k]} -> sampled {len(buckets_list[k])}/{N} (crit garantiti {min(critical_counts[k], len(buckets_list[k]))}){' [significant_only]' if significant_only else ''}")
    visited_unique_total = len(set().union(*[set(v) for v in buckets_list.values()])) if any(buckets_list.values()) else 0

    json_ref = get_mdp_paths(seed, size, out_dir if out_dir else CACHE_DIR)
    data = {
        "seed": seed,
        "size": size,
        "gamma": gamma,
        "hparams": {"alpha": alpha, "gamma": gamma, "epsilon_decay": epsilon_decay, "epsilon_min": epsilon_min, "n_actions": n_actions},
        "total_episodes": total_episodes,
        "final_sr_window": float(np.mean(sr_window)) if len(sr_window) else 0.0,
        "final_greedy_sr": float(final_greedy_sr),
        "thresholds": THRESHOLDS,
        "bucket_labels": BUCKETS,
        "buckets": buckets_list,
        "limit_per_bucket": N,
        "significant_only": bool(significant_only),
        "buckets_all_counts": {k: len(v) for k, v in buckets_all.items()},
        "critical_counts": critical_counts,
        "critical_ids": {k: sorted(list(critical_all[k] & buckets_all[k])) for k in BUCKETS},
        "graph_ref": str(json_ref),
        "visited_unique_total": visited_unique_total,
    }
    return data


def load_or_train(
    seed: int = DEFAULT_SEED,
    size: int = DEFAULT_SIZE,
    out_dir: Path | str | None = None,
    force: bool = False,
    **train_kwargs,
) -> dict:
    """
    Se file snapshot esiste e non force, carica; altrimenti train e salva.
    Ritorna dict con buckets iniziale/intermedio/avanzato (node_id).
    """
    json_path = get_qstates_paths(seed, size, out_dir)
    if json_path.exists() and not force:
        print(f"[cache] carico {json_path}")
        return load_qstates(json_path)
    print(f"[train] Q-learning seed={seed} size={size} -> {json_path}")
    data = train_until_convergence(seed=seed, size=size, out_dir=out_dir, **train_kwargs)
    save_qstates(data, json_path)
    print(f"  salvato json: {json_path} ({json_path.stat().st_size/1024:.1f} KB)")
    print(f"  buckets: " + ", ".join(f"{k}={len(v)}" for k, v in data['buckets'].items()))
    return data


def main():
    parser = argparse.ArgumentParser(description="Q-learning tabulare seed singolo con bucket iniziale/intermedio/avanzato (max N node_id, crit garantiti)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed unico (default 1337)")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, choices=[6, 8, 16], help="size mappa")
    parser.add_argument("--force", action="store_true", help="forza retrain anche se esiste")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--epsilon-decay", type=float, default=0.998, dest="epsilon_decay")
    parser.add_argument("--max-episodes", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--limit-per-bucket", type=int, default=100, dest="limit_per_bucket", help="max stati per bucket, crit garantiti prima (default 100, min per tutti i cambio-stage ~8)")
    parser.add_argument("--significant-only", action="store_true", dest="significant_only", help="salva solo i critici cambio-stage calpestati (fino a limit-per-bucket)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data = load_or_train(
        seed=args.seed,
        size=args.size,
        force=args.force,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_decay=args.epsilon_decay,
        max_episodes=args.max_episodes,
        eval_every=args.eval_every,
        limit_per_bucket=args.limit_per_bucket,
        significant_only=args.significant_only,
    )
    print(f"\nSnapshot seed={data['seed']} size={data['size']} ep={data['total_episodes']} sr_win={data['final_sr_window']:.3f} greedy={data['final_greedy_sr']:.3f}")
    for k in BUCKETS:
        print(f"  {k}: {len(data['buckets'][k])} stati (max 100) es: {data['buckets'][k][:5]}")

    # self-check
    Nchk = int(data.get("limit_per_bucket", 100))
    assert all(len(v) <= Nchk for v in data["buckets"].values()), f"bucket >{Nchk}"
    if data.get("significant_only"):
        # solo critici
        for k in BUCKETS:
            assert set(data["buckets"][k]).issubset(set(data.get("critical_ids", {}).get(k, []))), f"significant_only ma bucket {k} ha non-critici"
    # verifica che node_id esistano nel grafo
    from graph.mdp_graph import load_mdp
    mdp_json = get_mdp_paths(args.seed, args.size)
    if mdp_json.exists():
        mdp = load_mdp(mdp_json)
        max_id = len(mdp["nodes"]) - 1
        for k, lst in data["buckets"].items():
            for nid in lst:
                assert 0 <= nid <= max_id, f"node_id {nid} fuori range"
        # garanzia: critical ⊆ buckets se N >= |crit|, altrimenti buck ⊆ crit (campionato)
        if "critical_ids" in data:
            Nchk2 = int(data.get("limit_per_bucket", Nchk))
            for k in BUCKETS:
                crit = set(data["critical_ids"].get(k, []))
                buck = set(data["buckets"].get(k, []))
                if len(crit) <= Nchk2:
                    assert crit.issubset(buck) or len(buck) == 0, f"critical non garantito in bucket {k}: {crit - buck}"
                else:
                    assert buck.issubset(crit), f"con N<{len(crit)} bucket {k} deve essere solo crit ma got {buck - crit}"
                # solo calpestati: critical è già & buckets_all, verifica
                if "buckets_all_counts" in data:
                    pass
    print("[self-check] OK")
    if "critical_counts" in data:
        print(f"critical_counts: {data['critical_counts']} | buckets_all_counts: {data.get('buckets_all_counts')}")


if __name__ == "__main__":
    main()
