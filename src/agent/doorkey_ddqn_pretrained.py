#!/usr/bin/env python3
"""DDQN con pretraining supervisionato su Q_LLM (stile DQfD-lite).

Input:  thesis/graph/data/llm_results_*.json  (righe (s, a, v_llm) = Q-function LLM)
Fasi:   1) regressione MSE Q(s,a) -> v_llm per epoche fino a convergenza (no env)
        2) training online DDQN con expert permanenti nel replay (mai sovrascritti,
           campionati con priorita' piu' alta). Nessuna loss aggiuntiva.
Output: thesis/graph/data/ddqn_pretrained_seed_X[_tag].json  (hist + meta)

Riutilizza DDQNNet/train/evaluate/to_vec esistenti; seed = quello del JSON.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import gymnasium as gym
import minigrid  # noqa: F401  (registra gli env MiniGrid)

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from ExperienceReplayBuffer import ExperienceReplayBuffer, Experience  # noqa: E402
from env.view_wrapper import DoorKeyViewSystem  # noqa: E402
from doorkey_state import encode, to_vec, build_potential  # noqa: E402
from doorkey_ddqn import DDQNAgent, train, evaluate  # noqa: E402

DATA_DIR = Path(__file__).parent.parent / "graph" / "data"
ACTION_IDX = {"left": 0, "right": 1, "forward": 2, "pickup": 3, "drop": 4, "toggle": 5}
STAGE_IDX = {"find_key": 0, "open_door": 1, "reach_goal": 2}
STAGE_TARGET = {"find_key": "key_pos", "open_door": "door_pos", "reach_goal": "goal_pos"}


class ProtectedPER(ExperienceReplayBuffer):
    """PER con primi n_expert slot permanenti (mai sovrascritti), stile DQfD.

    Gli expert occupano [0, n_expert); l'online scrive/wrappa solo su
    [n_expert, buffer_size). Il campionamento garantisce expert_frac del
    batch dalla zona expert (priorita' PER + boost iniziale); pesi IS invariati.
    """

    def __init__(self, batch_size, buffer_size, alpha, random_state,
                 n_expert=0, expert_frac=0.25):
        super().__init__(batch_size, buffer_size, alpha, random_state)
        assert 0 <= n_expert < buffer_size, "serve buffer_size > n_expert"
        self.n_expert = int(n_expert)
        self.expert_frac = float(expert_frac)
        self.last_expert_frac = None  # frazione expert nell'ultimo batch (per il log)
        self.expert_sampled_total = 0

    def _in_full(self):
        return self.n_expert and self._buffer_length >= self._buffer_size

    def add(self, experience):
        if self._in_full() and self._ptr < self.n_expert:
            self._ptr = self.n_expert  # wrap solo nella regione online
        super().add(experience)
        if self._in_full() and self._ptr < self.n_expert:
            self._ptr = self.n_expert

    def sample(self, beta):
        if not self.n_expert or self._buffer_length <= self.n_expert or self.expert_frac <= 0:
            idxs, exps, w = super().sample(beta)
            self.last_expert_frac = (1.0 if (self.n_expert and self._buffer_length <= self.n_expert)
                                     else 0.0)
            self.expert_sampled_total += int((idxs < self.n_expert).sum()) if self.n_expert else 0
            return idxs, exps, w
        ps = self._buffer[:self._buffer_length]["priority"].astype(np.float64)
        ps = np.clip(np.nan_to_num(ps, nan=1.0, posinf=1e6, neginf=1e-6), 1e-6, 1e6)
        pe = np.power(ps, self._alpha)
        s = pe.sum()
        probs = pe / s if np.isfinite(s) and s != 0 else np.ones_like(pe) / len(pe)
        k = min(int(round(self.batch_size * self.expert_frac)), self.n_expert)
        k = max(k, 1)
        es = probs[:self.n_expert].sum()
        e_p = probs[:self.n_expert] / es if es > 0 else np.ones(self.n_expert) / self.n_expert
        o_p = probs[self.n_expert:]
        os = o_p.sum()
        o_p = o_p / os if os > 0 else np.ones_like(o_p) / len(o_p)
        idxs = np.concatenate([
            self._random_state.choice(self.n_expert, size=k, replace=True, p=e_p),
            self._random_state.choice(
                np.arange(self.n_expert, self._buffer_length),
                size=self.batch_size - k, replace=True, p=o_p),
        ])
        self._random_state.shuffle(idxs)
        self.last_expert_frac = k / self.batch_size
        self.expert_sampled_total += int((idxs < self.n_expert).sum())
        experiences = self._buffer["experience"][idxs]
        w = (self._buffer_length * np.clip(probs[idxs], 1e-12, 1.0)) ** -beta
        w = np.nan_to_num(w, nan=1.0, posinf=1e6, neginf=1e-6)
        m = w.max()
        return idxs, experiences, w / m if m != 0 else w


def resolve_input(name_or_path):
    p = Path(name_or_path) if name_or_path else None
    if p is None:
        cands = sorted(DATA_DIR.glob("llm_results*.json"))
        assert cands, f"nessun llm_results*.json in {DATA_DIR}"
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
        for q in (p, DATA_DIR / p, DATA_DIR / p.name):
            if q.exists():
                return q
        assert False, f"file non trovato: {p}"
    assert p.exists(), f"file non trovato: {p}"
    return p


def load_expert(input_path):
    rows = json.loads(input_path.read_text())
    seeds = {r["seed"] for r in rows}
    assert len(seeds) == 1, f"seed multipli nel file: {seeds}"
    seed = seeds.pop()
    mdp_path = DATA_DIR / f"mdp_8x8_seed{seed}.json"
    assert mdp_path.exists(), f"serve {mdp_path} per key/door/goal_pos"
    gi = json.loads(mdp_path.read_text())["grid_info"]
    targets = {"key_pos": tuple(gi["key_pos"]), "door_pos": tuple(gi["door_pos"]),
               "goal_pos": tuple(gi["goal_pos"])}
    agg = {}  # (vec, action) -> [somma_v, conteggio]: dedup Q(s,a), media sui duplicati
    n_clipped = 0
    for r in rows:
        t = targets[STAGE_TARGET[r["stage"]]]
        st = (t[0] - r["x"], t[1] - r["y"], r["dir"], STAGE_IDX[r["stage"]])
        key = (tuple(float(x) for x in to_vec(st)), ACTION_IDX[r["action"]])
        v = float(r["v_llm"])
        assert v == v and abs(v) != float("inf"), f"v_llm non finito: {v}"
        if not 0.0 <= v <= 1.0:  # normalizza target allo stesso [0,1] dei ritorni DDQN
            v = min(max(v, 0.0), 1.0)
            n_clipped += 1
        if key in agg:
            agg[key][0] += v
            agg[key][1] += 1
        else:
            agg[key] = [v, 1]
    expert = [(np.array(k[0], dtype=np.float32), k[1], s / n) for k, (s, n) in agg.items()]
    info = {"n_rows": len(rows), "n_dups": len(rows) - len(expert), "n_clipped": n_clipped}
    return seed, expert, info


def pretrain(agent, expert, lr, max_epochs, patience, min_delta, log_every):
    # ponytail: done=True => target TD = v_llm qui, e nel replay online (stessa regressione)
    S = torch.as_tensor(np.array([e[0] for e in expert]))
    A = torch.tensor([e[1] for e in expert], dtype=torch.int64).unsqueeze(1)
    V = torch.tensor([e[2] for e in expert], dtype=torch.float32).unsqueeze(1)
    S, A, V = S.to(agent.device), A.to(agent.device), V.to(agent.device)
    opt = torch.optim.Adam(agent.policy.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(torch.initial_seed() % (2 ** 32))
    best, wait, n = float("inf"), 0, len(expert)
    for ep in range(1, max_epochs + 1):
        tot = 0.0
        for idx in torch.randperm(n, generator=gen).split(64):
            opt.zero_grad()
            loss = ((agent.policy(S[idx]).gather(1, A[idx]) - V[idx]) ** 2).mean()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        mse = tot / n
        if ep % log_every == 0 or ep == 1:
            print(f"  pre ep {ep:4d}: mse={mse:.6f}")
        if mse < best - min_delta:
            best, wait = mse, 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  pretrain convergito: ep={ep} mse={mse:.6f}")
                return ep, mse
    print(f"  pretrain max_epochs: mse={mse:.6f}")
    return max_epochs, mse


def main():
    ap = argparse.ArgumentParser(description="DDQN con pretraining supervisionato su Q_LLM")
    ap.add_argument("--input", default=None, help="file JSON in thesis/graph/data/ (o path)")
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--eps_decay", type=float, default=0.999)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--buffer_size", type=int, default=300000)
    ap.add_argument("--expert_frac", type=float, default=0.25)
    ap.add_argument("--boost", type=float, default=5.0)
    ap.add_argument("--pre_lr", type=float, default=1e-3)
    ap.add_argument("--pre_epochs", type=int, default=500)
    ap.add_argument("--pre_patience", type=int, default=20)
    ap.add_argument("--pre_min_delta", type=float, default=1e-5)
    ap.add_argument("--tag", default=None, help="suffisso nome output (default: slug modello)")
    ap.add_argument("--target_clip", type=float, nargs=2, default=[0.0, 1.0], metavar=("LO", "HI"),
                    help="clip dei target TD allo stesso intervallo dei v_llm")
    ap.add_argument("--env_seed", type=int, default=None,
                    help="seed env fisso per tutti gli episodi (default: mappe casuali)")
    ap.add_argument("--max_steps", type=int, default=450,
                    help="step max per episodio (0 = limite env, 640)")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    input_path = resolve_input(args.input)
    seed, expert, einfo = load_expert(input_path)
    print(f"Input: {input_path.name} | seed={seed} | "
          f"expert={len(expert)} univoci ({einfo['n_rows']} righe, "
          f"{einfo['n_dups']} duplicati mediati, {einfo['n_clipped']} clip)")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.selfcheck:
        assert len(expert) > 0, "nessun expert caricato"
        keys = [(tuple(s.tolist()), a) for s, a, _ in expert]
        assert len(set(keys)) == len(keys), "duplicati (s,a) residui!"
        assert all(v == v and 0.0 <= v <= 1.0 for _, _, v in expert)
        buf = ProtectedPER(8, len(expert) + 10, 0.4,
                           np.random.RandomState(seed), len(expert), 0.25)
        for s, a, v in expert:
            buf.add_with_priority(Experience(s, a, v, s, True), 5.0)
        snap = buf._buffer[:len(expert)]["priority"].copy()
        for i in range(len(expert) + 50):  # forza wrap multipli
            buf.add(Experience(expert[0][0], 0, 0.0, expert[0][0], False))
        assert (buf._buffer[:len(expert)]["priority"] == snap).all(), "expert sovrascritti!"
        _, _, _ = buf.sample(0.5)
        idxs, _, _ = buf.sample(0.5)
        assert (idxs < len(expert)).mean() > 0.1, "expert mai campionati"
        print(f"Selfcheck OK (expert={len(expert)}, frac_osservata={(idxs < len(expert)).mean():.2f})")
        env = DoorKeyViewSystem(gym.make("MiniGrid-DoorKey-8x8-v0"))
        env.reset(seed=seed)
        s0 = encode(env)
        env.reset(seed=seed)
        assert encode(env) == s0, "reset con stesso seed non deterministico"
        env.close()
        print(f"Selfcheck env_seed OK (layout seed={seed} deterministico: {s0})")
        return

    env_id = "MiniGrid-DoorKey-8x8-v0"
    base = gym.make(env_id)
    n_actions = int(base.action_space.n)
    base.close()
    env = DoorKeyViewSystem(gym.make(env_id))
    env.reset(seed=seed)

    agent = DDQNAgent(action_dim=n_actions, lr=args.lr, gamma=args.gamma,
                      eps_decay=args.eps_decay, buffer_size=args.buffer_size,
                      batch_size=args.batch_size, target_clip=tuple(args.target_clip))
    agent.memory = ProtectedPER(args.batch_size, args.buffer_size, 0.4,
                                np.random.RandomState(seed), len(expert), args.expert_frac)

    print(f"Pretraining su {len(expert)} coppie (s,a,V_LLM)...")
    pre_epochs, pre_mse = pretrain(agent, expert, args.pre_lr, args.pre_epochs,
                                   args.pre_patience, args.pre_min_delta, args.log_every)
    agent.update_target()  # allinea target alla policy preallenata
    for s, a, v in expert:  # zona permanente [0, n_expert), priorita' boostata
        agent.memory.add_with_priority(Experience(s, a, v, s, True), args.boost)
    print(f"Buffer: {len(agent.memory)} expert permanenti (frac={args.expert_frac}, boost={args.boost})")

    print(f"Training online: {args.episodes} episodi "
          f"({'env_seed fisso' if args.env_seed is not None else 'mappe casuali'})")
    hist = train(env, agent, args.episodes, build_potential({}), [], args.max_steps,
                 args.log_every, fixed_env_seed=args.env_seed)
    if args.env_seed is not None:
        eval_seeds, eval_mode = [args.env_seed] * 200, "fixed"
    else:
        eval_seeds, eval_mode = random.sample(range(100000), 200), "random"
    eval_sr = evaluate(env, agent, len(eval_seeds), eval_seeds)
    print(f"Eval SR: {eval_sr:.3f}")
    env.close()

    tag = args.tag
    if tag is None:  # slug modello dal nome file, es. gpt-oss_120b
        tag = input_path.stem.split(f"seed{seed}_", 1)[-1].strip("_") or None
    out = DATA_DIR / f"ddqn_pretrained_seed_{seed}{('_' + tag) if tag else ''}.json"
    out.write_text(json.dumps({
        "algo": "ddqn_pretrained", "seed": seed, "input_file": input_path.name,
        "env_seed": args.env_seed, "eval_seeds": eval_mode,
        "n_expert": len(expert), "expert_rows": einfo["n_rows"],
        "expert_dups_aggregated": einfo["n_dups"], "expert_clipped": einfo["n_clipped"],
        "pretrain": {"epochs": pre_epochs, "mse_final": pre_mse, "lr": args.pre_lr,
                     "max_epochs": args.pre_epochs, "patience": args.pre_patience},
        "hparams": {"episodes": args.episodes, "lr": args.lr, "gamma": args.gamma,
                    "eps_decay": args.eps_decay, "batch_size": args.batch_size,
                    "buffer_size": args.buffer_size, "expert_frac": args.expert_frac,
                    "boost": args.boost, "max_steps": args.max_steps,
                    "target_clip": list(args.target_clip)},
        "eval_sr": eval_sr, "eval_episodes": len(eval_seeds),
        "rewards": [float(x) for x in hist["rewards"]],
        "losses": [float(x) for x in hist["losses"]],
        "successes": [float(x) for x in hist["successes"]],
        "ep_lengths": [int(x) for x in hist["ep_lengths"]],
    }))
    print(f"Risultati salvati: {out}")


if __name__ == "__main__":
    main()
