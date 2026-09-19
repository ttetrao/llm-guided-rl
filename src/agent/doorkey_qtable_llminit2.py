#!/usr/bin/env python3
"""LLM-init (iperparametri miei) vs vanilla stessi-hp vs vanilla standard, un unico grafico.

Uso:
    python3 -m thesis.agent.doorkey_qtable_llminit2 --input <llm_results>.json --seed 1337 \\
        --episodes 1500 --alpha 0.5 --gamma 0.9 --compare
    python3 -m thesis.agent.doorkey_qtable_llminit2 --input ... --seed 1337
        # senza --compare: sola run LLM-init con i miei hparams

--compare, 3 run a pari episodi/seed/max_steps (asse x confrontabile):
  A = Q init da V_LLM + miei hparams (cli);
  B = Q da zero + STESSI hparams dei miei (vanilla a pari condizioni);
  C = Q da zero + standard (alpha=0.25 gamma=0.99 eps_decay=0.998 eps_min=0.05,
      dal pilota doorkey_states.py che converge sul seed singolo).
Un solo PNG 2x2 per confronto: SR train (ma100) + eval SR | |TD| medio |
agreement con la policy ottima (VI) | value loss vs V*.
Con --compare escono due PNG: LLM-init vs Vanilla-std (_vs_std) e LLM-init
vs Vanilla (same hp) (_vs_samehp). Tutti gli iperparametri nel box in basso.
"""
import argparse
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import gymnasium as gym
import minigrid  # noqa: F401, registers MiniGrid envs

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from env.view_wrapper import DoorKeyViewSystem  # noqa: E402
from doorkey_qtable_llminit import (  # noqa: E402
    load_qinit, QLearningAgent, train, evaluate, save_run,
    optimal_policy_metrics, footnote_for, load_opt_mdp,
    plot_cmp, COLORS, ma, cmp_footnote, STD, PAIRS, plot_pairs)
from doorkey_ddqn_pretrained import DATA_DIR, resolve_input  # noqa: E402


def make_title(file_seeds):
    n = len(file_seeds)
    s = ",".join(map(str, file_seeds))
    if len(s) > 24:  # tronca con ... se non ci sta
        s = f"{s[:21].rsplit(',', 1)[0]},..."
    return f"{n} seed{'s' if n > 1 else ''} ({s})"


# ponytail: plot_cmp, COLORS, ma, cmp_footnote importati da llminit (fonte unica)


def make_stem(file_seeds, seed, tag):
    base = (f"qtable_cmp_{len(file_seeds)}seed_s{seed}" if len(file_seeds) > 1
            else f"qtable_cmp_seed{seed}")
    return f"{base}_{tag}" if tag else base


def run_one(env, n_actions, qinit, hp, args, seed, log_tag):
    random.seed(seed)
    np.random.seed(seed)
    agent = QLearningAgent(n_actions, qinit, alpha=hp["alpha"], gamma=hp["gamma"],
                           epsilon_decay=hp["eps_decay"], epsilon_min=hp["eps_min"])
    print(f"Training {log_tag}: {args.episodes} episodi su seed {seed}, {hp}")
    hist = train(env, agent, args.episodes, seed, args.max_steps, args.log_every)
    ev = evaluate(env, agent, args.eval_episodes, seed, args.max_steps)
    return agent, hist, ev


def main():
    ap = argparse.ArgumentParser(description="LLM-init (miei hparams) vs vanilla (standard), un grafico")
    ap.add_argument("--input", default=None)
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--eps_decay", type=float, default=0.995)
    ap.add_argument("--eps_min", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max_init", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=450)
    ap.add_argument("--eval_episodes", type=int, default=100)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--plot", default=None)
    ap.add_argument("--no_plot", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:  # ponytail: unico check su ma + STD + stem + label
        assert list(ma([0, 1, 1, 1])) == [0.0, 0.5, 2 / 3, 0.75]
        assert set(STD) == {"alpha", "gamma", "eps_decay", "eps_min"}
        assert set(COLORS) == {"LLM-init", "Vanilla (same hp)", "Vanilla-std"}
        assert [(a, b, s) for a, b, s in PAIRS] == [
            ("LLM-init", "Vanilla-std", "_vs_std"),
            ("LLM-init", "Vanilla (same hp)", "_vs_samehp")]
        assert make_stem([1337], 1337, None) == "qtable_cmp_seed1337"
        assert make_stem([1, 2, 3, 4, 5], 1337, None) == "qtable_cmp_5seed_s1337"
        assert make_stem([1, 2], 2, "x") == "qtable_cmp_2seed_s2_x"
        assert make_title([1337]) == "1 seed (1337)"
        assert make_title([1, 2, 3, 4, 5]) == "5 seeds (1,2,3,4,5)"
        assert make_title(list(range(30))) == "30 seeds (0,1,2,3,4,5,6,7,8,9,...)"
        print("Selfcheck OK")
        return

    input_path = resolve_input(args.input)
    file_seeds, qinit, einfo = load_qinit(input_path, args.max_init)
    seed = args.seed if args.seed is not None else file_seeds[0]
    file_seed = file_seeds[0]
    print(f"Input: {input_path.name} | seed={seed} | coppie init={len(qinit)}")

    env = DoorKeyViewSystem(gym.make("MiniGrid-DoorKey-8x8-v0"))
    n_actions = int(env.action_space.n)
    mine = {"alpha": args.alpha, "gamma": args.gamma,
            "eps_decay": args.eps_decay, "eps_min": args.eps_min}
    fake = SimpleNamespace(episodes=args.episodes, eval_episodes=args.eval_episodes,
                           max_steps=args.max_steps, max_init=args.max_init, no_plot=True,
                           plot=None)
    stem = make_stem(file_seeds, seed, args.tag)
    if load_opt_mdp is None:
        print("WARN: thesis.graph non importabile, metriche policy ottima saltate")
        opt_mdp = None
    else:
        opt_mdp = load_opt_mdp(seed=seed, size=8, out_dir=DATA_DIR)

    def _opt_for(agent):
        if opt_mdp is None:
            return None
        try:
            return optimal_policy_metrics(agent.q, opt_mdp)
        except Exception as e:
            print(f"WARN: metriche policy ottima saltate: {e}")
            return None

    specs = [("LLM-init", list(qinit.items()), mine, f"{stem}_llminit", einfo)]
    if args.compare:
        specs += [("Vanilla (same hp)", [], mine, f"{stem}_vsame",
                   {**einfo, "n_rows": 0, "n_dups": 0, "n_clipped": 0}),
                  ("Vanilla-std", [], STD, f"{stem}_vstd",
                   {**einfo, "n_rows": 0, "n_dups": 0, "n_clipped": 0})]
    runs = []
    for label, init, hp, name, info in specs:
        agent, hist, ev = run_one(env, n_actions, init, hp, args, seed, label)
        opt = _opt_for(agent)
        save_run(agent, hist, ev, SimpleNamespace(**{**vars(fake), **hp}),
                 name, label, file_seed, seed, info, input_path, opt=opt)
        if opt is not None:
            print(f"{label}: agreement={opt['agreement']:.3f} "
                  f"loss={opt['mean_value_loss']:.4f}")
        runs.append({"label": label, "hist": hist, "ev": ev, "opt": opt,
                     "hp": hp, "n_init": agent.n_init})
    if not args.no_plot:
        base_title = (f"DoorKey 8x8 seed {seed} | {args.episodes} episodi | "
                      f"{make_title(file_seeds)} | {input_path.name}")
        if not args.compare:
            sel = runs
            footnotes = [cmp_footnote(r["label"], {**r["hp"], "max_init": args.max_init,
                                                  "episodes": args.episodes,
                                                  "eval_episodes": args.eval_episodes,
                                                  "max_steps": args.max_steps},
                                     r["n_init"], seed, input_path.name,
                                     r["ev"], r["opt"]) for r in sel]
            plot_cmp(sel, args.plot or (DATA_DIR / f"{stem}.png"),
                     base_title, footnotes)
        else:
            if args.plot:
                print("WARN: --plot ignorato con --compare "
                      "(due PNG: _vs_std e _vs_samehp)")
            plot_pairs(runs, stem, base_title, args, seed, input_path)
    env.close()


if __name__ == "__main__":
    main()
