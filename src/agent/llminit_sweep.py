#!/usr/bin/env python3
"""Sweep su quanti coppie inizializzare: un PNG che confronta piu' max_init.

Per ogni n in --n_init: load_qinit(input, n) (bottleneck first, top V_LLM),
Q-learning tabulare con gli stessi hparam e gli stessi seed di ogni altro
punto, quindi un unico grafico 2x2: SR train (ma100) + eval greedy | |TD| medio
| agreement con la policy ottima (VI) | value loss vs V*. n=0 e' il controllo
Q=0 a parita' di hparam, n=il totale e' la run LLM-init completa.

--env sceglie l'ambiente (riusa il llminit corrispondente, nessuna logica
duplicata): doorkey = MiniGrid-DoorKey-8x8, frozenlake = 4x4/8x8 slippery.
Griglia di default = frazioni 0, 1/8, 1/4, 1/2, 3/4, 1 del totale coppie
(720 su doorkey seed1337, 76 su frozenlake seed1337).

Con input multi-seed (senza --seed) allena su tutti i seed del file, 1 seed per
episodio, e valuta due volte: --eval-in-seed (default primo train) e
--eval-new-seed (default primo seed libero sopra max(train)).

Output: src/output/agents/llminit_sweep_{env}[_{map}]_s{seed}[_tag].png
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from paths import AGENTS_DIR, CACHE_DIR  # noqa: E402

GRID_FRACS = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
DEFAULTS = {"doorkey": {"max_steps": 450, "map": None, "title": "DoorKey 8x8"},
            "frozenlake": {"max_steps": 0, "map": "8x8",
                           "title": "FrozenLake slippery"}}


def env_adapter(name, map_name):
    """Modulo llminit dell'env + le 3 differenze: env, MDP ottimale, stem."""
    if name == "doorkey":
        import gymnasium as gym
        import doorkey_qtable_llminit as m
        from doorkey_state import resolve_input
        from env.view_wrapper import DoorKeyViewSystem
        return {"resolve_input": resolve_input, "m": m,
                "make_env": lambda: DoorKeyViewSystem(
                    gym.make("MiniGrid-DoorKey-8x8-v0")),
                "opt_mdp": lambda seed: m.load_opt_mdp(seed=seed, size=8,
                                                       out_dir=CACHE_DIR),
                "stem": "doorkey_8x8"}
    import frozenlake_qtable_llminit as m
    from env.frozenlake_factory import make_env
    return {"resolve_input": m.resolve_input, "m": m,
            "make_env": lambda: make_env(map_name=map_name, is_slippery=True),
            "opt_mdp": lambda seed: m.load_opt_mdp(
                seed=seed, map_name=map_name, is_slippery=True, out_dir=CACHE_DIR),
            "stem": f"frozenlake_{map_name}_slippery"}


def default_grid(n_pairs):
    """Frazioni del totale, arrotondate e deduppate (0 compreso)."""
    return sorted({int(round(f * n_pairs)) for f in GRID_FRACS})


def seed_plan(args, file_seeds):
    """(seed, train_seeds|None, eval_new_seed|None) come nei moduli llminit."""
    if args.seed is not None or len(file_seeds) == 1:
        return (args.seed if args.seed is not None else file_seeds[0]), None, None
    train_seeds = (sorted(set(args.train_seeds)) if args.train_seeds
                   else list(file_seeds))
    assert set(train_seeds) <= set(file_seeds), \
        f"--train-seeds {train_seeds} fuori dal file {file_seeds}"
    seed = args.eval_in_seed if args.eval_in_seed is not None else train_seeds[0]
    new = args.eval_new_seed
    if new is None:  # primo seed libero sopra max(train) e fuori dal file
        new = max(train_seeds) + 1
        while new in file_seeds:
            new += 1
    if new in train_seeds:
        print(f"WARN: eval_new_seed {new} dentro il train set "
              f"(generalizzazione non pura)")
    return seed, train_seeds, new


def paint(labels, m):
    """plot_cmp legge la COLORS del modulo llminit (fallback "C3"): la riempiamo
    qui, senza toccare quei file.

    Palette qualitativa (colori complementari, massimo contrasto anche fra
    n_init vicini): tab10 fino a 10 curve, poi hue equispaziati. Colori in
    posizione sulla griglia, quindi a parita' di --n_init la figura e' sempre
    uguale.
    """
    from matplotlib import colormaps
    n = len(labels)
    cmap = colormaps["tab10" if n <= 10 else "hsv"]
    xs = range(n) if n <= 10 else np.linspace(0, 1, n, endpoint=False)
    m.COLORS.update({lab: cmap(x) for lab, x in zip(labels, xs)})


def opt_metrics(m, agent, opt_mdp):
    if opt_mdp is None:
        return None
    try:
        return m.optimal_policy_metrics(agent.q, opt_mdp)
    except Exception as e:
        print(f"WARN: metriche policy ottima saltate: {e}")
        return None


def sweep_point(m, env, args, input_path, n, seed, train_seeds, eval_new_seed,
                opt_mdp):
    """Una run LLM-init con le prime n coppie. Ritorna la run per plot_cmp."""
    _, qinit, einfo = m.load_qinit(input_path, n)
    random.seed(seed)
    np.random.seed(seed)
    agent = m.QLearningAgent(env.action_space.n, list(qinit.items()),
                             alpha=args.alpha, gamma=args.gamma,
                             epsilon_decay=args.eps_decay,
                             epsilon_min=args.eps_min)
    bn = einfo["n_bottleneck_kept"]
    print(f"  n_init={n}: {len(qinit)} coppie ({bn} bottleneck + "
          f"{len(qinit) - bn} altre), {args.episodes} episodi seed {seed}")
    hist = m.train(env, agent, args.episodes, seed, args.max_steps,
                   args.log_every, train_seeds=train_seeds)
    ev = m.evaluate(env, agent, args.eval_episodes, seed, args.max_steps)
    ev_new = (m.evaluate(env, agent, args.eval_episodes, eval_new_seed,
                         args.max_steps) if eval_new_seed is not None else None)
    return {"label": f"n={n}", "hist": hist, "ev": ev, "n_init": n,
            "kept": len(qinit), "bn": bn, "ev_new": ev_new, "qinit": qinit,
            "opt": opt_metrics(m, agent, opt_mdp)}


def selfcheck(m, ad, grid, total, input_path):
    assert default_grid(720) == [0, 90, 180, 360, 540, 720], default_grid(720)
    assert default_grid(76) == [0, 10, 19, 38, 57, 76], default_grid(76)
    assert default_grid(1) == [0, 1]
    counts = [len(m.load_qinit(input_path, n)[1]) for n in grid]
    assert counts == sorted(counts), counts
    assert counts[-1] == total, (counts, total)
    seen = {}
    for n, k in zip(grid, counts):  # n diversi oltre il totale = curve identiche
        if k in seen:
            print(f"  (n={n} e n={seen[k]} tengono le stesse {k} coppie: "
                  f"curve sovrapposte)")
        seen[k] = n
    paint([f"n={n}" for n in grid], m)
    assert all(f"n={n}" in m.COLORS for n in grid)
    print(f"Selfcheck OK (griglia {grid} su {total} coppie, load_qinit monotono, "
          f"colori distinti: {ad['stem']})")


def main():
    ap = argparse.ArgumentParser(
        description="Confronto sweep del numero di coppie inizializzate (LLM-init)")
    ap.add_argument("--env", required=True, choices=["doorkey", "frozenlake"])
    ap.add_argument("--input", default=None,
                    help="file JSON LLM in src/output/llm/ (o path)")
    ap.add_argument("--map", default=None, choices=["4x4", "8x8"],
                    help="solo frozenlake")
    ap.add_argument("--n_init", type=int, nargs="+", default=None, dest="n_init",
                    help="coppie inizializzate per run (default: griglia 0, 1/8, "
                         "1/4, 1/2, 3/4, 1 del totale)")
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--eps_decay", type=float, default=0.995)
    ap.add_argument("--eps_min", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=None,
                    help="seed singolo per train ed eval (default: primo seed "
                         "del file; con file multi-seed si allena su tutti)")
    ap.add_argument("--train-seeds", type=int, nargs="+", default=None,
                    dest="train_seeds", help="sottoinsieme dei seed del file")
    ap.add_argument("--eval-in-seed", type=int, default=None, dest="eval_in_seed")
    ap.add_argument("--eval-new-seed", type=int, default=None, dest="eval_new_seed")
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--eval_episodes", type=int, default=100)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--plot", default=None, help="path PNG")
    ap.add_argument("--no_plot", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    d = DEFAULTS[args.env]
    if args.max_steps is None:
        args.max_steps = d["max_steps"]
    map_name = d["map"] if args.map is None else args.map
    ad = env_adapter(args.env, map_name)
    m = ad["m"]

    input_path = ad["resolve_input"](args.input)
    file_seeds, all_qinit, _ = m.load_qinit(input_path, None)
    total = len(all_qinit)
    grid = default_grid(total) if args.n_init is None else sorted(set(args.n_init))
    assert all(n >= 0 for n in grid), f"--n_init negativi: {grid}"
    print(f"Input: {input_path.name} | file_seeds={file_seeds} | {total} coppie "
          f"disponibili | griglia n_init={grid} | {len(grid)} run x "
          f"{args.episodes} episodi")

    if args.selfcheck:
        selfcheck(m, ad, grid, total, input_path)
        return

    seed, train_seeds, eval_new_seed = seed_plan(args, file_seeds)
    print(f"train/eval seed={seed}" + (f" | train_seeds={train_seeds}"
                                       f" | eval_new_seed={eval_new_seed}"
                                       if train_seeds else ""))
    env = ad["make_env"]()
    if m.load_opt_mdp is None:
        print("WARN: graph non importabile, metriche policy ottima saltate")
        opt_mdp = None
    else:
        opt_mdp = ad["opt_mdp"](seed)

    runs = [sweep_point(m, env, args, input_path, n, seed, train_seeds,
                        eval_new_seed, opt_mdp) for n in grid]
    if not args.no_plot:
        paint([r["label"] for r in runs], m)
        lines = [f"hp: episodes={args.episodes} alpha={args.alpha} "
                 f"gamma={args.gamma} eps_decay={args.eps_decay} "
                 f"eps_min={args.eps_min} max_steps={args.max_steps} "
                 f"eval_ep={args.eval_episodes} seed={seed} "
                 f"input={input_path.name}"]
        if train_seeds:
            lines.append(f"train_seeds={train_seeds} "
                         f"eval_new_seed={eval_new_seed}")
        for r in runs:
            line = (f"{r['label']}: {r['kept']} coppie ({r['bn']} bottleneck + "
                    f"{r['kept'] - r['bn']}) | eval SR={r['ev']['sr']:.3f} "
                    f"rew={r['ev']['mean_reward']:.3f} "
                    f"len={r['ev']['mean_len']:.0f}")
            if r["ev_new"] is not None:
                line += f" | new-seed SR={r['ev_new']['sr']:.3f}"
            if r["opt"]:
                line += (f" | agreement={r['opt']['agreement']:.3f} "
                         f"loss={r['opt']['mean_value_loss']:.4f}")
            lines.append(line)
        title = (f"{d['title']} seed {seed} | sweep n_init {grid} | "
                 f"{args.episodes} episodi")
        out = Path(args.plot) if args.plot else AGENTS_DIR / (
            f"llminit_sweep_{ad['stem']}_s{seed}"
            f"{('_' + args.tag) if args.tag else ''}.png")
        m.plot_cmp(runs, out, title, lines)
    env.close()


if __name__ == "__main__":
    main()
