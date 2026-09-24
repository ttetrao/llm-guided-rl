#!/usr/bin/env python3
"""Standard tabular Q-learning with Q(s,a) initialized from V_LLM(s,a).

The LLM values only seed the table on covered pairs, the rest starts at 0.
Learning is plain Q-learning: Q += alpha * (r + gamma * max Q' - Q).

Ogni run salva nel JSON le metriche di policy ottima (agreement tie-aware e
value loss della greedy appresa vs Q* da value iteration sul seed di eval) e
le riporta nel plot insieme a tutti gli iperparametri (box in basso).
--compare: 3 run (LLM-init, Vanilla stessi hp, Vanilla-std) + due PNG
_vs_std/_vs_samehp a 2 curve.

Single seed (--seed o input 1-seed) per train+eval come prima.
Input multi-seed (envelope con seeds in cima): senza --seed allena su tutti i
seed del file (1 seed/episodio, --train-seeds per sottoinsieme) e doppia eval
finale su --eval-in-seed (default primo train) + --eval-new-seed nuovo
(default max(train)+1 libero).
Output: src/output/agents/qtable_llminit_seed_X[_tag].json + .png plots.
"""
import argparse
import json
import random
import sys
from pathlib import Path
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import gymnasium as gym
import minigrid  # noqa: F401, registers MiniGrid envs

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from env.view_wrapper import DoorKeyViewSystem  # noqa: E402
from doorkey_state import encode, resolve_input  # noqa: E402
from doorkey_state import ACTION_IDX, STAGE_IDX, STAGE_TARGET  # noqa: E402
from paths import AGENTS_DIR, CACHE_DIR  # noqa: E402

try:  # policy ottima da VI (solo metriche; fallisce solo fuori dal package)
    from graph.mdp_graph import load_or_build as load_opt_mdp
except ImportError:
    load_opt_mdp = None  # type: ignore

TIE_EPS = 1e-6  # pareggio per l'agreement tie-aware con la policy ottima
# ponytail: standard = pilota doorkey_states.py/qlearning_states.py
STD = {"alpha": 0.25, "gamma": 0.99, "eps_decay": 0.998, "eps_min": 0.05}
PAIRS = (("LLM-init", "Vanilla-std", "_vs_std"),
         ("LLM-init", "Vanilla (same hp)", "_vs_samehp"))
# stage_idx relativo -> (has_key, door_open) assoluti
STAGE_ABS = {0: (False, False), 1: (True, False), 2: (True, True)}


def _unwrap_rows(input_path):
    """Accetta lista legacy o envelope nuovo {seeds, rows}."""
    data = json.loads(input_path.read_text())
    if isinstance(data, dict) and "rows" in data:
        return data["rows"]
    return data


def load_qinit(input_path, max_init=None):
    """Rows (s,a,v_llm) -> {(state, action): mean v}.

    Multi-seed: i target (key/door/goal) sono caricati per ogni seed dal suo
    mdp_8x8_seed{S}.json; gli stati sono relativi (dx,dy) quindi aggregabili.
    Ritorna (file_seeds, qinit, info). Con 1 seed il comportamento e' quello storico.

    max_init keeps exactly that many pairs (absolute total): bottleneck
    pairs first (top by V_LLM), then top checkpoint pairs by V_LLM.
    None keeps everything.
    """
    assert max_init is None or max_init >= 0, "max_init must be >= 0"
    rows = _unwrap_rows(input_path)
    seeds = sorted({r["seed"] for r in rows})
    assert seeds, "no seeds in file"
    targets_per_seed = {}
    for s in seeds:
        mdp_path = CACHE_DIR / f"mdp_8x8_seed{s}.json"
        assert mdp_path.exists(), f"missing {mdp_path} (key/door/goal positions)"
        gi = json.loads(mdp_path.read_text())["grid_info"]
        targets_per_seed[s] = {"key_pos": tuple(gi["key_pos"]),
                               "door_pos": tuple(gi["door_pos"]),
                               "goal_pos": tuple(gi["goal_pos"])}
    agg = {}  # key -> [sum_v, count, is_bottleneck]
    n_clipped = 0
    per_seed_counts = {s: 0 for s in seeds}
    for r in rows:
        t = targets_per_seed[r["seed"]][STAGE_TARGET[r["stage"]]]
        key = ((t[0] - r["x"], t[1] - r["y"], r["dir"], STAGE_IDX[r["stage"]]),
               ACTION_IDX[r["action"]])
        v = float(r["v_llm"])
        assert v == v and abs(v) != float("inf"), f"non-finite v_llm: {v}"
        if not 0.0 <= v <= 1.0:
            v = min(max(v, 0.0), 1.0)
            n_clipped += 1
        is_bn = r.get("bucket") == "bottleneck"
        if key in agg:
            agg[key][0] += v
            agg[key][1] += 1
            agg[key][2] = agg[key][2] or is_bn
        else:
            agg[key] = [v, 1, is_bn]
        per_seed_counts[r["seed"]] += 1
    full = {k: s / n for k, (s, n, _) in agg.items()}
    is_bn = {k: b for k, (_, _, b) in agg.items()}
    ordered = sorted(full, key=lambda k: (not is_bn[k], -full[k], k[0], k[1]))
    if max_init is None:
        kept = ordered
    else:
        kept = ordered[:max_init]
    qinit = {k: full[k] for k in kept}
    n_bn_kept = sum(1 for k in kept if is_bn[k])
    info = {"seeds": seeds, "n_rows": len(rows), "n_dups": len(rows) - len(full),
            "n_clipped": n_clipped, "n_bottleneck": sum(is_bn.values()),
            "n_bottleneck_kept": n_bn_kept,
            "n_checkpoint_kept": len(kept) - n_bn_kept,
            "per_seed_rows": per_seed_counts}
    return seeds, qinit, info


class QLearningAgent:
    """Tabular agent: constant alpha, epsilon-greedy, classic update."""

    def __init__(self, n_actions, qinit=(), alpha=0.15, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float32))
        for (st, a), v in qinit:
            self.q[st][a] = v
        self.n_init = len(qinit)

    def act(self, state, greedy=False):
        if not greedy and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        vals = self.q[state]
        # Random tie-break: plain argmax loops on tied values.
        m = np.max(vals)
        cands = np.flatnonzero(vals == m)
        return int(cands[0] if len(cands) == 1 else np.random.choice(cands))

    def update(self, s, a, r, s_next, done):
        best_next = 0.0 if done else np.max(self.q[s_next])
        td = r + self.gamma * best_next - self.q[s][a]
        self.q[s][a] += self.alpha * td
        return abs(td)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def optimal_policy_metrics(qtable, mdp, tie_eps=TIE_EPS):
    """Greedy appresa vs policy ottima da VI: agreement tie-aware + value loss.

    qtable: {(dx,dy,dir,stage): qvals} (stati relativi visitati); mdp: dict da
    mdp_graph.load_or_build per l'eval seed. Stati relativi -> assoluti via
    target di stage; terminali esclusi (Q* tutta zero: agreement banale).
    Ritorna dict JSON-safe.
    """
    gamma = float(mdp.get("gamma", 0.99))
    gi = mdp["grid_info"]
    targets = [tuple(gi["key_pos"]), tuple(gi["door_pos"]),
               tuple(gi["goal_pos"])]
    index = mdp["index"]
    nodes = {int(n["id"]): n for n in mdp["nodes"]}
    agree, losses, worst = 0, [], []
    n_skip, n_unmapped, n_term = 0, 0, 0
    for st, qv in qtable.items():
        try:
            dx, dy, d, stg = int(st[0]), int(st[1]), int(st[2]), int(st[3])
            tx, ty = targets[stg]
            s_abs = (tx - dx, ty - dy, d, *STAGE_ABS[stg])
        except Exception:
            n_skip += 1
            continue
        nid = index.get(s_abs)
        if nid is None:
            n_unmapped += 1
            continue
        n = nodes[int(nid)]
        if n.get("is_terminal"):
            n_term += 1
            continue
        try:
            al = int(np.argmax(np.asarray(list(qv), dtype=float)))
        except Exception:
            n_skip += 1
            continue
        qs = {int(a): float(t["reward"]) + gamma * float(
            nodes[int(t["next_id"])]["v_value"])
            for a, t in n["transitions"].items()}
        qmax = max(qs.values())
        if al in {a for a, q in qs.items() if qmax - q <= tie_eps}:
            agree += 1
        loss = float(n["v_value"]) - qs.get(al, 0.0)
        losses.append(loss)
        worst.append((str(s_abs), al, loss))
    n = len(losses)
    worst.sort(key=lambda t: -t[2])
    return {
        "mdp_gamma": gamma, "tie_eps": tie_eps,
        "n_mdp_states": len(nodes), "n_terminal_excluded": n_term,
        "n_visited_scored": n, "n_skipped": n_skip,
        "n_unmapped": n_unmapped,
        "agreement": (agree / n) if n else 0.0,
        "mean_value_loss": (sum(losses) / n) if n else 0.0,
        "max_value_loss": max(losses) if losses else 0.0,
        "worst_states": [{"state": s, "action": a, "loss": l}
                         for s, a, l in worst[:10]],
    }


def train(env, agent, episodes, train_seed, max_steps=0, log_every=100,
          train_seeds=None):
    """Un episodio = un seed: fisso (singolo) o pescato uniforme da train_seeds (multi)."""
    rewards, tds, successes, ep_lengths = [], [], [], []
    for ep in range(episodes):
        seed = random.choice(train_seeds) if train_seeds else train_seed
        env.reset(seed=seed)
        st = encode(env)
        ep_rew, ep_td, steps, term, trunc = 0.0, 0.0, 0, False, False
        while max_steps == 0 or steps < max_steps:
            a = agent.act(st)
            _, r, term, trunc, _ = env.step(a)
            nst = encode(env)
            done = term or trunc
            ep_td += agent.update(st, a, r, nst, done)
            ep_rew += r
            steps += 1
            st = nst
            if done:
                break
        agent.decay_epsilon()
        rewards.append(ep_rew)
        tds.append(ep_td / max(steps, 1))
        successes.append(1.0 if (term and not trunc and ep_rew > 0) else 0.0)
        ep_lengths.append(steps)
        if ep % log_every == 0:
            avg = np.mean(rewards[-100:])
            print(f"Ep {ep:5d}: reward={ep_rew:.2f} avg100={avg:.2f} "
                  f"sr100={np.mean(successes[-100:]):.3f} "
                  f"eps={agent.epsilon:.3f} stati={len(agent.q)}")
    return {"rewards": rewards, "losses": tds, "successes": successes,
            "ep_lengths": ep_lengths}


def evaluate(env, agent, episodes, eval_seed, max_steps=0):
    """Greedy policy on the single eval seed."""
    srs, rews, lens = [], [], []
    for _ in range(episodes):
        env.reset(seed=eval_seed)
        st = encode(env)
        ep_rew, steps, term, trunc = 0.0, 0, False, False
        while True:
            a = agent.act(st, greedy=True)
            _, r, term, trunc, _ = env.step(a)
            st = encode(env)
            ep_rew += r
            steps += 1
            if term or trunc or (max_steps and steps >= max_steps):
                break
        srs.append(1.0 if (term and not trunc and ep_rew > 0) else 0.0)
        rews.append(ep_rew)
        lens.append(steps)
    return {"sr": float(np.mean(srs)), "mean_reward": float(np.mean(rews)),
            "mean_len": float(np.mean(lens)), "successes": srs,
            "rewards": rews, "lengths": lens}


def plot_run(hist, ev, title, out_path, window=100, ev_new=None,
             footnote=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def ma(x):
        # Growing window over the first episodes so curves start at 0.
        x = np.asarray(x, dtype=float)
        if len(x) == 0:
            return x
        cs = np.cumsum(x)
        idx = np.arange(len(x))
        start = np.clip(idx - window + 1, 0, None)
        return (cs - np.where(start > 0, cs[start - 1], 0.0)) / (idx - start + 1)

    panels = [
        ("rewards", "Reward per episodio (media mobile)", "C0", "episodio", "reward"),
        ("successes", "Successo train vs eval greedy (sfruttamento)", "green", "episodio", "successo"),
        ("ep_lengths", "Lunghezza episodio", "orange", "episodio", "passi"),
        ("losses", "|TD-error| medio per episodio", "purple", "episodio", "|TD|"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    xs = range(len(hist["rewards"]))
    for axis, (key, heading, color, xl, yl) in zip(axes.flat, panels):
        axis.plot(hist[key], alpha=0.25)
        axis.plot(xs, ma(hist[key]), linewidth=2, color=color)
        axis.set_title(heading)
        axis.set_xlabel(xl)
        axis.set_ylabel(yl)
        axis.grid(True, alpha=0.3)
    axes.flat[1].axhline(ev["sr"], color="red", linestyle="--",
                         label=f"eval greedy SR={ev['sr']:.3f}")
    if ev_new is not None:
        axes.flat[1].axhline(ev_new["sr"], color="purple", linestyle=":",
                             label=f"eval new-seed SR={ev_new['sr']:.3f}")
    axes.flat[1].set_ylim(-0.05, 1.05)
    axes.flat[1].legend()
    sub = (f"{title} | eval: SR={ev['sr']:.3f} rew={ev['mean_reward']:.3f} "
           f"len={ev['mean_len']:.0f} su {len(ev['successes'])} ep")
    if ev_new is not None:
        sub += (f" | new-seed: SR={ev_new['sr']:.3f} rew={ev_new['mean_reward']:.3f} "
                f"len={ev_new['mean_len']:.0f}")
    fig.suptitle(sub)
    if footnote:
        fig.text(0.01, 0.01, footnote, fontsize=7, family="monospace",
                 va="bottom", ha="left",
                 bbox=dict(facecolor="0.96", edgecolor="0.8", boxstyle="round,pad=0.3"))
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot salvato: {out_path}")


def footnote_for(hp, ev, opt=None, extra=None):
    """Box iperparametri+risultati per i plot (una stringa, due righe)."""
    line1 = (f"hp: episodes={hp['episodes']} alpha={hp['alpha']} gamma={hp['gamma']} "
             f"eps_decay={hp['eps_decay']} eps_min={hp['eps_min']} "
             f"max_steps={hp['max_steps']} eval_ep={hp['eval_episodes']} "
             f"seed={hp['seed']} n_init={hp['n_init']} max_init={hp['max_init']} "
             f"input={hp['input']}")
    line2 = (f"eval: SR={ev['sr']:.3f} rew={ev['mean_reward']:.3f} "
             f"len={ev['mean_len']:.0f}")
    if opt is not None:
        line2 += (f" | policy ottima: agreement={opt['agreement']:.3f} "
                  f"loss={opt['mean_value_loss']:.4f} "
                  f"(su {opt['n_visited_scored']} stati, "
                  f"mdp_gamma={opt['mdp_gamma']})")
    if extra:
        line2 += f" | {extra}"
    return line1 + "\n" + line2


COLORS = {"LLM-init": "C0", "Vanilla (same hp)": "C1", "Vanilla-std": "C2"}


def ma(x, window=100):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    cs = np.cumsum(x)
    idx = np.arange(len(x))
    start = np.clip(idx - window + 1, 0, None)
    return (cs - np.where(start > 0, cs[start - 1], 0.0)) / (idx - start + 1)


def cmp_footnote(label, hp, n_init, seed, input_name, ev, opt):
    full = dict(hp, seed=seed, n_init=n_init, max_init=hp.get("max_init"),
                input=input_name)
    return footnote_for(full, ev, opt, extra=label)


def plot_cmp(runs, out_path, title, footnotes=None):
    """Figura 2x2 di confronto: SR train | |TD| | agreement ottima | value loss.

    runs: [{label, hist, ev, opt}]; footnotes: [str] box iperparametri.
    Usata da --compare (qui) e da llminit2.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    for r in runs:
        c = COLORS.get(r["label"], "C3")
        ax.plot(ma(r["hist"]["successes"]), linewidth=2, color=c,
                label=f"{r['label']} (eval {r['ev']['sr']:.2f})")
        ax.axhline(r["ev"]["sr"], color=c, linestyle="--", alpha=0.5)
    ax.set_title("Success rate train (ma100) + eval greedy")
    ax.set_xlabel("episodio")
    ax.set_ylabel("SR")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for r in runs:
        c = COLORS.get(r["label"], "C3")
        ax.plot(ma(r["hist"]["losses"]), linewidth=2, color=c, label=r["label"])
    ax.set_title("|TD-error| medio per episodio (ma100)")
    ax.set_xlabel("episodio")
    ax.set_ylabel("|TD|")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    labels = [r["label"] for r in runs]
    colors = [COLORS.get(l, "C3") for l in labels]
    ax = axes[1, 0]
    avals = [(r["opt"]["agreement"] if r["opt"] else 0.0) for r in runs]
    bars = ax.bar(range(len(runs)), avals, color=colors)
    for b, r in zip(bars, runs):
        txt = f"{r['opt']['agreement']:.3f}" if r["opt"] else "n/d"
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, txt,
                ha="center", fontsize=9)
    ax.set_title("Agreement con policy ottima (tie-aware)")
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 1]
    lvals = [(r["opt"]["mean_value_loss"] if r["opt"] else 0.0) for r in runs]
    bars = ax.bar(range(len(runs)), lvals, color=colors)
    for b, r in zip(bars, runs):
        txt = f"{r['opt']['mean_value_loss']:.4f}" if r["opt"] else "n/d"
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), txt,
                ha="center", va="bottom", fontsize=9)
    ax.set_title("Value loss media vs V*")
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(title)
    if footnotes:
        fig.text(0.01, 0.01, "\n".join(footnotes), fontsize=7,
                 family="monospace", va="bottom", ha="left",
                 bbox=dict(facecolor="0.96", edgecolor="0.8",
                           boxstyle="round,pad=0.3"))
    fig.tight_layout(rect=[0, 0.10, 1, 0.94])
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot salvato: {out_path}")


def make_agent(n_actions, qinit, args):
    return QLearningAgent(n_actions, qinit, alpha=args.alpha, gamma=args.gamma,
                          epsilon_decay=args.eps_decay,
                          epsilon_min=args.eps_min)


def mine_hp(args):
    """Hparams cli (usati da LLM-init e Vanilla same-hp)."""
    return {"alpha": args.alpha, "gamma": args.gamma,
            "eps_decay": args.eps_decay, "eps_min": args.eps_min}


def std_hp(args):
    """Hparams Vanilla-std (default STD, override con --std-* per multi-seed)."""
    return {"alpha": args.std_alpha, "gamma": args.std_gamma,
            "eps_decay": args.std_eps_decay, "eps_min": args.std_eps_min}


def plot_pairs(runs, stem, base_title, args, seed, input_path):
    """Due PNG _vs_std / _vs_samehp da runs [{label,hist,ev,opt,hp,n_init}].

    Fonte unica per --compare (qui) e llminit2.
    """
    for a, b, suffix in PAIRS:
        sel = [r for r in runs if r["label"] in (a, b)]
        footnotes = [cmp_footnote(r["label"], {**r["hp"],
                                               "episodes": args.episodes,
                                               "eval_episodes": args.eval_episodes,
                                               "max_steps": args.max_steps,
                                               "max_init": args.max_init},
                                  r["n_init"], seed, input_path.name,
                                  r["ev"], r["opt"]) for r in sel]
        plot_cmp(sel, AGENTS_DIR / f"{stem}{suffix}.png",
                 f"{base_title} | {a} vs {b}", footnotes)


def save_run(agent, hist, ev, args, name, label, file_seed, seed, einfo,
             input_path, train_seeds=None, eval_new=None,
             eval_in_seed=None, eval_new_seed=None, opt=None):
    out = AGENTS_DIR / f"{name}.json"
    # ponytail: seeds in cima; modo singolo identico a prima + chiave "seeds"
    hp = {"episodes": args.episodes, "alpha": args.alpha,
          "gamma": args.gamma, "eps_decay": args.eps_decay,
          "eps_min": args.eps_min, "max_steps": args.max_steps,
          "eval_episodes": args.eval_episodes,
          "max_init": args.max_init,
          "init_order": "bottleneck_first_top_v"}
    payload = {
        "algo": "qtable_llminit", "seeds": einfo.get("seeds", [file_seed]),
        "seed": seed, "file_seed": file_seed,
        "input_file": input_path.name, "label": label,
        "n_init": agent.n_init, "expert_rows": einfo["n_rows"],
        "expert_dups_aggregated": einfo["n_dups"],
        "expert_clipped": einfo["n_clipped"],
        "hparams": hp,
        "optimal": opt,
        "eval": ev, "n_visited_states": len(agent.q),
        "rewards": [float(x) for x in hist["rewards"]],
        "losses": [float(x) for x in hist["losses"]],
        "successes": [float(x) for x in hist["successes"]],
        "ep_lengths": [int(x) for x in hist["ep_lengths"]],
        "qtable": {f"{st[0]},{st[1]},{st[2]},{st[3]}": [float(x) for x in agent.q[st]]
                   for st in agent.q},
    }
    if train_seeds is not None:  # modo multi: train su N seed + doppia eval
        payload["train_seeds"] = list(train_seeds)
        payload["eval_in_seed"] = eval_in_seed
        payload["eval_new_seed"] = eval_new_seed
        payload["eval_in"] = ev
        payload["eval_new"] = eval_new
    out.write_text(json.dumps(payload))
    msg = (f"{label} eval greedy su seed {seed} x{args.eval_episodes}: "
           f"SR={ev['sr']:.3f} rew={ev['mean_reward']:.3f} "
           f"len={ev['mean_len']:.0f} | stati: {len(agent.q)}")
    if eval_new is not None:
        msg += (f" | new-seed {eval_new_seed}: SR={eval_new['sr']:.3f} "
                f"rew={eval_new['mean_reward']:.3f} len={eval_new['mean_len']:.0f}")
    print(msg)
    print(f"Risultati salvati: {out}")
    if opt is not None:
        print(f"Policy ottima: agreement={opt['agreement']:.3f} "
              f"loss={opt['mean_value_loss']:.4f} "
              f"(su {opt['n_visited_scored']} stati)")
    if not args.no_plot:
        hp_plot = dict(hp, seed=seed, n_init=agent.n_init,
                       input=input_path.name)
        plot_run(hist, ev, f"{label} (seed {seed})",
                 args.plot or (AGENTS_DIR / f"{name}.png"), ev_new=eval_new,
                 footnote=footnote_for(hp_plot, ev, opt))


def run_training(env, agent, args, seed, name, label, file_seed, einfo,
                 input_path, train_seeds=None, eval_new_seed=None,
                 opt_mdp=None):
    def _opt():
        if opt_mdp is None:
            return None
        try:
            return optimal_policy_metrics(agent.q, opt_mdp)
        except Exception as e:
            print(f"WARN: metriche policy ottima saltate: {e}")
            return None

    if train_seeds is None:
        print(f"Training {label}: {args.episodes} episodi su seed {seed}, "
              f"alpha={args.alpha}")
        hist = train(env, agent, args.episodes, seed, args.max_steps,
                     args.log_every)
        ev = evaluate(env, agent, args.eval_episodes, seed, args.max_steps)
        opt = _opt()
        save_run(agent, hist, ev, args, name, label, file_seed, seed, einfo,
                 input_path, opt=opt)
        return {"hist": hist, "ev": ev, "opt": opt, "agent": agent}
    else:
        print(f"Training {label}: {args.episodes} episodi su seed {list(train_seeds)} "
              f"(1 seed/episodio), alpha={args.alpha}")
        hist = train(env, agent, args.episodes, seed, args.max_steps,
                     args.log_every, train_seeds=list(train_seeds))
        ev_in = evaluate(env, agent, args.eval_episodes, seed, args.max_steps)
        ev_new = evaluate(env, agent, args.eval_episodes, eval_new_seed,
                          args.max_steps)
        opt = _opt()
        save_run(agent, hist, ev_in, args, name, label, file_seed, seed, einfo,
                 input_path, train_seeds=list(train_seeds), eval_new=ev_new,
                 eval_in_seed=seed, eval_new_seed=eval_new_seed, opt=opt)
        return {"hist": hist, "ev": ev_in, "opt": opt, "agent": agent}


def main():
    ap = argparse.ArgumentParser(
        description="Q-Learning tabulare standard con init Q=V_LLM")
    ap.add_argument("--input", default=None,
                    help="file JSON in src/output/llm/ (o path)")
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--eps_decay", type=float, default=0.995)
    ap.add_argument("--eps_min", type=float, default=0.05)
    ap.add_argument("--std-alpha", type=float, default=STD["alpha"], dest="std_alpha",
                    help="alpha del Vanilla-std (default standard single-seed)")
    ap.add_argument("--std-gamma", type=float, default=STD["gamma"], dest="std_gamma",
                    help="gamma del Vanilla-std (default standard single-seed)")
    ap.add_argument("--std-eps-decay", type=float, default=STD["eps_decay"], dest="std_eps_decay",
                    help="eps_decay del Vanilla-std (default standard single-seed)")
    ap.add_argument("--std-eps-min", type=float, default=STD["eps_min"], dest="std_eps_min",
                    help="eps_min del Vanilla-std (default standard single-seed)")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed singolo per train ed eval "
                         "(default: quello del file input; con input multi-seed "
                         "allena su tutti i seed del file)")
    ap.add_argument("--train-seeds", type=int, nargs="+", default=None,
                    dest="train_seeds",
                    help="sottoinsieme dei seed del file su cui allenare "
                         "(default: tutti)")
    ap.add_argument("--eval-in-seed", type=int, default=None, dest="eval_in_seed",
                    help="seed interno per eval finale (default: primo train seed)")
    ap.add_argument("--eval-new-seed", type=int, default=None, dest="eval_new_seed",
                    help="seed nuovo per eval generalizzazione "
                         "(default: max(train)+1 libero)")
    ap.add_argument("--max_init", type=int, default=None,
                    help="absolute total of init pairs: bottleneck first "
                         "(top by V_LLM), then top checkpoint by V_LLM "
                         "(default: keep all)")
    ap.add_argument("--env_seed", type=int, default=None,
                    help="deprecated alias of --seed")
    ap.add_argument("--eval_episodes", type=int, default=100)
    ap.add_argument("--max_steps", type=int, default=450)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--plot", default=None,
                    help="path PNG (default: accanto al JSON)")
    ap.add_argument("--no_plot", action="store_true")
    ap.add_argument("--compare", action="store_true",
                    help="3 run (LLM-init, Vanilla stessi hp, Vanilla-std) + "
                         "due PNG _vs_std/_vs_samehp, stesso seed")
    # Deprecated: accepted so old commands keep working, ignored.
    ap.add_argument("--alpha_mode", default="const",
                    help="deprecated, ignored (constant alpha)")
    ap.add_argument("--alpha_exp", type=float, default=0.8,
                    help="deprecated, ignored")
    ap.add_argument("--alpha0", type=float, default=0.5,
                    help="deprecated, ignored")
    ap.add_argument("--alpha_k", type=float, default=0.01,
                    help="deprecated, ignored")
    ap.add_argument("--init_pseudo", type=int, default=0,
                    help="deprecated, ignored")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    input_path = resolve_input(args.input)
    file_seeds, qinit, einfo = load_qinit(input_path, args.max_init)
    file_seed = file_seeds[0]
    if args.seed is not None:
        single = True
        seed = args.seed
        train_seeds, eval_new_seed = None, None
    elif args.env_seed is not None:
        single = True
        seed = args.env_seed
        train_seeds, eval_new_seed = None, None
    elif len(file_seeds) == 1:
        single = True
        seed = file_seeds[0]
        train_seeds, eval_new_seed = None, None
    else:
        single = False
        train_seeds = sorted(set(args.train_seeds)) if args.train_seeds else list(file_seeds)
        assert set(train_seeds) <= set(file_seeds), \
            f"--train-seeds {train_seeds} fuori dal file {file_seeds}"
        seed = args.eval_in_seed if args.eval_in_seed is not None else train_seeds[0]
        if args.eval_new_seed is not None:
            eval_new_seed = args.eval_new_seed
        else:  # primo seed libero sopra max(train) e fuori dal file
            cand = max(train_seeds) + 1
            while cand in file_seeds:
                cand += 1
            eval_new_seed = cand
        if eval_new_seed in train_seeds:
            print(f"WARN: eval_new_seed {eval_new_seed} dentro il train set "
                  f"(generalizzazione non pura)")
    print(f"Input: {input_path.name} | file_seeds={file_seeds} "
          f"seed_train_eval={seed} | coppie={len(qinit)} "
          f"({einfo['n_bottleneck_kept']} bottleneck + "
          f"{einfo['n_checkpoint_kept']} checkpoint, "
          f"{einfo['n_rows']} righe, {einfo['n_dups']} duplicati mediati, "
          f"{einfo['n_clipped']} clip)"
          + ("" if single else f" | train_seeds={train_seeds} "
             f"eval_new_seed={eval_new_seed}"))

    random.seed(seed)
    np.random.seed(seed)

    if args.selfcheck:
        ag = QLearningAgent(7, list(qinit.items())[:10], alpha=0.15)
        (st0, a0), v0 = list(qinit.items())[0]
        assert ag.q[st0][a0] == v0, "init not assigned"
        q_before = float(ag.q[st0][a0])
        best = float(np.max(ag.q[st0]))
        ag.update(st0, a0, 1.0, st0, False)
        expected = q_before + 0.15 * (1.0 + 0.99 * best - q_before)
        assert abs(ag.q[st0][a0] - expected) < 1e-6
        env = DoorKeyViewSystem(gym.make("MiniGrid-DoorKey-8x8-v0"))
        env.reset(seed=seed)
        s0 = encode(env)
        env.reset(seed=seed)
        assert encode(env) == s0, "same seed must give same start state"
        env.close()
        # metriche ottimalità su MDP sintetico: (0,1,0,2) -> goal (3,3) con a=2
        mdp_toy = {
            "gamma": 0.99,
            "grid_info": {"key_pos": (1, 1), "door_pos": (2, 2),
                          "goal_pos": (3, 3)},
            "index": {(3, 2, 0, True, True): 0, (3, 3, 0, True, True): 1},
            "nodes": [
                {"id": 0, "v_value": 1.0, "is_terminal": False,
                 "transitions": {
                    2: {"next_id": 1, "reward": 1.0, "done": True},
                    0: {"next_id": 0, "reward": 0.0, "done": False}}},
                {"id": 1, "v_value": 0.0, "is_terminal": True,
                 "transitions": {}},
            ],
        }
        m = optimal_policy_metrics({(0, 1, 0, 2): [0.0] * 2 + [1.0] + [0.0] * 4},
                                   mdp_toy)
        assert m["agreement"] == 1.0 and abs(m["mean_value_loss"]) < 1e-9, m
        m2 = optimal_policy_metrics({(0, 1, 0, 2): [0.0] * 7}, mdp_toy)
        assert m2["agreement"] == 0.0 and m2["mean_value_loss"] > 0.0, m2
        assert set(STD) == {"alpha", "gamma", "eps_decay", "eps_min"}
        assert std_hp(SimpleNamespace(std_alpha=0.1, std_gamma=0.9,
                                       std_eps_decay=0.5,
                                       std_eps_min=0.01)) == {
            "alpha": 0.1, "gamma": 0.9, "eps_decay": 0.5, "eps_min": 0.01}
        assert [(a, b, s) for a, b, s in PAIRS] == [
            ("LLM-init", "Vanilla-std", "_vs_std"),
            ("LLM-init", "Vanilla (same hp)", "_vs_samehp")]
        print("Selfcheck OK (formula classica, init assegnata, "
              "seed deterministico, metriche ottimalità)")
        return

    env = DoorKeyViewSystem(gym.make("MiniGrid-DoorKey-8x8-v0"))
    n_actions = int(env.action_space.n)
    if load_opt_mdp is None:
        print("WARN: graph non importabile, metriche policy ottima saltate")
        opt_mdp = None
    else:
        opt_mdp = load_opt_mdp(seed=seed, size=8, out_dir=CACHE_DIR)

    tag = args.tag
    if tag is None:
        if len(file_seeds) == 1:
            tag = input_path.stem.split(f"seed{file_seed}_", 1)[-1].strip("_") or None
        else:
            m = input_path.stem.split("seeds", 1)
            tag = ("seeds" + m[1]).strip("_") if len(m) > 1 else input_path.stem
    if single:
        stem = f"qtable_llminit_seed_{seed}{('_' + tag) if tag else ''}"
    else:
        stem = (f"qtable_llminit_multiseed_train{'-'.join(map(str, train_seeds))}"
                f"_evalin{seed}_evalnew{eval_new_seed}"
                f"{('_' + args.tag) if args.tag else ''}")

    if args.compare:
        specs = (("LLM-init", list(qinit.items()), mine_hp(args), "llminit"),
                 ("Vanilla (same hp)", [], mine_hp(args), "vsame"),
                 ("Vanilla-std", [], std_hp(args), "vstd"))
        runs = []
        for label, init, hp, suffix in specs:
            random.seed(seed)
            np.random.seed(seed)
            agent = QLearningAgent(n_actions, init, alpha=hp["alpha"],
                                   gamma=hp["gamma"],
                                   epsilon_decay=hp["eps_decay"],
                                   epsilon_min=hp["eps_min"])
            vargs = SimpleNamespace(**{**vars(args), **hp})
            res = run_training(env, agent, vargs, seed, f"{stem}_{suffix}", label,
                               file_seed, einfo, input_path,
                               train_seeds=None if single else train_seeds,
                               eval_new_seed=None if single else eval_new_seed,
                               opt_mdp=opt_mdp)
            runs.append({"label": label, "hist": res["hist"], "ev": res["ev"],
                         "opt": res["opt"], "n_init": res["agent"].n_init,
                         "hp": hp})
        if not args.no_plot:
            plot_pairs(runs, stem,
                       f"DoorKey 8x8 seed {seed} | {args.episodes} episodi | "
                       f"{input_path.name}", args, seed, input_path)
        env.close()
        return

    agent = make_agent(n_actions, list(qinit.items()), args)
    if single:
        print(f"Training Q-Learning LLM-init: {args.episodes} episodi su seed "
              f"{seed}, alpha={args.alpha} gamma={args.gamma}")
    run_training(env, agent, args, seed, stem, "LLM-init", file_seed, einfo,
                 input_path, train_seeds=None if single else train_seeds,
                 eval_new_seed=None if single else eval_new_seed,
                 opt_mdp=opt_mdp)
    env.close()


if __name__ == "__main__":
    main()
