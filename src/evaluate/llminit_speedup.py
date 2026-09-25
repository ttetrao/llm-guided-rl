#!/usr/bin/env python3
"""Speedup LLM-init vs Vanilla dai JSON llminit (sample-efficiency, read-only).

Per ogni tripletta *_llminit/_vsame/_vstd.json in src/output/agents calcola,
su MA100 di successes[]: TTT@thr (episodi e steps cumulati) per thr=0.8/0.9,
AUC-SR e speedup = TTT_vanilla/TTT_llm. Output: tabella stdout + CSV + un PNG
per gruppo con marker TTT. Non misura wall-clock (non salvato nei JSON).
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from paths import AGENTS_DIR  # noqa: E402

THRS = (0.8, 0.9)
SUFFIXES = ("llminit", "vsame", "vstd")
LABELS = {"llminit": "LLM-init", "vsame": "Vanilla (same hp)",
          "vstd": "Vanilla-std"}


def ma(x, window=100):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    cs = np.cumsum(x)
    idx = np.arange(len(x))
    start = np.clip(idx - window + 1, 0, None)
    return (cs - np.where(start > 0, cs[start - 1], 0.0)) / (idx - start + 1)


def ttt(successes, ep_lengths, thr, window=100):
    """(episodio_1based|None, steps_cumulati|None)."""
    m = ma(successes, window)
    idx = np.where(m >= thr)[0]
    if not len(idx):
        return None, None
    ep = int(idx[0]) + 1
    steps = int(np.cumsum(np.asarray(ep_lengths, dtype=float))[idx[0]])
    return ep, steps


def load_metrics(path, thrs, window):
    d = json.loads(Path(path).read_text())
    out = {"eval_sr": float(d["eval"]["sr"]), "episodes": len(d["successes"]),
           "auc": float(np.mean(ma(d["successes"], window)))}
    for t in thrs:
        ep, st = ttt(d["successes"], d["ep_lengths"], t, window)
        out[f"ttt_ep_{t}"] = ep
        out[f"ttt_steps_{t}"] = st
    return out, d


def discover(groups_filter=None):
    groups = {}
    for f in sorted(AGENTS_DIR.glob("*llminit*.json")):
        m = re.match(r"^(.*)_(llminit|vsame|vstd|vanilla)(\.json)?$",
                     f.name.replace(".json", ""))
        if not m:
            continue
        stem, suf = m.group(1), m.group(2)
        if suf == "vanilla":
            continue
        groups.setdefault(stem, {})[suf] = f
    rows = {s: v for s, v in groups.items()
            if "llminit" in v and ("vsame" in v or "vstd" in v)}
    if groups_filter:
        rows = {s: v for s, v in rows.items()
                if groups_filter in s}
    else:  # ponytail: smoke test da 20ep (verify/cmp) solo su richiesta
        rows = {s: v for s, v in rows.items()
                if "verify" not in s and "cmp" not in s}
    return rows


def speedup(a, b):
    # ponytail: None (mai converge) -> inf, entrambi None -> n/d
    if a and b:
        return round(b / a, 2)
    if a and not b:
        return float("inf")
    return None


def fmt(v):
    if v is None:
        return "—"
    if v == float("inf"):
        return "∞"
    return str(v)


def plot_group(stem, curves, thrs, window, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    for suf in SUFFIXES:
        if suf not in curves:
            continue
        m = ma(curves[suf]["successes"], window)
        ax.plot(m, label=f"{LABELS[suf]} (eval {curves[suf]['eval_sr']:.2f})")
    for t, ls in zip(thrs, ("-", ":")):
        for suf in SUFFIXES:
            if suf not in curves:
                continue
            ep = curves[suf].get(f"ttt_ep_{t}")
            if ep:
                ax.axvline(ep, linestyle=ls, alpha=0.5)
    ax.set_title(f"{stem} | TTT solid=@{thrs[0]} dotted=@{thrs[1]}")
    ax.set_xlabel("episodio")
    ax.set_ylabel("SR train (MA100)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(out)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Plot salvato: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thr", type=float, nargs="+", default=list(THRS))
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--groups", default=None, help="sottostringa filtro stem")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--csv", default=str(AGENTS_DIR / "speedup_summary.csv"))
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        assert list(ma([0, 1, 1, 1], window=2)) == [0.0, 0.5, 1.0, 1.0]
        ep, st = ttt([0] * 5 + [1] * 5, [10] * 10, 0.8, window=5)
        assert (ep, st) == (9, 90), (ep, st)
        assert speedup(492, 1308) == 2.66 and speedup(5, None) == float("inf")
        d, _ = load_metrics(
            AGENTS_DIR / "qtable_llminit_seed_1337_gpt-oss_120b_llminit.json",
            [0.8], args.window)
        assert d["ttt_ep_0.8"] == 492, d  # spot-check deterministico
        print("Selfcheck OK")
        return

    groups = discover(args.groups)
    assert groups, "nessun gruppo llminit+baseline trovato"
    hdr = ["gruppo"] + [f"{m}_{s}@ {t}".replace(" ", "")
                        for s in ("llm", "vsame", "vstd")
                        for m in ("TTTep", "TTTsteps") for t in args.thr] \
        + ["sp_ep_same@%s" % t for t in args.thr] \
        + ["sp_ep_std@%s" % t for t in args.thr] \
        + ["sp_steps_same@%s" % t for t in args.thr] \
        + ["sp_steps_std@%s" % t for t in args.thr] \
        + ["AUC_llm", "AUC_vsame", "AUC_vstd",
           "eval_llm", "eval_vsame", "eval_vstd"]
    table = []
    for stem, files in sorted(groups.items()):
        mets, raws = {}, {}
        for suf in SUFFIXES:
            if suf in files:
                mets[suf], raws[suf] = load_metrics(files[suf], args.thr,
                                                    args.window)
        g = {"llm": mets.get("llminit", {}), "vsame": mets.get("vsame", {}),
             "vstd": mets.get("vstd", {})}
        row = {"gruppo": stem}
        for s in ("llm", "vsame", "vstd"):
            for t in args.thr:
                row[f"TTTep_{s}@{t}"] = g[s].get(f"ttt_ep_{t}")
                row[f"TTTsteps_{s}@{t}"] = g[s].get(f"ttt_steps_{t}")
        for t in args.thr:
            row[f"sp_ep_same@{t}"] = speedup(g["llm"].get(f"ttt_ep_{t}"),
                                             g["vsame"].get(f"ttt_ep_{t}"))
            row[f"sp_ep_std@{t}"] = speedup(g["llm"].get(f"ttt_ep_{t}"),
                                            g["vstd"].get(f"ttt_ep_{t}"))
            row[f"sp_steps_same@{t}"] = speedup(
                g["llm"].get(f"ttt_steps_{t}"), g["vsame"].get(f"ttt_steps_{t}"))
            row[f"sp_steps_std@{t}"] = speedup(
                g["llm"].get(f"ttt_steps_{t}"), g["vstd"].get(f"ttt_steps_{t}"))
        for s in ("llm", "vsame", "vstd"):
            row[f"AUC_{s}"] = round(g[s]["auc"], 3) if g[s] else None
            row[f"eval_{s}"] = g[s].get("eval_sr")
        table.append(row)
        if not args.no_plot:
            plot_group(stem, {s: {"successes": json.loads(
                files[{"llm": "llminit", "vsame": "vsame",
                       "vstd": "vstd"}[s]].read_text())["successes"],
                "eval_sr": g[s].get("eval_sr", 0.0),
                **{f"ttt_ep_{t}": g[s].get(f"ttt_ep_{t}")
                   for t in args.thr}} for s in ("llm", "vsame", "vstd")
                if s in g and g[s]}, args.thr, args.window,
                AGENTS_DIR / f"{stem}_speedup.png")

    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[c.replace(" ", "")
                                          for c in hdr])
        w.writeheader()
        for r in table:
            w.writerow({c.replace(" ", ""): r.get(c.replace(" ", ""), "")
                        for c in hdr})
    print(f"CSV salvato: {args.csv} ({len(table)} gruppi)")
    cols = ["gruppo"] + [c for c in hdr if c.startswith("sp_ep")]
    print("| " + " | ".join(cols) + " |")
    for r in table:
        print("| " + " | ".join([r["gruppo"]] +
                                [fmt(r.get(c)) for c in cols[1:]]) + " |")


if __name__ == "__main__":
    main()
