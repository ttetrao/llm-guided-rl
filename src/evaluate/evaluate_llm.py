#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Valuta la bontà dei valori LLM generati via llm/query_gemma.py.

Formato JSON atteso (una riga per azione, 6 azioni per stato):
  id, seed, size, bucket, node_id, x, y, dir, has_key, door_open, stage,
  action, v_true, v_llm, analysis
dove v_true = V*(s') e v_llm = stima LLM (Q via V).

v3 (correzioni):
  - FIX CRASH: np.ma.masked_where non broadcasta la condizione (6,1) vs (6,6):
    maschera costruita con np.broadcast_to.
  - FIX LAYOUT: legende spostate FUORI dall'area dati (bbox_to_anchor a destra),
    box statistiche SOTTO l'asse x: non coprono più punti/barre.
    Nel plot accuratezza la legenda "random" è un'annotazione sulla linea.

v2 (design):
  1. Filtro per STATO (non per riga): uno stato è fallito solo se TUTTE le
     azioni hanno v_llm == 0.
  2. Argmax tie-aware (v_true/v_llm quantizzati su gamma^k): la scelta LLM è
     corretta se UNA QUALSIASI delle azioni a pari merito è ottima. L'accuracy
     "strict" (idxmax classico) è riportata a parte.
  3. Rank Top-k consapevole dei pareggi.
  4. R² OLS standard (non slope del CCC).
  5. Plot coerenti con dati quantizzati: residui, istogramma Δk,
    confusion matrix quadrata con righe mai-ottimali hatchate + pannello regret.
  6. Regret (value loss) come metrica primaria.
  7. Bootstrap per stato/seed.

Uso:
    python -m evaluate.evaluate_llm --path graph/data/llm_results_8x8_seed1337_gemma-4-26b-a4b-it.json --outdir evaluate/output
    python -m evaluate.evaluate_llm --path graph/data --gamma auto
    python -m evaluate.evaluate_llm --path ... --tie-eps 1e-6
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

plt.rcParams.update({
    "figure.figsize": (8, 5.5),
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": 9,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "0.85",
    "lines.linewidth": 1.8,
    "patch.linewidth": 0.7,
    "axes.titlepad": 12,
})

# Palette colorblind-safe (Okabe-Ito/Wong)
# storiche (qlearning_states) + nuove (doorkey_states: bottleneck + ckpt pilota)
BUCKET_COLORS = {"iniziale": "#0072B2", "intermedio": "#E69F00", "avanzato": "#009E73",
                 "bottleneck": "#CC79A7", "ckpt-0.3": "#56B4E9", "ckpt-0.5": "#009E73",
                 "ckpt-0.7": "#F0E442", "ckpt-0.8": "#E69F00", "ckpt-1": "#D55E00"}
_FALLBACK_COLORS = ("#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#999999")
ACTION_ORDER = ["left", "right", "forward", "pickup", "drop", "toggle"]
ACTION_COLORS = {"left": "#0072B2", "right": "#E69F00", "forward": "#009E73",
                 "pickup": "#CC79A7", "drop": "#999999", "toggle": "#56B4E9"}

DEFAULT_GAMMA = 0.99
TIE_EPS_DEFAULT = 1e-6     # tolleranza per considerare due valori pari merito
K_TOL = 2e-3               # tolleranza "on-grid" (v ≈ gamma^k)
DK_CLIP = 6                # clip dell'istogramma Δk


# ------------------------------------------------------------------ helper
def _f(v, fmt="{:.4f}"):
    """Formato nan-safe per il log e i box nei plot."""
    try:
        if v is None:
            return "n/d"
        v = float(v)
        if not np.isfinite(v):
            return "n/d"
        return fmt.format(v)
    except Exception:
        return "n/d"


# --------------------------------------------------------------- statistiche
def ccc(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    mx, my = x.mean(), y.mean(); sx, sy = x.std(ddof=1), y.std(ddof=1)
    if sx == 0 or sy == 0:
        return float("nan"), float("nan"), float("nan")
    rho = np.corrcoef(x, y)[0, 1]
    c = (2 * rho * sx * sy) / (sx**2 + sy**2 + (mx - my)**2)
    b = rho * sy / sx; a = my - b * mx
    return float(c), float(b), float(a)


def bootstrap_ci(func, x, y, n_boot=1000, seed=42, alpha=0.05):
    rng = np.random.RandomState(seed); x = np.asarray(x, float); y = np.asarray(y, float); n = len(x)
    point = func(x, y)
    if n < 5:
        return point, float("nan"), float("nan")
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            v = func(x[idx], y[idx])
            if np.isfinite(v):
                boots.append(v)
        except Exception:
            pass
    if len(boots) < 50:
        return point, float("nan"), float("nan")
    return float(point), float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


def bootstrap_seeded(df, col_true, col_llm, metric, n_boot=500, seed=42):
    """Bootstrap per metriche di valore (pearson/ccc/mae): per seed se >=3,
    altrimenti per riga."""
    rng = np.random.RandomState(seed)
    seeds = df["seed"].unique() if "seed" in df.columns else np.array([0])
    if len(seeds) < 3:
        def f_row(x_, y_):
            if metric == "pearson":
                try: return float(pearsonr(x_, y_)[0])
                except Exception: return float("nan")
            if metric == "ccc":
                try: return float(ccc(x_, y_)[0])
                except Exception: return float("nan")
            if metric == "mae":
                return float(np.mean(np.abs(x_ - y_)))
            return float("nan")
        return bootstrap_ci(f_row, df[col_true].values, df[col_llm].values, n_boot=n_boot, seed=seed)

    def f(x_, y_):
        try:
            if metric == "pearson": return float(pearsonr(x_, y_)[0])
            if metric == "ccc":     return float(ccc(x_, y_)[0])
            if metric == "mae":     return float(np.mean(np.abs(x_ - y_)))
        except Exception:
            return float("nan")
        return float("nan")

    point = f(df[col_true].values, df[col_llm].values)
    boots = []
    for _ in range(n_boot):
        ss = rng.choice(seeds, size=len(seeds), replace=True)
        xs, ys = [], []
        for s in ss:
            sub = df[df["seed"] == s]
            xs.append(sub[col_true].values); ys.append(sub[col_llm].values)
        xb = np.concatenate(xs) if xs else np.array([])
        yb = np.concatenate(ys) if ys else np.array([])
        v = f(xb, yb)
        if np.isfinite(v):
            boots.append(v)
    if len(boots) < 50:
        return point, float("nan"), float("nan")
    return float(point), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def bootstrap_grouped(vals, groups=None, n_boot=500, seed=42, alpha=0.05):
    """Bootstrap della MEDIA su unità statistiche corrette (gruppi/stati/seed),
    non su singole righe. groups=None o <3 gruppi validi -> resampling per riga.
    NaN-safe: se ci sono gruppi NaN si ricade nel resampling per riga."""
    vals = np.asarray(vals, float)
    finite = np.isfinite(vals)
    vals = vals[finite]
    if groups is not None:
        groups = np.asarray(groups)[finite]
    point = float(np.mean(vals)) if len(vals) else float("nan")
    if len(vals) < 5:
        return point, float("nan"), float("nan")
    rng = np.random.RandomState(seed)

    use_groups = False
    uniq = by_g = None
    if groups is not None:
        try:
            gf = np.asarray(groups, float)
            if not np.isnan(gf).any() and len(np.unique(gf)) >= 3:
                use_groups = True
                uniq = np.unique(gf)
                by_g = {g: vals[gf == g] for g in uniq}
        except Exception:
            gu = np.asarray(groups)
            if len(np.unique(gu)) >= 3:
                use_groups = True
                uniq = np.unique(gu)
                by_g = {g: vals[gu == g] for g in uniq}

    if use_groups:
        boots = []
        for _ in range(n_boot):
            gs = rng.choice(uniq, size=len(uniq), replace=True)
            boots.append(float(np.mean(np.concatenate([by_g[g] for g in gs]))))
    else:
        n = len(vals)
        boots = [float(np.mean(vals[rng.randint(0, n, n)])) for _ in range(n_boot)]
    if len(boots) < 50:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


# ------------------------------------------------------------------- I/O
def find_files(path):
    p = Path(path)
    if p.is_file():
        return [str(p)]
    files = sorted(glob.glob(os.path.join(path, "**", "llm_results*.json"), recursive=True))
    files += sorted(glob.glob(os.path.join(path, "**", "llm_results*.csv"), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(path, "**", "*.json"), recursive=True))
        files += sorted(glob.glob(os.path.join(path, "**", "*.csv"), recursive=True))
    return files


def load_df(path):
    if path.endswith(".json"):
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception:
            return None
        if isinstance(data, dict):
            if "rows" in data and isinstance(data["rows"], list):
                data = data["rows"]  # envelope nuovo query_{gpt,gemma}: seeds in cima
            elif "nodes" in data or "buckets" in data or "grid_info" in data:
                return None  # mdp/qstates, non llm_results
            elif "v_true" not in data and "v_llm" not in data:
                return None
            else:
                data = [data]
        if not isinstance(data, list) or not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(path)
    rename = {}
    for c in df.columns:
        if c in ("q_value", "v_value"): rename[c] = "v_true"
        if c == "q_llm":                rename[c] = "v_llm"
        if c == "type":                 rename[c] = "bucket"
    if rename:
        df = df.rename(columns=rename)
    if "bucket" not in df.columns and "type" in df.columns:
        df["bucket"] = df["type"]
    for c in ("v_true", "v_llm"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "v_true" in df.columns:
        df = df[df["v_true"].notna()].copy()
    # id di stato: senza action (un gruppo = tutte le azioni di uno stato)
    if "id" not in df.columns:
        if {"seed", "node_id", "bucket"} <= set(df.columns):
            df["id"] = [f"{s}_{int(n):04d}_{b}" for s, n, b in zip(df["seed"], df["node_id"], df["bucket"])]
        else:
            df["id"] = [f"row{i:06d}" for i in range(len(df))]
    return df


# ------------------------------------------------- gamma e quantizzazione k
def detect_gamma(values, lo=0.80, hi=0.9995, n_grid=400, tol=K_TOL, min_frac=0.75):
    """Stima il fattore di sconto dai dati: cerca il gamma che massimizza la
    frazione di valori 'on-grid' (v ≈ gamma^k, k intero). None se fallisce."""
    vals = np.unique(np.asarray(values, float))
    vals = vals[(vals > 1e-3) & (vals <= 1.0 + 1e-9)]
    if len(vals) < 4:
        return None
    best_g, best_f = None, -1.0
    for g in np.linspace(lo, hi, n_grid):
        lg = np.log(g)
        k = np.round(np.log(vals) / lg)
        f = float((np.abs(np.exp(k * lg) - vals) <= tol).mean())
        if f > best_f:
            best_f, best_g = f, float(g)
    if best_g is None or best_f < min_frac:
        return None
    # raffinamento locale
    for g in np.linspace(max(1e-6, best_g - 0.002), min(1 - 1e-6, best_g + 0.002), 161):
        lg = np.log(g)
        k = np.round(np.log(vals) / lg)
        f = float((np.abs(np.exp(k * lg) - vals) <= tol).mean())
        if f > best_f:
            best_f, best_g = f, float(g)
    return best_g if best_f >= min_frac else None


def value_to_k(v, gamma, tol=K_TOL):
    """k tale che v ≈ gamma^k; NaN se il valore non è on-grid."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(v) or v <= 0:
        return np.nan
    lg = np.log(gamma)
    if lg == 0:
        return np.nan
    kr = float(np.round(np.log(v) / lg))
    if abs(np.exp(kr * lg) - v) <= tol:
        return kr
    return np.nan


# ------------------------------------------------------------ metriche valore
def compute_value_metrics(df, gamma):
    x = df["v_true"].values.astype(float)
    y = df["v_llm"].values.astype(float)
    err = y - x; abs_err = np.abs(err); sq_err = err ** 2
    m = {"n": int(len(df))}

    # correlazioni / concordanza
    try: pr, pp = pearsonr(x, y)
    except Exception: pr, pp = float("nan"), float("nan")
    try: sr, sp = spearmanr(x, y)
    except Exception: sr, sp = float("nan"), float("nan")
    c, ccc_slope, ccc_intercept = ccc(x, y)

    # OLS standard: slope/intercept/R² coerenti con la retta disegnata
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() >= 3 and np.std(x[mask]) > 0:
        b1, b0 = np.polyfit(x[mask], y[mask], 1)
        yhat = b0 + b1 * x[mask]
        ss_res = float(np.sum((y[mask] - yhat) ** 2))
        ss_tot = float(np.sum((y[mask] - y[mask].mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    else:
        b1, b0, r2 = float("nan"), float("nan"), float("nan")

    # errore sull'esponente k (v = gamma^k)
    k_true = np.array([value_to_k(v, gamma) for v in x], dtype=float)
    k_llm = np.array([value_to_k(v, gamma) for v in y], dtype=float)
    m["on_grid_true"] = float(np.isfinite(k_true).mean()) if k_true.size else float("nan")
    m["on_grid_llm"] = float(np.isfinite(k_llm).mean()) if k_llm.size else float("nan")
    ok = np.isfinite(k_true) & np.isfinite(k_llm)
    if ok.any():
        dk = k_llm[ok] - k_true[ok]
        m.update({
            "k_n": int(ok.sum()),
            "k_dk_exact": float((dk == 0).mean()),
            "k_dk_abs1": float((np.abs(dk) == 1).mean()),
            "k_dk_abs2plus": float((np.abs(dk) >= 2).mean()),
            "k_dk_mean": float(dk.mean()),
            "k_dk_abs_mean": float(np.abs(dk).mean()),
            "dk_values": dk,
        })
    else:
        m.update({"k_n": 0, "k_dk_exact": float("nan"), "k_dk_abs1": float("nan"),
                  "k_dk_abs2plus": float("nan"), "k_dk_mean": float("nan"),
                  "k_dk_abs_mean": float("nan"), "dk_values": np.array([])})

    # errori
    m.update({
        "mae": float(abs_err.mean()) if len(abs_err) else float("nan"),
        "rmse": float(np.sqrt(sq_err.mean())) if len(sq_err) else float("nan"),
        "bias": float(err.mean()) if len(err) else float("nan"),
        "max_err": float(abs_err.max()) if len(abs_err) else float("nan"),
        "within_05": float((abs_err <= 0.05).mean() * 100) if len(abs_err) else float("nan"),
        "within_10": float((abs_err <= 0.10).mean() * 100) if len(abs_err) else float("nan"),
    })

    # CI
    m["pearson_r"], m["pearson_p"] = pr, pp
    m["spearman_r"], m["spearman_p"] = sr, sp
    m["ccc"], m["ccc_slope"], m["ccc_intercept"] = c, ccc_slope, ccc_intercept
    m["ols_slope"], m["ols_intercept"], m["r2"] = b1, b0, r2
    m["pearson_ci"] = bootstrap_seeded(df, "v_true", "v_llm", "pearson")
    m["ccc_ci"] = bootstrap_seeded(df, "v_true", "v_llm", "ccc")
    m["mae_ci"] = bootstrap_seeded(df, "v_true", "v_llm", "mae")
    try:
        m["rmse_ci"] = bootstrap_ci(lambda a, b: np.sqrt(np.mean((a - b) ** 2)), x, y)
    except Exception:
        m["rmse_ci"] = (float("nan"), float("nan"), float("nan"))

    # per bucket / action / stage
    dfa = df.assign(err=err, abs_err=abs_err, sq_err=sq_err)
    if "bucket" in dfa.columns and dfa["bucket"].notna().any():
        m["mae_by_bucket"] = dfa.groupby("bucket")["abs_err"].mean().sort_values(ascending=False)
        m["rmse_by_bucket"] = dfa.groupby("bucket")["sq_err"].mean().apply(np.sqrt).sort_values(ascending=False)
        m["bias_by_bucket"] = dfa.groupby("bucket")["err"].mean()
        m["count_by_bucket"] = dfa["bucket"].value_counts()
    else:
        m["mae_by_bucket"] = m["rmse_by_bucket"] = m["bias_by_bucket"] = m["count_by_bucket"] = None
    if "action" in dfa.columns:
        m["mae_by_action"] = dfa.groupby("action")["abs_err"].mean().sort_values(ascending=False)
        m["rmse_by_action"] = dfa.groupby("action")["sq_err"].mean().apply(np.sqrt).sort_values(ascending=False)
    else:
        m["mae_by_action"] = m["rmse_by_action"] = None
    if "stage" in dfa.columns and dfa["stage"].notna().any():
        m["mae_by_stage"] = dfa.groupby("stage")["abs_err"].mean().sort_values(ascending=False)
        m["rmse_by_stage"] = dfa.groupby("stage")["sq_err"].mean().apply(np.sqrt).sort_values(ascending=False)
    else:
        m["mae_by_stage"] = m["rmse_by_stage"] = None
    return m


# ---------------------------------------------------------- metriche azione
def compute_action_metrics(df, tie_eps=TIE_EPS_DEFAULT):
    """df: righe degli stati NON falliti (tutte le azioni, incluse v_llm=0).
    L'argmax tollera i pareggi entro tie_eps (i valori sono quantizzati)."""
    rows = []
    for id_, grp in df.groupby("id", sort=False):
        grp = grp.reset_index(drop=True)
        if len(grp) == 0:
            continue
        v_true = grp["v_true"].values.astype(float)
        v_llm = pd.to_numeric(grp["v_llm"], errors="coerce").values.astype(float)
        llm_finite = np.isfinite(v_llm)
        if not llm_finite.any():
            continue
        opt_v = float(np.nanmax(v_true))
        if not np.isfinite(opt_v):
            continue
        # insieme delle azioni ottime (a pari merito)
        opt_mask = v_true >= opt_v - tie_eps
        opt_set = set(grp.loc[opt_mask, "action"])
        opt_label = grp.loc[opt_mask, "action"].iloc[0]        # "prima" per ordine file
        # insieme top-LLM (a pari merito)
        llm_v = float(np.nanmax(v_llm))
        top_mask = llm_finite & (v_llm >= llm_v - tie_eps)
        if not top_mask.any():
            top_mask = llm_finite & np.isclose(v_llm, llm_v)
        if not top_mask.any():
            continue
        top = grp.loc[top_mask]
        llm_label = top["action"].iloc[0]                      # "prima" del top (idxmax classico)
        llm_tie = bool(len(top) > 1)
        match = bool(opt_set & set(top["action"]))             # MATCH TOLLERANTE AI PAREGGI
        # rank: 1 + n. azioni con v_llm strettamente superiore
        vllm_rank = np.where(llm_finite, v_llm, -np.inf)
        opt_pos = np.where(opt_mask)[0]
        ranks = [1 + int(np.sum(vllm_rank > vllm_rank[i] + tie_eps)) for i in opt_pos]
        rank = min(ranks) if ranks else int(len(grp))
        # regret: costo in v_true della scelta
        chosen_v = float(v_true[top_mask][0])
        value_loss = float(opt_v - chosen_v)
        value_loss_best = float(opt_v - np.max(v_true[top_mask]))
        rows.append({
            "id": id_,
            "seed": grp["seed"].iloc[0] if "seed" in grp.columns else np.nan,
            "bucket": grp["bucket"].iloc[0] if "bucket" in grp.columns else "n/d",
            "stage": grp["stage"].iloc[0] if "stage" in grp.columns else "n/d",
            "n_actions": int(len(grp)),
            "n_optimal": int(len(opt_set)),
            "optimal_action": opt_label,
            "llm_action": llm_label,
            "llm_tie": llm_tie,
            "match": match,
            "rank": int(rank),
            "value_loss": value_loss,
            "value_loss_best": value_loss_best,
        })
    df_act = pd.DataFrame(rows)

    out = {"n_states": 0, "df": df_act, "accuracy": float("nan"), "accuracy_strict": float("nan"),
           "accuracy_strict_unique": float("nan"), "n_unique_opt": 0,
           "top2": float("nan"), "top3": float("nan"), "mean_value_loss": float("nan"),
           "mean_value_loss_best": float("nan"), "tie_rate": float("nan"),
           "acc_by_bucket": None, "acc_by_stage": None, "cm": None, "per_action": None,
           "regret_cell": None, "acc_ci": (float("nan"),) * 3, "regret_ci": (float("nan"),) * 3}
    if df_act.empty:
        return out
    groups = df_act["seed"].values if "seed" in df_act.columns else None
    out.update({
        "n_states": int(len(df_act)),
        "accuracy": float(df_act["match"].mean()),                              # tie-aware
        "accuracy_strict": float((df_act["optimal_action"] == df_act["llm_action"]).mean()),
        # strict su ottimo unico: esclude gli stati con pareggi ground-truth (n_optimal>1),
        # dove lo strict confronta due "prime in ordine di file" arbitrarie
        "accuracy_strict_unique": float((df_act.loc[df_act["n_optimal"] == 1, "optimal_action"]
                                         == df_act.loc[df_act["n_optimal"] == 1, "llm_action"]).mean()),
        "n_unique_opt": int((df_act["n_optimal"] == 1).sum()),
        "top2": float((df_act["rank"] <= 2).mean()),
        "top3": float((df_act["rank"] <= 3).mean()),
        "mean_value_loss": float(df_act["value_loss"].mean()),
        "mean_value_loss_best": float(df_act["value_loss_best"].mean()),
        "tie_rate": float(df_act["llm_tie"].mean()),
        "acc_by_bucket": df_act.groupby("bucket")["match"].mean(),
        "acc_by_stage": df_act.groupby("stage")["match"].mean(),
        "cm": pd.crosstab(df_act["optimal_action"], df_act["llm_action"]),
        "per_action": df_act.groupby("optimal_action")["match"].agg(["mean", "count"]),
        "regret_cell": df_act.groupby(["optimal_action", "llm_action"])["value_loss"].mean(),
        "acc_ci": bootstrap_grouped(df_act["match"].astype(float).values, groups),
        "regret_ci": bootstrap_grouped(df_act["value_loss"].values, groups),
    })
    return out


# --------------------------------------------------------------------- plot
def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4, width=0.7)


def _save(fig, path, rect=None):
    try:
        if rect is not None:
            fig.tight_layout(rect=rect)
        else:
            fig.tight_layout(pad=0.6)
    except Exception:
        pass
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _bucket_iter(df):
    """Itera (label, colore, subset) con fallback se manca la colonna bucket."""
    if "bucket" in df.columns:
        seen = set()
        for b, c in BUCKET_COLORS.items():
            sub = df[df["bucket"] == b]
            if len(sub):
                seen.add(b)
                yield b, c, sub
        # fail-safe: label non mappate (mai plot vuoti e silenziosi)
        rest = sorted(set(df["bucket"].dropna().unique()) - seen, key=str)
        for i, b in enumerate(rest):
            sub = df[df["bucket"] == b]
            if len(sub):
                yield b, _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)], sub
    else:
        yield "dati", "#0072B2", df


def _legend_outside(ax):
    """Legenda FUORI dall'area dati (a destra): non copre mai i punti.
    bbox_inches='tight' in _save la include nel file salvato."""
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=8, framealpha=0.95, edgecolor="0.85",
              handletextpad=0.5, borderpad=0.4)


def _stats_below(ax, text, y=-0.26):
    """Box statistiche SOTTO l'area dati (coordinate assi): non copre i punti.
    bbox_inches='tight' in _save lo include nel file salvato."""
    ax.text(0.0, y, text, transform=ax.transAxes, va="top", ha="left",
            fontsize=8.5, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      edgecolor="0.85", alpha=0.96))


def plot_scatter(df, metrics, title, path):
    fig, ax = plt.subplots(figsize=(7.7, 6.3))
    for b, c, sub in _bucket_iter(df):
        ax.scatter(sub["v_true"], sub["v_llm"], s=34, alpha=0.82, label=b, color=c,
                   edgecolor="white", linewidth=0.5, zorder=3)
    vals = pd.concat([df["v_true"], df["v_llm"]]).astype(float)
    try:
        p_lo, p_hi = vals.quantile([0.01, 0.99])
        spread = float(p_hi - p_lo)
        if spread < 0.18:
            mid = float((p_lo + p_hi) / 2)
            lo, hi = max(-0.05, mid - 0.11), min(1.05, mid + 0.11)
        else:
            lo, hi = max(-0.05, float(p_lo) - 0.03), min(1.05, float(p_hi) + 0.03)
        if lo >= hi:
            lo, hi = -0.05, 1.05
    except Exception:
        lo, hi = -0.05, 1.05
    n_out = int(((df["v_true"] < lo) | (df["v_true"] > hi) |
                 (df["v_llm"] < lo) | (df["v_llm"] > hi)).sum())
    ax.plot([lo, hi], [lo, hi], color="#444444", ls="--", lw=1.4, alpha=0.9, label="y = x", zorder=2)
    xs = np.linspace(lo, hi, 120)
    ax.fill_between(xs, xs - 0.05, xs + 0.05, color="#009E73", alpha=0.09, label="±0.05", zorder=1)
    ax.fill_between(xs, xs - 0.10, xs + 0.10, color="#009E73", alpha=0.045, label="±0.10", zorder=1)
    if np.isfinite(metrics.get("ols_slope", np.nan)):  # regressione OLS (coerente con R²)
        ax.plot(xs, metrics["ols_intercept"] + metrics["ols_slope"] * xs, color="#D55E00", lw=2.2,
                alpha=0.95, label=f"OLS slope={metrics['ols_slope']:.2f}", zorder=4)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("v_true  (V*(s'))"); ax.set_ylabel("v_llm  (stima LLM)")
    ax.set_title(title, loc="left", fontsize=13, pad=14, fontweight="bold")
    ax.grid(True, alpha=0.22, linestyle=":", linewidth=0.6)
    _despine(ax)
    _legend_outside(ax)  # FUORI dall'area dati
    text = (f"Pearson r = {_f(metrics['pearson_r'], '{:.3f}')}  [{_f(metrics['pearson_ci'][1], '{:.3f}')}, {_f(metrics['pearson_ci'][2], '{:.3f}')}]\n"
            f"CCC = {_f(metrics['ccc'], '{:.3f}')}  [{_f(metrics['ccc_ci'][1], '{:.3f}')}, {_f(metrics['ccc_ci'][2], '{:.3f}')}]\n"
            f"MAE = {_f(metrics['mae'])}  [{_f(metrics['mae_ci'][1])}, {_f(metrics['mae_ci'][2])}]   RMSE = {_f(metrics['rmse'])}\n"
            f"bias = {_f(metrics['bias'], '{:+.4f}')}   fail = {_f(metrics.get('failure', float('nan')), '{:.1f}%')} "
            f"({metrics.get('n_fail', 'n/d')}/{metrics.get('n_total', 'n/d')} stati)")
    if n_out > 0:
        text += f"\noutlier fuori zoom: {n_out}"
    _stats_below(ax, text)  # SOTTO l'area dati
    _save(fig, path)


def plot_residuals(df, metrics, title, path):
    """Residui v_llm - v_true vs v_true: con valori quantizzati mostra
    direttamente gli 'scalini' off-by-one (~ -(1-gamma^k) ≈ -0.01)."""
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    for b, c, sub in _bucket_iter(df):
        r = sub["v_llm"].values - sub["v_true"].values
        ax.scatter(sub["v_true"], r, s=30, alpha=0.75, label=b, color=c,
                   edgecolor="white", linewidth=0.4, zorder=3)
    resid = df["v_llm"].values - df["v_true"].values
    try:
        q_lo, q_hi = np.nanpercentile(resid, [1, 99])
    except Exception:
        q_lo, q_hi = -0.02, 0.02
    half = max(0.02, 1.15 * max(abs(float(q_lo)), abs(float(q_hi))))
    ax.set_ylim(-half, half)
    xmin, xmax = float(df["v_true"].min()), float(df["v_true"].max())
    pad = max(0.01, 0.05 * (xmax - xmin))
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.fill_between([xmin - pad, xmax + pad], -0.05, 0.05, color="#009E73", alpha=0.08, zorder=1)
    ax.fill_between([xmin - pad, xmax + pad], -0.10, 0.10, color="#009E73", alpha=0.04, zorder=1)
    ax.axhline(0, color="#444444", lw=1.3, zorder=2)
    ax.set_xlabel("v_true  (V*(s'))"); ax.set_ylabel("residuo  v_llm − v_true")
    ax.set_title(title, loc="left", fontsize=13, pad=12, fontweight="bold")
    ax.grid(True, alpha=0.22, linestyle=":", linewidth=0.6)
    _despine(ax)
    _legend_outside(ax)  # FUORI dall'area dati
    txt = f"MAE = {_f(metrics['mae'])}\nbias = {_f(metrics['bias'], '{:+.4f}')}\n"
    if metrics.get("k_n", 0) > 0:
        txt += f"Δk = 0: {metrics['k_dk_exact']*100:.1f}%    |Δk| = 1: {metrics['k_dk_abs1']*100:.1f}%"
    _stats_below(ax, txt)  # SOTTO l'area dati
    _save(fig, path)


def plot_k_error(metrics, gamma, title, path):
    """Istogramma di Δk = k_llm - k_true (v = gamma^k). È la metrica naturale
    per errori quantizzati: 'sbagliare l'esponente di 1'."""
    dk = metrics.get("dk_values")
    if dk is None or len(dk) == 0:
        return False
    dk = np.asarray(dk, float)
    total = len(dk)
    dk_c = np.clip(dk, -DK_CLIP, DK_CLIP)
    centers = np.arange(int(max(-DK_CLIP, dk_c.min())), int(min(DK_CLIP, dk_c.max())) + 1)
    if len(centers) == 0:
        return False
    counts = np.array([int(np.sum(dk_c == c)) for c in centers], float)
    if counts.sum() == 0:
        return False
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    colors = ["#CC3311" if abs(c) >= 2 else ("#E69F00" if abs(c) == 1 else "#009E73") for c in centers]
    bars = ax.bar(centers, counts, color=colors, edgecolor="white", width=0.8, zorder=3)
    for bar, cnt in zip(bars, counts):
        if cnt > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, cnt + total * 0.012,
                    f"{int(cnt)}\n{cnt / total * 100:.1f}%", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
    tick_labels = [str(c) for c in centers]
    if dk.min() <= -DK_CLIP: tick_labels[0] = f"≤−{DK_CLIP}"
    if dk.max() >= DK_CLIP:  tick_labels[-1] = f"≥{DK_CLIP}"
    ax.set_xticks(centers); ax.set_xticklabels(tick_labels)
    ax.set_xlabel(f"Δk = k_llm − k_true   (v = γ^k,  γ = {gamma:g})")
    ax.set_ylabel("numero di stime")
    ax.set_ylim(0, counts.max() * 1.30 if counts.max() > 0 else 1)
    ax.set_title(title, loc="left", fontsize=13, pad=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.22, linestyle=":", linewidth=0.6)
    _despine(ax)
    txt = (f"Δk = 0: {metrics['k_dk_exact']*100:.1f}%   |Δk| = 1: {metrics['k_dk_abs1']*100:.1f}%   "
           f"|Δk| ≥ 2: {metrics['k_dk_abs2plus']*100:.1f}%\nE|Δk| = {metrics['k_dk_abs_mean']:.3f}   "
           f"bias Δk = {metrics['k_dk_mean']:+.3f}   (n = {metrics['k_n']})")
    _stats_below(ax, txt)  # SOTTO l'area dati (prima era in alto a destra, sui bar)
    _save(fig, path)
    return True


def plot_action_acc(act, title, path):
    if act["n_states"] == 0:
        return
    n_actions = len(ACTION_ORDER)
    rand = 1 / n_actions
    labels = ["Top-1\n(tie-aware)", "Top-1\n(strict)", "Top-2", "Top-3"]
    vals = [act["accuracy"], act["accuracy_strict"], act["top2"], act["top3"]]
    colors = ["#0072B2", "#999999", "#009E73", "#56B4E9"]
    fig, ax = plt.subplots(figsize=(7.0, 4.9))
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.8, width=0.55, zorder=3)
    ax.bar(labels, vals, color="none", edgecolor="black", linewidth=0.7, width=0.55, zorder=4)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, (v if np.isfinite(v) else 0) + 0.02,
                f"{v*100:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    # linea random ANNOTATA sulla linea stessa (niente legenda che copre le barre)
    ax.axhline(rand, color="#666666", linestyle="--", lw=1.2, alpha=0.85, zorder=2)
    ax.text(1.0, rand + 0.018, f"random {rand*100:.1f}%",
            transform=ax.get_yaxis_transform(),  # x in coord. assi, y in coord. dati
            ha="right", va="bottom", fontsize=8, color="#666666")
    ax.set_ylim(0, 1.12); ax.set_ylabel("accuratezza")
    ax.set_title(title, loc="left", fontsize=13, pad=12, fontweight="bold")
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{int(t*100)}%" for t in np.linspace(0, 1, 6)])
    ax.grid(True, axis="y", alpha=0.22, linestyle=":", linewidth=0.6)
    _despine(ax)
    txt = (f"n = {act['n_states']} stati\nregret medio = {_f(act['mean_value_loss'])} "
           f"[{_f(act['regret_ci'][1])}, {_f(act['regret_ci'][2])}]\n"
           f"acc. tie-aware 95% CI [{_f(act['acc_ci'][1]*100, '{:.1f}%')}, {_f(act['acc_ci'][2]*100, '{:.1f}%')}]\n"
           f"top LLM con pareggi: {_f(act['tie_rate'] * 100, '{:.1f}%')}")
    _stats_below(ax, txt, y=-0.34)  # SOTTO (tick label su 2 righe -> più in basso)
    _save(fig, path)


def plot_mae_rmse_bucket(metrics, tag, gdir):
    mb = metrics.get("mae_by_bucket"); rb = metrics.get("rmse_by_bucket")
    if mb is None or rb is None or mb.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle(f"Errore per bucket — {tag}", fontsize=13, fontweight="bold")
    for ax, data, ylabel, title in [(axes[0], mb, "MAE", "MAE per bucket"),
                                    (axes[1], rb, "RMSE", "RMSE per bucket")]:
        colors = [BUCKET_COLORS.get(k, "#6A6A6A") for k in data.index]
        bars = ax.bar(data.index, data.values, color=colors, edgecolor="white", linewidth=0.9, width=0.55, zorder=3)
        ax.bar(data.index, data.values, color="none", edgecolor="black", linewidth=0.7, width=0.55, zorder=4)
        for bar, v in zip(bars, data.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(title, fontsize=11, pad=10); ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(0, max(data.values) * 1.22 if len(data) and max(data.values) > 0 else 1)
        ax.grid(True, axis="y", alpha=0.22, linestyle=":", linewidth=0.6); _despine(ax)
    _save(fig, str(gdir / f"{tag}_mae_rmse_bucket.png"), rect=(0, 0, 1, 0.94))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(mb.index, mb.values, color=[BUCKET_COLORS.get(k, "#888888") for k in mb.index],
           edgecolor="white", linewidth=0.9, width=0.55, zorder=3)
    ax.bar(mb.index, mb.values, color="none", edgecolor="black", linewidth=0.7, width=0.55, zorder=4)
    for bar, v in zip(ax.patches[:len(mb)], mb.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title(f"MAE per bucket — {tag}", loc="left", fontsize=13, pad=12, fontweight="bold")
    ax.set_ylabel("MAE", fontsize=10)
    ax.set_ylim(0, max(mb.values) * 1.22 if len(mb) and max(mb.values) > 0 else 1)
    ax.grid(True, axis="y", alpha=0.22, linestyle=":", linewidth=0.6); _despine(ax)
    _save(fig, str(gdir / f"{tag}_mae_bucket.png"))


def plot_confusion(act, title, path, square=True):
    """Confusion matrix quadrata (stesse 6 azioni su entrambi gli assi).
    Le azioni mai ottime nei dati (riga a zero conteggi) vengono hatchate e
    annotate invece di restare bianche: è informazione, non un errore di plot.
    Pannello 2: regret medio per cella."""
    cm = act.get("cm")
    if cm is None or cm.empty:
        return
    # matrice tie-aware: solo stati con ottimo unico. Con pareggi ground-truth,
    # optimal_action è la "prima in ordine di file" e gli stati tie finirebbero
    # fuori diagonale anche con scelta ottima (artefatto, non confusione vera).
    df = act.get("df")
    n_tie_excl = 0
    reg = None  # se resta None, sotto si usa regret_cell globale
    if df is not None and len(df) and "n_optimal" in df.columns:
        n_tie_excl = int((df["n_optimal"] > 1).sum())
        dfu = df[df["n_optimal"] == 1]
        if len(dfu):
            cm = pd.crosstab(dfu["optimal_action"], dfu["llm_action"])
            reg = dfu.groupby(["optimal_action", "llm_action"])["value_loss"].mean()
        else:
            return
    rows = ACTION_ORDER if square else [a for a in ACTION_ORDER if a in cm.index]
    cols = ACTION_ORDER
    if not rows:
        return
    cm_r = cm.reindex(index=rows, columns=cols).fillna(0.0)

    # --- FIX CRASH: tutto in numpy; masked_where non broadcasta (6,1)->(6,6),
    #     quindi la maschera la costruisco con np.broadcast_to ---
    cm_vals = cm_r.values.astype(float)          # (n_rows, n_cols)
    row_sums = cm_vals.sum(axis=1)               # (n_rows,)
    safe = np.where(row_sums == 0, 1.0, row_sums)
    cm_frac = cm_vals / safe[:, None]            # frazione per riga (righe vuote = 0, mascherate)
    mask_empty = np.broadcast_to((row_sums == 0)[:, None], cm_frac.shape)  # (n_rows, n_cols)
    cm_frac_m = np.ma.masked_array(cm_frac, mask=mask_empty)

    # matrice regret per cella (se sopra è già stata ricalcolata sul subset tie-aware, riusa quella)
    cm_reg = np.full((len(rows), len(cols)), np.nan)
    if reg is None:
        reg = act.get("regret_cell")
    if reg is not None and len(reg):
        for (oa, la), v in reg.items():
            if oa in rows and la in cols:
                cm_reg[rows.index(oa), cols.index(la)] = float(v)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # ---- pannello 1: conteggi (frazione per riga) ----
    im = axes[0].imshow(cm_frac_m, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    axes[0].set_xticks(range(len(cols))); axes[0].set_xticklabels(cols, rotation=28, ha="right", fontsize=9)
    axes[0].set_yticks(range(len(rows))); axes[0].set_yticklabels(rows, fontsize=9)
    axes[0].set_xlabel("Azione scelta LLM"); axes[0].set_ylabel("Azione ottima (verità)")
    axes[0].set_title("Conteggi (frazione per riga)", fontsize=11, pad=10)
    for i in range(len(rows)):
        if row_sums[i] == 0:
            # riga mai osservata: hatch + annotazione (sopra l'immagine, che è
            # mascherata/trasparente in quella riga)
            axes[0].add_patch(plt.Rectangle((-0.5, i - 0.5), len(cols), 1, facecolor="#F2F2F2",
                                            hatch="//", edgecolor="#BBBBBB", lw=0.0, zorder=2))
            axes[0].text((len(cols) - 1) / 2, i, "mai ottima unica in questi dati\n(solo pareggi o dominata)",
                         ha="center", va="center", fontsize=8, color="#777777", style="italic", zorder=3)
            continue
        for j in range(len(cols)):
            v = cm_vals[i, j]; vn = cm_frac[i, j]
            if v > 0:
                axes[0].text(j, i, f"{int(v)}\n{vn*100:.0f}%", ha="center", va="center", fontsize=8.5,
                             fontweight="bold", color="white" if vn > 0.55 else "black", zorder=3)
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    cbar.set_label("frazione per riga", fontsize=9)

    # ---- pannello 2: regret medio per cella ----
    masked = np.ma.masked_invalid(cm_reg)
    finite_reg = cm_reg[np.isfinite(cm_reg)]
    vmax = float(finite_reg.max()) if finite_reg.size else 1e-3
    vmax = vmax if vmax > 0 else 1e-3
    im2 = axes[1].imshow(masked, cmap="OrRd", vmin=0, vmax=vmax, aspect="auto")
    axes[1].set_xticks(range(len(cols))); axes[1].set_xticklabels(cols, rotation=28, ha="right", fontsize=9)
    axes[1].set_yticks(range(len(rows))); axes[1].set_yticklabels(rows, fontsize=9)
    axes[1].set_xlabel("Azione scelta LLM"); axes[1].set_ylabel("Azione ottima (verità)")
    axes[1].set_title("Regret medio per cella" + ("" if finite_reg.size else " (nessun mismatch)"),
                      fontsize=11, pad=10)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = cm_reg[i, j]
            if np.isfinite(v):
                axes[1].text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8.5, fontweight="bold",
                             color="white" if v > 0.55 * vmax else "black", zorder=3)
    cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label("regret  (v* ottima − v* scelta)", fontsize=9)

    empty = [a for i, a in enumerate(rows) if row_sums[i] == 0]
    n_cm = int(row_sums.sum())  # stati a ottimo unico effettivamente in matrice (== len(dfu))
    footer = (f"matrice: n = {n_cm} stati a ottimo unico (da {act['n_states']} valutati) — Top-1 strict {_f(act['accuracy_strict'] * 100, '{:.1f}%')} vs "
              f"tie-aware {_f(act['accuracy'] * 100, '{:.1f}%')} — strict-unique {_f(act['accuracy_strict_unique'] * 100, '{:.1f}%')} — top LLM con pareggi: {_f(act['tie_rate'] * 100, '{:.1f}%')}")
    if n_tie_excl:
        footer += f" — {n_tie_excl} stati con pareggi esclusi dalla matrice (somma celle {n_cm} = {act['n_states']} − {n_tie_excl})"
    if empty:
        footer += f" — mai ottime: {', '.join(empty)} (atteso: no-op negli stage valutati)"
    fig.text(0.5, 0.015, footer, ha="center", fontsize=8.5, color="#444444")
    _save(fig, path, rect=(0, 0.045, 1, 0.93))


# --------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(
        description="Valuta LLM su grafo (v_true vs v_llm) — v3: tie-aware, filtro per stato, Δk")
    parser.add_argument("--path", type=str, required=True,
                        help="file JSON/CSV o cartella con llm_results")
    parser.add_argument("--outdir", type=str, default="evaluate/output", help="output dir")
    parser.add_argument("--gamma", type=str, default=str(DEFAULT_GAMMA),
                        help="fattore di sconto (v = gamma^k); 'auto' per stima dai dati (default 0.99)")
    parser.add_argument("--tie-eps", type=float, default=TIE_EPS_DEFAULT,
                        help=f"tolleranza per considerare due valori pari merito (default {TIE_EPS_DEFAULT})")
    args = parser.parse_args()

    files = [f for f in find_files(args.path) if "llm_results" in os.path.basename(f)]
    if not files:
        print(f"Nessun file trovato in {args.path}")
        return
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    gdir = outdir / "grafici"; gdir.mkdir(parents=True, exist_ok=True)

    log_lines = []
    log_lines.append("=" * 70)
    log_lines.append("VALUTAZIONE LLM (v_true vs v_llm) Q via V — v3")
    log_lines.append("filtro per stato | argmax tie-aware | regret primario | Δk = k_llm − k_true")
    log_lines.append(f"Generato {datetime.now():%Y-%m-%d %H:%M:%S}")
    log_lines.append(f"Sorgente {args.path} files={len(files)}")
    log_lines.append("=" * 70)
    all_results = []

    for f in files:
        print(f"Valuto {f}")
        df_raw = load_df(f)
        if df_raw is None or df_raw.empty:
            log_lines.append(f"\n[SALTATO] {f}: vuoto o formato non riconosciuto")
            continue
        missing = {"id", "action", "v_true", "v_llm"} - set(df_raw.columns)
        if missing:
            log_lines.append(f"\n[SALTATO] {f}: colonne mancanti {sorted(missing)}")
            continue

        # ---- FILTRO PER STATO (non per riga!) ----
        def _state_failed(s):
            v = pd.to_numeric(s, errors="coerce")
            return bool((v.isna() | (v == 0)).all())

        failed = df_raw.groupby("id")["v_llm"].apply(_state_failed)
        failed_ids = set(failed[failed].index)
        n_states_total = int(df_raw["id"].nunique())
        n_failed_states = len(failed_ids)
        failure = n_failed_states / n_states_total if n_states_total else float("nan")
        df_states = df_raw[~df_raw["id"].isin(failed_ids)].copy()
        if df_states.empty:
            log_lines.append(f"\n[SALTATO] {f}: tutti gli stati falliti (v_llm=0)")
            continue
        df_val = df_states[df_states["v_llm"].notna() & (df_states["v_llm"] != 0)].copy()
        n_zero_rows = int(len(df_states) - len(df_val))
        if df_val.empty:
            log_lines.append(f"\n[SALTATO] {f}: nessuna stima valida")
            continue

        # ---- gamma ----
        gamma_src = "cli"
        if str(args.gamma).strip().lower() in ("auto", "detect"):
            detected = detect_gamma(np.concatenate([df_val["v_true"].values, df_val["v_llm"].values]))
            if detected:
                gamma, gamma_src = float(detected), "auto"
            else:
                gamma, gamma_src = DEFAULT_GAMMA, "auto (fallback: detection fallita)"
        else:
            gamma = float(args.gamma)

        # ---- metriche ----
        metrics = compute_value_metrics(df_val, gamma)
        metrics.update({
            "n_total": n_states_total, "n_fail": n_failed_states, "failure": failure,
            "n_states_valid": n_states_total - n_failed_states, "n_zero_rows": n_zero_rows,
            "gamma": gamma, "gamma_src": gamma_src,
        })
        act = compute_action_metrics(df_states, tie_eps=args.tie_eps)
        tag = Path(f).stem

        # ---- plot ----
        plot_scatter(df_val, metrics, f"v_true vs v_llm — {tag}", str(gdir / f"{tag}_scatter.png"))
        plot_residuals(df_val, metrics, f"Residui v_llm − v_true — {tag}", str(gdir / f"{tag}_residuals.png"))
        ok_k = plot_k_error(metrics, gamma, f"Errore sull'esponente k — {tag}", str(gdir / f"{tag}_k_error.png"))
        plot_action_acc(act, f"Accuratezza azione — {tag}", str(gdir / f"{tag}_accuracy.png"))
        plot_mae_rmse_bucket(metrics, tag, gdir)
        plot_confusion(act, f"Confusion matrix — {tag}", str(gdir / f"{tag}_confusion.png"))

        # ---- log ----
        log_lines.append("\n" + "-" * 70)
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d %H:%M:%S")
        except OSError:
            mtime = "n/d"
        log_lines.append(f"FILE: {f} (mtime {mtime}, righe file {len(df_raw)})")
        # guard: stati incompleti (attese 6 azioni/stato) — file parziale da query incrementale
        cnt_raw = df_raw.groupby("id").size()
        n_incomplete = int((cnt_raw != len(ACTION_ORDER)).sum())
        if n_incomplete:
            log_lines.append(f"  WARN: {n_incomplete} stati incompleti (<>{len(ACTION_ORDER)} azioni, file parziale?)")
        log_lines.append(f"Gamma: {gamma:g} ({gamma_src}) — on-grid v_true={_f(metrics['on_grid_true'], '{:.1%}')}, "
                         f"v_llm={_f(metrics['on_grid_llm'], '{:.1%}')}")
        log_lines.append(f"Stati: totali={n_states_total}  falliti (v_llm=0 su tutte le azioni)={n_failed_states} "
                         f"({_f(failure * 100, '{:.1f}%')})  validi={n_states_total - n_failed_states}")
        log_lines.append(f"Righe: stime valide={metrics['n']}  v_llm=0/nan in stati validi={n_zero_rows}")
        if metrics.get("k_n", 0) > 0:
            log_lines.append(f"Δk (n={metrics['k_n']}): Δk=0 {metrics['k_dk_exact']*100:.1f}%  "
                             f"|Δk|=1 {metrics['k_dk_abs1']*100:.1f}%  |Δk|≥2 {metrics['k_dk_abs2plus']*100:.1f}%  "
                             f"E|Δk|={metrics['k_dk_abs_mean']:.3f}  bias Δk={metrics['k_dk_mean']:+.3f}")
            if not ok_k:
                log_lines.append("  (plot Δk non generato)")
        else:
            log_lines.append("Δk: non calcolabile (valori non on-grid con questo gamma)")
        log_lines.append(f"VALORI: MAE={_f(metrics['mae'])} [{_f(metrics['mae_ci'][1])}, {_f(metrics['mae_ci'][2])}] "
                         f"RMSE={_f(metrics['rmse'])} [{_f(metrics['rmse_ci'][1])}, {_f(metrics['rmse_ci'][2])}] "
                         f"bias={_f(metrics['bias'], '{:+.4f}')} max_err={_f(metrics['max_err'])}")
        log_lines.append(f"Entro soglia: ≤0.05={_f(metrics['within_05'], '{:.1f}%')}  "
                         f"≤0.10={_f(metrics['within_10'], '{:.1f}%')}")
        log_lines.append(f"Pearson r={_f(metrics['pearson_r'], '{:.3f}')} "
                         f"[{_f(metrics['pearson_ci'][1], '{:.3f}')}, {_f(metrics['pearson_ci'][2], '{:.3f}')}]  "
                         f"Spearman ρ={_f(metrics['spearman_r'], '{:.3f}')}  "
                         f"CCC={_f(metrics['ccc'], '{:.3f}')} "
                         f"[{_f(metrics['ccc_ci'][1], '{:.3f}')}, {_f(metrics['ccc_ci'][2], '{:.3f}')}]")
        log_lines.append(f"OLS: slope={_f(metrics['ols_slope'], '{:.3f}')}  "
                         f"intercept={_f(metrics['ols_intercept'], '{:+.4f}')}  R²={_f(metrics['r2'], '{:.3f}')}  "
                         f"(slope CCC={_f(metrics['ccc_slope'], '{:.3f}')}, solo per confronto)")
        n_act = len(ACTION_ORDER)
        log_lines.append(f"AZIONI (n={act['n_states']} stati): Top-1 tie-aware="
                         f"{_f(act['accuracy'] * 100, '{:.1f}%')} "
                         f"[{_f(act['acc_ci'][1] * 100, '{:.1f}%')}, {_f(act['acc_ci'][2] * 100, '{:.1f}%')}]  "
                          f"Top-1 strict={_f(act['accuracy_strict'] * 100, '{:.1f}%')}  "
                          f"Top-1 strict-unique={_f(act['accuracy_strict_unique'] * 100, '{:.1f}%')} "
                          f"(n={act['n_unique_opt']})  "
                          f"Top-2={_f(act['top2'] * 100, '{:.1f}%')}  Top-3={_f(act['top3'] * 100, '{:.1f}%')}  "
                         f"(random {100 / n_act:.1f}% su {n_act} azioni)")
        log_lines.append(f"  regret medio (prima riga del top LLM)={_f(act['mean_value_loss'])} "
                         f"[{_f(act['regret_ci'][1])}, {_f(act['regret_ci'][2])}]  "
                         f"regret (tie favorevole)={_f(act['mean_value_loss_best'])}  "
                         f"top LLM con pareggi={_f(act['tie_rate'], '{:.1f}%')}")
        # ponytail: denominatore matrice esplicito (somma celle = unici, non totale)
        n_tie = int(act['n_states'] - act['n_unique_opt'])
        log_lines.append(f"  confusion: somma celle={act['n_unique_opt']} (stati a ottimo unico) = "
                         f"{act['n_states']} valutati − {n_tie} con pareggi esclusi")

        if metrics["mae_by_bucket"] is not None:
            log_lines.append("Valori per bucket:")
            for b in metrics["mae_by_bucket"].index:
                mae_v = metrics["mae_by_bucket"][b]
                rmse_v = metrics["rmse_by_bucket"][b] if metrics["rmse_by_bucket"] is not None else float("nan")
                bias_v = metrics["bias_by_bucket"][b] if metrics["bias_by_bucket"] is not None else float("nan")
                cnt = metrics["count_by_bucket"][b] if metrics["count_by_bucket"] is not None else 0
                log_lines.append(f"  {b:12s} n={cnt:4d} MAE={mae_v:.4f} RMSE={rmse_v:.4f} bias={bias_v:+.4f}")
        if act["acc_by_bucket"] is not None and len(act["acc_by_bucket"]):
            log_lines.append("Selezione azione per bucket:")
            for b, v in act["acc_by_bucket"].items():
                log_lines.append(f"  {b:12s} acc(tie-aware)={v*100:5.1f}%")
        if metrics["mae_by_action"] is not None:
            log_lines.append("Valori per action:")
            for a in metrics["mae_by_action"].index:
                mae_v = metrics["mae_by_action"][a]
                rmse_v = metrics["rmse_by_action"][a] if metrics["rmse_by_action"] is not None else float("nan")
                log_lines.append(f"  {a:10s} MAE={mae_v:.4f} RMSE={rmse_v:.4f}")
        if metrics["mae_by_stage"] is not None:
            log_lines.append("Valori per stage:")
            for s in metrics["mae_by_stage"].index:
                mae_v = metrics["mae_by_stage"][s]
                rmse_v = metrics["rmse_by_stage"][s] if metrics["rmse_by_stage"] is not None else float("nan")
                log_lines.append(f"  {s:14s} MAE={mae_v:.4f} RMSE={rmse_v:.4f}")
        all_results.append((f, metrics, act))

    log_path = outdir / "log_valutazione.txt"
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write("\n".join(log_lines))
    print(f"\nLog salvato {log_path}, grafici in {gdir}")


if __name__ == "__main__":
    main()
