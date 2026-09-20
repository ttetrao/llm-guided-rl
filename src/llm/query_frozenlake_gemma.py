#!/usr/bin/env python3
"""
Interroga Gemini AI Studio (gemma-4-26b-a4b-it) su FrozenLake slippery (MDP stocastico).

Clone di llm/query_gemma.py (plumbing Gemini) con logica FrozenLake da
llm/query_frozenlake_gpt.py, solo mode q:
- Sorgente stati: bucket da frozenlake_qstates_{map}_slippery_seed{seed}.pkl
  (default bottleneck: goal_entry + on-policy radi + 1-ring con quota H)
  + grafo frozenlake_mdp_{map}_slippery_seed{seed}.pkl per mappe s e outcome s'
- Prompt: Q* via media pesata -> Q*(s,a)=Σ p(s',r|s,a)[r+γV*(s')] con 4 azioni
  (left/down/right/up); ogni <a> contiene <expected> (esito voluto) + <alternative>
  (slip) con mappe s' (p da <documentation> success_rate, mai scritte, NON inferire)
- Batch: 1 stato per request, retry, ThreadPool; pacing 65s (rate Gemini)
- Output: JSON unico {seeds, bucket, rows}, una entry per (state,action); dedup
- v_true = Q* stocastico = Σ p·[r+γV*] (0.0 su done)
- id al LLM solo NNNN, bucket ricostruito al salvataggio (id=NNNN_bucket)

Uso:
    python -m llm.query_frozenlake_gemma --map 8x8 --seed 1337 --bucket bottleneck --dry-run
    python -m llm.query_frozenlake_gemma --map 8x8 --seed 1337 --bucket bottleneck --limit 5
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path

_THIS = Path(__file__).resolve()
_SRC = _THIS.parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# monorepo disponibile solo in env con api keys; import lazy per dry-run
try:
    from monorepo import GeminiLLM, load_api_keys
except ImportError:
    GeminiLLM = None
    load_api_keys = lambda: None

from graph.frozenlake_mdp_graph import load_or_build as load_mdp
from graph.frozenlake_qlearning_states import get_qstates_paths, load_qstates

# ---------------------------------------------------------------------------
# Config — plumbing come query_gemma.py (doorkey/Gemini)
# ---------------------------------------------------------------------------
MODEL_ID = "gemma-4-26b-a4b-it"
HISTORY_PER_REQUEST = 1  # uno stato per request (4 azioni x 1-3 outcome + current)
RETRY_LIMIT = 3
RETRY_DELAY = 10
LAUNCH_INTERVAL_SEC = 65
MAX_INFLIGHT = 5
DEFAULT_MAP = "8x8"
DEFAULT_SEED = 1337

BUCKETS_WANTED = ["bottleneck"]
ACTION_ORDER = ["left", "down", "right", "up"]  # 0,1,2,3


@dataclass(frozen=True)
class ResultRow:
    id: str  # salvato come f"{state:04d}_{bucket}"; al LLM va solo f"{state:04d}"
    seed: int
    map_name: str
    bucket: str
    state: int
    r: int
    c: int
    ch: str
    action: str
    v_true: float
    v_llm: float
    analysis: str


def get_output_paths(
    map_name: str,
    seed: int,
    model_id: str = MODEL_ID,
    out_dir: Path | str | None = None,
):
    if out_dir is None:
        out_dir = Path(__file__).parent.parent / "graph" / "data"
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"frozenlake_llm_{map_name}_slippery_q_seed{seed}_{model_id.replace(':','_').replace('/','_')}"
    return out_dir / f"{base}.json"


_FROZEN_STATES_RE = re.compile(
    r"frozenlake_qstates_(\d+x\d+)_(slippery|deterministic)_seed(\d+)$"
)


def _resolve_states_path(states_path, map_name, seed, model_id=MODEL_ID):
    """--path: file .pkl stati. map/slip/seed derivati dal nome salvo espliciti."""
    p = Path(states_path)
    m = _FROZEN_STATES_RE.match(p.stem)
    is_slippery = True
    if m:
        if map_name is None:
            map_name = m.group(1)
        is_slippery = m.group(2) == "slippery"
        if seed is None:
            seed = int(m.group(3))
    if map_name is None:
        map_name = DEFAULT_MAP
    if seed is None:
        seed = DEFAULT_SEED
    if m is None:
        print(f"  warn: nome {p.name} non riconosciuto, uso map={map_name} seed={seed}")
    out_json = (
        p.parent
        / f"frozenlake_llm_{p.stem}_q_{model_id.replace(':','_').replace('/','_')}.json"
    )
    return p, map_name, is_slippery, seed, out_json


def load_docs(lang: str = "it"):
    base = Path(__file__).parent.parent / "docs" / "frozenlake" / lang

    def read(p):
        try:
            return open(base / p, encoding="utf-8").read()
        except FileNotFoundError:
            fb = Path(__file__).parent.parent / "docs" / "frozenlake" / "it"
            return open(fb / p, encoding="utf-8").read()

    prompt_tmpl = read("prompt.txt")
    v_def = read("v-function_definition.md")
    q_def = read("q-function_definition.md")
    try:
        doc = read("FrozenLakeDocumentation.md")
    except FileNotFoundError:
        doc = read("MiniGridDocumentation.md")
    legend = read("legend.txt")
    try:
        t_def = read("transition-model.md")
    except FileNotFoundError:
        t_def = ""
    return prompt_tmpl, v_def, q_def, doc, legend, t_def


def build_base_prompt(
    prompt_tmpl: str, v_def: str, q_def: str, doc: str, legend: str, t_def: str = ""
) -> str:
    out = prompt_tmpl
    out = out.replace(
        "<v-definition></v-definition>", f"<v-definition>\n{v_def}\n</v-definition>"
    )
    out = out.replace(
        "<q-definition></q-definition>", f"<q-definition>\n{q_def}\n</q-definition>"
    )
    out = out.replace(
        "<documentation></documentation>", f"<documentation>\n{doc}\n</documentation>"
    )
    out = out.replace("<legend></legend>", f"<legend>\n{legend}\n</legend>")
    if t_def:
        out = out.replace(
            "<transition-model></transition-model>",
            f"<transition-model>\n{t_def}\n</transition-model>",
        )
    return out


DIR_DELTA = {"left": (0, -1), "down": (1, 0), "right": (0, 1), "up": (-1, 0)}


def build_prompt_for_state(base_prompt: str, mdp: dict, state: int, code: str) -> str:
    node = mdp["nodes"][state]
    cur_map = node["map"]
    nrow, ncol = int(mdp["nrow"]), int(mdp["ncol"])
    slip = "true" if mdp.get("is_slippery", True) else "false"
    prompt = base_prompt.replace(
        '<current_state id="xxxx" is_slippery="true"></current_state>',
        f'<current_state id="{code}" is_slippery="{slip}">\n{cur_map}\n</current_state>',
    )
    prompt = prompt.replace(  # fallback template legacy senza attributo
        '<current_state id="xxxx"></current_state>',
        f'<current_state id="{code}" is_slippery="{slip}">\n{cur_map}\n</current_state>',
    )
    for act_name in ACTION_ORDER:
        aid = next(
            (int(k) for k, v in mdp["action_names"].items() if v == act_name), None
        )
        if aid is None:
            continue
        trans_list = (
            node["transitions"].get(aid) or node["transitions"].get(str(aid)) or []
        )
        # esito voluto: outcome che cade nella cella della direzione azione (muro = fermo)
        dr, dc = DIR_DELTA[act_name]
        tr, tc = node["r"] + dr, node["c"] + dc
        if not (0 <= tr < nrow and 0 <= tc < ncol):
            tr, tc = node["r"], node["c"]
        exp, alt, seen = [], [], set()
        for t in trans_list:
            if float(t["prob"]) <= 1e-9:
                continue
            nid = int(t["next_id"])
            if nid in seen:
                continue
            seen.add(nid)
            m = mdp["nodes"][nid]
            # solo mappe, niente probabilità: p si leggono in <documentation>
            if (m["r"], m["c"]) == (tr, tc) and not exp:
                exp.append(m["map"])
            else:
                alt.append(m["map"])
        if not exp and alt:
            exp = [alt.pop(0)]  # fallback: expected mai vuoto
        if not exp and not alt:
            block = f"<{act_name}>\n(no outcome)\n</{act_name}>"
        else:
            inner = "<expected>\n" + "\n".join(exp) + "\n</expected>\n"
            inner += "".join(f"<alternative>\n{m}\n</alternative>\n" for m in alt)
            block = f"<{act_name}>\n{inner}</{act_name}>"
        # anchor a <expected>: nel testo intro <left>/<down>/... compaiono senza chiusura
        prompt = re.sub(
            rf"<{act_name}>\s*<expected>.*?</{act_name}>",
            lambda _: block,
            prompt,
            flags=re.DOTALL,
        )
    return prompt


def q_true_for(mdp: dict, state: int, act: str) -> float:
    """Q* stocastico: Σ p·[r+γV*] (0.0 su done)."""
    node = mdp["nodes"][state]
    aid = next((int(k) for k, v in mdp["action_names"].items() if v == act), None)
    if aid is None:
        return float(node["v_value"])
    lst = node["transitions"].get(aid) or node["transitions"].get(str(aid)) or []
    gamma = float(mdp.get("gamma", 0.99))
    return sum(
        float(t["prob"])
        * (
            float(t["reward"])
            + (
                0.0
                if t["done"]
                else gamma * float(mdp["nodes"][int(t["next_id"])]["v_value"])
            )
        )
        for t in lst
    )


def call_with_retry(client, prompt, retries=RETRY_LIMIT, delay=RETRY_DELAY):
    for attempt in range(retries):
        try:
            resp = client.ask(prompt=prompt)
            if resp and resp.strip():
                print(f"  risposta grezza preview: {resp[:400].replace(chr(10), ' ')}...")
                return resp
            print(f"  risposta vuota (tentativo {attempt+1}/{retries})")
        except Exception as e:
            print(f"  errore richiesta (tentativo {attempt+1}/{retries}): {e}")
        if attempt < retries - 1:
            time.sleep(delay)
    return None


def parse_response(response: str, batch_codes: list[str]):
    """Atteso: [{"code": str, "q-function-values": {"left": {"analysis": str, "value": float}, ...}}].
    Tollerante a varianti e fence ```json. Ritorna (code, action, v_llm, analysis)."""
    try:
        results = json.loads(response)
    except json.JSONDecodeError:
        m = re.search(r"```json\s*(\[\s*\{.*?\}\s*\])\s*```", response, re.DOTALL)
        if m:
            try:
                results = json.loads(m.group(1))
            except json.JSONDecodeError:
                results = None  # type: ignore
        if m is None or results is None:
            m = re.search(r"\[\s*\{.*\}\s*\]", response, re.DOTALL)
            if not m:
                return []
            try:
                results = json.loads(m.group())
            except json.JSONDecodeError:
                return []
    if isinstance(results, dict):
        results = [results]
    if not isinstance(results, list):
        return []
    out = []
    for item in results:
        code = str(item.get("code", "")).strip()
        if not code:
            continue
        if code not in batch_codes:
            code2 = code.removeprefix("history_")
            if code2 in batch_codes:
                code = code2
            else:
                continue
        qvals = item.get("q-function-values") or item.get("q_function_values") or {}
        if not isinstance(qvals, dict):
            continue
        for act in ACTION_ORDER:
            obj = qvals.get(act)
            if obj is None:
                continue
            v, analysis = None, ""
            if isinstance(obj, dict):
                for key in ["value", "v", "q-value", "q_value", "v-function-value"]:
                    if key in obj:
                        try:
                            v = float(obj[key])
                            break
                        except (TypeError, ValueError):
                            pass
                for ak in ["analysis", "analisys", "anlaysis"]:
                    if ak in obj:
                        analysis = str(obj[ak])
                        break
                if v is None:
                    try:
                        v = float(list(obj.values())[0])
                    except (TypeError, ValueError, IndexError):
                        pass
            elif isinstance(obj, (int, float)):
                v = float(obj)
            elif isinstance(obj, str):
                try:
                    v = float(obj)
                except ValueError:
                    pass
            if v is None:
                continue
            out.append((code, act, v, analysis))
    return out


_file_lock = threading.Lock()


def save_results(
    path_json: Path,
    rows: list[ResultRow],
    seed: int | None = None,
    bucket_filter: str | None = None,
):
    seeds = sorted({int(r.seed) for r in rows} if seed is None else {int(seed)})
    with _file_lock:
        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "seeds": seeds,
                    "bucket": bucket_filter or "all",
                    "rows": [asdict(r) for r in rows],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )


def load_existing_rows(path_json: Path):
    rows: list[ResultRow] = []
    try:
        with open(path_json, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  warn: impossibile leggere JSON esistente: {e}")
        return rows
    items = data.get("rows", []) if isinstance(data, dict) and "rows" in data else data
    if isinstance(items, dict):
        items = [items]
    for r in items:
        try:
            rows.append(
                ResultRow(
                    id=r["id"],
                    seed=int(r.get("seed", DEFAULT_SEED)),
                    map_name=r["map_name"],
                    bucket=r["bucket"],
                    state=int(r["state"]),
                    r=int(r["r"]),
                    c=int(r["c"]),
                    ch=r["ch"],
                    action=r["action"],
                    v_true=float(r["v_true"]),
                    v_llm=float(r["v_llm"]),
                    analysis=r["analysis"],
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return rows


def run(
    map_name: str | None = DEFAULT_MAP,
    seed: int | None = DEFAULT_SEED,
    lang: str = "it",
    limit: int | None = None,
    limit_per_bucket: int | None = None,
    dry_run: bool = False,
    force: bool = False,
    out_dir: Path | str | None = None,
    model_id: str = MODEL_ID,
    launch_interval: int = LAUNCH_INTERVAL_SEC,
    max_inflight: int = MAX_INFLIGHT,
    states_path: Path | str | None = None,
    bucket: str | None = None,
):
    data_dir = Path(__file__).parent.parent / "graph" / "data"
    if states_path is not None:
        qstates_pkl, map_name, is_slippery, seed, out_json = _resolve_states_path(
            states_path, map_name, seed, model_id
        )
        if not qstates_pkl.exists():
            print(f"ERRORE: stati non trovato {qstates_pkl}.")
            return
        qdata = load_qstates(qstates_pkl)
        mdp = load_mdp(
            seed=seed,
            map_name=map_name,
            is_slippery=is_slippery,
            out_dir=out_dir if out_dir else data_dir,
        )
    else:
        if map_name is None:
            map_name = DEFAULT_MAP
        if seed is None:
            seed = DEFAULT_SEED
        mdp = load_mdp(
            seed=seed,
            map_name=map_name,
            is_slippery=True,
            out_dir=out_dir if out_dir else data_dir,
        )
        qstates_pkl, _ = get_qstates_paths(
            map_name, True, seed, out_dir if out_dir else data_dir
        )
        if not qstates_pkl.exists():
            print(
                f"ERRORE: qstates non trovato {qstates_pkl}. "
                f"Esegui prima graph.frozenlake_qlearning_states"
            )
            return
        qdata = load_qstates(qstates_pkl)

    wanted = [bucket] if bucket else list(BUCKETS_WANTED)
    for b in wanted:
        if b not in qdata["buckets"]:
            print(
                f"ERRORE: bucket '{b}' assente. Disponibili: {sorted(qdata['buckets'])}"
            )
            return
    selected: list[tuple[str, int]] = []
    for b in wanted:
        ids = list(qdata["buckets"].get(b, []))
        if limit_per_bucket is not None:
            k = min(limit_per_bucket, len(ids))
            ids = random.Random(seed).sample(ids, k) if k < len(ids) else ids[:k]
        selected.extend([(b, nid) for nid in ids])
    if limit is not None and len(selected) > limit:
        random.Random(seed).shuffle(selected)
        selected = selected[:limit]

    prompt_tmpl, v_def, q_def, doc, legend, t_def = load_docs(lang)
    base_prompt = build_base_prompt(prompt_tmpl, v_def, q_def, doc, legend, t_def)

    if states_path is None:
        out_json = get_output_paths(map_name, seed, model_id, out_dir)
    existing: set[tuple[str, int, str]] = set()
    rows: list[ResultRow] = []
    if out_json.exists() and not force:
        print(f"Trovato output esistente {out_json}, carico per dedup...")
        rows = load_existing_rows(out_json)
        for r in rows:
            existing.add((r.bucket, r.state, r.action))
        print(f"  già salvate {len(rows)} righe")

    cnt = Counter((b, nid) for b, nid, _ in existing)
    to_query = [(b, nid) for b, nid in selected if cnt[(b, nid)] < len(ACTION_ORDER)]
    print(
        f"Selezionati {len(selected)} stati ({', '.join(wanted)}), "
        f"da processare {len(to_query)} dopo dedup (limit={limit})"
    )
    if not to_query:
        print("Nulla da processare (tutto già salvato).")
        return rows

    if dry_run:
        print(
            f"DRY_RUN: preparo {len(to_query)} stati, 1 stato/request "
            f"model={model_id}"
        )
        for b, nid in to_query[: min(3, len(to_query))]:
            node = mdp["nodes"][nid]
            code = f"{nid:04d}"  # ponytail: niente bucket nel prompt, ricostruito al salvataggio
            prompt = build_prompt_for_state(base_prompt, mdp, nid, code)
            qs = " | ".join(
                f"{a} q_true={q_true_for(mdp, nid, a):.4f}" for a in ACTION_ORDER
            )
            print(
                f"  {code} [{b}] V*={node['v_value']} {qs} prompt len {len(prompt)} ~{len(prompt)//4} tokens"
            )
        s0, b0 = to_query[0][1], to_query[0][0]
        fake = json.dumps(
            [
                {
                    "code": f"{s0:04d}",
                    "q-function-values": {
                        a: {"analysis": "test", "value": 0.5} for a in ACTION_ORDER
                    },
                }
            ]
        )
        parsed = parse_response(fake, [f"{s0:04d}"])
        assert len(parsed) == len(ACTION_ORDER), "parse_response sanity failed"
        print(f"  [self-check] parse_response OK ({len(parsed)} azioni)")
        return rows

    random.Random(seed).shuffle(to_query)
    batches = [[item] for item in to_query]
    total = len(batches)
    print(
        f"Avvio {total} batch (1 stato/batch), pacing 1/{launch_interval}s, "
        f"model {model_id}, lang {lang}"
    )

    if GeminiLLM is None:
        print("ERRORE: monorepo.GeminiLLM non disponibile. Usa --dry-run")
        return rows
    load_api_keys()
    client = GeminiLLM(model_id=model_id, temperature=0.7)

    def process_batch(batch: list[tuple[str, int]], batch_idx: int):
        assert len(batch) == 1
        b0, nid0 = batch[0]
        code = f"{nid0:04d}"  # ponytail: niente bucket nel prompt, ricostruito al salvataggio
        full_prompt = build_prompt_for_state(base_prompt, mdp, nid0, code)
        print(f"[Batch {batch_idx}/{total}] Lanciato {code} len {len(full_prompt)}")
        t0 = time.time()
        resp = call_with_retry(client, full_prompt)
        print(f"[Batch {batch_idx}/{total}] Risposta in {time.time()-t0:.1f}s")
        if resp is None:
            return []
        batch_rows = []
        for code2, act, v_llm, analysis in parse_response(
            resp, [code, f"{nid0:04d}_{b0}"]  # 2°: tollera vecchio "NNNN_bucket"
        ):
            try:
                nid2 = int(str(code2).split("_")[0])  # tollera vecchio "NNNN_bucket"
            except (ValueError, IndexError):
                continue
            if nid2 != nid0:
                continue
            b2 = b0
            if (b2, nid2, act) in existing:
                continue
            node = mdp["nodes"][nid2]
            row = ResultRow(
                id=f"{nid2:04d}_{b2}",
                seed=seed,
                map_name=map_name,
                bucket=b2,
                state=nid2,
                r=node["r"],
                c=node["c"],
                ch=node["ch"],
                action=act,
                v_true=q_true_for(mdp, nid2, act),
                v_llm=float(v_llm),
                analysis=analysis,
            )
            batch_rows.append(row)
            existing.add((b2, nid2, act))
        return batch_rows

    def on_done(fut):
        try:
            br = fut.result()
            if br:
                rows.extend(br)
                save_results(out_json, rows, seed=seed, bucket_filter=bucket)
                print(f"  + Salvato {len(br)} righe, totale {len(rows)}")
        except Exception as e:
            print(f"  Errore batch: {e}")

    with ThreadPoolExecutor(max_workers=max_inflight) as pool:
        futures = []
        for idx, batch in enumerate(batches, start=1):
            fut = pool.submit(process_batch, batch, idx)
            fut.add_done_callback(on_done)
            futures.append(fut)
            if idx < total:
                time.sleep(launch_interval)
        print("\nTutti i batch lanciati, attendo completamento...")
        for f in futures:
            try:
                f.result()
            except Exception as e:
                print(f"future error {e}")

    save_results(out_json, rows, seed=seed, bucket_filter=bucket)
    print(f"\nFatto. {len(rows)} righe totali in {out_json} [JSON]")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query Gemini gemma su FrozenLake slippery (Q* stocastica via V*)"
    )
    parser.add_argument("--map", type=str, default=None, choices=["4x4", "8x8"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lang", type=str, default="it", choices=["it", "en"])
    parser.add_argument(
        "--bucket",
        type=str,
        default=None,
        help="1 bucket solo (default bottleneck); disponibili in qstates",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--limit-per-bucket", type=int, default=None, dest="limit_per_bucket"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--model", type=str, default=MODEL_ID, dest="model_id")
    parser.add_argument(
        "--interval", type=int, default=LAUNCH_INTERVAL_SEC, dest="launch_interval"
    )
    parser.add_argument(
        "--max-inflight", type=int, default=MAX_INFLIGHT, dest="max_inflight"
    )
    parser.add_argument("--path", type=str, default=None, dest="states_path")
    args = parser.parse_args()
    run(
        map_name=args.map,
        seed=args.seed,
        lang=args.lang,
        limit=args.limit,
        limit_per_bucket=args.limit_per_bucket,
        dry_run=args.dry_run,
        force=args.force,
        model_id=args.model_id,
        launch_interval=args.launch_interval,
        max_inflight=args.max_inflight,
        states_path=args.states_path,
        bucket=args.bucket,
    )
