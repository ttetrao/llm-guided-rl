#!/usr/bin/env python3
"""
Interroga Gemini AI Studio (gemma-4-26b-a4b-it) usando i file generati in graph
e il prompt in docs/doorkey/{it,en}/prompt.txt con placeholder sostituiti.

- Sorgente stati: solo bucket 'iniziale' e 'avanzato' (0-0.33 e 0.8-1.0) da
  qlearning_states_8x8_seed1337.json (max 100 per bucket, campionati random)
  + grafo mdp_8x8_seed1337.json per mappe s e s'=T(s,a) esplicite
- Prompt: Q via V -> Q(s,a)=V*(s') con s' già in <left>..<toggle> (done ignorata, 6 s' per stato, drop=self-loop)
- Batch: 1 stato per request per limiti dimensione, rate come bak/llm/gconnection.py (pacing 65s, retry, ThreadPool)
- Output: file unico JSON con entrambi i bucket, una entry per (node,action) per leggibilità
- Arg --limit per scegliere numero più contenuto di stati per seed (es. 10)
- Controllo dedup: non riprocessa node/action già salvati su JSON

Uso:
    python -m llm.query_gemma --seed 1337 --size 8 --limit 10 --dry-run
    python -m llm.query_gemma --seed 1337 --limit 20
    python -m llm.query_gemma --seed 1337 --limit-per-bucket 5
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
import threading
from collections import defaultdict
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

from graph.mdp_graph import load_or_build
from graph.qlearning_states import get_qstates_paths, load_qstates
from paths import LLM_DIR

# ---------------------------------------------------------------------------
# Config come gconnection.py — 1 stato per request per limiti dimensione
# ---------------------------------------------------------------------------
MODEL_ID = "gemma-4-26b-a4b-it"
HISTORY_PER_REQUEST = 1  # uno stato per request (6 s' + current) per rispettare limite dimensione
RETRY_LIMIT = 3
RETRY_DELAY = 10
LAUNCH_INTERVAL_SEC = 65
MAX_INFLIGHT = 5
DEFAULT_SEED = 1337
DEFAULT_SIZE = 8

BUCKETS_WANTED = ["bottleneck", "ckpt-0.3", "ckpt-0.5", "ckpt-0.7", "ckpt-0.8", "ckpt-1"]  # doorkey_states (bottleneck+ckpt pilota)

@dataclass(frozen=True)
class ResultRow:
    id: str  # code = f"{seed}_{node_id:04d}_{bucket}_{action}"
    seed: int
    size: int
    bucket: str
    node_id: int
    x: int
    y: int
    dir: int
    has_key: bool
    door_open: bool
    stage: str
    action: str
    v_true: float
    v_llm: float
    analysis: str

def get_output_paths(seed: int, size: int, model_id: str = MODEL_ID, out_dir: Path | str | None = None):
    if out_dir is None:
        out_dir = LLM_DIR
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"llm_results_{size}x{size}_seed{seed}_{model_id.replace(':','_')}"
    # ponytail: JSON unico per leggibilità (CSV rimosso)
    return out_dir / f"{base}.json"

_STATES_RE = re.compile(r"(?:qlearning_states|doorkey_states)_(\d+)x(\d+)_seed(\d+)$")
_STATES_MULTI_RE = re.compile(r"(?:qlearning_states|doorkey_states)_(\d+)x(\d+)_seeds([\d\-]+)$")

def _resolve_states_path(states_path: Path | str, seed: int | None, size: int | None, model_id: str = MODEL_ID):
    """--path: file .json stati (dir+nome). seed/size derivati dal nome salvo espliciti.
    Supporta anche file multi-seed ..._seedsA-B-C.json (seed dal contenuto).
    Ritorna (qstates_json, seed, size, out_json accanto all'input)."""
    p = Path(states_path)
    m = _STATES_RE.match(p.stem)
    mm = None if m else _STATES_MULTI_RE.match(p.stem)
    if m:
        if size is None:
            size = int(m.group(1))
        if seed is None:
            seed = int(m.group(3))
    elif mm:
        if size is None:
            size = int(mm.group(1))
        # seed singolo dal nome non disponibile: resta None -> derivato dal contenuto
    if seed is None and mm is None:
        seed = DEFAULT_SEED
    if size is None:
        size = DEFAULT_SIZE
    if m is None and mm is None:
        print(f"  warn: nome {p.name} non riconosciuto, uso seed={seed} size={size} per MDP/output")
    out_json = p.parent / f"llm_results_{p.stem}_{model_id.replace(':','_')}.json"
    return p, seed, size, out_json


def _states_seed_buckets(qdata: dict, default_seed: int | None):
    """Normalizza stati singoli o multi -> (seeds, {seed: buckets})."""
    if "per_seed" in qdata:
        per = {int(k): v for k, v in qdata["per_seed"].items()}
        seeds = sorted(per)
        return seeds, {s: per[s]["buckets"] for s in seeds}
    seeds = [int(default_seed) if default_seed is not None else int(qdata.get("seed", DEFAULT_SEED))]
    return seeds, {seeds[0]: qdata["buckets"]}

def load_docs(lang: str = "it"):
    # ponytail: docs riorganizzati per ambiente e lingua → docs/doorkey/{it,en}
    base = Path(__file__).parent.parent / "docs" / "doorkey" / lang
    if not base.exists():
        # fallback legacy docs/{lang} per compatibilità temporanea
        base = Path(__file__).parent.parent / "docs" / lang
        if not base.exists():
            base = Path(__file__).parent.parent / "docs"
    def read(p):
        try:
            return open(base / p, encoding="utf-8").read()
        except FileNotFoundError:
            # fallback a docs/doorkey/it (canonico) poi docs root
            for fb in [Path(__file__).parent.parent / "docs" / "doorkey" / "it",
                       Path(__file__).parent.parent / "docs" / p]:
                try:
                    return open(fb / p if fb.is_dir() else fb, encoding="utf-8").read()
                except FileNotFoundError:
                    continue
            raise
    prompt_tmpl = read("prompt.txt")
    v_def = read("v-function_definition.md")
    q_def = read("q-function_definition.md")
    doc = read("MiniGridDocumentation.md")
    legend = read("legend.txt")
    return prompt_tmpl, v_def, q_def, doc, legend

def build_base_prompt(prompt_tmpl: str, v_def: str, q_def: str, doc: str, legend: str) -> str:
    # sostituisce i 4 placeholder vuoti in prompt.txt
    # prompt.txt ha <v-definition></v-definition> ecc. alla fine
    out = prompt_tmpl
    out = out.replace("<v-definition></v-definition>", f"<v-definition>\n{v_def}\n</v-definition>")
    out = out.replace("<q-definition></q-definition>", f"<q-definition>\n{q_def}\n</q-definition>")
    out = out.replace("<documentation></documentation>", f"<documentation>\n{doc}\n</documentation>")
    out = out.replace("<legend></legend>", f"<legend>\n{legend}\n</legend>")
    return out

def build_prompt_for_state(base_prompt: str, mdp: dict, node_id: int, code: str) -> str:
    node = mdp["nodes"][node_id]
    cur_map = node["map"]
    # base prompt + current_state
    prompt = base_prompt.replace('<current_state id="xxxx"></current_state>', f'<current_state id="{code}">\n{cur_map}\n</current_state>')
    # per ogni azione, map di s' esplicito — done esclusa, drop incluso (self-loop)
    for act_name in ["left", "right", "forward", "pickup", "drop", "toggle"]:
        # trova action id da nome
        act_id = None
        for aid, aname in mdp["action_names"].items():
            if aname == act_name:
                act_id = aid
                break
        if act_id is None:
            continue
        trans = node["transitions"].get(act_id)
        if trans is None:
            # fallback: cerca per nome
            for t in node["transitions"].values():
                if t["action_name"] == act_name:
                    trans = t
                    break
        if trans is None:
            next_map = "(no trans)"
        else:
            nid = trans["next_id"]
            next_map = mdp["nodes"][nid]["map"]
        # sostituisce <left></left> ecc. con mappa
        # prompt ha <left></left> vuoti
        prompt = prompt.replace(f"<{act_name}></{act_name}>", f"<{act_name}>\n{next_map}\n</{act_name}>")
    return prompt

def call_with_retry(client, prompt, retries=RETRY_LIMIT, delay=RETRY_DELAY):
    for attempt in range(retries):
        try:
            resp = client.ask(prompt=prompt)
            if resp and resp.strip():
                return resp
            print(f"  risposta vuota (tentativo {attempt+1}/{retries})")
        except Exception as e:
            print(f"  errore richiesta (tentativo {attempt+1}/{retries}): {e}")
        if attempt < retries - 1:
            time.sleep(delay)
    return None

def parse_response(response: str, batch_codes: list[str], batch_nodes: list[dict], mdp: dict):
    """
    Atteso: [{"code": str, "q-function-values": {"left": {"anlaysis": str, "value": float}, ...}}, ...]
    Tollerante a varianti: analisys/anlaysis/analysis, value/v-function-value, flat decimal.
    Ritorna lista di (code, action, v_llm, analysis)
    """
    try:
        results = json.loads(response)
    except json.JSONDecodeError:
        m = re.search(r"\[\s*\{.*\}\s*\]", response, re.DOTALL)
        if not m:
            return []
        try:
            results = json.loads(m.group())
        except json.JSONDecodeError:
            return []
    if not isinstance(results, list):
        return []
    out = []
    for item in results:
        code = str(item.get("code", "")).strip()
        if not code:
            continue
        # trova batch entry corrispondente
        # batch_codes contiene code attesi
        if code not in batch_codes:
            # prova con prefisso history_
            code2 = code.removeprefix("history_")
            if code2 in batch_codes:
                code = code2
            else:
                continue
        qvals = item.get("q-function-values") or item.get("q_function_values") or {}
        if not isinstance(qvals, dict):
            continue
        for act in ["left","right","forward","pickup","drop","toggle"]:
            obj = qvals.get(act)
            if obj is None:
                continue
            v = None
            analysis = ""
            if isinstance(obj, dict):
                # nested {anlaysis/analysis, value} o {analisys,value}
                for key in ["value", "v", "q-value", "q_value", "v-function-value"]:
                    if key in obj:
                        try:
                            v = float(obj[key])
                            break
                        except: pass
                # cerca analysis con typo tolleranza
                for ak in ["analysis","analisys","anlaysis"]:
                    if ak in obj:
                        analysis = str(obj[ak])
                        break
                if v is None:
                    # prova se obj è direttamente { "value": 0.8 } ma con typo
                    try:
                        # se contiene solo value
                        v = float(list(obj.values())[0])
                    except: pass
            elif isinstance(obj, (int,float)):
                v = float(obj)
            elif isinstance(obj, str):
                try:
                    v = float(obj)
                except: pass
            if v is None:
                continue
            out.append((code, act, v, analysis))
    return out

_file_lock = threading.Lock()

def save_results(path_json: Path, rows: list[ResultRow], seeds: list[int] | None = None,
                 bucket_filter: str | None = None):
    # ponytail: header seeds in cima; loader accetta anche lista legacy
    seeds = sorted({int(r.seed) for r in rows} if seeds is None else {int(s) for s in seeds})
    with _file_lock:
        with open(path_json, "w", encoding="utf-8") as f:
            json.dump({"seeds": seeds, "bucket": bucket_filter or "all",
                       "rows": [asdict(r) for r in rows]}, f, indent=2, ensure_ascii=False)


def load_existing_rows(path_json: Path):
    """Legge output esistente sia nuovo (dict seeds/rows) che legacy (lista)."""
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
            rows.append(ResultRow(
                id=r["id"], seed=int(r["seed"]), size=int(r["size"]), bucket=r["bucket"],
                node_id=int(r["node_id"]), x=int(r["x"]), y=int(r["y"]), dir=int(r["dir"]),
                has_key=bool(r["has_key"]), door_open=bool(r["door_open"]),
                stage=r["stage"], action=r["action"], v_true=float(r["v_true"]), v_llm=float(r["v_llm"]), analysis=r["analysis"]
            ))
        except Exception:
            continue
    return rows

def run(seed: int | None = DEFAULT_SEED, size: int | None = DEFAULT_SIZE, lang: str = "it",
        limit: int | None = None, limit_per_bucket: int | None = None,
        dry_run: bool = False, force: bool = False, out_dir: Path | str | None = None,
        states_path: Path | str | None = None, bucket: str | None = None):
    # 1. carica grafo e qstates (--path: file diretto, seed/size dal nome salvo espliciti)
    mdps: dict[int, dict] = {}
    if states_path is not None:
        qstates_json, seed, size, out_json = _resolve_states_path(states_path, seed, size)
        if not qstates_json.exists():
            print(f"ERRORE: stati non trovato {qstates_json}.")
            return
        qdata = load_qstates(qstates_json)
        seeds, seed_buckets = _states_seed_buckets(qdata, seed)
        for s in seeds:
            mdps[s] = load_or_build(seed=s, size=size)
        seed = seeds[0] if seed is None else seed
    else:
        if seed is None:
            seed = DEFAULT_SEED
        if size is None:
            size = DEFAULT_SIZE
        mdp = load_or_build(seed=seed, size=size)
        mdps[seed] = mdp
        qstates_json = get_qstates_paths(seed, size)
        if not qstates_json.exists():
            print(f"ERRORE: qstates non trovato {qstates_json}. Esegui prima graph.qlearning_states")
            return
        qdata = load_qstates(qstates_json)
        seeds, seed_buckets = _states_seed_buckets(qdata, seed)
    # 2. seleziona stati dai bucket richiesti (--bucket: 1 solo, default tutti)
    wanted = [bucket] if bucket else list(BUCKETS_WANTED)
    for b in wanted:
        if not any(b in bk for bk in seed_buckets.values()):
            print(f"ERRORE: bucket '{b}' assente. Disponibili: {sorted({k for bk in seed_buckets.values() for k in bk})}")
            return
    selected: list[tuple[int, str, int]] = []  # (seed, bucket, node_id)
    for s in seeds:
        bk = seed_buckets[s]
        for b in wanted:
            ids = list(bk.get(b, []))
            if limit_per_bucket is not None:  # per bucket per seed (bilanciato tra seed)
                k = min(limit_per_bucket, len(ids))
                ids = random.Random(s).sample(ids, k) if k < len(ids) else ids[:k]
            selected.extend([(s, b, nid) for nid in ids])
    # se --limit totale, riduci ulteriormente bilanciato
    if limit is not None:
        # limit è totale per seed (es. 10 -> 5+5)
        # se già limit_per_bucket, limit prevale
        if len(selected) > limit:
            # bilancia tra bucket
            # prendi limit//2 per ciascuno, resto ad avanzato
            random.Random(seeds[0] if seeds else DEFAULT_SEED).shuffle(selected)
            selected = selected[:limit]
            # opzionale: ri-bilancia per bucket uniformità
            # per ora shuffle totale va bene
    # 3. carica prompt base
    prompt_tmpl, v_def, q_def, doc, legend = load_docs(lang)
    base_prompt = build_base_prompt(prompt_tmpl, v_def, q_def, doc, legend)

    # 4. prepara output paths e dedup (JSON unico; con --path: già risolto accanto all'input)
    if states_path is None:
        out_json = get_output_paths(seed, size, MODEL_ID, out_dir)
    existing: set[tuple[int, str, int, str]] = set()  # (seed, bucket, node_id, action)
    rows: list[ResultRow] = []
    if out_json.exists() and not force:
        print(f"Trovato output esistente {out_json}, carico per dedup...")
        rows = load_existing_rows(out_json)
        for r in rows:
            existing.add((r.seed, r.bucket, r.node_id, r.action))
        print(f"  già salvate {len(rows)} righe, {len(existing)} (seed,bucket,node,action)")

    # filtra selected già completamente salvati (tutte 6 azioni, done esclusa)
    from collections import Counter
    cnt = Counter((s, b, nid) for s, b, nid, _ in existing)
    to_query: list[tuple[int, str, int]] = []
    for s, b, nid in selected:
        if cnt[(s, b, nid)] >= 6:
            continue
        to_query.append((s, b, nid))
    print(f"Selezionati {len(selected)} stati ({', '.join(wanted)}), seeds={seeds}, da processare {len(to_query)} dopo dedup (limit={limit})")

    if not to_query:
        print("Nulla da processare (tutto già salvato).")
        return rows

    if dry_run:
        print(f"DRY_RUN: preparo {len(to_query)} stati, 1 stato per request (HISTORY_PER_REQUEST=1)")
        for s, b, nid in to_query[: min(3, len(to_query))]:
            mdp = mdps[s]
            node = mdp["nodes"][nid]
            code = f"{s}_{nid:04d}_{b}"
            prompt = build_prompt_for_state(base_prompt, mdp, nid, code)
            print(f"  {code} prompt len {len(prompt)} stage {node['stage']} v_true sample {node['v_value']} -> ~{len(prompt)//4} tokens")
        return rows

    # 5. prepara batch — 1 stato per request per limiti dimensione richiesta/risposta
    random.Random(seeds[0] if seeds else DEFAULT_SEED).shuffle(to_query)
    batches = [[item] for item in to_query]  # HISTORY_PER_REQUEST=1
    total = len(batches)
    print(f"Avvio {total} batch (1 stato/batch), pacing 1/{LAUNCH_INTERVAL_SEC}s, model {MODEL_ID}, lang {lang}")

    # client
    if GeminiLLM is None:
        print("ERRORE: monorepo.GeminiLLM non disponibile, impossibile chiamare API. Usa --dry-run")
        return rows
    load_api_keys()
    client = GeminiLLM(model_id=MODEL_ID, temperature=0.7)

    def process_batch(batch: list[tuple[int, str, int]], batch_idx: int):
        # 1 stato per request: costruisce prompt per singolo stato (current + 6 s' espliciti: left/right/forward/pickup/drop/toggle)
        assert len(batch) == 1, "HISTORY_PER_REQUEST deve essere 1"
        s0, b0, nid0 = batch[0]
        mdp0 = mdps[s0]
        code = f"{s0}_{nid0:04d}_{b0}"
        codes = [code]
        full_prompt = build_prompt_for_state(base_prompt, mdp0, nid0, code)
        print(f"[Batch {batch_idx}/{total}] Lanciato {', '.join(codes)} len {len(full_prompt)}")
        t0 = time.time()
        resp = call_with_retry(client, full_prompt)
        elapsed = time.time()-t0
        print(f"[Batch {batch_idx}/{total}] Risposta in {elapsed:.1f}s")
        if resp is None:
            return []
        # parse
        # per trovare v_true, serve mdp
        parsed = parse_response(resp, codes, batch, mdp0)
        # converti a ResultRow
        batch_rows = []
        for code2, act, v_llm, analysis in parsed:
            # trova seed/bucket e nid da code
            # code format seed_nid_bucket
            try:
                s_str, nid_str, b2 = code2.rsplit("_", 2)
                s2, nid2 = int(s_str), int(nid_str)
            except:
                continue
            # dedup per action
            if (s2, b2, nid2, act) in existing:
                continue
            mdp2 = mdps.get(s2, mdp0)
            node = mdp2["nodes"][nid2]
            # v_true = q_*(s,a) = r + gamma*V*(s') per quell'azione (MDP deterministico)
            # trova trans
            act_id = None
            for aid, aname in mdp2["action_names"].items():
                if aname==act:
                    act_id=aid
                    break
            if act_id is not None and act_id in node["transitions"]:
                trans = node["transitions"][act_id]
                nid_next = trans["next_id"]
                r = float(trans.get("reward", 0.0))
                gamma = float(mdp2.get("gamma", 0.99))
                v_true = r + gamma * float(mdp2["nodes"][nid_next]["v_value"])
            else:
                v_true = float(node["v_value"])
            row = ResultRow(
                id=code2, seed=s2, size=size, bucket=b2, node_id=nid2,
                x=node["x"], y=node["y"], dir=node["dir"], has_key=node["has_key"],
                door_open=node["door_open"], stage=node["stage"], action=act,
                v_true=v_true, v_llm=v_llm, analysis=analysis
            )
            batch_rows.append(row)
            existing.add((s2, b2, nid2, act))
        return batch_rows

    # ThreadPool come gconnection
    def on_done(fut):
        try:
            br = fut.result()
            if br:
                rows.extend(br)
                save_results(out_json, rows, seeds=seeds, bucket_filter=bucket)
                print(f"  + Salvato {len(br)} righe, totale {len(rows)}")
        except Exception as e:
            print(f"  Errore batch: {e}")

    with ThreadPoolExecutor(max_workers=MAX_INFLIGHT) as pool:
        futures=[]
        for idx, batch in enumerate(batches, start=1):
            fut = pool.submit(process_batch, batch, idx)
            fut.add_done_callback(on_done)
            futures.append(fut)
            if idx < total:
                time.sleep(LAUNCH_INTERVAL_SEC)
        print("\nTutti i batch lanciati, attendo completamento...")
        for f in futures:
            try:
                f.result()
            except Exception as e:
                print(f"future error {e}")

    # salva finale (JSON unico)
    save_results(out_json, rows, seeds=seeds, bucket_filter=bucket)
    print(f"\nFatto. {len(rows)} righe totali in {out_json} [JSON]")
    return rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query Gemini AI Studio su grafo+ qstates (Q via V)")
    parser.add_argument("--seed", type=int, default=None, help="default 1337 o derivato da --path")
    parser.add_argument("--size", type=int, default=None, choices=[6,8,16], help="default 8 o derivato da --path")
    parser.add_argument("--lang", type=str, default="it", choices=["it","en"])
    parser.add_argument("--bucket", type=str, default=None, help="1 bucket solo (es. bottleneck); default tutti")
    parser.add_argument("--limit", type=int, default=None, help="numero totale stati per seed (es. 10) bilanciato tra iniziale/avanzato")
    parser.add_argument("--limit-per-bucket", type=int, default=None, dest="limit_per_bucket", help="numero per bucket per seed (es. 5)")
    parser.add_argument("--dry-run", action="store_true", help="non chiama API, solo verifica prompt")
    parser.add_argument("--force", action="store_true", help="ignora dedup e riprocessa tutto")
    parser.add_argument("--path", type=str, default=None, dest="states_path", help="file .json stati (es. src/output/cache/doorkey_states_8x8_seed1337.json o ..._seeds1337-42.json); seed/size dal nome, output accanto")
    args = parser.parse_args()
    # se --limit dato, ignora limit_per_bucket? No, limit totale prevale ma per semplicità li combiniamo già sopra
    run(seed=args.seed, size=args.size, lang=args.lang, limit=args.limit, limit_per_bucket=args.limit_per_bucket, dry_run=args.dry_run, force=args.force, states_path=args.states_path, bucket=args.bucket)

