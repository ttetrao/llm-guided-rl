#!/usr/bin/env python3
"""
Interroga gpt-oss:120b via Ollama (cloud) usando i file generati in graph
e il prompt in docs/doorkey/{it,en}/prompt.txt con placeholder sostituiti.

Clone di query_gemma.py con solo client sostituito -> Ollama come in bak/llm/ollama_cloud_connection*.py

- Sorgente stati: solo bucket 'iniziale' e 'avanzato' (0-0.33 e 0.8-1.0) da
  qlearning_states_8x8_seed1337.json (max 100 per bucket, campionati random)
  + grafo mdp_8x8_seed1337.json per mappe s e s'=T(s,a) esplicite
- Prompt: Q via V -> Q(s,a)=V*(s') con s' già in <left>..<toggle> (done ignorata, 6 s' per stato, drop=self-loop)
- Batch: 1 stato per request, retry, ThreadPool; pacing ridotto vs Gemini (Ollama cloud più veloce)
- Output: file unico JSON con entrambi i bucket, una entry per (node,action)
- Dedup: non riprocessa node/action già salvati su JSON

Uso:
    python -m llm.query_gpt --seed 1337 --size 8 --limit 10 --dry-run
    python -m llm.query_gpt --seed 1337 --limit 20
    python -m llm.query_gpt --seed 1337 --limit-per-bucket 5
    OLLAMA_API_KEY=xxx python -m llm.query_gpt --seed 1337 --limit 10
    OLLAMA_HOST=https://ollama.com python -m llm.query_gpt --limit 10  # default cloud
    OLLAMA_HOST=http://localhost:11434 python -m llm.query_gpt --model gpt-oss:20b --limit 10  # locale
"""
from __future__ import annotations

import argparse
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

# ollama disponibile solo se pip install ollama; import lazy per dry-run
try:
    from ollama import Client  # type: ignore
except ImportError:
    Client = None  # type: ignore

from graph.mdp_graph import load_or_build
from graph.qlearning_states import get_qstates_paths, load_qstates
from paths import LLM_DIR

# ---------------------------------------------------------------------------
# Config — come bak/llm/ollama_cloud_connection.py + pacing gemma
# ---------------------------------------------------------------------------
MODEL_ID = "gpt-oss:120b"  # stesso di bak/llm/ollama_cloud_connection*.py:17
HISTORY_PER_REQUEST = 1  # uno stato per request (6 s' + current)
RETRY_LIMIT = 3
RETRY_DELAY = 5  # bak usa 2s, gemma 10s -> compromesso 5s
LAUNCH_INTERVAL_SEC = 10  # gemma 65s per rate-limit; ollama cloud più veloce -> 10s default
MAX_INFLIGHT = 5
TEMPERATURE = 0.7  # bak 0.8 per V, 0.5 per q2; gemma 0.7 -> allineato a gemma per comparabilità
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
    base = f"llm_results_{size}x{size}_seed{seed}_{model_id.replace(':','_').replace('/','_')}"
    # JSON unico per leggibilità (CSV rimosso)
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
    if seed is None and mm is None:
        seed = DEFAULT_SEED
    if size is None:
        size = DEFAULT_SIZE
    if m is None and mm is None:
        print(f"  warn: nome {p.name} non riconosciuto, uso seed={seed} size={size} per MDP/output")
    out_json = p.parent / f"llm_results_{p.stem}_{model_id.replace(':','_').replace('/','_')}.json"
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
    # docs riorganizzati per ambiente e lingua → docs/doorkey/{it,en}
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
    out = prompt_tmpl
    out = out.replace("<v-definition></v-definition>", f"<v-definition>\n{v_def}\n</v-definition>")
    out = out.replace("<q-definition></q-definition>", f"<q-definition>\n{q_def}\n</q-definition>")
    out = out.replace("<documentation></documentation>", f"<documentation>\n{doc}\n</documentation>")
    out = out.replace("<legend></legend>", f"<legend>\n{legend}\n</legend>")
    return out

def build_prompt_for_state(base_prompt: str, mdp: dict, node_id: int, code: str) -> str:
    node = mdp["nodes"][node_id]
    cur_map = node["map"]
    prompt = base_prompt.replace('<current_state id="xxxx"></current_state>', f'<current_state id="{code}">\n{cur_map}\n</current_state>')
    for act_name in ["left", "right", "forward", "pickup", "drop", "toggle"]:
        act_id = None
        for aid, aname in mdp["action_names"].items():
            if aname == act_name:
                act_id = aid
                break
        if act_id is None:
            continue
        trans = node["transitions"].get(act_id)
        if trans is None:
            for t in node["transitions"].values():
                if t["action_name"] == act_name:
                    trans = t
                    break
        if trans is None:
            next_map = "(no trans)"
        else:
            nid = trans["next_id"]
            next_map = mdp["nodes"][nid]["map"]
        prompt = prompt.replace(f"<{act_name}></{act_name}>", f"<{act_name}>\n{next_map}\n</{act_name}>")
    return prompt

def get_ollama_client(host: str | None = None, api_key: str | None = None):
    """Crea Client Ollama come in bak/llm/ollama_cloud_connection.py:131"""
    if Client is None:
        return None
    host = host or os.environ.get("OLLAMA_HOST", "https://ollama.com")
    api_key = api_key if api_key is not None else os.environ.get("OLLAMA_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    # Client accetta headers anche per locale (ignorati)
    return Client(host=host, headers=headers)

def _extract_content(resp) -> str:
    # gestisce sia dict che ChatResponse/Message object (ollama 0.5+)
    try:
        if isinstance(resp, dict):
            m = resp.get("message", {})
            if isinstance(m, dict):
                return m.get("content", "") or ""
            return getattr(m, "content", "") or ""
        # object ChatResponse
        m = getattr(resp, "message", None)
        if m is None:
            try:
                m = resp["message"]  # __getitem__ fallback
            except Exception:
                return ""
        if isinstance(m, dict):
            return m.get("content", "") or ""
        return getattr(m, "content", "") or ""
    except Exception:
        return ""

def call_with_retry(client, prompt, model_id: str = MODEL_ID, temperature: float = TEMPERATURE, retries=RETRY_LIMIT, delay=RETRY_DELAY):
    for attempt in range(retries):
        try:
            resp = client.chat(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                options={"temperature": temperature},
            )
            content = _extract_content(resp)
            if not content:
                # fallback str(resp) contiene Message repr — prova estrazione grezza
                try:
                    content = str(resp)
                    # se str contiene content='[...]’, estrai tra content=' e thinking
                    if "content='" in content and "q-function-values" in content:
                        # non affidabile, ma tenta regex su str repr
                        pass
                except: pass
            # debug come bak: anteprima
            if content:
                preview = content[:400].replace("\n"," ")[:400]
                print(f"  risposta grezza preview: {preview}...")
            if content and content.strip():
                return content
            print(f"  risposta vuota (tentativo {attempt+1}/{retries})")
        except Exception as e:
            print(f"  errore richiesta Ollama (tentativo {attempt+1}/{retries}): {e}")
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
        # prova blocco ```json ... ``` prima di regex greedy
        m = re.search(r"```json\s*(\[\s*\{.*?\}\s*\])\s*```", response, re.DOTALL)
        if m:
            try:
                results = json.loads(m.group(1))
            except json.JSONDecodeError:
                results = None  # type: ignore
            if results is not None:
                pass
            else:
                m = None  # fallback sotto
        if m is None or 'results' not in locals() or results is None:
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
        for act in ["left","right","forward","pickup","drop","toggle"]:
            obj = qvals.get(act)
            if obj is None:
                continue
            v = None
            analysis = ""
            if isinstance(obj, dict):
                for key in ["value", "v", "q-value", "q_value", "v-function-value"]:
                    if key in obj:
                        try:
                            v = float(obj[key])
                            break
                        except: pass
                for ak in ["analysis","analisys","anlaysis"]:
                    if ak in obj:
                        analysis = str(obj[ak])
                        break
                if v is None:
                    try:
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
        model_id: str = MODEL_ID, ollama_host: str | None = None, temperature: float = TEMPERATURE,
        launch_interval: int = LAUNCH_INTERVAL_SEC, max_inflight: int = MAX_INFLIGHT,
        states_path: Path | str | None = None, bucket: str | None = None):
    # 1. carica grafo e qstates (--path: file diretto, seed/size dal nome salvo espliciti)
    mdps: dict[int, dict] = {}
    if states_path is not None:
        qstates_json, seed, size, out_json = _resolve_states_path(states_path, seed, size, model_id)
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
        mdps[seed] = load_or_build(seed=seed, size=size)
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
    if limit is not None:
        if len(selected) > limit:
            random.Random(seeds[0] if seeds else DEFAULT_SEED).shuffle(selected)
            selected = selected[:limit]
    # 3. carica prompt base
    prompt_tmpl, v_def, q_def, doc, legend = load_docs(lang)
    base_prompt = build_base_prompt(prompt_tmpl, v_def, q_def, doc, legend)

    # 4. prepara output paths e dedup (JSON unico; con --path: già risolto accanto all'input)
    if states_path is None:
        out_json = get_output_paths(seed, size, model_id, out_dir)
    existing: set[tuple[int, str, int, str]] = set()  # (seed, bucket, node_id, action)
    rows: list[ResultRow] = []
    if out_json.exists() and not force:
        print(f"Trovato output esistente {out_json}, carico per dedup...")
        rows = load_existing_rows(out_json)
        for r in rows:
            existing.add((r.seed, r.bucket, r.node_id, r.action))
        print(f"  già salvate {len(rows)} righe, {len(existing)} (seed,bucket,node,action)")

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
        print(f"DRY_RUN: preparo {len(to_query)} stati, 1 stato per request (HISTORY_PER_REQUEST=1) model={model_id} host={ollama_host or os.environ.get('OLLAMA_HOST','https://ollama.com')}")
        for s, b, nid in to_query[: min(3, len(to_query))]:
            mdp = mdps[s]
            node = mdp["nodes"][nid]
            code = f"{s}_{nid:04d}_{b}"
            prompt = build_prompt_for_state(base_prompt, mdp, nid, code)
            print(f"  {code} prompt len {len(prompt)} stage {node['stage']} v_true sample {node['v_value']} -> ~{len(prompt)//4} tokens")
        # sanity check parsing su fake response
        s0, b0, nid0 = to_query[0]
        fake = json.dumps([{"code": f"{s0}_{nid0:04d}_{b0}", "q-function-values": {a: {"anlaysis": "test", "value": 0.5} for a in ["left","right","forward","pickup","drop","toggle"]}}])
        parsed = parse_response(fake, [f"{s0}_{nid0:04d}_{b0}"], [], mdps[s0])
        assert len(parsed) == 6, "parse_response sanity failed"
        print(f"  [self-check] parse_response OK ({len(parsed)} azioni)")
        return rows

    # 5. prepara batch — 1 stato per request
    random.Random(seeds[0] if seeds else DEFAULT_SEED).shuffle(to_query)
    batches = [[item] for item in to_query]  # HISTORY_PER_REQUEST=1
    total = len(batches)
    print(f"Avvio {total} batch (1 stato/batch), pacing 1/{launch_interval}s, model {model_id}, lang {lang}, temp {temperature}")

    if Client is None:
        print("ERRORE: pip install ollama mancante. `pip install ollama` poi riprova. Usa --dry-run per test senza API.")
        return rows
    client = get_ollama_client(host=ollama_host)
    if client is None:
        print("ERRORE: impossibile creare Ollama Client. Verifica OLLAMA_HOST / OLLAMA_API_KEY. Usa --dry-run")
        return rows
    # check key per cloud
    if (ollama_host or os.environ.get("OLLAMA_HOST","https://ollama.com")).startswith("https://") and not os.environ.get("OLLAMA_API_KEY"):
        print("WARN: OLLAMA_API_KEY non impostata ma host è https://ollama.com — la chiamata fallirà senza key. Esporta OLLAMA_API_KEY.")

    def process_batch(batch: list[tuple[int, str, int]], batch_idx: int):
        assert len(batch) == 1, "HISTORY_PER_REQUEST deve essere 1"
        s0, b0, nid0 = batch[0]
        mdp0 = mdps[s0]
        code = f"{s0}_{nid0:04d}_{b0}"
        codes = [code]
        full_prompt = build_prompt_for_state(base_prompt, mdp0, nid0, code)
        print(f"[Batch {batch_idx}/{total}] Lanciato {', '.join(codes)} len {len(full_prompt)}")
        t0 = time.time()
        resp = call_with_retry(client, full_prompt, model_id=model_id, temperature=temperature)
        elapsed = time.time()-t0
        print(f"[Batch {batch_idx}/{total}] Risposta in {elapsed:.1f}s")
        if resp is None:
            return []
        parsed = parse_response(resp, codes, batch, mdp0)
        batch_rows = []
        for code2, act, v_llm, analysis in parsed:
            try:
                s_str, nid_str, b2 = code2.rsplit("_", 2)
                s2, nid2 = int(s_str), int(nid_str)
            except:
                continue
            if (s2, b2, nid2, act) in existing:
                continue
            mdp2 = mdps.get(s2, mdp0)
            node = mdp2["nodes"][nid2]
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

    def on_done(fut):
        try:
            br = fut.result()
            if br:
                rows.extend(br)
                save_results(out_json, rows, seeds=seeds, bucket_filter=bucket)
                print(f"  + Salvato {len(br)} righe, totale {len(rows)}")
        except Exception as e:
            print(f"  Errore batch: {e}")

    with ThreadPoolExecutor(max_workers=max_inflight) as pool:
        futures=[]
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

    save_results(out_json, rows, seeds=seeds, bucket_filter=bucket)
    print(f"\nFatto. {len(rows)} righe totali in {out_json} [JSON]")
    return rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query Ollama gpt-oss:120b su grafo+qstates (Q via V) — clone di query_gemma per Ollama cloud")
    parser.add_argument("--seed", type=int, default=None, help="default 1337 o derivato da --path")
    parser.add_argument("--size", type=int, default=None, choices=[6,8,16], help="default 8 o derivato da --path")
    parser.add_argument("--lang", type=str, default="it", choices=["it","en"])
    parser.add_argument("--bucket", type=str, default=None, help="1 bucket solo (es. bottleneck); default tutti")
    parser.add_argument("--limit", type=int, default=None, help="numero totale stati per seed (es. 10) bilanciato tra iniziale/avanzato")
    parser.add_argument("--limit-per-bucket", type=int, default=None, dest="limit_per_bucket", help="numero per bucket per seed (es. 5)")
    parser.add_argument("--dry-run", action="store_true", help="non chiama API, solo verifica prompt")
    parser.add_argument("--force", action="store_true", help="ignora dedup e riprocessa tutto")
    parser.add_argument("--model", type=str, default=MODEL_ID, dest="model_id", help="model id Ollama (default gpt-oss:120b, come bak)")
    parser.add_argument("--ollama-host", type=str, default=None, dest="ollama_host", help="OLLAMA_HOST override (default env OLLAMA_HOST o https://ollama.com)")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="temperature Ollama")
    parser.add_argument("--interval", type=int, default=LAUNCH_INTERVAL_SEC, dest="launch_interval", help="pacing sec tra batch (default 10, gemma 65)")
    parser.add_argument("--max-inflight", type=int, default=MAX_INFLIGHT, dest="max_inflight", help="max thread concorrenti")
    parser.add_argument("--path", type=str, default=None, dest="states_path", help="file .json stati (es. src/output/cache/doorkey_states_8x8_seed1337.json o ..._seeds1337-42.json); seed/size dal nome, output accanto")
    args = parser.parse_args()
    run(seed=args.seed, size=args.size, lang=args.lang, limit=args.limit, limit_per_bucket=args.limit_per_bucket, dry_run=args.dry_run, force=args.force,
        model_id=args.model_id, ollama_host=args.ollama_host, temperature=args.temperature, launch_interval=args.launch_interval, max_inflight=args.max_inflight, states_path=args.states_path, bucket=args.bucket)
