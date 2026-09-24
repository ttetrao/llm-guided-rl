"""Unico punto di verità per le directory di output (sotto src/).

- CACHE_DIR: cache MDP + qstates (rigenerabili, ignorati da git)
- LLM_DIR: risultati delle query LLM
- AGENTS_DIR: run degli agenti (json + png)
- EVAL_DIR / EVAL_FL_DIR: valutazioni doorkey / frozenlake (log + grafici)

Ogni modulo usa queste costanti per i default: i lanci restano organizzati da soli.
Argomenti espliciti (--path, --out, --outdir) accettano comunque qualsiasi posizione.
"""
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SRC_DIR / "output"

CACHE_DIR = OUTPUT_DIR / "cache"
LLM_DIR = OUTPUT_DIR / "llm"
AGENTS_DIR = OUTPUT_DIR / "agents"
EVAL_DIR = OUTPUT_DIR / "evaluate"
EVAL_FL_DIR = OUTPUT_DIR / "evaluate_frozenlake"


def ensure_dirs() -> None:
    for d in (CACHE_DIR, LLM_DIR, AGENTS_DIR, EVAL_DIR, EVAL_FL_DIR):
        d.mkdir(parents=True, exist_ok=True)
