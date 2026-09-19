# Docs per ambiente e lingua

Struttura riorganizzata (ponytail lite):

```
docs/
  doorkey/
    it/  → MiniGridDocumentation.md (it, Rewards normalizzati), legend.txt, prompt.txt (Q via V* 6 azioni), q/v-definition.md (it)
    en/  → MiniGridDocumentation.md (en, Rewards normalizzati), legend.txt, prompt.txt (Q via r+γV* 6 azioni, mirror it), q/v-definition.md (en estesa)
  frozenlake/
    it/  → FrozenLakeDocumentation.md (dettagliata success_rate=1/3), legend.txt, prompt.txt (Q it con p=1/3), prompt_json.txt (Q+V), q/v-definition.md (it)
    en/  → FrozenLakeDocumentation.md (stessa dettagliata), legend.txt, prompt.txt (Q en mirror it), prompt_v.txt (V singola), prompt_json.txt (Q+V), q/v-definition.md (en estesa)
```

- `doorkey` = MiniGrid-DoorKey deterministico, reward `r=1` solo su goal, `γ=0.99`.
- `frozenlake` = FrozenLake slippery stocastico, `p=1/3` uniforme, hole terminale `0`.

Duplicati eliminati: root `docs/*.md/*.txt` e vecchie cartelle `docs/it`, `docs/en`, `docs/frozenlake/*.md` root sono stati rimossi; `bak/` ignorato.
Lingua determina prompt e definizioni; per `frozenlake` `mode` determina il file (`q`→`prompt.txt`, `v`→`prompt_v.txt`) con fallback incrociato it/en.

Loader aggiornati: `thesis/llm/query_gemma.py` → `docs/doorkey/{lang}`, `query_frozenlake.py` → `docs/frozenlake/{lang}`.
