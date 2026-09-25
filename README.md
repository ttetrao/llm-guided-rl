# LLM-Guided Reinforcement Learning

Questo progetto esplora l'utilizzo di modelli linguistici di grandi dimensioni (LLM) per inizializzare la tabella dei valori in algoritmi di Apprendimento per Rinforzo (Q-learning). In particolare, il lavoro dimostra come stimare la funzione valore ottima su un sottoinsieme di stati critici, accelerando l'apprendimento su ambienti con ricompensa sparsa come **MiniGrid-DoorKey-8x8**.

## Cosa è stato fatto

- Estrazione di un grafo di stati (Processo Decisionale di Markov) e identificazione dei nodi critici (bottleneck).
- Interrogazione strutturata di modelli LLM (GPT e Gemma) per ottenere stime numeriche della distanza dall'obiettivo per ogni possibile azione.
- Inizializzazione della Q-table basata sulle risposte dei modelli linguistici.
- Confronto delle prestazioni dell'agente potenziato rispetto al Q-learning standard, misurando tempi di convergenza e metriche di errore.
- Strutturazione di una pipeline per l'ambiente MiniGrid e una predisposizione architetturale per FrozenLake.

## Per saperne di più

Il dettaglio teorico, la metodologia e i risultati completi sono descritti nel documento di tesi.
Puoi consultare il file compilato in: [`thesis/elaborato.pdf`](thesis/elaborato.pdf).
Il codice sorgente dell'agente, della valutazione e della generazione dei grafi risiede all'interno della directory `src/`.

## Installazione

L'ambente di lavoro è Python 3.10. Le dipendenze si installano con conda; i due client LLM richiedono poi `pip`.

```bash
conda create -n llm-guided-rl python=3.10 -y
conda activate llm-guided-rl
conda install -c conda-forge numpy pandas matplotlib scipy gymnasium minigrid -y
pip install ollama
pip install "monorepo @ git+https://github.com/e-zorzi/monorepo"
```

| Pacchetto | Serve per |
| --- | --- |
| `numpy` | grafo MDP, Q-table, metriche |
| `pandas` | `evaluate/`: aggregazione per stato, bucket e azione |
| `matplotlib` | grafici in `evaluate/` e `agent/` (backend `Agg`, senza schermo) |
| `scipy` | `pearsonr` e `spearmanr` in `evaluate/` |
| `gymnasium` | ambienti `MiniGrid-DoorKey-8x8-v0` e `FrozenLake-v1` |
| `minigrid` | registrazione degli env MiniGrid e wrapper di rendering |
| `ollama` (pip) | `llm.query_gpt` contro Ollama Cloud |
| `monorepo` (pip) | `llm.query_gpt` / `llm.query_gemma` e lettura delle API key |

Note: `monorepo` non è su conda-forge e la versione omonima su PyPI non è quella del progetto, quindi va installata dal repository. `pygame` serve solo per aprire finestre di rendering: il codice usa il backend `Agg` e funziona senza. Le interrogazioni all'LLM richiedono credenziali (`monorepo.load_api_keys()`, oppure `OLLAMA_API_KEY` e `OLLAMA_HOST` per Ollama); senza chiave i moduli partono comunque in `--dry-run`.

## Comandi per generare i file

Il codice è organizzato in pacchetti python e va eseguito partendo dalla directory `src/`.
Spostati nella cartella principale dei sorgenti:
```bash
cd src
```

### 1. Costruzione del Grafo e Selezione Stati
Per generare il grafo e identificare gli stati critici dell'ambiente:
```bash
python3 -m graph.mdp_graph
python3 -m graph.doorkey_states
```

### 2. Interrogazione del Modello LLM
Per avviare la stima dei valori tramite API (richiede credenziali e configurazione nei file preposti):
```bash
python3 -m llm.query_gpt
# oppure
python3 -m llm.query_gemma
```

### 3. Valutazione e Metriche
Per analizzare la qualità delle risposte dell'LLM e generare i grafici a dispersione e sui residui:
```bash
python3 -m evaluate.evaluate_llm
```

### 4. Esecuzione Agenti e Grafici di Confronto
Per avviare l'addestramento degli agenti tabellari e creare i plot di paragone delle prestazioni:
```bash
python3 -m agent.doorkey_qtable_llminit
```
