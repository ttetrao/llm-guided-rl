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
python3 -m agent.doorkey_qtable_llminit2
```
