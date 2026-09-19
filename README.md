# Inizializzare l'apprendimento per rinforzo con stime di un LLM: il caso MiniGrid-DoorKey

Appunti strutturati per la tesi. Tutti i numeri riportati sono misurati sui file in `src/thesis/graph/data/` e `src/thesis/evaluate/output/`; i riferimenti al codice indicano il file e la riga.

## 1. Problema

In ambienti con reward sparso il Q-learning tabellare impiega molti episodi prima di osservare il primo successo, e la stima si propaga lentamente a ritroso dagli stati terminali. L'idea esaminata qui: chiedere a un language model di stimare la funzione valore ottima su un piccolo insieme di stati, e usare quelle stime per inizializzare la Q-table invece di partire da zero. La domanda empirica è se, e in che misura, questo riduce gli episodi necessari a convergere.

## 2. Ambiente e MDP di riferimento

L'ambiente è `MiniGrid-DoorKey-8x8-v0`. L'agente deve raccogliere la chiave, aprire la porta e raggiungere il goal. Lo stato è la quintupla `(x, y, dir, has_key, door_open)`; le azioni utili sono sei (`left, right, forward, pickup, drop, toggle`, l'azione `done` è un self-loop e viene esclusa dalle query). Il reward è 1 solo entrando nella cella goal, 0 altrove; gli episodi sono troncati a 450 passi negli esperimenti con agente (1000 nel pilota su grafo).

Poiché layout e dinamica sono deterministici a meno del seed, il lavoro costruisce una volta sola il grafo completo dell'MDP (`src/thesis/graph/mdp_graph.py`): enumerazione degli stati validi, transizione deterministica per ognuna delle 7 azioni, reward e flag `done` per arco. Sul seed 1337 il grafo ha 480 nodi (chiave in (3,3), porta in (4,3), goal in (6,6), partenza in (3,5), 33 muri). La `V*` di riferimento è calcolata con value iteration (`_value_iteration`, `mdp_graph.py:150`) con gamma 0.99. Questo fornisce la verità di base contro cui si giudicano sia il pilota di raccolta stati sia le stime dell'LLM.

Gli stadi di avanzamento (`src/thesis/env/view_wrapper.py`) sono tre: `find_key`, `open_door`, `reach_goal`, derivati direttamente da posizione e flag dello stato.

## 3. Selezione degli stati

Interrogare l'LLM su tutti i 480 stati costerebbe troppo e servirebbe a poco: la maggior parte degli stati è corridoio. Con un budget di 150 stati la selezione (`src/thesis/graph/doorkey_states.py`) combina due sorgenti complementari: stati che per struttura obbligano una decisione, e stati che il learner ha effettivamente attraversato mentre imparava. I primi non dipendono da alcun training, i secondi sì.

### 3.1 Le sei famiglie di colli di bottiglia

I colli di bottiglia (`extract_bottleneck`, `doorkey_states.py:83`) sono estratti dal grafo senza eseguire alcun episodio. Le sei regole, nell'ordine in cui compaiono nel codice:

1. `on_key` (riga 90): stare sulla cella della chiave. Conta 8 nodi su ogni seed: 4 direzioni per 2 valori di `door_open`. La combinazione con `has_key=False` non esiste perché il grafo esclude la partenza senza chiave dalla cella chiave (`mdp_graph.py:275`).
2. `on_door` (riga 91): stare sulla cella della porta. Conta 8 nodi: 4 direzioni per 2 valori di `has_key`. Esiste solo con `door_open=True`, perché la porta chiusa è un muro invalicabile (`mdp_graph.py:277`).
3. `adj_door` (righe 93-99): celle adiacenti alla porta, con direzione puntata verso di essa e porta chiusa. Il ciclo prova le 4 direzioni cardinali, ma il risultato è 3-4 nodi invece di 8: la porta è inserita in una colonna di muro, quindi due dei quattro approcci cadono nel muro e non esistono come stati; inoltre sul seed 1337 il lato ovest coincide con la cella chiave (chiave in (3,3), porta in (4,3)) e contribuisce un solo nodo invece di due.
4. `door_toggle` (righe 101-103): stati con porta chiusa in cui l'azione `toggle` (indice 5) porta a uno stato con porta aperta. Sono sempre 2 su tutti i seed: i due lati praticabili della porta con chiave in mano. È un sottoinsieme di `adj_door`.
5. `key_take` (riga 105): stati in cui `pickup` (indice 3) cambia davvero lo stato, cioè le angolazioni di presa della chiave. Massimo 4 lati spaziali per 2 valori di `door_open` = 8; il valore osservato va da 2 a 8 a seconda della posizione della chiave (in un angolo resta un solo lato praticabile).
6. `goal_entry` (righe 107-109): stati non terminali con almeno una transizione a reward positivo, cioè le angolazioni di arrivo al goal. Il goal è in un angolo, quindi gli ingressi spaziali sono 1 o 2, moltiplicati per 4 combinazioni di flag: 4 oppure 8 nodi. La variazione dipende dalla colonna del muro con la porta, che cambia con il seed e può occupare una delle due celle di approccio (sul seed 69896 il muro è in x=5, la cella (5,6) non esiste e resta il solo ingresso da (6,5)).

I terminali sul goal sono esclusi in partenza (assorbenti, nessuna decisione da valutare). Ogni famiglia da sola è piccola; l'unione deduplica le sovrapposizioni (`door_toggle` dentro `adj_door`, la cella chiave che è anche adiacente alla porta sul seed 1337) e il risultato sta tra 30 e 34 nodi.

### 3.2 I numeri sui sei seed

| seed | chiave | porta | on_key | on_door | adj | tog | take | goal | unione |
|---|---|---|---|---|---|---|---|---|---|
| 1337 | (3,3) | (4,3) | 8 | 8 | 3 | 2 | 7 | 8 | 32 |
| 47952 | (1,5) | (2,1) | 8 | 8 | 4 | 2 | 4 | 8 | 32 |
| 69896 | (2,3) | (5,3) | 8 | 8 | 4 | 2 | 8 | 4 | 32 |
| 74839 | (2,5) | (3,4) | 8 | 8 | 4 | 2 | 6 | 8 | 34 |
| 80970 | (3,4) | (5,4) | 8 | 8 | 4 | 2 | 8 | 4 | 32 |
| 93045 | (1,1) | (2,4) | 8 | 8 | 4 | 2 | 2 | 8 | 30 |

`on_key`, `on_door` e `door_toggle` sono costanti (8, 8, 2): dipendono solo dalla struttura fissa del problema, non dal layout. La variabilità sta tutta in `key_take` (posizione della chiave) e `goal_entry` (colonna del muro). Il goal è sempre in (6,6); i nodi totali del grafo sono sempre 480.

### 3.3 Il pilota: checkpoint lungo l'apprendimento

I bottleneck dicono dove stanno le decisioni, non cosa vede un agente mentre impara. A questo serve il pilota (`run_pilot`, `doorkey_states.py:141`): un Q-learning tabellare standard (alpha 0.3, gamma 0.99, epsilon da 1.0 con decadimento 0.998 e minimo 0.05) che gira direttamente sul grafo, senza ambiente gym, partendo dallo stato iniziale del seed.

Ad ogni episodio il pilota registra gli stati visitati e l'esito; mantiene una finestra mobile degli ultimi 100 episodi sia per la success rate sia per lo storico `(visitati, successo)`. Quando la success rate tocca una delle soglie 0.3, 0.5, 0.7, 0.8, 1.0 (`DEFAULT_THRESHOLDS`, riga 61), congela un pool con gli stati visitati nella finestra. La regola di congelamento è doppia (`capture`, righe 166-176): sotto la soglia 0.8 il pool è l'unione diretta dei visitati, perché la policy è ancora esplorativa; da 0.8 in su il pool è l'unione dei visitati allargata con random walk di 3 passi (`_random_walk`, riga 113) più gli stati degli episodi falliti, perché a quel punto la policy è quasi ottima e i visitati da soli collasserebbero sul percorso migliore. Le soglie mai raggiunte entro 10000 episodi prendono comunque uno snapshot finale, marcato come non raggiunto (righe 202-205). Ogni 50 episodi una valutazione greedy serve solo al log. Sul seed 1337 il pilota tocca le cinque soglie agli episodi 93, 115, 137, 150 e 175, e ogni pool contiene poco più di 300 stati.

Due seed su sei (47952 e 74839) mostrano un caso limite onesto: il pilota arriva al goal al primo episodio per fortuna esplorativa, la success rate a finestra vale subito 1.0 e tutti e cinque i checkpoint collassano sullo stesso istante, con pool piccoli (62 e 88 stati) e greedy finale a 0.0. I checkpoint ereditano la fortuna del pilota; la sezione 9 riprende il punto.

Combinare bottleneck strutturali con stati campionati da checkpoint a diversi tassi di successo è coerente con la Reverse Curriculum Generation di Florensa et al. (2017), che usa soglie di success rate per espandere progressivamente la distribuzione di training a partire dal goal; la logica degli scaglioni 0.3, 0.5, 0.7, 0.8, 1.0 è dello stesso spirito. Il pilota è un run preliminare con l'algoritmo "puro" (senza LLM) sullo stesso ambiente: è la stessa struttura dei metodi teacher-student di curriculum learning, quindi una fase di preprocessing offline — non una loss né uno shaping applicato durante il training valutato — e resta compatibile con il vincolo "nessun reward shaping".

### 3.4 Budget e campionamento

Pool interi significherebbero 1500+ stati: il budget (`allocate`, `doorkey_states.py:230`) li riduce con due regole. Il bucket bottleneck è protetto e tenuto intero; il resto è ripartito tra i checkpoint in proporzione alle dimensioni con il metodo dei resti maggiori (`_proportional`, riga 216), usando un generatore seeded per riproducibilità. Se il budget fosse minore dello stesso bottleneck, la protezione cade e tutto diventa proporzionale. Con `--total 150` sul seed 1337 i pool 32 / 303 / 307 / 307 / 308 / 306 diventano bucket da 32 / 23 / 24 / 24 / 24 / 23, che sommano esattamente a 150. Senza tetto (`--total 0`) si ottengono tutti i pool interi. Il file prodotto conserva pool originali, bucket finali, episodi e success rate di ogni soglia e il riferimento al grafo, così ogni numero resta tracciabile (`extract`, righe 283-313).

### 3.5 Dal bucket alla query

Gli script di interrogazione non prendono necessariamente tutto il pool: leggono solo i bucket elencati in `BUCKETS_WANTED` (`query_gemma.py:63`, identico in `query_gpt.py:66`), poi applicano un campionamento seeded (`random.Random(seed).sample`) guidato da `--limit` (totale per seed) o `--limit-per-bucket`, e saltano le coppie stato-azione già presenti nel JSON di output (dedup). Il run gpt-oss ha coperto 145 dei 150 stati (870 righe, 6 azioni per stato, a meno di uno stato fallito); il run gemma si è fermato a 28 stati per limite esplicito (168 righe). Ogni riga conserva `node_id`, coordinate, direzione, flag, stadio, `v_true` e `v_llm`, quindi ogni stima resta riconducibile allo stato del grafo che l'ha generata.

### 3.6 Il precedente a bucket, e perché è stato sostituito

Prima di questo schema esisteva una selezione per fasi di apprendimento (`src/thesis/graph/qlearning_states.py`): tre bucket — iniziale, intermedio, avanzato — per success rate 0-0.33, 0.33-0.8 e 0.8-1.0, con al massimo 100 stati visitati campionati a caso per bucket e un sottoinsieme garantito di stati critici (cambi di stadio o terminali effettivamente calpestati, righe 213-222), con opzione `significant_only` per tenere solo quelli. Il problema di quell'approccio è che i bucket per fase mescolano punti decisionali e corridoio senza distinzione, e dipendono interamente dalla traiettoria del pilota. Lo schema attuale separa le due cose: i bottleneck fissano dove stanno le decisioni indipendentemente da come si impara, i checkpoint fotografano cosa si vede mentre si impara. Il vecchio modulo resta nel perimetro solo come dipendenza, perché fornisce `QLearningAgent` al pilota.

## 4. Stima dei valori con l'LLM

Il protocollo è Q-via-V (`src/thesis/llm/query_gemma.py`, `query_gpt.py`; prompt in `src/thesis/docs/doorkey/it/prompt.txt`). Per ogni stato il prompt mostra la mappa ASCII dello stato corrente e, già calcolate, le mappe dei sei successori `s' = T(s,a)`: il modello non deve inferire le transizioni, deve solo stimare `V*(s')` come `gamma^d` con `d` passi al goal e restituire `Q(s,a) = r + 0.99 * V*(s')`. L'output è JSON vincolato, con un'analisi per azione che cita definizione usata, fatto dell'ambiente e conto esplicito; analisi identiche tra azioni diverse invalidano la risposta.

Modelli interrogati: `gemma-4-26b-a4b-it` (28 stati, 168 righe: run parziale) e `gpt-oss 120b` (145 stati, 870 righe: copertura completa dei 150 stati del pool). Una richiesta per stato, pacing di 65 secondi tra gli avvii, 3 tentativi, deduplica su file: il collo è la latenza delle API, non il calcolo.

## 5. Qualità delle stime

La valutazione (`src/thesis/evaluate/evaluate_llm.py`, esiti in `src/thesis/evaluate/output/log_valutazione.txt`) confronta `v_llm` con `v_true = V*(s')` su tre piani: valori (MAE, RMSE, Pearson, Spearman, CCC con intervalli bootstrap), selezione dell'azione (accuratezza tie-aware: la scelta è corretta se una qualsiasi delle azioni a pari merito quantizzate su `gamma^k` è ottima) e regret come metrica primaria.

| modello | stati | MAE | Pearson r | Top-1 tie-aware | regret medio |
|---|---|---|---|---|---|
| gemma-4-26b-a4b-it | 28 | 0.0098 | 0.932 | 92.9% | 0.0010 |
| gpt-oss 120b | 144 | 0.0146 | 0.948 | 96.5% | 0.0011 |

Il 97% delle stime cade entro 0.05 dal vero e il bias è sistematicamente negativo di circa -0.01 (il modello sottostima di un passo di sconto). L'accuratezza resta sopra il 90% anche sul solo bucket bottleneck, che è il più difficile (MAE quasi doppia rispetto ai checkpoint). In termini pratici: i valori assoluti sono approssimati, l'ordinamento delle azioni è quasi sempre quello giusto, e il costo di una scelta sbagliata in valore atteso è intorno a un millesimo.

## 6. Inizializzazione del Q-learning e speedup

L'agente (`src/thesis/agent/doorkey_qtable_llminit.py`) lavora in coordinate relative al bersaglio di stadio `(dx, dy, dir, stage)` (`doorkey_state.py:14`), così gli stati di seed diversi con la stessa geometria relativa condividono la riga della tabella. Le righe `(stato, azione, v_llm)` vengono aggregate per media e scritte nella Q-table sulle coppie coperte, il resto parte da zero; l'apprendimento resta Q-learning ordinario. Confronto a parità di iperparametri (1500 episodi, alpha 0.2, gamma 0.95, epsilon con decadimento 0.99, seed 1337, stime di gpt-oss 120b):

| init | coppie inizializzate | primo successo (ep.) | SR ultimi 100 ep. | SR eval (100 ep.) | passi medi a convergenza |
|---|---|---|---|---|---|
| LLM, 720 coppie | 720 / 870 | 14 | 1.00 | 1.00 | 13 (ottimo) |
| LLM, 50 coppie | 50 / 870 | 40 | — | 1.00 | 13 (ottimo) |
| da zero | 0 | 127 | 0.07 | 0.00 | — (non converge) |

Con 720 coppie il primo successo arriva all'episodio 14 contro 127, e la policy converge al percorso ottimo di 13 passi mentre il baseline in 1500 episodi non risolve l'ambiente (7% di successi negli ultimi 100 episodi, 0% in valutazione). Anche con sole 50 coppie inizializzate il risultato finale è identico, solo più lento all'avvio: la dose di conoscenza iniziale sposta la curva, non il punto di arrivo. Si nota un dettaglio onesto: l'agente sconta con gamma 0.95 mentre le stime sono prodotte con 0.99; l'inizializzazione è quindi approssimata due volte, e funziona lo stesso.

### 6.1 Sensibilità agli iperparametri: alpha alto, gamma basso

A parità di inizializzazione (720 coppie, seed 1337, 1500 episodi, epsilon con decadimento 0.99) sono stati provati sei abbinamenti di learning rate e sconto (file `qtable_llminit_seed_1337_sweep_*.json`, comando nella mappa in fondo). Tutti convergono a SR 1.0 in valutazione; cambiano velocità di avvio, velocità di consolidamento e qualità del percorso finale.

| alpha | gamma | primo successo (ep.) | ep. con SR100 a 1.0 | passi medi in eval |
|---|---|---|---|---|
| 0.2 | 0.95 | 14 | 407 | 13 |
| 0.5 | 0.95 | 16 | 289 | 13 |
| 0.9 | 0.95 | 14 | 350 | 13 |
| 0.2 | 0.90 | 15 | 314 | 13 |
| 0.2 | 0.80 | 22 | 324 | 13 |
| 0.5 | 0.90 | 5 | 286 | 13 |
| 0.2 | 0.99 | 12 | 770 | 16 |

Tre osservazioni. Primo, la combinazione migliore è alpha 0.5 con gamma 0.9: primo successo all'episodio 5 e finestra a 1.0 già all'episodio 286, con percorso finale ottimo. Secondo, gamma 0.99 — lo stesso sconto con cui sono calcolate le stime iniziali — è il peggiore per consolidamento (770 episodi) e l'unico che converge a un percorso subottimo di 16 passi: con sconto alto il paesaggio dei valori è piatto (stati adiacenti differiscono di un fattore 0.99) e la greedy discrimina poco, mentre uno sconto più basso rende ripido il gradiente del prior LLM e le decisioni diventano nette prima. Terzo, alpha 0.9 parte veloce ma consolida più lentamente di alpha 0.5 (350 contro 289 episodi): aggiornamenti troppo aggressivi fanno oscillare la tabella intorno ai valori iniziali invece di assestarla. In sintesi, l'inizializzazione dà la direzione e alpha alto con gamma basso la rende utilizzabile in fretta; lo sconto dell'agente va scelto per la dinamica di apprendimento, non per coerenza con quello delle stime.

## 7. Prova multi-seed e generalizzazione

Addestrando su 5 seed con rappresentazione relativa e valutando sul primo seed di training e su un seed mai visto (file `qtable_llminit_multiseed_*.json`): in-seed SR 1.00 con 25 passi medi, nuovo seed SR 0.01. Le stime dell'LLM accelerano l'apprendimento sul layout visto ma non trasferiscono la competenza a un layout nuovo: la coordinata relativa riduce lo spazio ma non astrae la posizione di muri e strozzature, che cambiano con il seed. È il limite principale del metodo così formulato.

## 8. Esperimento DDQN: esito negativo

La stessa idea è stata provata con approssimazione di funzione (`src/thesis/agent/doorkey_ddqn_pretrained.py`): regressione supervisionata della rete sui valori LLM (MSE finale 0.003 dopo 500 epoche) seguita da 3000 episodi online con transizioni esperte permanenti nel replay. Risultato: 965 successi su 3000 episodi ma SR 0.00 in valutazione su 200 episodi. La rete imita i valori senza stabilizzare la policy greedy sotto reward sparso. Il contrasto con il caso tabellare è istruttivo: l'inizializzazione aiuta l'esplorazione, non sostituisce la convergenza dell'approssimatore.

## 9. Limiti

Primo, copertura: 150 stati su 480 bastano per DoorKey 8x8 ma la frazione utile scala male con la dimensione della mappa. Secondo, costo: le query LLM restano l'anello lento della pipeline (una richiesta per stato). Terzo, generalizzazione assente tra layout diversi (sezione 7). Quarto, i checkpoint ereditano la fortuna del pilota: sui seed 47952 e 74839 il successo al primo episodio fa collassare tutte le soglie sullo stesso istante, e quei checkpoint fotografano rumore esplorativo invece di una traiettoria di apprendimento (sezione 3.3). Quinto, il DDQN con pretraining non converge in questa configurazione (sezione 8): il risultato positivo vale per il caso tabellare.

## 10. Mappa dei file citati

- `src/thesis/graph/mdp_graph.py` — grafo MDP completo e `V*` via value iteration.
- `src/thesis/graph/doorkey_states.py` — bottleneck strutturali (riga 83), pilota con checkpoint (riga 141), budget con resti maggiori (righe 216-242).
- `src/thesis/graph/qlearning_states.py` — precedente approccio a bucket; resta come dipendenza (fornisce `QLearningAgent` al pilota).
- `src/thesis/llm/query_gemma.py`, `query_gpt.py` — interrogazione dei modelli.
- `src/thesis/docs/doorkey/{it,en}/` — prompt, legenda, definizioni di `v_*` e `q_*`.
- `src/thesis/evaluate/evaluate_llm.py` — metriche su valori, azioni e regret; `evaluate/output/` contiene grafici e `log_valutazione.txt`.
- `src/thesis/agent/doorkey_qtable_llminit.py`, `doorkey_state.py` — Q-learning con init da LLM; `doorkey_ddqn_pretrained.py`, `doorkey_ddqn.py`, `ExperienceReplayBuffer.py` — ramo DDQN. Comando dello sweep (sezione 6.1), da `src/`: `python3 -m thesis.agent.doorkey_qtable_llminit --input llm_results_doorkey_states_8x8_seed1337_gpt-oss_120b.json --seed 1337 --episodes 1500 --eps_decay 0.99 --eps_min 0.05 --max_steps 450 --eval_episodes 100 --no_plot --alpha A --gamma G --tag sweep_aAgG`; gli esiti sono in `graph/data/qtable_llminit_seed_1337_sweep_*.json`.
- `src/thesis/env/view_wrapper.py`, `doorkey_events.py` — stadi e osservazione.
- `src/stash/` — tutto il materiale escluso dal perimetro (FrozenLake, varianti di agenti superate, backup): non citato sopra.

## 11. Bibliografia ragionata minima

1. Sutton, R. S., Barto, A. G., *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018 — MDP, Bellman, TD, Q-learning; update `Q ← Q + α[r + γ max Q' − Q]`.
2. Watkins, C. J. C. H., Dayan, P., "Q-learning", *Machine Learning* 8, 1992 — convergenza tabulare a `Q*`; l'init arbitraria non cambia il punto fisso sotto visite infinite e Robbins-Monro.
3. Wiewiora, E., "Potential-Based Shaping and Q-Value Initialization are Equivalent", *JAIR* 19, 2003 — `Q0(s,a) = Φ(s)` equivale in condizioni specifiche al potential shaping `F = γΦ(s') − Φ(s)`; i valori LLM vanno descritti come sola init a `t=0`, senza reward aggiuntivi né modifiche al target TD.
4. Mnih, V. et al., "Human-level Control through Deep RL", *Nature* 518, 2015 — DQN: replay, target network, instabilità di bootstrapping + off-policy + approssimazione.
5. Van Hasselt, H., Guez, A., Silver, D., "Deep RL with Double Q-learning", *AAAI*, 2016 — target DDQN con selezione online e valutazione target; riduce il bias del max (`doorkey_ddqn.py:94-97`).
6. Schaul, T. et al., "Prioritized Experience Replay", *ICLR*, 2016 — priorità `p ∝ |δ|^α` con pesi IS (`ExperienceReplayBuffer.py`); quota expert e `priority_boost` sono interventi distinti (cfr. sezione 8).
7. Hester, T. et al., "Deep Q-learning from Demonstrations", *AAAI*, 2018 — solo come delimitazione: DQfD usa pre-training + replay permanente + loss con margine e target multi-step; il metodo qui è *LLM-based Q-init with persistent expert replay*, non DQfD completo.
8. McGovern, A., Barto, A. G., "Automatic Discovery of Subgoals...", *ICML*, 2001 — solo la definizione di bottleneck (stati frequenti nei successi); il metodo qui è a regole strutturali, non diverse-density (cfr. sezione 3.1).
9. Florensa, C. et al., "Reverse Curriculum Generation for RL", *CoRL*, 2017 — solo analogia: stati iniziali espansi dal goal per soglie di successo; stessa logica degli scaglioni 0.3–1.0, ma qui i checkpoint servono a campionare query, non a generare start-state (cfr. sezione 3.3).
10. Chevalier-Boisvert, M. et al., "MiniGrid & MiniWorld", *NeurIPS 36*, 2023 — riferimento per DoorKey-8x8, reward sparso, osservabilità parziale.
11. Henderson, P. et al., "Deep RL that Matters", *AAAI*, 2018 — un solo run non basta: seed e dettagli implementativi cambiano gli esiti (cfr. sezioni 3.3, 7, 9).

Scartati dopo controllo del codice: Even-Dar e Mansour 2001 (init ottimistica — mai usata: il resto parte da zero e le stime LLM hanno bias negativo, sezione 5; il flag `optimistic_init` esiste solo in `stash` FrozenLake, fuori perimetro); Even-Dar e Mansour 2003 (schedule `α = 1/N^ω` — mai usato: alpha sempre costante e gli args `alpha_mode` sono deprecated e ignorati, `doorkey_qtable_llminit.py:355-364`); Ng, Harada e Russell 1999 (gli shape `F = γΦ' − Φ` non sono mai aggiunti nei run valutati — `build_potential({})` vuoto nel ramo pretrained; la forma è già richiamata nella voce 3); Colas et al. 2018 e Agarwal et al. 2021 (nessuna power analysis né IQM nel codice: gli intervalli in `evaluate_llm.py` sono bootstrap semplici; restano lavoro futuro per la valutazione finale).
