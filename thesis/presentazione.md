# Presentazione: inizializzazione LLM per Q-learning

Obiettivo: spiegare in 15 minuti cosa è stato fatto, come funziona e cosa supportano gli artefatti correnti.
Tempo: 15 minuti di parlato, senza domande. Un messaggio principale per slide; le note danno il dettaglio.

## Slide 1 - Titolo (0:40)

**Messaggio:** Inizializzare una Q-table con stime di un LLM per accelerare l'apprendimento in un compito con reward sparso.

**Mostra:** `src/output/agents/doorkey_8x8_seed1337.png`

**Note per il relatore:** Il run principale usa il seed 1337. La presentazione distinguerà sempre i risultati sui layout visti da quelli su un layout non visto, così da non trasformare un caso favorevole in una promessa di generalizzazione.

## Slide 2 - Il problema e l'idea (0:55)

**Messaggio:** Il Q-learning parte da una tabella vuota; quando il goal riceve reward solo occasionalmente, i valori utili tardano ad arrivare. Qui l'LLM agisce solo come punto di partenza: stima i valori iniziali, poi l'agente continua con Q-learning ordinario.

**Note per il relatore:** Non si aggiunge shaping e non si modifica la ricompensa durante l'addestramento. L'inizializzazione è applicata una volta, all'inizio; il resto del training resta il Q-learning tabellare.

## Slide 3 - Ambiente e verità di base (0:55)

**Messaggio:** In `MiniGrid-DoorKey-8x8-v0` l'agente deve prendere la chiave, aprire la porta e raggiungere il goal; il reward è concentrato sul risultato finale.

**Mostra:** `src/output/agents/doorkey_8x8_seed1337.png`

**Note per il relatore:** Il grafo deterministico del seed 1337 contiene 480 stati. Per il confronto si usa value iteration con `gamma=0.99`; gli episodi hanno un limite di 450 passi. La mappa serve a orientare il pubblico, non a rappresentare tutti gli stati osservabili.

## Slide 4 - Stati selezionati e query all'LLM (1:00)

**Messaggio:** L'LLM viene interrogato su un insieme esplicito di stati, non su tutta la tabella.

- Nel run corrente del seed 1337 il selettore considera 100 stati.
- Il file di valutazione del seed 1337 riporta 99 stati richiesti: 85 validi e 14 falliti.
- Nella valutazione su cinque layout il file contiene 155 stati richiesti, tutti nel bucket `bottleneck`; 111 sono validi e 44 falliti.

**Note per il relatore:** Il selettore attuale considera 100 stati, mentre le figure dello scenario base incorporano una valutazione precedente con 99 richieste: per questi numeri uso sempre i campi del grafico, non il JSON corrente. Nel multi-layout, gli artefatti documentano 155 stati, tutti bottleneck.

## Slide 5 - Valutazione sul seed 1337 (1:10)

**Messaggio:** Sul run base le stime sono abbastanza precise da orientare bene le azioni, pur senza trasformarle in una verità assoluta.

- 85 stati validi su 99 richiesti; 14 falliti, pari al 14.1%.
- MAE `0.0122`; Pearson `0.941`; CCC `0.908`.
- Top-1 tie-aware `94.1%`; Top-1 strict `84.7%`.
- Regret medio `0.0011`.

**Mostra:** `src/output/evaluate/grafici/llm_results_doorkey_states_8x8_seed1337_gpt-oss_120b_scatter.png`

**Note per il relatore:** Se serve un secondo supporto, usare `src/output/evaluate/grafici/llm_results_doorkey_states_8x8_seed1337_gpt-oss_120b_accuracy.png`. Il vantaggio del tie-aware è che non penalizza un'azione quando il vero optimum è ambiguo; per questo lo tengo distinto dallo strict.

## Slide 6 - Valutazione su cinque layout (1:10)

**Messaggio:** La valutazione su cinque layout è informativa, ma il 28.4% delle richieste fallisce.

- 155 stati richiesti; 44 falliti (`28.4%`), 111 validi.
- MAE `0.0153`; RMSE `0.0519`; bias `-0.0116`.
- Pearson `0.707`; CCC `0.657`.
- Top-1 tie-aware `91.9%`; strict `82.0%`; regret `0.0020`.

**Mostra:** `src/output/evaluate/grafici/llm_results_doorkey_states_8x8_seeds13507-19920-32455-54937-69693_gpt-oss_120b_scatter.png`

**Note per il relatore:** I 111 stati validi di questo file sono tutti bottleneck. Non presento questi numeri come un confronto tra tutti i tipi di stato e non parlo di un degrado dei checkpoint: non è ciò che il log corrente supporta. Uso i conteggi del log, non le etichette percentuali del plot.

## Slide 7 - Inizializzazione sul seed 1337 (1:20)

**Messaggio:** Nel run principale, l'inizializzazione combina un passo di aggiornamento piccolo con una tabella già informata.

- 2000 episodi; `alpha=0.04`; `gamma=0.95`; 20 coppie stato-azione inizializzate.
- Valutazione greedy finale su 100 episodi. Il campo `len` dei grafici è la lunghezza media di queste partite.
- LLM-init: SR `1`, `len=13`, agreement `0.576`, loss `0.0068`.
- Same-hp: SR `1`, `len=16`, agreement `0.527`, loss `0.0075`.
- Standard, con altri iperparametri: SR `1`, `len=16`, agreement `0.556`, loss `0.0066`.

**Mostra:** `src/output/agents/qtable_llminit_seed_1337_gpt-oss_120b_llminit.png`

**Note per il relatore:** Il vantaggio osservato riguarda la combinazione di inizializzazione e iperparametri di questo run. Non attribuisco il risultato ad `alpha` da solo: nel repository non è rimasto uno sweep riproducibile che consenta di isolare l'effetto del singolo parametro.

## Slide 8 - La curva di convergenza (1:00)

**Messaggio:** La curva del run con inizializzazione si stabilizza molto prima; il controllo same-hp alla fine raggiunge comunque SR `1`.

**Mostra:** `src/output/agents/qtable_llminit_seed_1337_gpt-oss_120b_vs_samehp.png`

**Note per il relatore:** Qui il vantaggio è soprattutto di stabilizzazione anticipata e percorso più corto: `13` contro `16` passi medi. Non significa che il baseline same-hp non risolva, né che `alpha=0.04` sia universalmente migliore. Il dato è osservazione di un run, non una legge sugli iperparametri.

## Slide 9 - Il run addestrato su cinque seed (1:20)

**Messaggio:** L'evidenza più ampia riguarda un run addestrato su cinque layout, con una valutazione separata su un layout visto e uno non visto.

- 5000 episodi; `alpha=0.15`; `gamma=0.95`; 400 coppie stato-azione inizializzate.
- Seed di addestramento: `13507`, `19920`, `32455`, `54937`, `69693`.
- Seed `13507`, visto: LLM-init ottiene SR `1`, `len=21`.
- Seed `69694`, non visto: LLM-init ottiene SR `0`, `len=450`.

**Mostra:** `src/output/agents/qtable_llminit_multiseed_train13507-19920-32455-54937-69693_evalin13507_evalnew69694_llminit.png`

**Note per il relatore:** È un run multi-seed, non uno studio con repliche indipendenti. La distinzione tra seed `13507` e `69694` è essenziale: il primo fa parte del training, il secondo no.

## Slide 10 - Confronto sul seed di addestramento 13507 (1:20)

**Messaggio:** Sul seed già incontrato durante l'addestramento, l'inizializzazione produce un vantaggio netto rispetto ai controlli.

- LLM-init: agreement `0.671`, loss `0.0038`, SR `1`, `len=21`.
- Same-hp: SR `0`, `len=450`, agreement `0.464`, loss `0.0079`.
- Standard, con altri iperparametri: SR `0`, `len=450`, agreement `0.587`, loss `0.0067`.

**Mostra:** `src/output/agents/qtable_llminit_multiseed_train13507-19920-32455-54937-69693_evalin13507_evalnew69694_vs_samehp.png`

**Note per il relatore:** Il confronto con lo standard è visibile anche in `src/output/agents/qtable_llminit_multiseed_train13507-19920-32455-54937-69693_evalin13507_evalnew69694_vs_std.png`. La formulazione corretta è «molto meglio sul layout visto», non «sempre meglio»: il risultato unseen introduce un limite immediato.

## Slide 11 - Il limite: un layout non visto (1:20)

**Messaggio:** L'inizializzazione aiuta moltissimo sui layout già addestrati, ma il layout `69694` non viene risolto in questo run.

- LLM-init sul nuovo seed `69694`: SR `0`, `len=450`.
- Il controllo standard sul nuovo seed ottiene `17%` di SR: non tutti i controlli falliscono completamente.
- Un solo layout non visto non basta per concludere che il metodo generalizzi.
- Il run multi-seed è comunque un ottimo punto di partenza e permette di progettare una verifica più ampia, con più layout non visti e repliche indipendenti.

**Mostra:** `src/output/agents/qtable_llminit_multiseed_train13507-19920-32455-54937-69693_evalin13507_evalnew69694_llminit.png`

**Note per il relatore:** Non attribuisco il fallimento a una singola causa non verificata. La conclusione prudente è sul comportamento osservato: seen molto migliore, unseen non risolto dallo stesso agente in questo run.

## Slide 12 - FrozenLake: stime deboli, inizializzazione utile (1:00)

**Messaggio:** Su FrozenLake le stime dei valori sono deboli, ma l'inizializzazione aiuta comunque l'agente in questo singolo seed e run.

- MAE `0.1228`; Top-1 tie-aware `55.6%`.
- LLM-init: SR finale `1`, `len=17`.
- Same-hp: `len=23`.
- Standard: `len=22`.

**Mostra:** `src/output/agents/frozenlake_qtable_llminit_8x8_slippery_seed_1337_gpt-oss_120b_vs_samehp.png`

**Note per il relatore:** È FrozenLake 8x8 slippery, un ambiente stocastico. Qui non parlo di fallimento totale: dico che la valutazione delle stime è modesta mentre il percorso finale dell'agente inizializzato è più corto. Evidenza limitata a un seed e a una run.

## Slide 13 - Conclusioni (0:55)

**Messaggio:** I dati mostrano un contributo concreto sui layout incontrati, ma lasciano aperta la questione della generalizzazione.

**Quello che regge:**
- l'inizializzazione LLM è fattibile e mantiene il Q-learning tabellare;
- nel run considerato la convergenza è più precoce e i percorsi finali sono più corti;
- nel run multi-seed il vantaggio sul seed di addestramento `13507` è ampio e riproducibile dai grafici e dai JSON mostrati.

**Quello che non sappiamo ancora:**
- se il metodo generalizzi: sul solo seed non visto `69694` LLM-init ottiene SR `0`;
- quanto variano i risultati fra esecuzioni indipendenti, perché i cinque layout condividono una sola tabella Q.

**Note per il relatore:** Chiudere su questa distinzione. L'inizializzazione è un ottimo punto di partenza e il multi-seed offre una base concreta per verifiche più larghe, ma non è ancora una soluzione completa al trasferimento.

## Slide 14 - Chiusura (0:40)

**Messaggio:** Nei layout addestrati, l'LLM fornisce una direzione utile al Q-learning e riduce il lavoro necessario per raggiungere una strategia stabile.

**Note per il relatore:** Ripetere la distinzione finale: vantaggio chiaro sui layout visti; limite aperto sul layout non visto. Il passo successivo è una valutazione multi-seed più ampia, con repliche indipendenti e una misura della varianza.

## Slide 15 - Grazie per l'attenzione (0:15)

Domande.

## Checklist figure, JSON e log

### Valutazione seed 1337

- `src/output/evaluate/grafici/llm_results_doorkey_states_8x8_seed1337_gpt-oss_120b_scatter.png`
- `src/output/evaluate/grafici/llm_results_doorkey_states_8x8_seed1337_gpt-oss_120b_accuracy.png`
- `src/output/evaluate/grafici/llm_results_doorkey_states_8x8_seed1337_gpt-oss_120b_mae_bucket.png`
- `src/output/evaluate/grafici/llm_results_doorkey_states_8x8_seed1337_gpt-oss_120b_residuals.png`
- `src/output/llm/llm_results_doorkey_states_8x8_seed1337_gpt-oss_120b.json`

### Valutazione cinque layout

- `src/output/evaluate/grafici/llm_results_doorkey_states_8x8_seeds13507-19920-32455-54937-69693_gpt-oss_120b_scatter.png`
- `src/output/evaluate/grafici/llm_results_doorkey_states_8x8_seeds13507-19920-32455-54937-69693_gpt-oss_120b_accuracy.png`
- `src/output/evaluate/grafici/llm_results_doorkey_states_8x8_seeds13507-19920-32455-54937-69693_gpt-oss_120b_mae_bucket.png`
- `src/output/evaluate/grafici/llm_results_doorkey_states_8x8_seeds13507-19920-32455-54937-69693_gpt-oss_120b_residuals.png`
- `src/output/evaluate/log_valutazione.txt`
- `src/output/llm/llm_results_doorkey_states_8x8_seeds13507-19920-32455-54937-69693_gpt-oss_120b.json`

### Agente seed 1337

- `src/output/agents/qtable_llminit_seed_1337_gpt-oss_120b_llminit.png`
- `src/output/agents/qtable_llminit_seed_1337_gpt-oss_120b_vs_samehp.png`
- `src/output/agents/qtable_llminit_seed_1337_gpt-oss_120b_vs_std.png`
- `src/output/agents/qtable_llminit_seed_1337_gpt-oss_120b_llminit.json`
- `src/output/agents/qtable_llminit_seed_1337_gpt-oss_120b_vsame.json`
- `src/output/agents/qtable_llminit_seed_1337_gpt-oss_120b_vstd.json`

### Agente cinque seed

- `src/output/agents/qtable_llminit_multiseed_train13507-19920-32455-54937-69693_evalin13507_evalnew69694_llminit.png`
- `src/output/agents/qtable_llminit_multiseed_train13507-19920-32455-54937-69693_evalin13507_evalnew69694_vs_samehp.png`
- `src/output/agents/qtable_llminit_multiseed_train13507-19920-32455-54937-69693_evalin13507_evalnew69694_vs_std.png`
- `src/output/agents/qtable_llminit_multiseed_train13507-19920-32455-54937-69693_evalin13507_evalnew69694_llminit.json`
- `src/output/agents/qtable_llminit_multiseed_train13507-19920-32455-54937-69693_evalin13507_evalnew69694_vsame.json`
- `src/output/agents/qtable_llminit_multiseed_train13507-19920-32455-54937-69693_evalin13507_evalnew69694_vstd.json`

### FrozenLake

- `src/output/agents/frozenlake_qtable_llminit_8x8_slippery_seed_1337_gpt-oss_120b_vs_samehp.png`
- `src/output/evaluate_frozenlake/log_valutazione.txt`
- `src/output/agents/frozenlake_qtable_llminit_8x8_slippery_seed_1337_gpt-oss_120b_llminit.json`
- `src/output/agents/frozenlake_qtable_llminit_8x8_slippery_seed_1337_gpt-oss_120b_vsame.json`
- `src/output/agents/frozenlake_qtable_llminit_8x8_slippery_seed_1337_gpt-oss_120b_vstd.json`
