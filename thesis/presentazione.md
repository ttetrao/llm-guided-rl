# Presentazione repo: canovaccio

Obiettivo: spiegare in 15 minuti cosa hai fatto, come, perché, e cosa è venuto fuori.
Tempo: 14 minuti di parlato + domande. Se sfori, taglia la tabella dello sweep in slide 9 e tieni solo i tre motivi.
Pubblico: chi non ha letto la tesi. Un grafico per slide. Poche parole per slide.

## Slide 1 - Titolo (1 min)

Di cosa si tratta in una riga: inizializzare il Q-learning con stime di un LLM su MiniGrid-DoorKey-8x8.
Mostra: titolo, nome, seed principale 1337, link al repo.
Dire: il problema è il reward sparso. Parti da zero e ci metti 127 episodi prima del primo goal. Con le stime LLM ne bastano 14. Lascia questi due numeri sullo schermo, torneranno alla fine.

## Slide 2 - Perché farlo (1 min)

Il Q-learning tabellare propaga il valore a ritroso dal goal. Se il goal si vede di rado, la tabella resta a zero per molto tempo.
L'idea provata qui: chiedi al modello una stima di V* su pochi stati e usala come Q0. Poi apprendi in modo normale, senza shaping.
Riferimento nel README §1 e §11 punto 3 (Wiewiora 2003: init e shaping sono legati solo a certe condizioni, qui fai solo init a t=0).

## Slide 3 - Ambiente e verità di base (1 min 30)

Ambiente `MiniGrid-DoorKey-8x8-v0`: prendi la chiave, apri la porta, vai al goal in (6,6). Reward 1 solo sul goal, 0 altrove. Episodi troncati a 450 passi.
Stato usato nel grafo: (x, y, dir, has_key, door_open). 480 nodi sul seed 1337.
Cosa vuol dire, in una riga per elemento: x e y dicono la cella, dir da che parte guarda l'agente (serve perché avanti e gira dipendono da quello), has_key e door_open dicono a che punto del compito sei. Il resto non serve: a parità di seed il layout è fisso, muri compresi (33 sul 1337), quindi da questi cinque valori ricostruisci tutto.
Il grafo si costruisce una volta sola con `src/graph/mdp_graph.py`: per ogni stato e ognuna delle 7 azioni calcoli transizione deterministica, reward e flag done. V* viene da value iteration con gamma 0.99 ed è la verità di base per giudicare l'LLM. I terminali sul goal sono assorbenti e restano fuori dalle query, l'azione done è un self-loop e la si esclude.
Mostra: una mappa ASCII da `src/docs/doorkey/it/` con chiave in (3,3), porta in (4,3).

## Slide 4 - Quali stati, quanti, cosa sono (2 min)

Messaggio da far passare: usi pochi stati e sai dire quali sono.
Totale grafo: 480 nodi per seed. Budget: 150 stati per seed, cioè il 31% del totale. I corridoi restano fuori.
Dentro i 150, due sorgenti. Codice in `src/graph/doorkey_states.py`:
- bottleneck strutturali: 30-34 nodi per seed, circa il 7% dei 480. Sei regole fisse: on_key 8 nodi (stare sulla chiave), on_door 8 (stare sulla porta), adj_door 3-4 (celle davanti alla porta chiusa), door_toggle 2 (dove toggle apre davvero), key_take 2-8 (angoli di presa chiave), goal_entry 4-8 (angoli di ingresso goal). Sul 1337 l'unione fa 32 nodi perché alcune famiglie si sovrappongono.
- checkpoint dal pilota: Q-learning senza LLM sul grafo. Quando la success rate a finestra tocca 0.3, 0.5, 0.7, 0.8, 1.0 congeli i visitati. Sul 1337 tocca le soglie agli episodi 93, 115, 137, 150, 175, pool da circa 300 stati l'uno. Da questi pool escono i restanti 118 stati del budget con ripartizione a resti maggiori: 32 + 23/24/24/24/23 = 150.
Mostra: tabella seed del README §3.2 più una riga con le frazioni 32/480 e 150/480. Chi ascolta deve ricordare solo due frazioni: budget un terzo del grafo, bottleneck sotto il 10%.

## Slide 5 - Come chiedi i valori (1 min)

Protocollo Q-via-V. Prompt in `src/docs/doorkey/it/prompt.txt`, script `src/llm/query_gpt.py` e `query_gemma.py`.
Per stato mostri la mappa attuale e le sei mappe dei successori già calcolate. Il modello non indovina le transizioni, stima solo V*(s') come gamma^d e rende Q = r + 0.99 V*.
Output JSON vincolato, una riga per coppia stato-azione con node_id, stadio, v_true, v_llm.
Run reali: gpt-oss 120b sul 1337 copre 145 query (870 righe), pari a 125 stati fisici unici perché lo stesso nodo compare in più bucket; di questi 32 sono bottleneck (192 righe). Gemma si ferma a 28 stati (168 righe, run parziale). Sul run a 5 seed hai 200 stati totali (40 per seed), di cui 160 bottleneck e 40 checkpoint: lì i bottleneck sono l'80% del valutato.

## Slide 6 - Le stime sono buone? Valori (1 min)

File da citare: `evaluate/output/log_valutazione.txt` del 2026-09-23, run gpt-oss su 5 seed (200 stati, 195 validi). Attenzione: 960 righe su 1200 sono bottleneck, quindi la media è trascinata dai bottleneck. È voluto, sono gli stati che contano.
Numeri da dire:
- MAE 0.0186, RMSE 0.0730, bias -0.0080. Il 96.4% entro 0.05.
- Bottleneck più duro: MAE 0.0141. I checkpoint stanno tra 0.0101 e 0.0134, tranne ckpt-0.8 che ha un outlier (MAE 0.1078 su 60 righe).
Figure da mettere:
- `..._scatter.png`: punti addensati sulla diagonale y=x, banda ±0.05.
- `..._residuals.png`: residui quasi tutti nella fascia ±0.05, bias leggero sotto zero.
- `..._mae_bucket.png`: bottleneck peggio dei checkpoint.
Nota onesta: il README §5 cita un run vecchio su seed 1337 (MAE 0.0146). I numeri sopra sono del run a 5 seed, su più layout. Dillo.

## Slide 7 - Le stime scelgono l'azione giusta? (1 min)

Stesso log, righe 18-20 e 28-34:
- Top-1 tie-aware 90.3%, Top-2 94.9%. Random sarebbe 16.7%.
- Regret medio 0.0020. Sbagli poco e quando sbagli costa poco.
- Bottleneck acc 88.4%. I checkpoint 0.3-0.8 sono al 100%, ckpt-1 a 88.9%.
Figure da mettere:
- `..._accuracy.png`: barre Top-1 90.3%, Top-2 94.9%.
- `..._confusion.png` e `..._k_error.png`: tienile di scorta per le domande. Δk=0 solo 8.8%, |Δk|=1 al 63.5%: i valori assoluti tornano, la distanza in passi meno.

## Slide 8 - L'init accelera davvero? (2 min)

Agente in `src/agent/doorkey_qtable_llminit.py`. Lavora in coordinate relative al bersaglio di stadio, così righe uguali su seed diversi condividono la tabella. Le stime LLM entrano come media per coppia coperta, il resto parte da zero. Poi Q-learning normale. Ordine di init: `bottleneck_first_top_v` in `load_qinit`, prima i bottleneck con V_llm più alta.
Punto chiave dei grafici buoni: il run `qtable_llminit_seed_1337_gpt-oss_120b_llminit.json` usa `max_init=20`, cioè 20 coppie (stato relativo, azione) prese dai bottleneck. Sono circa metà dei 32 bottleneck del seed 1337, e il 4% dei 480 nodi del grafo. Con così poco arrivi a SR 1.00 in eval con 17 passi medi.
Confronto a pari iperparametri, seed 1337, 1500 episodi, alpha 0.2, gamma 0.95 (run storici del README §6):
- LLM 720 coppie: primo successo ep. 14, SR ultimi 100 a 1.00, eval 1.00, 13 passi (ottimo).
- LLM 50 coppie: primo successo ep. 40, eval 1.00, 13 passi.
- da zero: primo successo ep. 127, SR ultimi 100 a 0.07, eval 0.00.
Dettaglio da dire: le stime usano gamma 0.99, l'agente 0.95. Init approssimata due volte, funziona lo stesso.
Figure da mettere: `src/graph/data/qtable_llminit_seed_1337_gpt-oss_120b_llminit.png` (reward, successo train vs eval, lunghezza episodio, TD-error) e `..._vs_samehp.png` per il confronto diretto.

## Slide 9 - Iperparametri e perché funzionano (1 min 30)

I grafici buoni usano alpha 0.07, gamma 0.9, eps_decay 0.99, eps_min 0.05, 2000 episodi. Perché reggono:
- gamma 0.9 e non 0.99: con 0.99 due stati vicini differiscono di un fattore 0.99, il paesaggio dei valori è piatto e la greedy sceglie a fatica. Con 0.9 il gradiente è ripido e le decisioni diventano nette prima. Lo sweep del README §6.1 lo misura: gamma 0.99 consolida solo a ep. 770 con percorso da 16 passi, gamma 0.9 con alpha 0.5 arriva a ep. 286 con 13 passi ottimi.
- alpha basso (0.07): la tabella parte già vicina al giusto grazie all'init, quindi servono passi piccoli che rifiniscono senza distruggere il prior. Alpha 0.9 parte veloce ma oscilla intorno ai valori iniziali e consolida dopo (350 contro 289 episodi).
- eps_decay 0.99 con minimo 0.05: esplorazione lunga ma non infinita, il minimo evita di bloccarsi presto su una greedy ancora grezza. Max steps 450 lascia spazio alle deviazioni dei primi episodi (l'ottimo è 13 passi, all'inizio servono centinaia).
Prova che il merito è dell'init e non degli iperparametri: a pari iperparametri il vanilla da zero non risolve (SR 0.00, 450 passi medi, file `..._vsame.json`). Il vanilla standard con altri iperparametri risolve ma con 16 passi, peggio dei 13-17 dell'init.
Comando in README §10. Se hai tempo mostra solo la tabella dello sweep, non i json (che tra l'altro non sono versionati).

## Slide 10 - Dove si rompe, parte 1 (1 min)

Due limiti, tutti misurati:
- Nuovo layout: train su 5 seed, eval su seed mai visto. SR in-seed 1.00 con 25 passi, su nuovo seed 0.01. File `qtable_llminit_multiseed_*.json` + png. La coordinata relativa non basta, muri e strozzature cambiano. È il limite principale.
- Prossimo passo naturale: allargare il pool di seed e di stati. Più seed coprono più righe della tabella relativa e più bottleneck visti dall'LLM. Resta da provare, non da promettere: la coordinata relativa riduce lo spazio ma non descrive dove sono muri e strozzature, quindi più dati aiutano solo se la rappresentazione regge. Conta anche il costo, una query LLM per stato.
- Pilota fortunato: sui seed 47952 e 74839 il goal arriva al primo episodio per caso, la finestra vale subito 1.0 e i 5 checkpoint collassano sullo stesso istante (pool da 62 e 88 stati). Quei checkpoint fotografano rumore, non apprendimento. Greedy finale a 0.0.
Mostra il png del multiseed. Prenditi tempo qui, è la slide che il relatore ricorda.

## Slide 11 - Dove si rompe, parte 2 + cenno a FrozenLake (1 min 30)

- DDQN: pretraining a MSE 0.003 dopo 500 epoche, poi 3000 episodi online con replay esperto. 965 successi in train ma SR 0.00 in eval su 200 episodi. La rete copia i valori senza fissare la policy. Il risultato positivo vale per il tabellare.
- FrozenLake, cenno veloce: stessa idea su `FrozenLake-v1` 8x8 slippery, ambiente stocastico senza stadi. Pipeline pronta (grafo MDP, query bottleneck, eval, agente in `src/agent/frozenlake_qtable_llminit.py`). Log in `evaluate/output_frozenlake/log_valutazione.txt`, gpt-oss 120b su 18 stati bottleneck: MAE 0.1228, acc tie-aware 55.6% contro 25% random, regret 0.0060. Numeri deboli e su pochi stati, dati non versionati: citalo come lavoro aperto, non come risultato. Se chiedono perché va peggio, risposta onesta: valori stocastici Q* con slip, una sola cifra di stima per tre esiti possibili.

## Slide 12 - Chiusura (30 secondi)

Ripeti tre numeri: MAE 0.0186 con 90.3% di scelte giuste, primo successo da 127 a 14 episodi, eval da 0.00 a 1.00.
Poi il limite: niente trasferimento a layout nuovi.
Chiudi con cosa faresti dopo: copertura su mappe più grandi, costo query, IQM su più run (Henderson et al. 2018, citato in README §11 punto 11).

## Checklist figure

Da `evaluate/output/grafici/`: scatter, residuals, accuracy, mae_bucket. Confusion e k_error solo se chiedono.
Da `src/graph/data/`: `qtable_llminit_seed_1337_gpt-oss_120b_llminit.png`, `..._vs_samehp.png`, `qtable_llminit_multiseed_train47952-69896-74839-80970-93045_evalin47952_evalnew93046_llminit.png`.
Log da avere aperti: `evaluate/output/log_valutazione.txt` righe 11-19 e 21-34.
