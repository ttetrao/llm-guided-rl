# I grafici LLM-init vs Vanilla — materiale per il colloquio

> **Numeri verificati il 27/09 2026 (commit `1048dc4`).**
> Ricalcolati dai JSON con la stessa media mobile dei plot (`ma` finestra 100,
> `src/evaluate/llminit_speedup.py:27-45`). Se un numero qui non torna con il
> comando di §9, il comando ha ragione e questo file è vecchio: i run degli
> agenti sono stati rilanciati più volte durante la preparazione e i valori
> cambiano a ogni rilancio.

Copre le **sei** figure di confronto già incluse nella tesi
(`thesis/portable/doc.tex`, Figure 7, 8, 10, 11, 14, 15): tre gruppi di run ×
due coppie. Non copre i PNG `*_speedup.png` (vedi §8, punto 4) né gli
`llminit_sweep_*` (vedi §6, risposta alla domanda 2).

---

## 1. Grammatica del grafico (una volta sola)

Tutte e sei le figure hanno lo stesso identico impianto 2×2, generato da
`plot_cmp` (`src/agent/doorkey_qtable_llminit.py:383`, gemello in
`src/agent/frozenlake_qtable_llminit.py:363`), `figsize=(14,9)`, `dpi=150`.

| pannello | cosa mostra | come leggerlo |
|---|---|---|
| **alto-sx** | tasso di successo in addestramento, media mobile su 100 episodi | è la curva dell'efficienza di campionamento. Una `axhline` tratteggiata per ogni run segna la **SR greedy finale** (100 episodi, seed fisso) |
| **alto-dx** | errore TD medio per episodio, media mobile 100 | `|TD|` medio per passo. Scende quando l'agente smette di essere sorpreso |
| **basso-sx** | barre: **accordo** con la policy ottima (tie-aware, ε=1e-6) | quanto spesso l'argmax della tabella appresa coincide con l'argmax della value iteration. 1.0 = perfetto |
| **basso-dx** | barre: **perdita media di valore** | `V*(s) − (r + γ·V*(s′))` dell'azione che l'agente sceglie in quello stato. Più basso = meglio |
| **box in basso** | hp, n_init, file input, SR/rew/len dell'eval, agreement/loss, n stati scorsi | **è la fonte autorevole**: se il testo e il box del grafico non concordano, il box ha ragione |

Cose da sapere prima di guardare i numeri, e che nessun pannello mostra:

- **`ma100` non è una media a 100 fissi.** È una finestra *crescente* nei primi
  100 episodi, poi fissa a 100 (`doorkey_qtable_llminit.py:367-374`). Serve a far
  partire le curve da 0 invece che da un valore gonfiato dai primi 100 punti.
- **Le barre hanno denominatori diversi.** L'accord e la perdita si calcolano
  solo sugli stati visitati *e* mappabili nel MDP, quindi ogni run ha un
  denominatore proprio: su DoorKey 1337 sono 285 / 289 / 275 stati per
  LLM-init / same-hp / std. Le barre sono confrontabili come stima, non come
  identità.
- **Non ci sono bande di errore, repliche, o bande di confidenza.** Un solo
  seed per configurazione: questi grafici mostrano un effetto, non la sua
  varianza.
- **Gli stati terminali sono esclusi** dal calcolo di accordo e perdita
  (`doorkey_qtable_llminit.py:201-203`): lì `Q*` è tutto zero e l'accordo sarebbe
  banale.
- **La linea tratteggiata non è "la convergenza"**: è la valutazione greedy
  finale, 100 episodi sullo stesso seed, con policy deterministica. Può stare
  sopra la coda della curva di addestramento (che include ancora esplorazione
  casuale e, su FrozenLake slippery, il rumore dello scivolamento).

---

## 2. Le due coppie: cosa è controllato e cosa non lo è

Questo è l'argomento che regge tutta la difesa, e va detto **per primo**.

| coppia | cosa differisce | che tipo di esperimento è |
|---|---|---|
| `_vs_samehp` | **solo l'inizializzazione.** Stessa tabella, stessi α/γ/decay/εmin, stesso seed, stessi episodi | **controllato.** È l'unico confronto che attribuisce un effetto alla sola inizializzazione |
| `_vs_std` | **inizializzazione + iperparametri.** Il Vanilla ha α e γ diversi | **non controllato.** Una differenza qui è un mix di due cause; attribuirla all'init è un errore |

Attenzione alla terza colonna della tabella qui sotto: **"Vanilla-std" non
significa sempre la stessa cosa** nei tre gruppi. La sigla vuol dire "una
seconda configurazione di parametri", e in due casi su tre i valori non sono
quelli standard del pilota (`STD` in `doorkey_qtable_llminit.py:46` e
`frozenlake_qtable_llminit.py:45`):

| gruppo | Vanilla-std usato | è lo `STD` del pilota? |
|---|---|---|
| DoorKey 1337 | α=0.25, γ=0.99, decay=0.998, εmin=0.05 | **sì**, esattamente |
| DoorKey multi-seed | α=0.15, γ=0.99, decay=0.9995, εmin=0.05 | **no** (STD sarebbe α=0.25) |
| FrozenLake | α=0.1, γ=0.99, decay=0.998, εmin=0.01 | **no** (STD sarebbe decay=0.9995) |

Se ti chiedono "e il Vanilla con i parametri standard?", la risposta è: nel
gruppo DoorKey 1337 sì, negli altri due no — lì il confronto è con una seconda
configurazione, scelta diversamente nel contesto di quel run.

---

## 3. I sei grafici, uno per uno

Numeri di riferimento (media mobile 100, soglia **SR ≥ 0.8**; soglia 0.9 solo
dove arriva):

| gruppo | run | episodi alla soglia 0.8 | passi cumulati | AUC-SR | SR greedy finale | passi finali | accord | perdita |
|---|---|---|---|---|---|---|---|---|
| **DK 1337** | LLM-init | **414** | 151 715 | 0.796 | 1.000 | 13 | 0.565 | 0.0072 |
| | Vanilla (same hp) | **mai** | — | 0.006 | 0.000 | 450 | 0.498 | 0.0080 |
| | Vanilla-std | 1308 | 546 119 | 0.365 | 1.000 | 16 | 0.556 | **0.0066** |
| **DK multi** | LLM-init | **2011** | 728 350 | 0.681 | 1.000 | 21 | **0.671** | **0.0038** |
| | Vanilla (same hp) | **mai** | — | 0.005 | 0.000 | 450 | 0.464 | 0.0079 |
| | Vanilla-std | 4371 | 1 534 499 | 0.325 | 1.000 | 23 | 0.588 | 0.0064 |
| **FL slippery** | LLM-init | **461** | 26 394 | 0.888 | 1.000 | **54** | 0.580 | 0.0136 |
| | Vanilla (same hp) | 1422 (0.9: 1603) | 35 689 (0.9: 38 669) | 0.645 | 1.000 | **15** | 0.358 | 0.0454 |
| | Vanilla-std | 1968 (0.9: 2016) | 90 333 (0.9: 93 201) | 0.626 | 1.000 | 22 | 0.472 | 0.0338 |

**Attenzione a due celle**, perché sono quelle che ti salteranno all'occhio e su
cui ti potrebbero chiedere conto:

- **DoorKey 1337 / Vanilla-std: perdita 0.0066 < 0.0072 dell'LLM-init.** L'init
  vince su 2 dei 3 confronti di perdita di valore, non su 3.
- **FrozenLake / LLM-init: 54 passi finali contro 15 del same-hp.** L'init
  arriva prima *e cammina più lento*.

---

### 3.1 DoorKey 8x8, seed 1337 — `_vs_samehp`

Riquadro: 2000 episodi, α=0.04, γ=0.99, decay 0.99, εmin 0.05, limite 450 passi,
20 coppie iniziali, eval 100 episodi, `gpt-oss-120b`.

- **Curva LLM-init**: resta sotto 0.04 fino a ~200 episodi, poi sale: 0.68 a 400,
  0.98 a 600, e da lì un pianoro stabile a 0.97-0.98 fino alla fine. La salita è
  un gradino, non una rampa.
- **Curva same-hp**: piatta a 0.00-0.01 per tutti i 2000 episodi. Lunghezza
  media degli episodi 448.9 su 450: **ogni episodio finisce al timeout.** Non è
  un caso lento, è un agente che non ha mai toccato il goal in modo utile.
- **Pannelli bassi**: accordo 0.565 contro 0.498, perdita 0.0072 contro 0.0080.
  Entrambi peggiori per l'init, e la differenza è dello stesso ordine di
  grandezza della differenza tra i due Vanilla (0.498 vs 0.556).
- **Cosa non si conclude**: niente su α=0.04 in generale. Il same-hp potrebbe
  essere lento perché α=0.04 è troppo basso per 2000 episodi, non (o non solo)
  perché manca l'inizializzazione. È esattamente l'obiezione giusta, e la
  risposta è in §6 domanda 2.

### 3.2 DoorKey 8x8, seed 1337 — `_vs_std`

Stesse due curve più il Vanilla-std (α=0.25, γ=0.99, decay 0.998, εmin 0.05),
stesso seed, stessi 2000 episodi.

- **Curva Vanilla-std**: piatta a 0.01-0.04 fino a ~1200 episodi, poi **un salto
  improvviso** fra 1200 (0.04) e 1400 (0.96), e un pianoro a 0.93-0.98. Una
  curva a due fasi, tipica di un α alto che accumula in silenzio e poi sfonda.
- **Lunghezza media episodi**: 95.7 (init) contro 286.9 (std) contro 448.9
  (same-hp). Anche in addestramento, non solo in valutazione, l'init percorre
  percorsi molto più brevi.
- **Pannello basso-dx**: qui il Vanilla-std batte l'init, 0.0066 contro 0.0072.
- **Cosa non si conclude**: questo confronto cambia due variabili insieme. Se il
  vantaggio è dell'init o di α=0.04, **non lo distingue**. Dirlo è gratis e ti
  fa guadagnare credibilità per tutto il resto.

### 3.3 DoorKey 8x8, multi-seed — `_vs_samehp`

Riquadro: 5000 episodi, α=0.15, γ=0.95, decay 0.995, 400 coppie iniziali, un
seed per episodio pescato da {13507, 19920, 32455, 54937, 69693}. Valutazione
riportata sul seed **13507**, che è dentro l'addestramento.

- **Curva LLM-init**: 0.06 a 500, 0.18 a 1000, 0.60 a 1500, 0.79 a 2000, 0.92 a
  2500, 0.95-0.98 da 3000 in poi. Rampa graduale, niente salto.
- **Curva same-hp**: 0.00-0.02 per tutti i 5000 episodi, lunghezza media 449.2 su
  450. **Di nuovo, mai un successo utile.**
- **Pannelli bassi**: accordo 0.671 contro 0.464, perdita 0.0038 contro 0.0079.
  Qui il divario è netto: +0.21 di accordo, perdita dimezzata. È il confronto più
  pulito del gruppo, perché i due agenti hanno visitato un numero di stati simile
  (613 contro 638) e la differenza non è un artefatto del denominatore.
- **Il seed non visto**: la valutazione sul 69694 dà **SR=0.000 e 450 passi per
  tutti e tre gli agenti**, LLM-init compreso. Va detto, non nascosto (§5, punto 4).

### 3.4 DoorKey 8x8, multi-seed — `_vs_std`

Le stesse due curve più il Vanilla-std di questo run, che **non** è lo STD del
pilota: α=0.15 (uguale all'init), γ=0.99, decay 0.9995, εmin 0.05.

- **Curva Vanilla-std**: sale a 0.12 a 2000, 0.44 a 2500, 0.48 a 3000, 0.69 a
  3500, **ricade a 0.11 a 4000**, poi 0.79 a 4500 e 0.81 a 5000. Instabile.
- **Il vantaggio dell'init non è solo la velocità, è la stabilità**: la curva
  dell'init è monotona e ferma a 0.97, quella del Vanilla-std oscilla e non
  raggiunge la soglia 0.9 in 5000 episodi. Chi guarda le due figure in fila vede
  subito la differenza di comportamento, non solo di tempo.
- **Pannelli bassi**: accordo 0.671 contro 0.588, perdita 0.0038 contro 0.0064.
  Entrambi a favore dell'init, con l'init che è il best complessivo dei tre run
  del gruppo.
- **Correzione importante**: la caption attuale della tesi (Figure 11) dice che
  questo Vanilla-std finisce a SR=0.000 e 450 passi. Il JSON dice **SR=1.000,
  23 passi**. Se ti presentano quella figura, usa i numeri del box (§8, punto 1).

### 3.5 FrozenLake 8x8 slippery, seed 1337 — `_vs_samehp`

Riquadro: 5000 episodi, α=0.05, γ=0.99, decay 0.99, εmin 0.01, **nessun limite
di passi**, 76 coppie iniziali, 19 stati valutati, `gpt-oss-120b`.

- **Curva LLM-init**: 0.31 a 313, 0.46 a 461, **0.94 già a 500 episodi**, 0.99 a
  1000. Poi oscilla fra 0.78 e 0.98 fino alla fine. L'oscillazione è il rumore
  dello slippery con l'esplorazione residua (εmin=0.01): non è instabilità
  dell'apprendimento.
- **Curva same-hp**: 0.00 fino a ~1000 episodi, poi 0.87 a 1500 e un pianoro
  instabile fra 0.87 e 0.95. Qui il Vanilla **converge**, diversamente da
  DoorKey.
- **Pannelli bassi**: accordo 0.580 contro 0.358 (**+0.22**, il divario più
  grande dei sei confronti), perdita 0.0136 contro 0.0454 (**un terzo**). Su
  questo ambiente l'inizializzazione non accelera solo: migliora
  nettamente la tabella appresa.
- **La controintuitività**: la valutazione greedy finale dà **54 passi medi con
  l'init contro 15 senza**. Entrambe SR=1.000. Se te lo chiedono, la risposta è
  in §6 domanda 5.
- **Nota di metodo**: su slippery la media mobile non raggiunge mai 1.0 pulita
  (oscilla 0.88-0.98) mentre la valutazione greedy dà esattamente 1.000. Non è
  una contraddizione: la valutazione è greedy su un seed fisso, la curva di
  addestramento include l'1% di azioni casuali e il rumore dello scivolamento.

### 3.6 FrozenLake 8x8 slippery, seed 1337 — `_vs_std`

Le stesse curve più il Vanilla-std di questo run: α=0.1, γ=0.99, decay 0.998,
εmin 0.01 (decay diverso dallo `STD` del modulo, che sarebbe 0.9995).

- **Curva Vanilla-std**: 0.00 a 500, 0.01 a 1000, 0.42 a 1500, 0.89 a 2000, poi
  0.88-0.95. Converge, ma con il ritardo più lungo dei tre.
- **Pannelli bassi**: accordo 0.580 contro 0.472, perdita 0.0136 contro 0.0338.
  Anche qui l'init vince su entrambi.
- **È il confronto con il divario più netto in assoluto**: 4.27× in episodi e
  3.42× in passi cumulati per raggiungere la soglia 0.8, o 4.11× e 3.29× per la
  soglia 0.9.

---

## 4. Cosa funziona: i sei punti, ciascuno con la sua prova

1. **L'efficienza di campionamento migliora ovunque, di un fattore 2-4.** Alle
   soglie: DoorKey 1337 414 contro 1308 episodi (3.16×) e 151 715 contro 546 119
   passi (3.60×); DoorKey multi 2011 contro 4371 (2.17×) e 728 350 contro
   1 534 499 (2.11×); FrozenLake 461 contro 1422 (3.08×) e 26 394 contro 35 689
   (1.35×), 1968 contro l'init con 4.27× in episodi contro Vanilla-std. Nei due
   confronti *same-hp* su DoorKey il Vanilla **non arriva mai** alla soglia.
2. **L'AUC-SR vince 3 su 3.** È un numero solo che somma tutta la curva, quindi
   regge anche a chi obietta che la soglia è arbitraria: 0.796 / 0.681 / 0.888
   contro 0.006 / 0.005 / 0.645 (same-hp) e 0.365 / 0.325 / 0.626 (std).
3. **L'accordo con la policy ottima vince 3 su 3.** 0.565 vs 0.556, 0.671 vs
   0.588, 0.580 vs 0.472 contro i rispettivi Vanilla-std, e +0.07 / +0.21 / +0.22
   contro i same-hp. L'init non solo arriva prima: **la tabella che produce è
   più vicina all'ottimo**, e su FrozenLake il divario è netto.
4. **L'effetto si replica su due ambienti e in due regimi.** GridWorld
   deterministico e labirinto stocastico, addestramento su un seed e su cinque,
   20 e 400 coppie iniziali. Non è un artefatto di una configurazione.
5. **Il meccanismo è spiegabile e verificabile, non solo empirico.** Con
   `Q ≡ 0` e γ<1 l'unico target non nullo è `r + γ·max Q'(s')`, cioè zero finché
   l'agente non tocca il goal: nessuna ricompensa, nessun gradiente, greedy
   tutta da pareggi, scelta casuale. L'unica via d'uscita è la fortuna. I due
   controlli same-hp che restano a zero su DoorKey **sono** quel deadlock,
   visibile in grafico.
6. **La valutazione greedy finale è 1.000 in 5 confronti su 6**, e l'unica
   eccezione è il caso peggiore per l'init (il same-hp di DoorKey, che non ha
   mai imparato). Nessun run regredisce rispetto a un Vanilla.

## 5. Cosa è migliorabile: sei punti, ciascuno con la frase di mitigazione

1. **Un solo seed per configurazione, nessuna stima della varianza.** Non ci
   sono repliche, quindi i confronti sono evidenza di fattibilità, non un test
   statistico.
   *«Non ho stimato la varianza perché ogni configurazione è stata eseguita una
   volta sola. È il primo limite dichiarato e il primo lavoro futuro.»*
2. **I confronti `_vs_std` confondono inizializzazione e iperparametri.** Solo
   `_vs_samehp` è un esperimento controllato, e peraltro il controllo same-hp non
   converge su DoorKey, quindi non misura "init migliore di no-init in regime
   convergente" ma "init vs deadlock".
   *«I due confronti servono a misurare cose diverse: il controllo a parità di
   parametri isola l'inizializzazione, quello con i parametri standard risponde
   alla domanda che mi facevo prima, "quanto bene fa il Q-learning di
   manuale". Li presento come due esperimenti, non come due conferme.»*
3. **Su FrozenLake l'init arriva prima ma cammina più lento in valutazione
   finale: 54 passi contro 15.** Entrambe le policy ottime, quindi la metrica
   giusta è il tempo alla soglia, non la lunghezza finale — ma il dato è
   controproducente e va mostrato per primo da te, non scoperto dalla
   commissione.
   *«Con 5000 episodi e γ=0.99 i tre agenti convergono a un ottimo diverso fra
   policy ottime equivalenti. Su 100 episodi di valutazione la differenza di
   lunghezza è rumore; la differenza che misura qualcosa è 461 contro 1422
   episodi per arrivare a SR≥0.8.»*
4. **La generalizzazione non è risolta: sul seed non visto 69694 tutti e tre gli
   agenti danno SR=0.000 e 450 passi.**
   *«Il vantaggio che ho misurato è sui layout incontrati. Il seed non visto dà
   zero a tutti, quindi il lavoro non dimostra generalizzazione: è un limite del
   metodo e non dell'esperimento, perché l'ho misurato esplicitamente invece di
   ometterlo.»*
5. **La perdita di valore è persa in un confronto su tre** (DoorKey 1337:
   0.0066 std contro 0.0072 init). L'accordo vince 3 su 3, la perdita no.
   *«Su quel gruppo il Vanilla con α=0.25 ha una perdita di valore migliore ma
   arriva alla soglia tre volte più tardi: miglior tabella e peggior efficienza
   non vanno nella stessa direzione, e per un agente che deve imparare conta il
   secondo.»*
6. **I cinque layout del multi-seed condividono una sola tabella Q: non sono
   cinque repliche indipendenti dell'esperimento, sono cinque mappe su un agente.**
   *«I cinque semi danno varietà sui layout allenati, non varietà sull'esito
   sperimentale. Per quello servirebbero repliche indipendenti con lo stesso
   budget.»*
   *Aggiunta di precisione sulle barre*: accord e perdita sono calcolati su
   denominatori diversi per ogni run (275-337 stati nei tre gruppi), quindi la
   differenza più piccola (DK 1337: 0.565 vs 0.556) è dentro il rumore di
   denominatore. I confronti solidi sono quelli con divario ≥ 0.2.

## 6. Domande del commissione → risposta

**1. «Tutti e tre finiscono a SR=1.000. Allora l'inizializzazione serve a
nulla?»**
No: il confronto giusto non è la SR finale, che per costruzione converge, ma il
tempo alla soglia. FrozenLake: 491 contro 1603 e 2016 episodi per SR≥0.9; DoorKey
1337: 414 contro 1308, e il controllo a parità di parametri non ci arriva proprio.
La destinazione la raggiungono tutti, il tempo cambia di 2-4×.

**2. «Il vantaggio è l'inizializzazione o α basso? Non potete distinguerli.»**
La prima metà la distingo. Nei confronti a parità di parametri l'unica differenza
fra le due curve è l'inizializzazione, e l'effetto c'è: 414 contro mai e 2011
contro mai. La seconda metà l'ho affrontata con uno sweep sul numero di coppie
inizializzate (`src/agent/llminit_sweep.py`, con il caso `n_init=0` come
controllo a Q≡0): esiste proprio per separare le due cause, e la tesi lo dichiara
come lavoro in corso.

**3. «Perché su FrozenLake funziona meglio che su DoorKey?»**
Perché il collo di bottiglia è più duro. Su FrozenLake slippery un percorso
ottimo è lungo e ogni passo può scivolare, quindi la finestra di ricompensa è
rarissima e il deadlock da Q≡0 è severo: l'init ha molto più margine. Su DoorKey
il compito è più guidato. E infatti su FrozenLake il divario sull'accordo con
la policy ottima è il più grande dei sei confronti (+0.22).

**4. «Il vantaggio è robusto o è un caso?»**
Quello che regge è il meccanismo, non il numero. Il deadlock da tabella a zero
è dimostrabile e lo si vede nei due controlli same-hp che restano a zero. Il
fattore 2-4× è una singola tiratura per configurazione: non presentarlo come
statistica.

**5. «Su FrozenLake il vostro agente impiega 54 passi contro 15. È peggio.»**
Entrambe le policy sono ottime, SR=1.000 in entrambi i casi, quindi non è un
peggioramento dell'apprendimento: a γ=0.99 e 5000 episodi si converge a un
ottimo diverso fra policy ottime equivalenti. Su 100 episodi la differenza di
lunghezza è rumore. La metrica che distingue le due è il tempo alla soglia, e
lì l'init vince 3.08×.

**6. «I valori dell'LLM sono sbagliati in assoluto (MAE 0.17), come fate a
dire che funziona?»**
Perché l'inizializzazione usa l'**ordine** delle stime, non la scala. L'errore
dell'LLM è in gran parte una costante per stato, che non cambia l'argmax; la
policy iniziale non deve essere buona, deve solo **far arrivare il primo successo**,
e da lì il valore si propaga a ritroso. L'apprendimento poi cancella l'errore di
calibrazione — ma non cancella i primi 500 episodi.

**7. «Come sapete che l'init non vi dia solo una policy fortuna?»**
Tre filtri indipendenti, nessuno dei quali dipende dalla lunghezza finale:
l'accordo con la policy ottima calcolata per value iteration (3 su 3), la perdita
di valore contro V* (2 su 3), e la lunghezza media degli episodi **durante
l'addestramento** su DoorKey (95.7 con init contro 286.9 e 448.9 senza).

**8. «Il gioco è stocastico, come fate i confront?»**
Seed fisso 1337 per addestramento e valutazione, così l'unica differenza fra le
curve è l'inizializzazione. Lo slippery resta dentro il modello: è l'ambiente a
essere casuale, non il confronto. È un limite, e con un solo seed non si stima la
varianza.

## 7. Cosa NON dire

- **Non presentare lo speedup come statistica.** Un seed per configurazione.
- **Non dire che l'inizializzazione migliora la policy finale.** Non lo fa e non
  può: su FrozenLake allunga il percorso finale. Riduce il tempo per arrivarci.
- **Non leggere le curve `_vs_std` come un effetto dell'inizializzazione.**
  Cambiano due variabili.
- **Non usare le figure `*_speedup.png`.** Non contengono la curva LLM-init (§8,
  punto 4): mostrano solo i due Vanilla e non possono reggere un'affermazione di
  speedup.
- **Non citare le caption della tesi come fonte dei numeri.** Contraddicono i
  box dei grafici (§8, punto 1). La fonte autorevole è il box in basso a ogni
  figura, che è generato dagli stessi JSON.

## 8. Checklist prima della discussione

1. **Le caption della tesi contraddicono le loro figure.** I numeri da correggere,
   con la riga di `thesis/portable/doc.tex` (e `thesis/doc.tex`, che è
   allineato):
   - **riga 199 (fig:samehp)**: dice "entrambi gli agenti hanno SR=1.00, 13 e 16
     passi, accordo 0.576 contro 0.527". La figura dice `Vanilla (same hp)
     (eval 0.00)`, SR=0.000, 450 passi, accordo 0.498. Dalla tesi: **entrambi
     SR=0.00, 450 passi, accordo 0.565 contro 0.498**, e l'init raggiunge la
     soglia a 414 episodi contro "mai".
   - **riga 206 (fig:vstd)**: accordo e perdita da correggere come sopra
     (0.565 / 0.556, perdita 0.0072 / 0.0066), γ da 0.95 a 0.99.
   - **riga 229 (fig:multivstd)**: dice "SR=0.000, len=450, agreement=0.587,
     loss=0.0067" per il Vanilla-std. Il JSON dice **SR=1.000, 23 passi, accordo
     0.588, perdita 0.0064**.
   - **righe 281 e 288 (fig:fl_samehp, fig:fl_vstd)**: dicono accordo 0.490 /
     0.0252 e 17 passi finali. La figura dice **accordo 0.429, perdita 0.0292,
     59 passi** (è una run diversa da quella dei JSON attuali, che ora danno
     0.580 / 0.0136 / 54 passi).
   - **righe 30, 44, 188, 237**: γ=0.95 nel testo contro **γ=0.99** nei dati.
2. **`doc.tex` e `portable/doc.tex` non sono allineati**: la riga 170 riporta
   144 stati / Top-1 96.5% in uno e 85 stati / 94.1% nell'altro. Se leggi
   entrambi durante la discussione, scegline uno.
3. Le due figure FrozenLake incluse nella tesi sono copie di versioni **precedenti**
   dei PNG in `src/output/agents/`: se rilanci gli agenti, le copie in
   `thesis/portable/img/agents/` non si aggiornano da sole e i numeri del box
   cambieranno senza che la figura cambi.
4. **Bug noto, non ancora corretto**: nei PNG `*_speedup.png` la curva LLM-init
   **non viene tracciata**. `llminit_speedup.py:189` costruisce il dizionario con
   le chiavi `llm/vsame/vstd`, mentre `plot_group` cerca `llminit` a `:104` e fa
   `continue` se non la trova. Le due speedup su disco mostrano solo i Vanilla.
   Non usarle finché non è risolto.
5. `llminit_speedup.py --selfcheck` **fallisce** con i JSON attuali: asserisce
   `ttt_ep_0.8 == 492` (`llminit_speedup.py:146`) e il valore reale è 414.

## 9. Ricalcolo dei numeri

I tre gruppi, rigenerati al volo. Il comando non modifica nulla: legge i JSON e
stampa la tabella di §3.

```bash
cd /home/pietro/Documenti/llm-guided-rl && python3 - <<'EOF'
import json, numpy as np
from pathlib import Path
A = Path("src/output/agents")

def ma(x, w=100):
    x = np.asarray(x, float)
    cs = np.cumsum(x); i = np.arange(len(x))
    st = np.clip(i - w + 1, 0, None)
    return (cs - np.where(st > 0, cs[st - 1], 0.0)) / (i - st + 1)

def ttt(s, L, thr, w=100):
    i = np.where(ma(s, w) >= thr)[0]
    return (int(i[0]) + 1, int(np.cumsum(np.asarray(L, float))[i[0]])) if len(i) else (None, None)

G = [("DK1337", "qtable_llminit_seed_1337_gpt-oss_120b"),
     ("DKmulti", "qtable_llminit_multiseed_train13507-19920-32455-54937-69693_evalin13507_evalnew69694"),
     ("FL1337", "frozenlake_qtable_llminit_8x8_slippery_seed_1337_gpt-oss_120b")]

for lab, stem in G:
    print("=" * 72, lab)
    for suf in ("llminit", "vsame", "vstd"):
        d = json.loads((A / f"{stem}_{suf}.json").read_text())
        hp, ev, op = d["hparams"], d["eval"], d["optimal"]
        e8, s8 = ttt(d["successes"], d["ep_lengths"], 0.8)
        e9, s9 = ttt(d["successes"], d["ep_lengths"], 0.9)
        print(f"  {suf:8s} ep={len(d['successes']):5d} n_init={d['n_init']:3d} "
              f"a={hp['alpha']} g={hp['gamma']} d={hp['eps_decay']} m={hp['eps_min']} "
              f"max_steps={hp['max_steps']}")
        print(f"           TTT0.8={e8}/{s8} TTT0.9={e9}/{s9} "
              f"AUC={np.mean(ma(d['successes'])):.3f} | eval SR={ev['sr']:.3f} "
              f"len={ev['mean_len']:.0f} | agr={op['agreement']:.3f} "
              f"loss={op['mean_value_loss']:.4f} su {op['n_visited_scored']} stati")
        if "eval_new" in d:
            print(f"           eval_new seed={d['eval_new_seed']} "
                  f"SR={d['eval_new']['sr']:.3f} len={d['eval_new']['mean_len']:.0f}")
EOF
```

E i comandi che rigenerano le figure. Sono **ricostruiti dagli hp contenuti nei
JSON**, non sono i comandi originali (i run sono stati rilanciati più volte), ma
sono verificati: producono lo stesso seed di addestramento, lo stesso
`eval_new_seed` e le stesse coppie iniziali dei JSON attuali.

```bash
cd src   # i moduli vanno lanciati da src/, non dalla radice

# DoorKey seed 1337 -> qtable_llminit_seed_1337_gpt-oss_120b_{vs_samehp,vs_std}.png
python3 -m agent.doorkey_qtable_llminit --compare \
  --input llm_results_doorkey_states_8x8_seed1337_gpt-oss_120b.json \
  --episodes 2000 --alpha 0.04 --gamma 0.99 --eps_decay 0.99 \
  --eps_min 0.05 --max_steps 450 --max_init 20

# DoorKey multi-seed -> qtable_llminit_multiseed_train13507-..._evalnew69694_*.png
#   train_seeds ed eval_new_seed (=69694) vengono scelti dal modulo, non vanno passati
python3 -m agent.doorkey_qtable_llminit --compare \
  --input llm_results_doorkey_states_8x8_seeds13507-19920-32455-54937-69693_gpt-oss_120b.json \
  --episodes 5000 --alpha 0.15 --gamma 0.95 --eps_decay 0.995 --max_init 400 \
  --std-alpha 0.15 --std-gamma 0.99 --std-eps-decay 0.9995

# FrozenLake 8x8 slippery -> frozenlake_..._seed_1337_gpt-oss_120b_{vs_samehp,vs_std}.png
python3 -m agent.frozenlake_qtable_llminit --compare --map 8x8 \
  --input frozenlake_llm_8x8_slippery_q_seed1337_gpt-oss_120b.json \
  --episodes 5000 --alpha 0.05 --gamma 0.99 --eps_decay 0.99 --eps_min 0.01 \
  --max_steps 0 --max_init 76 \
  --std-alpha 0.1 --std-gamma 0.99 --std-eps-decay 0.998 --std-eps-min 0.01
```

Attenzione: le flag dei parametri del Vanilla-std usano il **trattino**
(`--std-alpha`), mentre `--max_init`, `--eps_decay`, `--eps_min` e `--max_steps`
usano il **trattino basso**. Mescolarle fa fallire argparse.

Entrambi i moduli hanno `--selfcheck` (aggiungilo a uno dei comandi sopra per
provare senza addestrare): verifica la formula dell'aggiornamento, l'assegnazione
dell'inizializzazione, il determinismo del seed e il calcolo delle metriche di
ottimalità.
