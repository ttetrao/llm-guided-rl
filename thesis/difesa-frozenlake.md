# FrozenLake — materiale per il colloquio

> **Numeri verificati il 27/09 2026, 17:40.**
> I run degli agenti sono stati rilanciati più volte durante la preparazione (7000 episodi
> all'ultima lettura, prima erano 10000, prima ancora 5000) e i valori **cambiano a ogni
> rilancio**. Prima della presentazione ricalcola con i comandi di §6: ogni numero di
> questo file ha accanto il comando che lo rigenera. Se un numero qui non torna con il
> comando, il comando ha ragione e questo file è vecchio.

Ambiente: `FrozenLake-v1` 8x8 slippery, seed 1337. Modello: `gpt-oss-120b`.
Stato = intero 0..63, 4 azioni (left, down, right, up).

---

## 1. Il paradosso, in tre righe

Se il prof chiede "quindi l'inizializzazione funziona o no?", la risposta è una, prima che
la faccia lui:

> **Funziona, ma non perché l'LLM abbia indovinato i valori. L'ha indovinati male in
> assoluto e bene nell'ordine. E l'inizializzazione usa solo l'ordine.**

Tutto il resto del file è la prova di questa frase.

| cosa dice il prof | cosa rispondi |
|---|---|
| MAE 0.17, il modello sbaglia | sì, sulla scala: 0.17 è quasi tutto una costante per stato, che non cambia la scelta |
| Top-1 52.6% | sì, e il caso è 25%: è 2.1× il caso, e in regret il 16% di quello che butta via una moneta |
| e allora l'init funziona | funziona: 3.3× in episodi sul tempo alla convergenza, stesso SR finale |

---

## 2. I tre slide, testi pronti

Sostituiscono le slide 10-12 del pptx (`thesis/presentazione.pptx`), che oggi stanno nella
sezione "Limiti" e raccontano il contrario.

### Slide 1 — FrozenLake slippery: il paradosso
*sezione: Risultati → FrozenLake slippery*

- `FrozenLake-v1` 8x8 slippery, seed 1337: 64 stati × 4 azioni = **256 coppie**
- LLM su **19 stati bottleneck = 76 coppie** (30% della tabella)
- Qualità delle stime: **MAE 0.170**, bias +0.051, **Top-1 52.6%** (caso 25%)
- Eppure l'init dà **3.3× sul tempo alla convergenza**
- **La domanda della slide:** com'è possibile?

> *da dire:* "Qui l'errore sui valori è alto e l'azione giusta viene scelta solo in un caso
> su due. Eppure l'inizializzazione accelera l'apprendimento di tre volte. Il punto delle
> prossime due slide è che non sono due fatti in contraddizione."

**Figura:** `src/output/evaluate_frozenlake/grafici/frozenlake_llm_8x8_slippery_q_seed1337_gpt-oss_120b_accuracy.png`

### Slide 2 — Il 94% dell'errore non cambia la scelta
*sezione: Risultati → FrozenLake slippery*

| | |
|---|---|
| MAE grezzo | 0.170 |
| **MAE centrato per stato** (solo l'ordinamento fra le 4 azioni) | **0.034** |
| **quota dell'errore che è una costante per stato** | **94%** |

- spostare tutte le Q di `c` cambia il TD di `(γ−1)c` ≈ 5·10⁻⁴, e il terminale vale 0:
  l'offset sparisce in pochi update
- una costante su tutte e 4 le azioni **non sposta l'argmax**
- **regret** (la metrica che l'init usa davvero): LLM **0.0114** · azione casuale **0.0731**
  → butta via il **16%** di quello che butta via una moneta
- il 52.6% è a **toleranza zero**: a ε=0.01 è 78.9%, a ε=0.05 è 94.7%
- **8 dei 9 "errori" costano < 2.5%** del valore, e l'LLM non sceglie **mai** l'azione peggiore
- **10 stati su 19** hanno regret esattamente 0
- esempi: stato 17 (riga 2, col 1) V\*=0.394, l'LLM dice 0.842 · stato 39 (4,7) V\*=0.690,
  l'LLM dice 0.961 → **valore sbagliato, ordinamento giusto**

> *da dire:* "Il MAE misura la calibrazione, cioè la scala. L'inizializzazione usa solo
> l'ordinamento: quale delle quattro azioni metto più in alto. Sono due cose diverse, e
> qui divergono in modo netto: l'errore di scala è 5× più grande di quello di ordinamento."

**Figura:** `.../frozenlake_llm_8x8_slippery_q_seed1337_gpt-oss_120b_scatter.png`
(la nuvola fuori dalla diagonale *è* l'offset)

### Slide 3 — L'effetto è sulla velocità, non sulla qualità
*sezione: Risultati → FrozenLake slippery*

| run | episodi fino a SR≥0.9 | passi cumulati | **SR greedy finale** |
|---|---|---|---|
| LLM-init | **491** | 28 307 | 1.000 |
| Vanilla (stessi hp) | 1603 | 38 669 | 1.000 |
| Vanilla (hp standard) | 5270 | 249 792 | 1.000 |

- con Q≡0 la greedy è un pareggio uniforme → **nessun gradiente**: SR cumulativa a 500
  episodi = 0.002, a 1000 = 0.001, piatta
- rollout greedy sulla tabella **non addestrata**: con init **SR 4.1%**, con Q=0 **0.2%**
- l'init non deve essere una buona policy (il 95% dei suoi episodi finisce in un buco):
  deve **far arrivare il primo successo**, poi il resto lo fa Q-learning
- **tutti e tre arrivano a SR=1.000**: l'init cambia *quando*, non *cosa*

> *da dire:* "L'inizializzazione non deve risolvere il problema. Deve rompere il deadlock:
> con la tabella a zero l'agente non ha nessun segnale di ricompensa e non impara niente.
> Con l'init il primo successo arriva, e da lì il valore si propaga a ritroso. L'apprendimento
> poi cancella l'errore di calibrazione, ma non cancella i primi 500 episodi."

**Figura:** `src/output/agents/frozenlake_qtable_llminit_8x8_slippery_seed_1337_gpt-oss_120b_vs_samehp.png`

> **Attenzione — da 20 secondi, due fix sulle slide che già ci sono:**
> 1. sulla slide 11 c'è scritto **"MAE = 0.338"**: è il numero di **gemma**, ma le figure
>    sotto sono di **gpt-oss** (0.170). Correggi la label.
> 2. **togli** dalla slide 10 la frase "LLM fatica su ambienti non deterministici come
>    FrozenLake". Con questi tre slide il messaggio è l'opposto, e se la lasci lì il prof
>    ti obietta e non hai una slide per rispondere.

---

## 3. Le definizioni che ti serviranno

**Coppia (s,a) ≠ stato.** `max_init` conta coppie, non stati: `load_qinit` ordina le chiavi
`(state, action_idx)` e taglia `ordered[:max_init]`
(`src/agent/frozenlake_qtable_llminit.py:117-119`). Su FrozenLake 8x8: 64 × 4 = 256 coppie,
di cui l'LLM ne stima 76. Su DoorKey l'LLM dà 870 righe ma 720 coppie, perché le righe
identiche da posizioni assolute diverse collassano sulla stessa coppia relativa.

**Regret.** `regret(s) = max_a' Q*(s,a') − Q*(s,a')`, cioè quanto valore hai buttato via
scegliendo l'azione scelta invece della migliore. Nel codice è `value_loss`
(`src/evaluate/evaluate_frozenlake_llm.py:480`) ed è la stessa identica cosa del "value loss"
dei pannelli degli agenti.

*Perché serve invece dell'accuracy:* l'accuracy è tutto-o-niente. Se due azioni valgono 0.690
e 0.689, sbagliare la scelta ti costa 0.001 di valore ma l'accuracy ti conta un errore
pieno. Il regret è graduale, e risponde alla domanda giusta per un'init: non "ha scelto
l'azione giusta?" ma "ha scelto male?".

*Unità:* il regret è in **unità di valore**, non in percentuale. In questo 8x8 V\* sta tra
0.185 e 0.878, quindi 0.011 è un numero piccolo in assoluto. A contare è il confronto con
la baseline casuale, non il valore assoluto.

**Il punto cieco dell'accuracy.** L'accuracy che riporti è a **tolleranza zero**:
`accuratezza(ε) = P(gap ≤ ε)` con `gap = V*ottimo − V*(azione scelta)`, e `ε = 0` significa
che un'azione che manca l'ottima di 0.001 conta come errore pieno.

| ε | LLM | casuale |
|---|---|---|
| 0 (esatto) | **52.6%** | 0.0% |
| 0.001 | 57.9% | 0.0% |
| 0.005 | 68.4% | 10.5% |
| 0.010 | **78.9%** | 31.6% |
| 0.020 | 84.2% | 47.4% |
| 0.050 | **94.7%** | 52.6% |
| 0.100 | 94.7% | 63.2% |

Nota tecnica: nel repo `tie_eps=1e-6`, cioè **solo l'uguaglianza esatta** conta come pareggio.
In slippery i pareggi esatti non ci sono quasi mai (Q\* è stocastico e i valori sono
tutti diversi), quindi l'accuracy è di fatto un match esatto su una quantità continua.
Non è sbagliata: è più severa di quanto suggerisca il nome.

**Calibrazione vs ordinamento.** Il MAE è una metrica di calibrazione: dice quanto il
numerico è lontano dalla verità, scala compresa. Ma l'init non usa il numerico, usa
`argmax_a Q(s,a)`, e l'argmax è cieco a qualunque costante aggiunta a tutte le azioni.
Scomponendo l'errore per stato in "costante + ordinamento", la costante porta il 94%
dell'energia dell'errore, e l'ordinamento si sbaglia di 0.034 invece di 0.170.

**Deadlock di bootstrap.** Con Q≡0 e γ<1 l'unico target non nullo è `r + γ·max Q'(s')` nei
stati non terminali, cioè 0 finché l'agente non tocca il goal. Quindi: nessuna ricompensa,
nessun Q non nullo, nessun gradiente, e la greedy resta un pareggio uniforme (= casuale).
L'unica via d'uscita è la fortuna. È quello che si vede nella colonna "Vanilla" nei primi
1000 episodi: SR 0.001, piatta.

---

## 4. Le domande probabili, con la risposta

**"Perché il MAE è alto se l'inizializzazione funziona?"**
Perché sono due metriche diverse. Il MAE guarda il numero, l'init guarda solo quale delle 4
azioni è più alta. Il 94% dell'errore è una costante per stato, che non cambia l'argmax e
che Q-learning cancella da sola perché il terminale vale 0. La parte che cambia la scelta
sbaglia di 0.034, e il regret relativo è 0.011 contro 0.073 di un'azione casuale.

**"Che cos'è il regret?"** → vedi §3.

**"Il 52.6% è buono o scarso?"** Dipende dalla base: il caso è 25%, quindi è 2.1× il caso,
non il doppio. E in regret è 6.4× meglio del caso.

Ma la difesa vera è un'altra, e sta nel fatto che il 52.6% è a **toleranza zero**:
"non è l'azione migliore" vuol spesso dire "manca di 0.001". Su 19 stati l'LLM sbaglia
9 volte e:

- **8 di quelle 9 costano meno del 2.5% del valore** (mediano del gap 0.0074)
- **non sceglie mai l'azione peggiore**: 1ª in 10 stati, 2ª in 5, 3ª in 4, **4ª in zero**
- la media 0.0114 è **trainata da un solo stato** (il 21, gap 0.139 = 28% del valore):
  escluso quello la media degli altri 18 è **0.0043, il 6% del regret casuale**
- scegliendo *sempre* la 2ª migliore il regret sarebbe 0.0741 (6.5× peggio), *sempre* la
  peggiore 0.1223

I 9 stati con gap > 0, per intero:

| stato | V\* ottimo | scelta | **gap** | 2ª migliore | 4ª (peggiore) | % del valore | rank |
|---|---|---|---|---|---|---|---|
| 21 | 0.4938 | right | **0.1392** | 0.1392 | 0.1852 | 28.2% | 2ª |
| 22 | 0.5612 | up | 0.0248 | 0.0171 | 0.0304 | 4.4% | 3ª |
| 12 | 0.4832 | down | 0.0234 | 0.0121 | 0.0303 | 4.8% | 3ª |
| 9 | 0.4212 | down | 0.0110 | 0.0059 | 0.0144 | 2.6% | 3ª |
| 17 | 0.3938 | left | 0.0074 | 0.0074 | 0.0225 | 1.9% | 2ª |
| 8 | 0.4117 | down | 0.0059 | 0.0049 | 0.0081 | 1.4% | 3ª |
| 13 | 0.5135 | up | 0.0035 | 0.0035 | 0.0206 | 0.7% | 2ª |
| 23 | 0.5859 | right | 0.0013 | 0.0013 | 0.0234 | 0.2% | 2ª |
| 0 | 0.4146 | down | 0.0010 | 0.0010 | 0.0051 | 0.2% | 2ª |

*Frase pronta:* "Il 52.6% è un criterio a tolleranza zero: in slippery i valori delle quattro
azioni stanno tutti in una fascia stretta, e 'non è la migliore' vuol spesso dire 'mancano
di 0.001'. Su 19 stati l'LLM sbaglia 9 volte, ma 8 di quelle costano meno del 2.5% del
valore e non sceglie mai l'azione peggiore. Per questo uso il regret: il 52.6% risponde
a 'ha scelto l'azione giusta', il regret a 'ha scelto male'."

**"Perché solo 19 stati su 64? Non è una fetta minima?"** Sono i 19 stati bottleneck scelti
dal grafo (celle della chiave, della porta, adiacenti frontali, predecessori di pickup e di
ingresso al goal): sono le celle in cui la decisione cambia la fase del task. 19 su 53
camminabili. Il resto della tabella parte da zero ed è appreso normalmente, e per misurare
l'effetto dell'init quei 19 sono quelli che contano.

**"Quanti stati hai inizializzato?"** → **tre numeri diversi, non dare il primo che capita**:
- coppie inizializzate: **76** (`n_init` nel JSON)
- stati distinti coperti: **19**
- stati toccati dal training: **50** (`n_visited_states`)
Attenzione: la riga di log `stati: 53` che vedi in console stampa `len(agent.q)`, cioè gli
stati *visitati*, non quelli inizializzati.

**"L'effetto è robusto o è un caso?"** Onesto: è una **prova di fattibilità in singola
tiratura**. Un seed, 19 stati, un modello. Quello che regge è il meccanismo, non la stima:
l'offset viene cancellato, il primo successo arriva prima, e le curve di apprendimento
sono coerenti. Non è un test statistico e non lo presentare come tale.

**"Perché qui funziona meglio che su DoorKey?"** Perché il collo di bottiglia è più
duro. Su FrozenLake slippery un percorso ottimo è lungo e ogni passo può scivolare, quindi
la finestra di ricompensa è rarerissima e il deadlock del Q=0 è severo: l'init ha molto
più margine. Su DoorKey il compito è più guidato e l'init porta un vantaggio minore.

**"Tutti e tre finiscono a SR=1.000, allora l'init serve a nulla?"** Serve, e il punto è
esattamente che l'init non può cambiare la destinazione, solo il tempo per arrivarci. Il
confronto giusto non è lo SR finale (identico per costruzione: con 7000 episodi e un
α=0.05 tutti convergono) ma il tempo alla soglia: 491 contro 1603 e 5270 episodi.

**"Il gioco è stocastico, come fate i confront?"** Seed fisso (1337) per train ed eval, così
l'unica differenza fra le curve è l'inizializzazione. Lo slippery resta dentro il modello:
è l'ambiente che è casuale, non il confronto. È un limite: con un solo seed non si stima
la varianza.

**"Il modello sbaglia di 0.17 e sceglie bene: come è possibile?"** Con un esempio.
Stato 17 (riga 2, col 1): il valore vero della sua azione migliore è 0.394, l'LLM stima
0.842. La stima è sbagliata di 0.45 e sarebbe un disastro se usata come valore. Ma le
*altre tre* azioni di quello stato hanno stime tutte più basse, quindi l'argmax cade
sull'azione giusta: regret 0.0074. Lo stesso per lo stato 39: V\*=0.690, stima 0.961,
regret 0. Il numero assoluto è rumore, l'ordinamento è segnale.

**"Perché l'init non è solo una euristica tipo 'vai verso il goal'?"** Perché l'euristica
non funziona in slippery: nel rollout la mia euristica più semplice dà SR 0.4%, e
l'inizializzazione dell'LLM dà 4.1%, dieci volte tanto. Il vantaggio non è "sa dove sta
il goal" (lo sa chiunque legga la mappa), è che sa **da quale lato conviene aggirare i
buchi**, che è l'informazione che l'apprendimento fa più lentamente.

---

## 5. Le trappole: dove i tuoi numeri non tornano fra loro

Non sono errori di calcolo, sono file e versioni diverse. Se il prof li mette a confronto
devi saperli spiegare.

1. **`src/output/evaluate_frozenlake/log_valutazione.txt` è vecchio.** Dice MAE 0.1228,
   Top-1 55.6%, 18 stati / 72 righe, e punta a `graph/data/...`. Il JSON corrente dà
   **0.1695 / 52.6% / 19 stati / 76 righe**. Il log è del 23/09, su una versione precedente
   del file LLM. Se citi 0.1228 devi dire da dove viene.

2. **Il "MAE = 0.338" della slide 11 è di gemma, non di gpt-oss.** Le due figure sotto
   quella slide sono di gpt-oss (`frozenlake_llm_..._gpt-oss_120b_*.png`). 0.338 è il MAE
   di gemma. Curioso: gemma ha il MAE peggiore (0.338, Pearson −0.16, R² 0.03) ma
   l'accuratezza migliore (77.8%). Se te lo fa notare: con 19 stati e valori quasi tutti
   sotto 0.1 la differenza è nel rumore, e l'accuratezza lì è gonfiata dai pareggi
   (28% dei top dell'LLM sono pareggi contro 11% di gpt-oss).

3. **Il report (`thesis/portable/doc.tex`, appendice FrozenLake) è su una run diversa.**
   Dice 3000 episodi, α=0.05 γ=**0.9**, 18 stati, MAE 0.1228, Top-1 55.6%, e una tabella
   "inizializzata su 40 coppie". I JSON attuali dicono 7000 episodi, γ=**0.99**, 19 stati,
   MAE 0.1695, Top-1 52.6%, **76 coppie**. Sono due run diverse: i plot sulla slide sono
   della run vecchia, i numeri qui sono della nuova.

4. **"17 passi medi contro 23" non è più vero, e si è invertito.** Nella run attuale la
   valutazione greedy finale dà **54 passi medi con l'init contro 15 del controllo**. Entrambe
   SR=1.000, cioè entrambe ottime: l'init converge a una policy *più lenta ma ugualmente
   ottima*, e su 100 episodi di eval la differenza è rumore. Se il prof chiede perché
   l'init "impiega più passi", la risposta è che a γ=0.99 e 7000 episodi converge a un
   ottimo diverso fra policy ottime equivalenti, e che la metrica da confrontare è il tempo
   alla soglia, non la lunghezza finale.

---

## 6. Ricalcolare i numeri

Tutti e quattro riusano il codice del repo, quindi non possono divergere dalle metriche
ufficiali. Stampa la riga da confrontare con la tabella corrispondente qui sopra.

**§1 e §2 — qualità dell'LLM, MAE, Top-1, regret**

```bash
cd /home/pietro/Documenti/llm-guided-rl && python3 - <<'EOF'
import sys; sys.path.insert(0, 'src')
from evaluate.evaluate_frozenlake_llm import load_df, compute_value_metrics, compute_action_metrics
df = load_df('src/output/llm/frozenlake_llm_8x8_slippery_q_seed1337_gpt-oss_120b.json')
m = compute_value_metrics(df, 0.99); a = compute_action_metrics(df, 1e-6)
print(f"n_stati {a['n_states']} MAE {m['mae']:.4f} bias {m['bias']:+.4f} "
      f"Top1 {a['accuracy']*100:.1f}% top2 {a['top2']*100:.1f}% "
      f"regret {a['mean_value_loss']:.5f} CI[{a['regret_ci'][1]:.5f},{a['regret_ci'][2]:.5f}]")
EOF
```

**§2 — decomposizione offset / ordinamento**

```bash
cd /home/pietro/Documenti/llm-guided-rl && python3 - <<'EOF'
import sys, collections, numpy as np
sys.path.insert(0, 'src')
from evaluate.evaluate_frozenlake_llm import load_df
df = load_df('src/output/llm/frozenlake_llm_8x8_slippery_q_seed1337_gpt-oss_120b.json')
by = collections.defaultdict(dict)
for r in df.to_dict('records'):
    by[r['id']][r['action']] = (float(r['v_true']), float(r['v_llm']))
raw, cen, off, reg = [], [], [], []
for ac in by.values():
    t = np.array([v[0] for v in ac.values()]); l = np.array([v[1] for v in ac.values()])
    raw += list(abs(t - l))
    cen += list(abs((t - t.mean()) - (l - l.mean())))
    off.append(abs(l.mean() - t.mean()))
    reg.append(float(t.max() - t[int(np.argmax(l))]))
raw, cen = np.array(raw), np.array(cen)
rnd = np.mean([max(v[0] for v in ac.values()) - np.mean([v[0] for v in ac.values()])
               for ac in by.values()])
print(f"MAE grezzo {raw.mean():.4f} | MAE centrato {cen.mean():.4f} | "
      f"energia costante {100*(1-(cen**2).mean()/(raw**2).mean()):.0f}% | offset medio {np.mean(off):.4f}")
print(f"stati con regret 0: {sum(1 for x in reg if x==0)}/{len(reg)} | regret max {max(reg):.4f}")
print(f"regret LLM {np.mean(reg):.5f} | regret casuale {rnd:.5f} -> {100*np.mean(reg)/rnd:.0f}% del casuale")
EOF
```

**§3 — tempo alla soglia sugli agenti** (usa la media mobile del repo, coerente coi plot)

```bash
cd /home/pietro/Documenti/llm-guided-rl && python3 -c "
import sys, json, numpy as np; sys.path.insert(0, 'src')
from evaluate.llminit_speedup import ttt
A = 'src/output/agents/frozenlake_qtable_llminit_8x8_slippery_seed_1337_gpt-oss_120b_'
r = {}
for n, l in (('llminit', 'LLM-init'), ('vsame', 'Vanilla same-hp'), ('vstd', 'Vanilla-std')):
    d = json.load(open(A + n + '.json')); r[l] = ttt(d['successes'], d['ep_lengths'], 0.9, 100)
    e = d['eval']
    print(f'{l:15s} n_init {d[\"n_init\"]:3d} TTT {r[l][0]:5d} ep {r[l][1]:7d} step | eval SR {e[\"sr\"]:.3f} len {e[\"mean_len\"]:.0f}')
i = r['LLM-init']
print(f'speedup ep:   same {r[\"Vanilla same-hp\"][0]/i[0]:.1f}x  std {r[\"Vanilla-std\"][0]/i[0]:.1f}x')
print(f'speedup step: same {r[\"Vanilla same-hp\"][1]/i[1]:.1f}x  std {r[\"Vanilla-std\"][1]/i[1]:.1f}x')
"
```

**§3 e §4 — distribuzione del gap e accuratezza in funzione della tolleranza**

```bash
cd /home/pietro/Documenti/llm-guided-rl && python3 - <<'EOF'
import sys, collections, numpy as np
sys.path.insert(0, 'src')
from evaluate.evaluate_frozenlake_llm import load_df
df = load_df('src/output/llm/frozenlake_llm_8x8_slippery_q_seed1337_gpt-oss_120b.json')
by = collections.defaultdict(dict)
for r in df.to_dict('records'):
    by[r['id']][r['action']] = (float(r['v_true']), float(r['v_llm']))
rows = []
for s, ac in by.items():
    t = {a: v[0] for a, v in ac.items()}; l = {a: v[1] for a, v in ac.items()}
    ch = max(l, key=lambda a: l[a]); opt = max(t.values())
    order = sorted(t, key=lambda a: -t[a])
    rows.append(dict(s=s, ch=ch, opt=opt, gap=opt - t[ch], rank=order.index(ch) + 1,
                     gap2=opt - t[order[1]], gap4=opt - t[order[3]],
                     rnd=opt - np.mean(list(t.values()))))
g = np.array([r['gap'] for r in rows]); rnd = np.mean([r['rnd'] for r in rows])
print("stato  ottimo  scelta   gap LLM   2a      4a       %val  rank")
for r in sorted(rows, key=lambda r: -r['gap']):
    if r['gap'] > 0:
        print(f"{r['s'][:4]:>5} {r['opt']:.4f} {r['ch']:>6} {r['gap']:8.4f} "
              f"{r['gap2']:6.4f} {r['gap4']:6.4f} {100*r['gap']/r['opt']:6.1f}% {r['rank']:>4}")
print(f"\ngap 0 esatto: {(g==0).sum()}/{len(g)} | mediano dei 9: "
      f"{np.median([r['gap'] for r in rows if r['gap']>0]):.4f} | max {g.max():.4f}")
print(f"regret LLM {g.mean():.4f} | sempre 2a {np.mean([r['gap2'] for r in rows]):.4f} "
      f"| sempre 4a {np.mean([r['gap4'] for r in rows]):.4f} | casuale {rnd:.4f}")
print(f"media escluso lo stato peggiore: {np.mean(sorted(g)[:-1]):.4f} "
      f"({100*np.mean(sorted(g)[:-1])/rnd:.0f}% del casuale)")
print("rank scelta:", dict(sorted(collections.Counter(r['rank'] for r in rows).items())))
print("\neps        LLM     casuale")
for eps in (1e-9, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1):
    print(f"  {eps:<9.3f} {(g<=eps+1e-12).mean()*100:6.1f}%  {np.mean([r['rnd']<=eps for r in rows])*100:6.1f}%")
EOF
```

**Rollout greedy sulla tabella non addestrata** (la prova del deadlock; vuole la cache
`src/output/cache/frozenlake_mdp_8x8_slippery_seed1337.json`, se non c'è rigenera il grafo)

```bash
cd /home/pietro/Documenti/llm-guided-rl && python3 - <<'EOF'
import json, numpy as np
g = json.load(open('src/output/cache/frozenlake_mdp_8x8_slippery_seed1337.json'))
nodes = {n['state']: n for n in g['nodes']}
ACT = ['left', 'down', 'right', 'up']
d = json.load(open('src/output/llm/frozenlake_llm_8x8_slippery_q_seed1337_gpt-oss_120b.json'))
rows = d['rows'] if isinstance(d, dict) and 'rows' in d else d
q = {(r['state'], ACT.index(r['action'])): min(max(r['v_llm'], 0.0), 1.0) for r in rows}
def sim(qi, n=2000, seed=1):
    rng = np.random.RandomState(seed); ok = hole = 0; L = []
    for _ in range(n):
        s = 0; st = 0; done = False
        while not done and st < 200:
            v = [qi.get((s, a), 0.0) for a in range(4)]; mx = max(v)
            c = [i for i, x in enumerate(v) if x >= mx - 1e-9]
            ts = nodes[s]['transitions'][str(int(rng.choice(c)))]
            p = np.array([t['prob'] for t in ts]); p /= p.sum()
            t = ts[int(rng.choice(len(ts), p=p))]
            s = int(t['next_id']); done = bool(t['done']); st += 1
        if done and nodes[s]['is_goal']: ok += 1
        elif done: hole += 1
        L.append(st)
    return ok / n, hole / n, float(np.mean(L))
for lab, qi in (('Q_LLM (76 coppie)', q), ('Q=0', {})):
    a, b, c = sim(qi)
    print(f"{lab:20s} SR {a:.3f} | buco {b:.3f} | len {c:.1f}")
EOF
```

---

## 7. Cosa NON dire

- **Non dire "l'LLM indovina la soluzione"**: non la conosce, e la tabella inizializzata è
  una policy che finisce in un buco il 95% delle volte.
- **Non dire "il MAE non conta"**: conta, solo che non è la metrica della domanda che stai
  rispondendo. Il MAE misura la qualità di una stima di valore; l'init usa l'ordinamento.
- **Non citare 0.1228 o 55.6%** senza dire che vengono dal log del 23/09.
- **Non presentare la speedup come statistica**: un seed. Se il prof chiede la varianza,
  la risposta è che non è stata stimata ed è lavoro futuro.
- **Non dire che l'init migliora la policy finale**: non lo fa, non lo può fare. Riduce il
  tempo per arrivarci.
