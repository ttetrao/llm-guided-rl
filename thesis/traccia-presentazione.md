# Traccia della presentazione — 20 slide, 15 minuti

> **Calibrata su: 15:00 di parlato, 130 parole al minuto, 1 voce.**
> I conteggi di questo file sono **misurati**, non stimati: ogni volta che hai
> cambiato una riga, il numero nella tabella è quello che esce contando le
> parole. Comandi di ricalcolo in §5.

## 1. Perché 130 parole al minuto, e che cosa NON ci è dentro

130 è una velocità da relazione orale sobria, non un ritmo da annuncio. La
media italiana parlata sta fra 120 e 160 parole al minuto, ma in una discussione
si scende sotto i 140 perché si guarda il foglio e si scandiscono i numeri.

Il conteggio include **solo le parole da dire**. **Non** include:

- le pause e i silenzi
- il tempo di girare la slide
- il tempo di indicare la figura con il mouse o il dito
- l'eventuale "uh", i riavvii, il guardare la commissione

Su questo tipo di presentazione, con 4-5 grafici da indicare, le pause valgono
**realisticamente il 10% del tempo parlato**. Quindi:

- **parole da dire = minuti reali × 130 ÷ 1,10**
- per 15:00 reali servono **≈ 1.770 parole**, non 1.950

Formula: secondi di parlato = parole / 130 × 60 = parole × 0,4615.
Secondi reali = secondi di parlato × 1,10.

**La traccia completa, 20 slide, sta su 1.932 parole = 14:52 di parlato, cioè
~16:20 reali. Non entra in 15 minuti.** La configurazione che entra è la **B**
di §2: 18 slide, 1.701 parole, 13:05 di parlato, ~14:20 reali, con 40 secondi di
margine.

## 2. Mappa dei tempi (conteggi misurati sul testo in §4)

**Configurazione A — tutte e 20 le slide: 1.932 parole = 14:52 di parlato,
~16:20 reali. Serve uno slot da 17 minuti.**

| # | slide | blocco | parole | parlato | cum. A | stato |
|---|---|---|---|---|---|---|
| 1 | Titolo | Apertura | 57 | 0:26 | 0:26 | — |
| 2 | Perché farlo | Apertura | 93 | 0:43 | 1:09 | — |
| 3 | Reinforcement Learning | Metodo | 73 | 0:34 | 1:43 | riducibile |
| 4 | DoorKey | Metodo | 87 | 0:40 | 2:23 | — |
| 5 | Stati usati | Metodo | 83 | 0:38 | 3:01 | — |
| 6 | Generazione Q-Function | Metodo | 99 | 0:46 | 3:47 | — |
| 7 | Prompt | Metodo | 78 | 0:36 | 4:23 | riducibile |
| 8 | Errore per bucket | Risultati LLM | 77 | 0:36 | 4:59 | — |
| 9 | Correlazione lineare | Risultati LLM | 65 | 0:30 | 5:29 | **TAGLIARE** |
| 10 | Accuratezza policy | Risultati LLM | 74 | 0:34 | 6:03 | — |
| 11 | DoorKey init vs std | **Grafici** | 150 | 1:09 | 7:12 | — |
| 12 | DoorKey init vs no-init | **Grafici** | 145 | 1:07 | 8:19 | — |
| 13 | DoorKey 5 seed | **Grafici** | 143 | 1:06 | 9:25 | — |
| 14 | FrozenLake slippery | Risultati LLM | 71 | 0:33 | 9:58 | — |
| 15 | Regret FrozenLake | Risultati LLM | 92 | 0:42 | 10:40 | **TAGLIARE** |
| 16 | FrozenLake init vs std | **Grafici** | 140 | 1:05 | 11:45 | — |
| 17 | FrozenLake init vs no-init | **Grafici** | 144 | 1:06 | 12:51 | — |
| 18 | Limiti | Chiusura | 116 | 0:54 | 13:45 | — |
| 19 | Futuro | Chiusura | 74 | 0:34 | 14:19 | **TAGLIARE** |
| 20 | Chiusura | Chiusura | 71 | 0:33 | 14:52 | — |

**Configurazione B — la raccomandata per 15 minuti.** Si tolgono le tre slide
marcate TAGLIARE (9, 15, 19), che sono 231 parole:

- **1.701 parole = 13:05 di parlato, 14:24 reali** (13:05 × 1,10)
- cum. B: 0:26 → 1:09 → 1:43 → 2:23 → 3:01 → 3:47 → 4:23 → 4:59 → 5:33 →
  6:42 → 7:49 → 8:55 → 9:28 → 10:32 → 11:39 → 12:32 → 13:05
- **margine residuo: 36 secondi.** Se sei in ritardo, taglia anche una frase
  dalla 3 e dalla 7 (le due uniche riducibili) e ne recuperi altre 15.

**Se devi togliere una terza slide, non toccare mai i grafici** (11, 12, 13, 16,
17: 722 parole, il 38% della traccia). Sono il cuore del lavoro e il posto dove
ti fanno le domande. Se il tempo è corto, si accorcia il metodo (3-7), non i
risultati.

**Se il tempo è invece generoso** (slot da 20 minuti): usa la configurazione A e
aggiungi la 6ª figura mancante (§6 punto 1) parlando per 1 minuto su di essa.

## 3. Dove respirare

Non leggere tutto di corsa. I tre respiri sono in corrispondenza delle tre
transizioni del discorso:

| dopo slide | perché lì | cosa fai |
|---|---|---|
| 10 | finisce la parte "quanto sbaglia il modello" | **respiro lungo**, cambio voce: da "ho sbagliato" a "cosa cambia quando lo uso" |
| 13 | finisce DoorKey | **respiro lungo**, poi guardi la commissione: qui stai per dire una cosa scomoda sul seed non visto |
| 17 | finisce FrozenLake | **respiro lungo**, e qui ti compri il tempo per la domanda più probabile |

Le slide 11, 13, 16, 17 sono le uniche con numeri a due cifre da scandire
lentamente. Non accelerare su "quattordicimilacentosessantuno passi": è proprio
lì che la commissione sta guardando.

---

## 4. Il testo, slide per slide

Il testo in `>` è **esattamente** quello da dire. Le righe fuori dal `>` non si
dicono: sono indicazioni per te.

### Slide 1 — Titolo

> Buongiorno. Il mio lavoro riguarda l'uso di modelli linguistici per migliorare
> l'apprendimento per rinforzo. In pratica: ho chiesto a un modello linguistico
> di stimare la funzione di valore di un agente, e l'ho usata per inizializzare
> la tabella Q del Q-learning tabulare. Tutto il codice è su GitHub, il link è
> in basso.

*Non dire la data ad alta voce. Prima slide, guardi, aspetti, poi parli.*

### Slide 2 — Introduzione / Perché farlo

> Il punto di partenza è questo: il Q-learning propaga il valore a ritroso, e
> funziona bene solo se l'agente incontra spesso una ricompensa. Se il goal si
> vede di rado, l'agente converge con molta fatica, e nelle prime migliaia di
> episodi non impara quasi nulla. L'idea che ho voluto provare è semplice: se il
> segnale è scarso, si può chiedere a un modello linguistico di generare la
> funzione di valore su un piccolo insieme di stati significativi, e usarla come
> punto di partenza. Non sostituisce l'apprendimento: lo anticipa.

*L'ultima frase è la thesis della tesi in 10 parole. Se te la chiedono a metà,
è già detta.*

### Slide 3 — Reinforcement Learning

> Brevissimo richiamo. L'ambiente è un MDP: stati, azioni, transizione,
> ricompensa e fattore di sconto. L'agente mappa gli stati in azioni per
> massimizzare il ritorno atteso. Qui la transizione e la ricompensa dipendono
> solo dallo stato e dall'azione correnti, e il ritorno è scontato con gamma, che
> nel mio setup vale zero virgola novantanove. Ho scelto il Q-learning tabulare
> perché è l'implementazione più trasparente e riproducibile di questo problema.

*Se il relè è caldo, taglia "Ho scelto il Q-learning tabulare perché…". La
scelta la giustifichi solo se chiedono.*

### Slide 4 — DoorKey

> L'ambiente principale è MiniGrid DoorKey 8x8. Lo stato è la posizione relativa
> alla porta, la direzione, se ho la chiave e se la porta è aperta. Le azioni
> sono cinque. La ricompensa è uno solo sul goal e zero altrove: una ricompensa
> rarissima, e questo è il punto. L'ambiente è gerarchico, si può scomporre in
> fasi: prendere la chiave, aprire la porta, arrivare al goal. Sfruttando questa
> struttura l'agente impara le fasi in ordine. Come riferimento ho calcolato la
> ground truth con value iteration.

*Il "riferimento con value iteration" è importante: è quello con cui confronti
tutte le metriche di ottimalità. Se non lo dici qui, dopo ti chiedono "e come
sapete quale azione è giusta".*

### Slide 5 — Stati usati

> Non ho dato al modello tutti gli stati. Per un singolo seed gli stati sono 480,
> e ne ho estratti due famiglie. I bottleneck: gli stati in cui cambia lo stage,
> cioè dove l'agente supera una delle tre fasi; sono le transizioni che decidono
> se l'episodio si chiude. E i checkpoint: stati scelti fra quelli visitati
> durante l'addestramento con Q-learning, a scaglioni di success rate. Sul seed
> singolo mi sono fermato ai soli bottleneck, perché sono quelli che contano.

### Slide 6 — Generazione Q-Function

> Per ogni stato raccolgo l'osservazione dell'ambiente e la trasformo in una mappa
> ASCII: a sinistra l'agente, il cancelletto con la chiave, il muro con la porta,
> il goal in basso a destra. Poi chiedo al modello il valore di ogni azione,
> cioè la V dei sette stati successivi. Ho usato due modelli, gpt-oss 120b e
> Gemma 4 26b. Lo script scarta le risposte anomale, e in più valido sempre il
> risultato con la stessa value iteration. Così ho sia l'errore assoluto sia
> l'errore nella scelta dell'azione, che è quello che mi interessa.

*La mappa ASCII è la cosa che i prof chiedono sempre ("gli dai cosa
esattamente?"). Se hai tempo, descrivila a mano sulla slide.*

### Slide 7 — Prompt

> Il prompt è in pseudo-XML e ha quattro blocchi statici iniettati dallo script:
> la documentazione dell'ambiente, la legenda dei simboli ASCII con le regole di
> calcolo, la definizione della funzione di V e quella di Q. Per ogni stato
> vengono iniettate sette mappe ASCII, quella corrente e le sette possibili
> successive, una per azione. Ho anche forzato una sezione di analisi, per far
> ragionare il modello prima di dare i numeri. L'output richiesto è JSON.

*Se il tempo stringe, taglia "Ho anche forzato una sezione di analisi…". La
forced chain-of-thought è un dettaglio implementativo, non un contributo.*

### Slide 8 — Errore per bucket

> Primo risultato, e il più scomodo: l'errore assoluto del modello è alto. Su
> DoorKey l'errore assoluto medio è di circa zero virgola quattordici sui
> bottleneck. Quindi il modello, in valore assoluto, sbaglia. Però questo errore
> grande in valore assoluto è in parte una costante per stato, cioè uno scarto di
> scala che non cambia quale azione sembra la migliore. E per giudicare un agente
> conta l'errore nella scelta dell'azione, non il valore assoluto.

*Non scusarti qui. Dì "il modello sbaglia in valore assoluto" e vai avanti: la
slide 10 è la risposta. Se ti scusi, la difesa ti blocca su questo per due
minuti.*

### Slide 9 — Correlazione lineare · **tagliare se il tempo stringe**

> Il grafico successivo mette in relazione l'errore assoluto del modello con
> l'errore assoluto di un agente allenato per ventimila episodi, sugli stessi
> stati. I punti si dispongono lungo una retta con pendenza bassa: chi sbaglia in
> assoluto sbaglia anche contro l'ottimo, ma non necessariamente di più. Serve a
> capire se l'errore del modello è rumore o un errore sistematico di scala.

*Slide di servizio. Se la tiri, la commissione probabilmente non se ne accorge.
Se la lasci, non(commentarla: guardala un secondo e vai alla 10.*

### Slide 10 — Accuratezza policy

> Qui la metrica cambia completamente. Ho calcolato l'accuratezza dell'argmax del
> modello rispetto all'azione ottima, calcolata per value iteration. La versione
> tie-aware, che considera corrette anche le azioni a pari merito, dà circa il
> novantaquattro per cento. La versione stretta, che accetta solo la prima azione
> in ordine, dà circa l'ottantaquattro per cento. La differenza fra i due numeri è
> il peso dei pareggi, e in questa mappa sono tanti.

*Qui puoi permetterti un sorriso: è il primo risultato positivo, e la frase
"la metrica cambia completamente" è la svolta del racconto.*

### Slide 11 — DoorKey init vs std · **cuore del lavoro**

> Adesso arriviamo al risultato centrale: cosa succede quando questi valori
> inizializzano la tabella Q. In alto a sinistra c'è il tasso di successo in
> addestramento, in media mobile su cento episodi; la linea tratteggiata è la
> valutazione greedy finale. La curva blu sale e si stabilizza attorno a zero
> virgola novantotto entro seicento episodi. La curva arancione, l'agente con i
> parametri standard, resta piatta fino a circa milleduecento, poi sale di colpo
> e arriva a ottantasei per cento dopo un tremilaquattrocento. In termini di
> tempo alla soglia, tre virgola due volte più tardi. In basso a destra, però,
> l'agente standard ha una perdita di valore migliore: zero virgola zero sei sei
> contro zero virgola zero sette due. Arriva dopo, ma la tabella che produce è
> leggermente più vicina all'ottimo. E il confronto, ve lo dico subito, cambia
> due parametri insieme, quindi non attribuisce tutto all'inizializzazione.

*Indica il pannello in alto a sinistra quando dici "in alto a sinistra", e il
basso a destra quando dici "in basso a destra". L'ultima frase è l'onestà che ti
compra il resto: dopo quella non ti fermano.*

### Slide 12 — DoorKey init vs no-init · **il confronto che regge**

> Questo è il confronto che invece è pulito, perché cambia una sola cosa: la
> tabella inizializzata, e basta. Stessi parametri, stesso seed, stessi duemila
> episodi. E qui il risultato non è che l'agente inizializzato sia migliore: è che
> l'altro non impara. La curva arancione resta a zero per duemila episodi, con
> una lunghezza media di quattrocentoquarantanove passi su quattrocentocinquanta,
> cioè ogni episodio finisce per timeout. Non ha mai toccato il goal. E il motivo
> è proprio il meccanismo del Q-learning con la tabella a zero: finché l'agente
> non arriva al goal, ogni valore resta a zero, non c'è nessun gradiente, e la
> policy greedy è un pareggio, cioè una scelta casuale. L'inizializzazione non
> deve risolvere il problema, deve rompere il blocco. Da lì in poi il valore si
> propaga a ritroso, e l'apprendimento cancella l'errore di calibrazione.

*Se guardi solo una figura in tutta la presentazione, guarda questa: è il
risultato che tiene da solo.*

### Slide 13 — DoorKey 5 seed

> Per dare un'idea oltre il singolo seed, ho ripetuto l'esperimento allenando su
> cinque layout diversi, con un seed per episodio, quattrocento coppie iniziali
> e cinquemila episodi. La curva dell'agente inizializzato sale gradualmente e si
> stabilizza a zero virgola novantasette. Quella dell'agente senza
> inizializzazione, di nuovo, non parte mai. Sui pannelli in basso il divario è
> netto: accordo con la policy ottima zero virgola sei sette uno contro zero
> virgola quattro sei, e perdita di valore dimezzata, zero virgola zero tre otto
> contro zero virgola zero sette nove. Qui però l'agente con cui confronto ha
> parametri diversi, quindi anche questo non è un esperimento controllato. E c'è
> un limite importante: la valutazione riportata è sul seed 13507, che fa parte
> dell'addestramento. Sul seed mai visto, il 69694, tutti e tre danno zero.
> Quindi questo non dimostra generalizzazione.

*Il numero "quattrocento coppie iniziali", non tremila: correggi la slide se
riporta tremila. La penultima frase la dici con il grafico ancora acceso, non
con la mano già sulla slide 14.*

### Slide 14 — FrozenLake (Slippery)

> Ho ripetuto l'esperimento su FrozenLake 8x8 slippery, che introduce un elemento
> nuovo: ogni azione viene eseguita con probabilità di un terzo, quindi la
> transizione è stocastica. Anche qui la ricompensa è solo sul goal. Su questo
> ambiente l'accuratezza del modello cala: l'errore assoluto è zero virgola
> diciassette, e la selezione corretta dell'azione intorno al cinquantaquattro per
> cento. Quindi qui il modello è oggettivamente peggio che su DoorKey.

*Non anticipare qui che l'inizializzazione funziona meglio. Tienilo per le slide
16-17: è il colpo di scena.*

### Slide 15 — Regret FrozenLake · **tagliare se il tempo stringe**

> Aggiungo il regret, che è la metrica più onesta per un agente: quanto valore si
> perde scegliendo l'azione che il modello ha valutato migliore invece di quella
> ottima. Su questi stati il regret medio è zero virgola zero uno uno, e zero
> virgola zero zero quattro se escludo il caso peggiore. Per dare la scala, ho
> inizializzato gli stessi stati ventimila volte a caso: il regret medio è zero
> virgola zero sette tre. Quindi l'inizializzazione del modello produce un
> segnale sei volte migliore di una casuale, pur sbagliando in assoluto.

*Se la tiri, non riprendere la parola "onesta": se hai appena detto che il
modello è peggio, la difesa chiederà perché il regret è basso. Se la tiri,
taglia anche la slide 18-19 correspondingly no: la 18 resta, perché è la
dichiarazione dei limiti.*

### Slide 16 — FrozenLake init vs std

> Sulla curva di FrozenLake l'agente inizializzato passa lo zero virgola otto a
> quattrocentosessantuno episodi. L'agente con i parametri standard ci arriva a
> millenovecentosessantotto: quattro virgola tre volte di ritardo, e in passi
> cumulati tre virgola quattro volte. I pannelli in basso dicono la stessa cosa:
> accordo zero virgola cinque otto contro zero virgola quattro sette due, perdita
> zero virgola zero uno tre sei contro zero virgola zero tre tre otto. Qui
> l'inizializzazione non accelera soltanto: migliora anche la tabella. Ha senso
> che il vantaggio sia maggiore che su DoorKey: il collo di bottiglia è più duro,
> un percorso ottimo è lungo e ogni passo può scivolare, quindi la finestra di
> ricompensa è rarissima e l'inizializzazione ha molto più margine. Da notare che
> qui i due agenti usano parametri diversi, quindi anche questo confronto mescola
> due cause.

*Le frasi "qui il modello è peggio" (slide 14) e "il vantaggio è maggiore" (qui)
devono stare vicine nella mente della commissione, non per forza nel testo: è il
passaggio che ti salva la domanda "allora su FrozenLake l'LLM non funziona".*

### Slide 17 — FrozenLake init vs no-init · **il picco**

> L'ultimo grafico è il confronto controllato, stessi parametri, e la differenza
> è netta. L'agente inizializzato passa lo zero virgola otto a
> quattrocentosessantuno episodi, quello senza inizializzazione a
> milletrecentoventidue: tre virgola zero volte. Nei pannelli in basso l'accordo
> con la policy ottima è zero virgola cinque otto contro zero virgola tre cinque
> otto, il divario più grande di tutti i confronti, e la perdita di valore è un
> terzo. Un dato però va detto chiaramente: l'agente inizializzato, alla fine,
> cammina più lento nella valutazione finale, cinquantaquattro passi contro
> quindici. Entrambe le policy sono ottime, con success rate uno, quindi non è un
> peggioramento: a cinquemila episodi si converge a un ottimo diverso fra soluzioni
> equivalenti, e su cento episodi la differenza di lunghezza è rumore. La metrica
> che distingue le due è il tempo alla soglia, e lì l'inizializzazione vince.

*Se ti fanno la domanda "perché impiega più passi", hai appena già dato la
risposta, per iscritto, prima che la facciano. È la mossa più forte della
presentazione: non nascondere mai un dato controproducente, mettilo tu con la
spiegazione.*

### Slide 18 — Limiti

> I limiti li dico io, prima che me li chiediate. Primo: ogni configurazione è
> stata eseguita una sola volta, con un solo seed, quindi non ho una stima della
> varianza. Secondo: due dei tre confronti cambiano insieme inizializzazione e
> parametri, e solo quelli a parità di parametri sono un esperimento controllato.
> Terzo: i cinque seed del multi-seed condividono una sola tabella Q, quindi non
> sono cinque repliche indipendenti. Quarto, ed è il più importante: la
> generalizzazione non è risolta. Sul seed mai visto tutti gli agenti danno zero.
> E correggo una cosa che avevo scritto: sull'ambiente stocastico il modello
> sbaglia di più nella stima, ma l'inizializzazione lì aiuta più che altrove,
> non meno.

*L'ultima frase serve a te più che alla slide: la slide 18 dice ancora "LLM
fatica su ambienti non deterministici come FrozenLake", che è un'affermazione
che ti va contro. Se la lasci com'è e non aggiungi questa frase, la
commissione ti chiederà perché l'hai scritta e ti mangerà 30 secondi.*

### Slide 19 — Futuro · **tagliare se il tempo stringe**

> Quello che viene dopo è chiaro. Primo, un modello locale fine-tuned, per
> togliere il costo e la latenza delle API. Secondo, valutare molti più stati su
> molti più seed, per misurare davvero la generalizzazione. Terzo, e per me il più
> interessante: generare la funzione di valore online, durante l'addestramento,
> quando l'agente incontra uno stato che non ha mai visto. È il modo naturale per
> superare il limite che ho appena descritto.

*L'ultimo punto è il più difendibile: risponde al limite che hai appena
dichiarato. Se tagli la slide 19, il limite della slide 18 resta senza risposta
e perdi punti. Se il tempo stringe, taglia invece 9 e 15 e tieni 19.*

### Slide 20 — Chiusura

> Concludo. Ho verificato che un insieme ridotto di stime fornite da un modello
> linguistico può essere usato come punto di partenza per il Q-learning tabulare,
> e che accelera l'apprendimento di due a quattro volte sui layout incontrati,
> senza peggiorare la valutazione finale. Non dimostra la generalizzazione, ed è
> un punto di partenza, non una conclusione. Il codice è su GitHub, il link è in
> basso. Grazie per l'attenzione.

*Non aggiungere niente dopo "Grazie". Il primo "eh" dopo aver finito vale un
penalty. Vai a sederti.*

## 5. Ricalcolo dei conteggi

Il file è la fonte: conta le parole di ogni blocco `>` (da `### Slide N` al
`###` successivo) e riapplica la formula. **Attenzione all'unità**: `wpm` sono
parole *al minuto*, quindi i minuti sono `parole / wpm`, non `parole / wpm / 60`.

```bash
cd /home/pietro/Documenti/llm-guided-rl && python3 - <<'EOF'
import re
from pathlib import Path

WPM, PAUSE, BUDGET = 130.0, 1.10, 15 * 60   # wpm, pausa 10%, 15 minuti reali

txt = Path("thesis/traccia-presentazione.md").read_text()
parts = re.split(r"^### Slide (\d+) — (.+)$", txt, flags=re.M)

rows, tot_w, tot_s = [], 0, 0.0
for i in range(1, len(parts), 3):
    n, title, body = int(parts[i]), parts[i + 1], parts[i + 2]
    said = "\n".join(l[2:] for l in body.splitlines() if l.startswith("> "))
    w = len(re.findall(r"[\wÀ-ÿ]+", said))
    tot_w += w
    tot_s += w / WPM * 60
    rows.append((n, title.strip(), w, tot_s / 60, "tagliare" in title.lower()))

print(f"{'#':>2} {'parole':>6} {'parlato':>8} {'cum.':>7}  slide")
for n, t, w, cum, cut in rows:
    print(f"{n:>2} {w:>6} {w/WPM:>6.2f}m {cum:>6.2f}m  {t[:44]:<46}"
          f"{'TAGLIARE' if cut else ''}")

keep = [r for r in rows if not r[4]]
kw, ks = sum(r[2] for r in keep), sum(r[2] for r in keep) / WPM * 60
cap = BUDGET / PAUSE * WPM / 60           # parole massime entro 15:00 reali
print(f"\nA) tutte le slide   {tot_w:5d} parole = {tot_s/60:5.2f} min parlato "
      f"= {tot_s/60*PAUSE:5.2f} min reali  {'ENTRA' if tot_s*PAUSE <= BUDGET else '>>> NON ENTRA'}")
print(f"B) senza i tagli    {kw:5d} parole = {ks/60:5.2f} min parlato "
      f"= {ks/60*PAUSE:5.2f} min reali  {'ENTRA' if ks*PAUSE <= BUDGET else '>>> NON ENTRA'}")
print(f"tetto per 15:00 reali = {cap:.0f} parole | margine in B = {cap-kw:.0f} parole")
assert ks * PAUSE <= BUDGET, "la configurazione B sfora: taglia un'altra frase"
EOF
```

Tetto per 15:00 reali: **1.773 parole**. La configurazione B ne usa 1.701, con
**72 parole di margine** (più i ~40 secondi di pausa già contati). Se allunghi
qualcosa, togli il margine da un'altra parte: **il file non deve superare 1.770
parole in configurazione B**.

## 6. Tre correzioni al deck, prima di presentare

1. **Manca una delle sei figure.** Le slide 11, 12, 13, 16, 17 contengono
   rispettivamente `qtable_llminit_seed_1337_gpt-oss_120b_vs_std.png`,
   `…_vs_samehp.png`, `qtable_llminit_multiseed_…_vs_std.png`,
   `frozenlake_…_vs_std.png` e `frozenlake_…_vs_samehp.png`. **Non c'è il
   `_vs_samehp` multi-seed**, che è il confronto più forte in assoluto
   (accordo 0.671 contro 0.464, perdita dimezzata) ed è l'unico `vs_samehp`
   insieme a FrozenLake. Se hai tempo per una modifica sola, è questa.
2. **La slide 16 ha un testo fantasma**: "Esempio su 5 seed" è rimasto dentro una
   slide che mostra il seed singolo su FrozenLake. Se la proietti, la commissione
   chiederà quali sono i cinque seed.
3. **La slide 18 contraddice le slide 16-17**: dice "LLM fatica su ambienti non
   deterministici come FrozenLake", mentre sui dati FrozenLake è
   l'ambiente dove l'inizializzazione aiuta di più. Riscrivila (vedi §4 slide 18)
   o cancella quel punto: lasciandolo, ti chiederanno di riconciliarlo.
