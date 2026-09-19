## Definizione di \(v\)-function

Nel libro _Reinforcement Learning: An Introduction_ di Richard S. Sutton e Andrew G. Barto (seconda edizione), la **state-value function** (o \(v\)-function) di uno stato \(s\) sotto una policy \(\pi\), denotata \(v\_\pi(s)\), è il ritorno atteso partendo dallo stato \(s\) e seguendo la policy \(\pi\) da lì in poi.

Formalmente, per i processi decisionali di Markov (MDP):

\[
v_\pi(s) \doteq \mathbb{E}_\pi[G_t \mid S_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \Bigm| S_t = s\right], \quad \text{per tutti } s \in \mathcal{S}
\]

dove:

- \(G_t\) è il ritorno (return) a partire dal tempo \(t\)
- \(\gamma \in [0,1]\) è il fattore di sconto
- \(\mathbb{E}\_\pi[\cdot]\) indica il valore atteso sotto la policy \(\pi\)
- \(\mathcal{S}\) è lo spazio degli stati

## Significato intuitivo

La \(v\)-function misura quanto è “buono” uno stato nel lungo periodo, non solo per la ricompensa immediata. Mentre il segnale di reward dice cosa è buono nell’immediato, la value function dice cosa è buono nel lungo termine: il totale di reward che un agente può aspettarsi di accumulare partendo da quello stato e seguendo la policy.

## Relazione con la policy

Una policy \(\pi\) è una mappa da stati ad azioni (eventualmente stocastica). La \(v_\pi\) dipende sempre dalla policy: cambia se cambi il modo in cui l’agente sceglie le azioni. Due policy si confrontano proprio tramite le loro value function: \(\pi \geq \pi'\) se e solo se \(v_\pi(s) \geq v_{\pi'}(s)\) per tutti gli stati \(s\).

## Value function ottimale

Per MDP finiti esiste sempre almeno una policy ottimale \(\pi_*\). Tutte le policy ottimali condividono la stessa state-value function ottimale:

\[
v_*(s) \doteq \max_\pi v_\pi(s), \quad \text{per tutti } s \in \mathcal{S}
\]

Questa è la massima ricompensa attesa ottenibile partendo da \(s\).

## Equazione di Bellman

Una proprietà fondamentale è che \(v\_\pi\) soddisfa l’equazione di Bellman (relazione ricorsiva):

\[
v_\pi(s) = \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} \mid S_t = s]
\]

cioè il valore di uno stato è uguale al reward immediato atteso più il valore scontato dello stato successivo, sempre sotto la stessa policy.

## Modello di transizione

Il modello di transizione dell'MDP è

\[
p(s',r \mid s,a) \doteq \Pr\{S_{t+1}=s',\, R_{t+1}=r \mid S_t=s,\, A_t=a\},
\]

definito per ogni \(s,s'\in\mathcal{S},\, r\in\mathcal{R},\, a\in\mathcal{A}\). Con esso l'aspettativa \(\mathbb{E}[\,\cdot\mid S_t=s, A_t=a]\) diventa \(\sum\_{s',r} p(s',r\mid s,a)[\,\cdot\,]\).

## Equazione di Bellman di ottimalità per \(v\_\*\)

Per la funzione ottimale:

\[
v_*(s) = \max_a \sum_{s',r} p(s',r\mid s,a)\,[\,r + \gamma v_*(s')\,]
= \max_a \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1}) \mid S_t=s, A_t=a].
\]

Quando \(p\) è deterministica la somma collassa su un unico \((s',r)\).
