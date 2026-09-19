Il **modello di transizione** (_transition model_) descrive la dinamica dell'ambiente: la distribuzione congiunta di stato successivo e reward dati stato corrente e azione.

\[
p(s',r \mid s,a) \doteq \Pr\{S_{t+1}=s',\, R_{t+1}=r \mid S_t=s,\, A_t=a\},
\]

definita per ogni \(s,s'\in\mathcal{S},\, r\in\mathcal{R},\, a\in\mathcal{A}\). Sotto l'**ipotesi di Markov** dipende solo da \((s,a)\), non dalla storia passata. Con essa l'aspettativa condizionata diventa una somma pesata:

\[
\mathbb{E}[\,\cdot \mid S_t=s, A_t=a] = \sum_{s',r} p(s',r\mid s,a)\,[\,\cdot\,].
\]

Questa unica forma copre ogni ambiente; deterministico e stocastico sono solo due istanze:

1. **Deterministico** (es. DoorKey): \(p\) è degenere su un unico \((s',r)\), la somma collassa e \(q_*(s,a) = r + \gamma v_*(s')\).
2. **Stocastico** (es. FrozenLake slippery): \(p\) è distribuita su più outcome, \(q_*(s,a) = \sum_{s',r} p(s',r\mid s,a)\,[r + \gamma v_*(s')]\).

### Legame con \(v_*\) e \(q_*\)

Il transition model è il peso dentro le equazioni di ottimalità di Bellman:

\[
q_*(s,a) = \sum_{s',r} p(s',r\mid s,a)\,[r + \gamma v_*(s')] = \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1}) \mid s,a],
\]
\[
v_*(s) = \max_a q_*(s,a) = \max_a \sum_{s',r} p(s',r\mid s,a)\,[r + \gamma v_*(s')].
\]

Letta così, \(q_*(s,a)\) è la **media pesata** dei ritorni a un passo \([r + \gamma v_*(s')]\) con pesi \(p(s',r\mid s,a)\): ogni outcome contribuisce in proporzione alla sua probabilità. La \(v_*(s)\) finale è il massimo di queste medie sulle azioni disponibili, cioè il valore dell'azione il cui mix di outcome pesato è migliore.
