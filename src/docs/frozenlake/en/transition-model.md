The **transition model** describes the environment dynamics: the joint distribution of next state and reward given the current state and action.

\[
p(s',r \mid s,a) \doteq \Pr\{S_{t+1}=s',\, R_{t+1}=r \mid S_t=s,\, A_t=a\},
\]

defined for all \(s,s'\in\mathcal{S},\, r\in\mathcal{R},\, a\in\mathcal{A}\). Under the **Markov assumption** it depends only on \((s,a)\), not on the past history. With it, the conditional expectation becomes a weighted sum:

\[
\mathbb{E}[\,\cdot \mid S_t=s, A_t=a] = \sum_{s',r} p(s',r\mid s,a)\,[\,\cdot\,].
\]

This single form covers every environment; deterministic and stochastic are just two instances:

1. **Deterministic** (e.g. DoorKey): \(p\) is degenerate on a single \((s',r)\), the sum collapses and \(q_*(s,a) = r + \gamma v_*(s')\).
2. **Stochastic** (e.g. slippery FrozenLake): \(p\) is spread over multiple outcomes, \(q_*(s,a) = \sum_{s',r} p(s',r\mid s,a)\,[r + \gamma v_*(s')]\).

### Link with \(v_*\) and \(q_*\)

The transition model is the weight inside the Bellman optimality equations:

\[
q_*(s,a) = \sum_{s',r} p(s',r\mid s,a)\,[r + \gamma v_*(s')] = \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1}) \mid s,a],
\]
\[
v_*(s) = \max_a q_*(s,a) = \max_a \sum_{s',r} p(s',r\mid s,a)\,[r + \gamma v_*(s')].
\]

Read this way, \(q_*(s,a)\) is the **weighted average** of the one-step returns \([r + \gamma v_*(s')]\) with weights \(p(s',r\mid s,a)\): each outcome contributes in proportion to its probability. The final \(v_*(s)\) is the maximum of these averages over the available actions, i.e. the value of the action whose weighted outcome mix is best.
