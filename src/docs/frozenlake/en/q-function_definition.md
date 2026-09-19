## Definition of \(q\)-function

In the book *Reinforcement Learning: An Introduction* by Richard S. Sutton and Andrew G. Barto (second edition), the **action-value function** (or \(q\)-function) for the policy \(\pi\), denoted \(q_\pi(s, a)\), is the expected return starting from state \(s\), taking action \(a\), and following policy \(\pi\) thereafter.

Formally, for Markov Decision Processes (MDPs):

\[
q_\pi(s, a) \doteq \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \Bigm| S_t = s, A_t = a\right]
\]

where:
- \(G_t\) is the return starting from time \(t\)
- \(\gamma \in [0,1]\) is the discount factor
- \(\mathbb{E}_\pi[\cdot]\) denotes the expected value under policy \(\pi\)
- \(R_{t+k+1}\) is the reward received at time \(t+k+1\)
- \(\mathcal{S}\) is the state space and \(\mathcal{A}\) the action space

## Intuitive meaning

While the \(v\)-function measures how "good" a state is in general, the \(q\)-function specifically evaluates the value of a *particular action* taken in that state. It indicates the total reward an agent can expect to accumulate by starting its transition with action \(a\) from state \(s\) and subsequently following policy \(\pi\).

## Optimal value function

For finite MDPs there always exists at least one optimal policy \(\pi_*\). All optimal policies share the same optimal action-value function:

\[
q_*(s, a) \doteq \max_\pi q_\pi(s, a)
\]

This function provides the maximum expected reward obtainable starting from \(s\) and executing \(a\), then following the best possible policy. If \(q_*(s, a)\) is known, the optimal choice simply consists in choosing, in each state \(s\), the action \(a\) that maximizes the Q-value.

## Transition model

The transition model is \(p(s',r\mid s,a)\doteq\Pr\{S_{t+1}=s',\,R_{t+1}=r\mid S_t=s,A_t=a\}\) for \(s,s',r,a\). With it \(\mathbb{E}[\,\cdot\mid S_t=s,A_t=a]=\sum_{s',r}p(s',r\mid s,a)[\,\cdot\,]\).

## Bellman equation for the \(q\)-function

Like the \(v\)-function, the \(q\)-function also satisfies the Bellman equation. In particular, for the optimal \(q_*\):

\[
q_*(s, a) = \sum_{s',r} p(s',r\mid s,a)\,[\,r + \gamma \max_{a'} q_*(s', a')\,]
          = \mathbb{E}[R_{t+1} + \gamma \max_{a'} q_*(S_{t+1}, a') \mid S_t = s, A_t = a].
\]

The value of taking an action in a state equals the expected immediate reward plus the maximum discounted value reachable from the next state.
