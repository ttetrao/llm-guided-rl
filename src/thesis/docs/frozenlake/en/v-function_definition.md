## Definition of the \(v\)-function

In the book *Reinforcement Learning: An Introduction* by Richard S. Sutton and Andrew G. Barto (second edition), the **state-value function** (or \(v\)-function) of a state \(s\) under a policy \(\pi\), denoted \(v_\pi(s)\), is the expected return starting from state \(s\) and following policy \(\pi\) thereafter.

Formally, for Markov Decision Processes (MDPs):

\[
v_\pi(s) \doteq \mathbb{E}_\pi[G_t \mid S_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \Bigm| S_t = s\right], \quad \text{for all } s \in \mathcal{S}
\]

where:
- \(G_t\) is the return starting from time \(t\)
- \(\gamma \in [0,1]\) is the discount factor
- \(\mathbb{E}_\pi[\cdot]\) denotes the expected value under policy \(\pi\)
- \(\mathcal{S}\) is the state space

## Intuitive meaning

The \(v\)-function measures how "good" a state is in the long run, not just for the immediate reward. While the reward signal indicates what is good in the immediate sense, the value function indicates what is good in the long term: the total reward an agent can expect to accumulate starting from that state and following the policy.

## Relationship with the policy

A policy \(\pi\) is a mapping from states to actions (possibly stochastic). \(v_\pi\) always depends on the policy: it changes if the way the agent chooses actions changes. Two policies are compared precisely through their value functions: \(\pi \geq \pi'\) if and only if \(v_\pi(s) \geq v_{\pi'}(s)\) for all states \(s\).

## Optimal value function

For finite MDPs, there always exists at least one optimal policy \(\pi_*\). All optimal policies share the same optimal state-value function:

\[
v_*(s) \doteq \max_\pi v_\pi(s), \quad \text{for all } s \in \mathcal{S}
\]

This is the maximum expected reward obtainable starting from \(s\).

## Bellman equation

A fundamental property is that \(v_\pi\) satisfies the Bellman equation (a recursive relationship):

\[
v_\pi(s) = \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} \mid S_t = s]
\]

that is, the value of a state equals the expected immediate reward plus the discounted value of the next state, always under the same policy.

## Transition model

The MDP transition model is

\[
p(s',r \mid s,a) \doteq \Pr\{S_{t+1}=s',\, R_{t+1}=r \mid S_t=s,\, A_t=a\},
\]

for all \(s,s'\in\mathcal{S},\, r\in\mathcal{R},\, a\in\mathcal{A}\). With it, \(\mathbb{E}[\,\cdot\mid S_t=s,A_t=a] = \sum_{s',r} p(s',r\mid s,a)[\,\cdot\,]\).

## Bellman optimality for \(v_*\)

For the optimal value function:

\[
v_*(s) = \max_a \sum_{s',r} p(s',r\mid s,a)\,[\,r + \gamma v_*(s')\,]
      = \max_a \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1}) \mid S_t=s, A_t=a].
\]

When \(p\) is deterministic the sum collapses to a single \((s',r)\).
