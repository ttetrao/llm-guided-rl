# The Action-Value Function also known as Q-Function

In *Reinforcement Learning: An Introduction* by Richard S. Sutton and Andrew G. Barto (second edition), the action-value function of a policy $\pi$, denoted $q_\pi(s, a)$, is the expected return starting from state $s$, taking action $a$, and thereafter following $\pi$:
$q_\pi(s, a) \doteq \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a]$, where $G_t$ is the discounted return, $\gamma \in [0,1]$ the discount factor. While $v_\pi(s)$ evaluates a state, $q_\pi(s,a)$ evaluates a particular action in that state. The two are related by $v_\pi(s) = \sum_a \pi(a \mid s) \, q_\pi(s,a)$ and $q_\pi(s, a) = \mathbb{E}_\pi[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s, A_t = a]$.

## Bellman Equation for $q_\pi$

$q_\pi(s, a) = \sum_{s',r} p(s',r \mid s,a)\,[r + \gamma \sum_{a'} \pi(a' \mid s') \, q_\pi(s', a')]$: the value of an action equals the expected immediate reward plus the discounted expected value thereafter under $\pi$.

## Optimal Action-Value Function

For finite MDPs there exists at least one optimal policy $\pi_*$. All optimal policies share the same optimal action-value function $q_*(s, a) \doteq \max_\pi q_\pi(s, a)$: the maximum expected return from $(s,a). It is related to the optimal state-value function by $v_*(s) = \max_a q_*(s,a)$. If $q_*$ is known, an optimal policy acts greedily: $\pi_*(s) \in \arg\max_a q_*(s,a)$.

## Bellman Optimality Equation

$q_*(s, a) = \mathbb{E}[R_{t+1} + \gamma \max_{a'} q_*(S_{t+1}, a') \mid S_t = s, A_t = a]$
$= \sum_{s',r} p(s',r \mid s,a)\,[r + \gamma \max_{a'} q_*(s', a')]$: after the first step the agent behaves optimally.

## Transition model

$p(s',r \mid s,a) \doteq \Pr\{S_{t+1}=s',\, R_{t+1}=r \mid S_t=s,\, A_t=a\}$ for all $s,s',r,a$.

## Application note (DoorKey, not Sutton & Barto)

This MDP is deterministic: the sum over $(s',r)$ collapses to the single successor $(s',r)$, so $q_*(s,a) = r + \gamma v_*(s')$ with $r=1$ only when $s'$ is the goal.
