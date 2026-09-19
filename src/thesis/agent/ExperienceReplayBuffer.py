import collections
import typing
import numpy as np

_field_names = ["state", "action", "reward", "next_state", "done"]
Experience = collections.namedtuple("Experience", field_names=_field_names)


class ExperienceReplayBuffer:

    def __init__(
        self,
        batch_size: int,
        buffer_size: int,
        alpha: float,
        random_state: np.random.RandomState,
    ) -> None:
        self._batch_size = batch_size
        self._buffer_size = buffer_size
        self._alpha = alpha
        self._random_state = (
            np.random.RandomState() if random_state is None else random_state
        )
        self._buffer = np.empty(
            self.buffer_size,
            dtype=[("priority", np.float32), ("experience", Experience)],
        )
        self._buffer_length = 0
        self._ptr = 0
        self._max_priority = 1.0  # track max TD priority, not boosted inserts (avoid exponential)

    def __len__(self) -> int:
        """Current number of prioritized experience tuple stored in buffer."""
        return self._buffer_length

    @property
    def alpha(self):
        """Strength of prioritized sampling."""
        return self._alpha

    @property
    def batch_size(self) -> int:
        """Number of experience samples per training batch."""
        return self._batch_size

    @property
    def buffer_size(self) -> int:
        """Maximum number of prioritized experience tuples stored in buffer."""
        return self._buffer_size

    def is_empty(self) -> bool:
        """True if the buffer is empty; False otherwise."""
        return self._buffer_length == 0

    def is_full(self) -> bool:
        """True if the buffer is full; False otherwise."""
        return self._buffer_length == self._buffer_size

    def add(self, experience: Experience) -> None:
        """Add a new experience to memory."""
        # ponytail: usa _max_priority (TD max) non max del buffer (che può essere boosted) per evitare overflow
        priority = 1.0 if self.is_empty() else float(self._max_priority)
        if not np.isfinite(priority) or priority <= 0:
            priority = 1.0
        priority = float(np.clip(priority, 1e-6, 1e6))
        self._buffer[self._ptr] = (np.float32(priority), experience)

        self._ptr = (self._ptr + 1) % self._buffer_size
        self._buffer_length = min(self._buffer_length + 1, self._buffer_size)

    def add_with_priority(self, experience: Experience, priority: float) -> None:
        """Add with explicit priority (for PER boost on V_llm states)."""
        # ponytail: clip to avoid overflow/NaN (float32 max ~3e38, but TD ~1)
        p = float(priority)
        if not np.isfinite(p) or p <= 0:
            p = 1.0
        p = float(np.clip(p, 1e-6, 1e6))
        self._buffer[self._ptr] = (np.float32(p), experience)
        self._ptr = (self._ptr + 1) % self._buffer_size
        self._buffer_length = min(self._buffer_length + 1, self._buffer_size)
        # don't update _max_priority here for boosted inserts to avoid exponential

    def add_boosted(self, experience: Experience, is_known: bool, boost: float = 5.0) -> None:
        """Add with boosted priority if state is in V_llm dict.
        boost=5.0 -> ~1.9x sampling with alpha=0.4, ~2.6x with alpha=0.6 (Schaul 2015).
        ponytail: base = _max_priority (TD max), not current max (which may be boosted),
                  evita crescita esponenziale 5^k che causava overflow/NaN.
        """
        boost = float(boost)
        if self.is_empty():
            base = 1.0
        else:
            base = float(self._max_priority)
            if not np.isfinite(base) or base <= 0:
                base = 1.0
        if is_known:
            # per stati V_llm: priorità più alta di base*boost, ma clip
            prio = base * boost
        else:
            prio = base
        # also ensure at least 1.0
        prio = float(np.clip(prio, 1e-6, 1e6))
        self.add_with_priority(experience, prio)

    def sample(self, beta: float) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of experiences from memory."""
        # use sampling scheme to determine which experiences to use for learning
        ps = self._buffer[: self._buffer_length]["priority"].astype(np.float64)
        # ensure finite positives
        ps = np.nan_to_num(ps, nan=1.0, posinf=1e6, neginf=1e-6)
        ps = np.clip(ps, 1e-6, 1e6)
        # power
        ps_alpha = np.power(ps, self._alpha)
        denom = np.sum(ps_alpha)
        if not np.isfinite(denom) or denom == 0:
            # fallback uniform
            sampling_probs = np.ones_like(ps_alpha) / len(ps_alpha)
        else:
            sampling_probs = ps_alpha / denom
            # ensure normalized and no NaN
            sampling_probs = np.nan_to_num(sampling_probs, nan=1.0/len(ps_alpha))
            s = sampling_probs.sum()
            if s != 0:
                sampling_probs = sampling_probs / s
        idxs = self._random_state.choice(
            np.arange(ps.size), size=self._batch_size, replace=True, p=sampling_probs
        )

        # select the experiences and compute sampling weights
        experiences = self._buffer["experience"][idxs]
        # avoid zero probs
        probs = np.clip(sampling_probs[idxs], 1e-12, 1.0)
        weights = (self._buffer_length * probs) ** -beta
        # handle inf/nan
        weights = np.nan_to_num(weights, nan=1.0, posinf=1e6, neginf=1e-6)
        wmax = weights.max()
        normalized_weights = weights / wmax if wmax != 0 else weights

        return idxs, experiences, normalized_weights

    def update_priorities(self, idxs: np.ndarray, priorities: np.ndarray) -> None:
        """Update the priorities associated with particular experiences."""
        # clip and ensure finite
        p = np.asarray(priorities, dtype=np.float32)
        p = np.nan_to_num(p, nan=1.0, posinf=1e6, neginf=1e-6)
        p = np.clip(p, 1e-6, 1e6)
        self._buffer["priority"][idxs] = p.astype(np.float32)
        # track max for future inserts
        try:
            m = float(np.max(p))
            if np.isfinite(m) and m > self._max_priority:
                self._max_priority = m
            # also consider existing buffer max may be higher (e.g., from previous boosts)
            # but we keep _max_priority as max TD, not boosted, so don't update from buffer
        except Exception:
            pass
