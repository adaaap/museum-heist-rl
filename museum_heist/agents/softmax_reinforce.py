"""The guard's learning agent: a softmax policy trained with REINFORCE.

The guard keeps one number per room, which says how much it wants to watch that
room. A softmax turns those numbers into watch probabilities, and a temperature
``tau`` sets how sharp the choice is. After each game the guard nudges the
numbers toward the rooms that helped catch the thief; this is REINFORCE. It also
subtracts a running average of past results (a baseline), which makes learning
steadier without changing what it learns.
"""
from __future__ import annotations

import numpy as np


class SoftmaxReinforceAgent:
    def __init__(self, n_actions: int, *, tau: float = 1.0, alpha: float = 0.1,
                 gamma: float = 0.99, baseline: bool = True, batch_size: int = 1,
                 seed: int | None = None) -> None:
        if n_actions < 1:
            raise ValueError(f"n_actions must be >= 1, got {n_actions}.")
        if tau <= 0:
            raise ValueError(f"tau must be > 0, got {tau}.")
        if not (0 < alpha <= 1):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}.")
        if not (0 < gamma <= 1):
            raise ValueError(f"gamma must be in (0, 1], got {gamma}.")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
        self.n_actions = int(n_actions)
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.baseline = bool(baseline)
        self.batch_size = int(batch_size)
        self.theta = np.zeros(self.n_actions, dtype=np.float64)  # one logit per room
        self._rng = np.random.default_rng(seed)
        self._episode: list[tuple[int, float]] = []   # (action, reward) for the current episode
        self._batch_grads: list[np.ndarray] = []       # per-trajectory gradients awaiting a batch step
        self._baseline_value = 0.0                      # running mean of trajectory returns
        self._baseline_count = 0

    def reset_rng(self, seed: int | None) -> None:
        """Reseed the action-sampling RNG so the harness can make a run reproducible."""
        self._rng = np.random.default_rng(seed)

    # --------------------------- policy --------------------------- #
    def action_probabilities(self) -> np.ndarray:
        """Current softmax distribution over rooms (numerically stable)."""
        logits = self.theta / self.tau
        logits = logits - logits.max()  # max-subtraction for stability
        weights = np.exp(logits)
        return weights / weights.sum()

    def score(self, action: int, probs: np.ndarray | None = None) -> np.ndarray:
        """``grad_theta log pi_theta(action) = (e_action - pi_theta) / tau``."""
        if probs is None:
            probs = self.action_probabilities()
        grad = -probs.copy()
        grad[action] += 1.0
        return grad / self.tau

    def select_action(self, state, *, greedy: bool = False) -> int:
        if greedy:
            # Deterministic argmax tie-break. It uses no RNG, so greedy eval
            # depends only on the env seed.
            best = np.flatnonzero(self.theta == self.theta.max())
            return int(best[0])
        probs = self.action_probabilities()
        return int(self._rng.choice(self.n_actions, p=probs))

    # ------------------------- learning --------------------------- #
    def update(self, state, action, reward, next_state, terminated, *, next_action=None) -> None:
        # Monte-Carlo: just buffer the transition; the gradient step is at episode end.
        self._episode.append((int(action), float(reward)))

    def end_episode(self) -> None:
        if self._episode:
            self._batch_grads.append(self._trajectory_gradient())
            if len(self._batch_grads) >= self.batch_size:
                mean_grad = np.mean(self._batch_grads, axis=0)
                self.theta += self.alpha * mean_grad  # gradient ASCENT on J(theta)
                self._batch_grads.clear()
        self._episode = []

    def _trajectory_gradient(self) -> np.ndarray:
        """REINFORCE-with-baseline estimator: ``(sum_t grad log pi(a_t)) * (R - b)``.

        The behaviour policy is the current theta, because we only update between
        episodes. So recomputing the scores here is on-policy and unbiased. The
        baseline ``b`` is the running mean of *previous* trajectory returns. It does
        not depend on this trajectory, so the estimator stays unbiased and has lower
        variance.
        """
        actions = [a for a, _ in self._episode]
        rewards = [r for _, r in self._episode]
        probs = self.action_probabilities()
        traj_return = sum((self.gamma ** t) * r for t, r in enumerate(rewards))
        baseline = self._baseline_value if self.baseline else 0.0
        score_sum = np.zeros(self.n_actions)
        for a in actions:
            score_sum += self.score(a, probs)
        self._update_baseline(traj_return)
        return score_sum * (traj_return - baseline)

    def _update_baseline(self, traj_return: float) -> None:
        # Running mean of the discounted trajectory return. It is used as the
        # baseline for the NEXT trajectory, so the current estimate stays unbiased.
        self._baseline_count += 1
        self._baseline_value += (traj_return - self._baseline_value) / self._baseline_count
