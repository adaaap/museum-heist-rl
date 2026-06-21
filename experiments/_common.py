"""Shared helper for the experiments: train one guard and report its scores.

It keeps the experiment scripts short. Every experiment trains the guard the
same way (REINFORCE with a baseline).
"""
from __future__ import annotations

from museum_heist.agents.softmax_reinforce import SoftmaxReinforceAgent
from museum_heist.envs.museum_heist_env import MuseumHeistEnv
from museum_heist.envs.topology import MuseumTopology
from museum_heist.utils.evaluation import evaluate_policy, policy_entropy
from museum_heist.utils.training import train

SWEEP_SEEDS = tuple(range(8))  # 0..7, the seed set used for the sweeps (many configs)


def train_and_eval(topology: MuseumTopology, *, tau: float = 1.0, beta: float = 1.0,
                   alpha: float = 0.2, gamma: float = 0.99, batch_size: int = 16,
                   n_episodes: int = 3000, max_steps: int = 50, eval_episodes: int = 600,
                   seed: int = 0) -> dict:
    """Train a REINFORCE-with-baseline guard and return its stochastic/greedy KPIs and entropy."""
    env = MuseumHeistEnv(topology, beta=beta)
    agent = SoftmaxReinforceAgent(
        topology.n_rooms, tau=tau, alpha=alpha, gamma=gamma,
        batch_size=batch_size, seed=seed,
    )
    train(agent, env, n_episodes, max_steps_per_episode=max_steps, seed=seed)
    # Paired comparison: score BOTH policies on the SAME episode stream (one eval
    # seed), so the stochastic-vs-greedy gap comes from the policy alone, not from
    # two different thief runs.
    eval_seed = 3000 + seed
    stochastic = evaluate_policy(agent, env, eval_episodes, max_steps=max_steps, seed=eval_seed, greedy=False)
    greedy = evaluate_policy(agent, env, eval_episodes, max_steps=max_steps, seed=eval_seed, greedy=True)
    return {
        "catch_rate_stochastic": stochastic.catch_rate,
        "catch_rate_greedy": greedy.catch_rate,
        "entropy": policy_entropy(agent.action_probabilities()),
        "agent": agent,
    }
