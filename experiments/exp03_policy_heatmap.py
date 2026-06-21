"""Experiment 3: the main result, a heatmap of where the guard learns to watch.

It trains the guard, draws how often it watches each room, and saves
``figures/exp03_policy_heatmap.png``.
"""
from __future__ import annotations

import os

from experiments.config import ExperimentConfig
from museum_heist.agents.softmax_reinforce import SoftmaxReinforceAgent
from museum_heist.envs.museum_heist_env import MuseumHeistEnv
from museum_heist.envs.topology import open_grid
from museum_heist.utils.evaluation import evaluate_policy
from museum_heist.utils.plotting import plot_room_probability_heatmap
from museum_heist.utils.training import train


def run() -> None:
    cfg = ExperimentConfig()
    topology = open_grid(cfg.grid_height, cfg.grid_width)
    env = MuseumHeistEnv(topology, beta=cfg.beta)
    agent = SoftmaxReinforceAgent(
        topology.n_rooms, tau=cfg.tau, alpha=cfg.alpha, gamma=cfg.gamma,
        batch_size=cfg.batch_size, seed=0,
    )
    train(agent, env, cfg.n_episodes, max_steps_per_episode=cfg.max_steps, seed=0)

    # Stochastic evaluation: the fair operating point for a strategic game.
    result = evaluate_policy(agent, env, 3000, max_steps=cfg.max_steps, seed=123, greedy=False)
    # Uniform baseline: an untrained agent (theta=0) watches every room the same.
    uniform = SoftmaxReinforceAgent(topology.n_rooms, tau=cfg.tau, seed=0)
    ubase = evaluate_policy(uniform, env, 3000, max_steps=cfg.max_steps, seed=123, greedy=False)
    print(f"uniform guard (stochastic): catch_rate={ubase.catch_rate:.3f}")
    print(f"trained guard (stochastic): catch_rate={result.catch_rate:.3f}  "
          f"escape_rate={result.escape_rate:.3f}  timeout_rate={result.timeout_rate:.3f}  "
          f"mean_time_to_catch={result.mean_time_to_catch:.2f}")

    figure = plot_room_probability_heatmap(
        agent.action_probabilities(), topology,
        title=f"Learned watch distribution - open {cfg.grid_height}x{cfg.grid_width}",
    )
    os.makedirs("figures", exist_ok=True)
    out_path = "figures/exp03_policy_heatmap.png"
    figure.savefig(out_path, dpi=120)
    print(f"saved {out_path}")


if __name__ == "__main__":
    run()
