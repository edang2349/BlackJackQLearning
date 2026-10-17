"""
Assignment 2 - Tabular Q-learning on Gymnasium Blackjack-v1

Observation: (player_sum, dealer_showing, usable_ace)
    player_sum      4..21 before the agent acts (up to 31 after a bust)
    dealer_showing  1..10 (1 = ace)
    usable_ace      0 or 1
Actions: 0 = stick, 1 = hit

The observation is already discrete, so no discretization is needed; the
Q-table is a dict keyed by the observation tuple.
"""

import random
from collections import defaultdict
from dataclasses import dataclass, asdict, replace

import gymnasium as gym
import numpy as np

STICK, HIT = 0, 1
ACTION_NAMES = {STICK: "Stick", HIT: "Hit"}


# ----------------------------------------------------------------------
# Hyperparameters
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    alpha: float = 0.01          # learning rate
    gamma: float = 1.0           # discount factor
    eps_start: float = 1.0       # epsilon at episode 0
    eps_end: float = 0.05        # epsilon after the decay finishes
    eps_decay_frac: float = 0.5  # fraction of training spent decaying (0 = constant eps_end)
    n_episodes: int = 200_000
    q_init: float = 0.0          # initial value of every Q(s, a)
    noise_p: float = 0.0         # probability the chosen action is flipped (part H)

    def epsilon(self, episode):
        """Linear decay from eps_start to eps_end, then constant."""
        decay_len = self.eps_decay_frac * self.n_episodes
        if decay_len <= 0:
            return self.eps_end
        frac = min(1.0, episode / decay_len)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def with_(self, **changes):
        return replace(self, **changes)

    def as_dict(self):
        return asdict(self)


BASELINE = Config()


# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------

class ActionNoise(gym.ActionWrapper):
    """With probability p the intended action is replaced by the other action.

    Blackjack is already stochastic (the deck), so this is a separate,
    controllable source of uncertainty in the agent's own actions.
    """

    def __init__(self, env, p, seed=None):
        super().__init__(env)
        self.p = p
        self.rng = np.random.default_rng(seed)

    def action(self, action):
        if self.p > 0 and self.rng.random() < self.p:
            return 1 - action
        return action


def make_env(noise_p=0.0, seed=None, render_mode=None):
    # Default Sutton & Barto rules: infinite deck, dealer sticks on 17,
    # natural blackjack pays 1 (not 1.5), no doubling or splitting.
    env = gym.make("Blackjack-v1", natural=False, sab=False, render_mode=render_mode)
    if noise_p > 0:
        env = ActionNoise(env, noise_p, seed=seed)
    return env


# ----------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------

class QLearningAgent:
    def __init__(self, n_actions=2, alpha=0.01, gamma=1.0, epsilon=1.0,
                 q_init=0.0, seed=None):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_init = q_init
        self.rng = random.Random(seed)
        self.q = defaultdict(self._new_row)

    def _new_row(self):
        return np.full(self.n_actions, self.q_init, dtype=float)

    def greedy(self, obs):
        """Best action; ties are broken at random."""
        row = self.q[obs]
        best = np.flatnonzero(row == row.max())
        return int(best[0]) if len(best) == 1 else self.rng.choice(best.tolist())

    def act(self, obs, explore=True):
        """Epsilon-greedy action. explore=False gives the greedy policy."""
        if explore and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_actions)
        return self.greedy(obs)

    def update(self, obs, action, reward, next_obs, terminated):
        """Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)].

        A terminal next state has value 0, so the target is just r.
        """
        target = reward
        if not terminated:
            target += self.gamma * self.q[next_obs].max()
        self.q[obs][action] += self.alpha * (target - self.q[obs][action])

    def policy_table(self):
        """Plain dict {obs: greedy action} for every visited state."""
        return {s: int(np.argmax(v)) for s, v in self.q.items()}

    def state_dict(self):
        return {"q": {s: v.tolist() for s, v in self.q.items()},
                "alpha": self.alpha, "gamma": self.gamma, "q_init": self.q_init}

    @classmethod
    def from_state_dict(cls, d):
        agent = cls(alpha=d["alpha"], gamma=d["gamma"], q_init=d["q_init"], epsilon=0.0)
        for s, v in d["q"].items():
            agent.q[tuple(s)] = np.array(v, dtype=float)
        return agent


# ----------------------------------------------------------------------
# Training and evaluation
# ----------------------------------------------------------------------

def play_hand(env, choose, agent=None):
    """Play one hand. Returns (reward, n_actions, last_info).

    choose(obs) -> action. If agent is given, every transition is used
    for a Q-learning update.
    """
    obs, _ = env.reset()
    total, steps = 0.0, 0
    while True:
        action = choose(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        if agent is not None:
            agent.update(obs, action, reward, next_obs, terminated)
        total += reward
        steps += 1
        obs = next_obs
        if terminated or truncated:
            return total, steps, obs


def evaluate(choose, n_episodes=10_000, seed=12345, noise_p=0.0):
    """Run a fixed policy (no learning, no exploration).

    Using the same seed for every policy means they all see the same
    card sequence (common random numbers), which makes comparisons sharper.
    """
    env = make_env(noise_p, seed=seed + 1)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    rewards = np.empty(n_episodes)
    steps = np.empty(n_episodes)
    busts = 0
    for i in range(n_episodes):
        r, k, final_obs = play_hand(env, choose)
        rewards[i] = r
        steps[i] = k
        if final_obs[0] > 21:
            busts += 1
    env.close()
    n = n_episodes
    return {
        "episodes": n,
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std(ddof=1)),
        "ci95": float(1.96 * rewards.std(ddof=1) / np.sqrt(n)),
        "win_rate": float((rewards > 0).mean()),
        "draw_rate": float((rewards == 0).mean()),
        "loss_rate": float((rewards < 0).mean()),
        "bust_rate": busts / n,
        "mean_steps": float(steps.mean()),
    }


def train(cfg, seed=0, eval_every=10_000, eval_episodes=5_000, eval_seed=777):
    """Train one agent. Returns (agent, history).

    history["rewards"]  reward of every training episode (with exploration)
    history["eval_x"], history["eval_y"]  greedy-policy reward measured
        every eval_every episodes on a fixed set of hands
    """
    env = make_env(cfg.noise_p, seed=seed + 10_000)
    env.reset(seed=seed)
    agent = QLearningAgent(alpha=cfg.alpha, gamma=cfg.gamma, epsilon=cfg.eps_start,
                           q_init=cfg.q_init, seed=seed)

    rewards = np.empty(cfg.n_episodes, dtype=np.float32)
    eval_x, eval_y = [], []
    greedy = lambda obs: agent.act(obs, explore=False)

    for ep in range(cfg.n_episodes):
        agent.epsilon = cfg.epsilon(ep)
        rewards[ep], _, _ = play_hand(env, agent.act, agent)
        if eval_every and ((ep + 1) % eval_every == 0 or ep == 0):
            res = evaluate(greedy, eval_episodes, seed=eval_seed, noise_p=cfg.noise_p)
            eval_x.append(ep + 1)
            eval_y.append(res["mean_reward"])
    env.close()
    return agent, {"rewards": rewards, "eval_x": np.array(eval_x), "eval_y": np.array(eval_y)}


# ----------------------------------------------------------------------
# Reference policies
# ----------------------------------------------------------------------

def basic_strategy(obs):
    """Hit/stick basic strategy for these rules (Sutton & Barto, Fig. 5.2)."""
    total, dealer, usable = obs
    if usable:
        if total >= 19:
            return STICK
        if total == 18:
            return STICK if 2 <= dealer <= 8 else HIT
        return HIT
    if total >= 17:
        return STICK
    if 13 <= total <= 16:
        return STICK if 2 <= dealer <= 6 else HIT
    if total == 12:
        return STICK if 4 <= dealer <= 6 else HIT
    return HIT


def dealer_rule(obs):
    """Copy the dealer: hit below 17."""
    return HIT if obs[0] < 17 else STICK


def random_policy_factory(seed=0):
    rng = random.Random(seed)
    return lambda obs: rng.randrange(2)


def decision_states():
    """Every state where the choice matters (player sum 12..21)."""
    return [(p, d, u) for u in (0, 1) for p in range(12, 22) for d in range(1, 11)]


def agreement_with_basic(choose):
    states = decision_states()
    return float(np.mean([choose(s) == basic_strategy(s) for s in states]))
