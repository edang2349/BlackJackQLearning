"""
Runs every experiment for the report (parts C-H) and writes plots and
tables to results/.

    python experiments.py              # full run (10 seeds, ~3 min on 16 cores)
    python experiments.py --seeds 3    # quicker
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from qlearning import (BASELINE, Config, QLearningAgent, agreement_with_basic,
                       basic_strategy, dealer_rule, evaluate,
                       random_policy_factory, train)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

VAL_SEED, VAL_EPISODES = 2024, 50_000      # used to pick the best configuration
TEST_SEED, TEST_EPISODES = 99_999, 100_000  # used only for the final report numbers
MA_WINDOW, MA_STRIDE = 5_000, 500

# Each experiment changes one setting of the baseline.
ALPHAS = [0.001, 0.01, 0.1, 0.5, 0.9]
GAMMAS = [0.5, 0.9, 0.99, 1.0]
EPS_SCHEDULES = {  # label -> (eps_start, eps_end, eps_decay_frac)
    "constant ε=0.1": (0.1, 0.1, 0.0),
    "decay over 10%": (1.0, 0.05, 0.1),
    "decay over 50%": (1.0, 0.05, 0.5),
    "decay over 90%": (1.0, 0.05, 0.9),
}
NOISE_LEVELS = [0.0, 0.1, 0.25, 0.4]

# Reference palette, light mode (categorical order is fixed)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "legend.frameon": False, "lines.linewidth": 2,
})


# ----------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------

def moving_average(x, w=MA_WINDOW, stride=MA_STRIDE):
    c = np.cumsum(np.insert(x.astype(float), 0, 0.0))
    ma = (c[w:] - c[:-w]) / w
    idx = np.arange(0, len(ma), stride)
    return idx + w, ma[idx]


def run_one(job):
    cfg, seed = job
    agent, hist = train(cfg, seed=seed)
    greedy = lambda obs: agent.act(obs, explore=False)
    val = evaluate(greedy, VAL_EPISODES, seed=VAL_SEED, noise_p=cfg.noise_p)
    ma_x, ma_y = moving_average(hist["rewards"])
    return {
        "cfg": cfg, "seed": seed,
        "ma_x": ma_x, "ma_y": ma_y,
        "eval_x": hist["eval_x"], "eval_y": hist["eval_y"],
        "val": val["mean_reward"],
        "agreement": agreement_with_basic(greedy),
        "agent": agent.state_dict(),
    }


def run_configs(configs, seeds, pool):
    jobs = [(c, s) for c in configs for s in seeds]
    out = {c: [] for c in configs}
    for r in pool.imap_unordered(run_one, jobs):
        out[r["cfg"]].append(r)
    for c in out:
        out[c].sort(key=lambda r: r["seed"])
    return out


def summarize(runs):
    """Numbers that go into the comparison tables."""
    ex = runs[0]["eval_x"]
    ey = np.array([r["eval_y"] for r in runs])
    val = np.array([r["val"] for r in runs])

    def at(n):
        return float(ey[:, np.searchsorted(ex, n)].mean())

    late = ey[:, ex >= ex[-1] // 2]
    return {
        "eval@10k": at(10_000), "eval@50k": at(50_000), "eval@final": float(ey[:, -1].mean()),
        "val_mean": float(val.mean()), "val_std_across_seeds": float(val.std(ddof=1)),
        # how much the greedy policy's score wobbles in the 2nd half of training
        "late_fluctuation": float(late.std(axis=1, ddof=1).mean()),
        "agreement_basic": float(np.mean([r["agreement"] for r in runs])),
    }


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def band(ax, x, ys, color, label):
    ys = np.asarray(ys)
    m, s = ys.mean(0), ys.std(0)
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, linewidth=0)
    ax.plot(x, m, color=color, label=label)


def comparison_plot(groups, title, fname, ref=None):
    """groups: list of (label, runs). Three panels: training reward,
    greedy evaluation reward, and final validation reward."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3),
                             gridspec_kw={"width_ratios": [1.15, 1.15, 0.7]})
    for i, (label, runs) in enumerate(groups):
        c = SERIES[i]
        band(axes[0], runs[0]["ma_x"], [r["ma_y"] for r in runs], c, label)
        band(axes[1], runs[0]["eval_x"], [r["eval_y"] for r in runs], c, label)
        vals = [r["val"] for r in runs]
        axes[2].bar(i, np.mean(vals), yerr=np.std(vals, ddof=1), color=c, width=0.65,
                    error_kw={"ecolor": INK2, "capsize": 3, "linewidth": 1})
        axes[2].scatter(np.full(len(vals), i) + np.linspace(-0.18, 0.18, len(vals)), vals,
                        s=10, color=INK, zorder=3, linewidths=0)

    axes[0].set_title("Training reward (ε-greedy, %d-episode moving avg)" % MA_WINDOW)
    axes[1].set_title("Greedy policy reward during training")
    axes[2].set_title("Final greedy reward (%dk hands)" % (VAL_EPISODES // 1000))
    for ax in axes[:2]:
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Mean reward per hand")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%dk" % (v / 1000)))
    if ref is not None:
        for ax in axes[1:]:
            ax.axhline(ref, color=INK2, linestyle="--", linewidth=1)
        axes[1].text(axes[1].get_xlim()[0], ref, "  basic strategy", va="bottom",
                     ha="left", color=INK2, fontsize=8)
    axes[2].set_xticks(range(len(groups)), [g[0] for g in groups], rotation=20, ha="right")
    axes[2].set_ylabel("Mean reward per hand")
    lo = min(min(r["val"] for r in runs) for _, runs in groups)
    axes[2].set_ylim(min(lo, ref if ref is not None else lo) - 0.03, 0)
    axes[2].grid(axis="x", visible=False)
    axes[1].legend(loc="lower right", fontsize=9)
    fig.suptitle(title, fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, fname), dpi=150)
    plt.close(fig)


def baseline_plot(runs, cfg, ref):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    ax = axes[0]
    band(ax, runs[0]["ma_x"], [r["ma_y"] for r in runs], SERIES[0], "ε-greedy training reward")
    band(ax, runs[0]["eval_x"], [r["eval_y"] for r in runs], SERIES[1], "greedy policy reward")
    ax.axhline(ref, color=INK2, linestyle="--", linewidth=1, label="basic strategy")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Mean reward per hand")
    ax.set_title("Reward vs. training episode (mean ± 1 sd over %d seeds)" % len(runs))
    ax.legend(loc="lower right", fontsize=9)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%dk" % (v / 1000)))

    ax = axes[1]
    x = np.arange(cfg.n_episodes)
    ax.plot(x, [cfg.epsilon(e) for e in x], color=SERIES[0])
    ax.set_xlabel("Training episode")
    ax.set_ylabel("ε")
    ax.set_title("Exploration schedule")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%dk" % (v / 1000)))
    fig.suptitle("Baseline: α=%g, γ=%g, ε %g→%g over first %d%%, %dk episodes"
                 % (cfg.alpha, cfg.gamma, cfg.eps_start, cfg.eps_end,
                    100 * cfg.eps_decay_frac, cfg.n_episodes // 1000),
                 fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "C_baseline_training.png"), dpi=150)
    plt.close(fig)


def policy_grid(choose, usable):
    """rows = player sum 21..12, cols = dealer A..10; 1 = hit."""
    return np.array([[choose((p, d, usable)) for d in range(1, 11)] for p in range(21, 11, -1)])


def policy_plot(policies, fname, title):
    """policies: list of (label, choose). One column per policy, two rows (ace / no ace)."""
    cmap = matplotlib.colors.ListedColormap([SERIES[0], SERIES[1]])
    fig, axes = plt.subplots(2, len(policies), figsize=(3.6 * len(policies), 7))
    axes = np.atleast_2d(axes).reshape(2, -1)
    for j, (label, choose) in enumerate(policies):
        for i, usable in enumerate((1, 0)):
            ax = axes[i, j]
            g = policy_grid(choose, usable)
            ax.imshow(g, cmap=cmap, vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(10), ["A"] + [str(d) for d in range(2, 11)])
            ax.set_yticks(range(10), [str(p) for p in range(21, 11, -1)])
            ax.set_xticks(np.arange(-0.5, 10), minor=True)
            ax.set_yticks(np.arange(-0.5, 10), minor=True)
            ax.grid(which="minor", color=SURFACE, linewidth=2)
            ax.grid(which="major", visible=False)
            ax.tick_params(which="minor", length=0)
            for (r, c), v in np.ndenumerate(g):
                ax.text(c, r, "H" if v else "S", ha="center", va="center", color="white", fontsize=7)
            ax.set_title("%s\n%s" % (label, "usable ace" if usable else "no usable ace"), fontsize=10)
            ax.set_xlabel("Dealer showing")
            if j == 0:
                ax.set_ylabel("Player sum")
    handles = [matplotlib.patches.Patch(color=SERIES[0], label="S = Stick"),
               matplotlib.patches.Patch(color=SERIES[1], label="H = Hit")]
    fig.legend(handles=handles, loc="upper right", ncol=2)
    fig.suptitle(title, fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(RESULTS, fname), dpi=150)
    plt.close(fig)


def noise_plot(noise_runs, robust, fname):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    for i, (p, runs) in enumerate(noise_runs.items()):
        band(ax, runs[0]["eval_x"], [r["eval_y"] for r in runs], SERIES[i], "p = %g" % p)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Mean reward per hand (under noise p)")
    ax.set_title("Greedy policy reward during training")
    ax.set_ylim(top=0.04)  # room for the legend above the p=0 curve
    ax.legend(loc="upper center", ncol=len(noise_runs), fontsize=9, bbox_to_anchor=(0.5, 1.0))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: "%dk" % (v / 1000)))

    ax = axes[1]
    ps = list(robust.keys())
    x = np.arange(len(ps))
    w = 0.26
    keys = [("trained_at_p", "Trained under p"), ("p0_policy", "p=0 policy"),
            ("basic_strategy", "Basic strategy")]
    for k, (key, label) in enumerate(keys):
        m = [robust[p][key]["mean_reward"] for p in ps]
        e = [robust[p][key]["ci95"] for p in ps]
        ax.bar(x + (k - 1) * w, m, w * 0.92, yerr=e, color=SERIES[k], label=label,
               error_kw={"ecolor": INK2, "capsize": 2, "linewidth": 1})
    ax.set_xticks(x, ["p = %g" % p for p in ps])
    ax.set_ylabel("Mean reward per hand (%dk test hands)" % (TEST_EPISODES // 1000))
    ax.set_title("Robustness: evaluated under noise p")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", fontsize=9)
    fig.suptitle("Action noise: chosen action is flipped with probability p",
                 fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, fname), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def fmt_table(rows, cols):
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "".join("| " + " | ".join(str(c) for c in r) + " |\n" for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    seeds = list(range(args.seeds))
    t0 = time.time()

    ref = evaluate(basic_strategy, VAL_EPISODES, seed=VAL_SEED)["mean_reward"]

    alpha_cfgs = {"α = %g" % a: BASELINE.with_(alpha=a) for a in ALPHAS}
    gamma_cfgs = {"γ = %g" % g: BASELINE.with_(gamma=g) for g in GAMMAS}
    eps_cfgs = {k: BASELINE.with_(eps_start=s, eps_end=e, eps_decay_frac=f)
                for k, (s, e, f) in EPS_SCHEDULES.items()}
    experiments = {"D_alpha": alpha_cfgs, "E_gamma": gamma_cfgs, "F_epsilon": eps_cfgs}
    all_cfgs = {BASELINE} | {c for exp in experiments.values() for c in exp.values()}

    with Pool(args.workers) as pool:
        print("Training %d configurations x %d seeds..." % (len(all_cfgs), len(seeds)))
        runs = run_configs(sorted(all_cfgs, key=str), seeds, pool)
        print("  done in %.0fs" % (time.time() - t0))

        # ---- C: baseline
        baseline_plot(runs[BASELINE], BASELINE, ref)

        # ---- D, E, F
        titles = {"D_alpha": "Learning rate α (γ=1, ε decay over 50%)",
                  "E_gamma": "Discount factor γ (α=0.01, ε decay over 50%)",
                  "F_epsilon": "Exploration schedule (α=0.01, γ=1)"}
        summary = {"baseline": BASELINE.as_dict(), "basic_strategy_val": ref, "experiments": {}}
        for name, cfgs in experiments.items():
            comparison_plot([(k, runs[c]) for k, c in cfgs.items()], titles[name], name + ".png", ref)
            summary["experiments"][name] = {k: summarize(runs[c]) for k, c in cfgs.items()}

        # ---- G: best configuration = highest mean validation reward over seeds
        label_of = {c: k for cfgs in experiments.values() for k, c in cfgs.items()}
        label_of[BASELINE] = "baseline"
        ranking = sorted(runs, key=lambda c: -summarize(runs[c])["val_mean"])
        best_cfg = ranking[0]
        best_run = max(runs[best_cfg], key=lambda r: r["val"])
        base_run = max(runs[BASELINE], key=lambda r: r["val"])
        summary["ranking"] = [{"config": label_of[c], **c.as_dict(), **summarize(runs[c])}
                              for c in ranking]
        best_agent = QLearningAgent.from_state_dict(best_run["agent"])
        base_agent = QLearningAgent.from_state_dict(base_run["agent"])
        best_pi = lambda o: best_agent.act(o, explore=False)
        base_pi = lambda o: base_agent.act(o, explore=False)

        test = {
            "best Q-learning agent": evaluate(best_pi, TEST_EPISODES, TEST_SEED),
            "baseline Q-learning agent": evaluate(base_pi, TEST_EPISODES, TEST_SEED),
            "basic strategy": evaluate(basic_strategy, TEST_EPISODES, TEST_SEED),
            "dealer rule (hit < 17)": evaluate(dealer_rule, TEST_EPISODES, TEST_SEED),
            "random": evaluate(random_policy_factory(0), TEST_EPISODES, TEST_SEED),
            "best agent, 30 hands": evaluate(best_pi, 30, TEST_SEED),
        }
        test["best Q-learning agent"]["agreement_basic"] = agreement_with_basic(best_pi)
        test["baseline Q-learning agent"]["agreement_basic"] = agreement_with_basic(base_pi)
        summary["best"] = {"config": label_of[best_cfg], **best_cfg.as_dict(),
                           "seed": best_run["seed"], "val": best_run["val"]}
        summary["baseline_agent_seed"] = base_run["seed"]
        summary["test"] = test
        with open(os.path.join(RESULTS, "best_agent.json"), "w") as f:
            json.dump({"config": best_cfg.as_dict(), "seed": best_run["seed"],
                       "q": [[list(s), v] for s, v in best_run["agent"]["q"].items()],
                       "alpha": best_cfg.alpha, "gamma": best_cfg.gamma,
                       "q_init": best_cfg.q_init}, f)
        policy_plot([("Best Q-learning agent", best_pi), ("Baseline agent", base_pi),
                     ("Basic strategy", basic_strategy)],
                    "G_policies.png", "Learned greedy policies vs. basic strategy")
        gamma_low = max(runs[gamma_cfgs["γ = 0.5"]], key=lambda r: r["val"])
        g_agent = QLearningAgent.from_state_dict(gamma_low["agent"])
        policy_plot([("γ = 0.5", lambda o: g_agent.act(o, explore=False)),
                     ("γ = 1", base_pi)], "E_gamma_policies.png",
                    "Effect of γ on the learned policy")

        # ---- H: action noise, trained with the best configuration
        print("Training under action noise...")
        noise_cfgs = {p: best_cfg.with_(noise_p=p) for p in NOISE_LEVELS}
        nr = run_configs(list(noise_cfgs.values()), seeds, pool)
        noise_runs = {p: nr[c] for p, c in noise_cfgs.items()}

    robust = {}
    p0_run = max(noise_runs[0.0], key=lambda r: r["val"])
    p0_agent = QLearningAgent.from_state_dict(p0_run["agent"])
    p0_pi = lambda o: p0_agent.act(o, explore=False)
    noisy_agents = {}
    for p, rs in noise_runs.items():
        best_p = max(rs, key=lambda r: r["val"])
        a = QLearningAgent.from_state_dict(best_p["agent"])
        noisy_agents[p] = a
        pi = lambda o, a=a: a.act(o, explore=False)
        robust[p] = {
            "trained_at_p": evaluate(pi, TEST_EPISODES, TEST_SEED, noise_p=p),
            "p0_policy": evaluate(p0_pi, TEST_EPISODES, TEST_SEED, noise_p=p),
            "basic_strategy": evaluate(basic_strategy, TEST_EPISODES, TEST_SEED, noise_p=p),
            "agreement_basic": agreement_with_basic(pi),
            "agreement_p0": float(np.mean([pi(s) == p0_pi(s) for s in
                                           [(q, d, u) for u in (0, 1) for q in range(12, 22)
                                            for d in range(1, 11)]])),
        }
        robust[p]["training"] = summarize(rs)
    noise_plot(noise_runs, robust, "H_noise.png")
    policy_plot([("Trained at p = %g" % p, lambda o, a=noisy_agents[p]: a.act(o, explore=False))
                 for p in NOISE_LEVELS], "H_noise_policies.png",
                "Greedy policies learned under different action-noise levels")
    summary["noise"] = {str(p): v for p, v in robust.items()}

    with open(os.path.join(RESULTS, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    write_tables(summary)
    print("All done in %.0fs. Results in %s" % (time.time() - t0, RESULTS))


def write_tables(s):
    out = ["# Results tables\n", "Basic-strategy reference (validation hands): %.4f\n" % s["basic_strategy_val"]]
    cols = ["setting", "greedy @10k", "greedy @50k", "greedy @200k", "val mean", "val sd (seeds)",
            "late fluctuation", "agree w/ basic"]
    for name, rows in s["experiments"].items():
        out.append("\n## %s\n\n" % name)
        out.append(fmt_table([[k, "%.4f" % v["eval@10k"], "%.4f" % v["eval@50k"],
                               "%.4f" % v["eval@final"], "%.4f" % v["val_mean"],
                               "%.4f" % v["val_std_across_seeds"], "%.4f" % v["late_fluctuation"],
                               "%.1f%%" % (100 * v["agreement_basic"])] for k, v in rows.items()], cols))
    out.append("\n## Ranking (by mean validation reward)\n\n")
    out.append(fmt_table([[r["config"], "%.4f" % r["val_mean"], "%.4f" % r["val_std_across_seeds"]]
                          for r in s["ranking"]], ["config", "val mean", "val sd"]))
    out.append("\n## G: test evaluation (best = %s, seed %d)\n\n" % (s["best"]["config"], s["best"]["seed"]))
    out.append(fmt_table([[k, v["episodes"], "%.4f ± %.4f" % (v["mean_reward"], v["ci95"]),
                           "%.1f%%" % (100 * v["win_rate"]), "%.1f%%" % (100 * v["draw_rate"]),
                           "%.1f%%" % (100 * v["loss_rate"]), "%.1f%%" % (100 * v["bust_rate"]),
                           "%.2f" % v["mean_steps"]] for k, v in s["test"].items()],
                         ["policy", "hands", "mean reward ± 95% CI", "win", "draw", "loss", "bust",
                          "actions/hand"]))
    out.append("\n## H: action noise (test hands, evaluated under the same p)\n\n")
    rows = []
    for p, v in s["noise"].items():
        rows.append([p, "%.4f" % v["trained_at_p"]["mean_reward"], "%.4f" % v["p0_policy"]["mean_reward"],
                     "%.4f" % v["basic_strategy"]["mean_reward"], "%.4f" % v["training"]["eval@10k"],
                     "%.4f" % v["training"]["eval@50k"], "%.1f%%" % (100 * v["agreement_p0"]),
                     "%.1f%%" % (100 * v["trained_at_p"]["bust_rate"])])
    out.append(fmt_table(rows, ["p", "trained at p", "p=0 policy", "basic strategy", "greedy @10k",
                                "greedy @50k", "agree w/ p=0 policy", "bust rate"]))
    with open(os.path.join(RESULTS, "tables.md"), "w", encoding="utf-8") as f:
        f.write("".join(out))


if __name__ == "__main__":
    main()
