# Plot each separate run on a different sub-axis, ordered by AUC

import pickle
from math import ceil
import seaborn as sns
import functools
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy
import json
import sys
import plot_utils as plot
import matplotlib as mpl
mpl.rcParams["font.size"] = 24

try:
    import hypers
    import runs
except ModuleNotFoundError:
    import utils.hypers
    import utils.runs

# Set up plots
params = {
      'axes.labelsize': 8,
      'axes.titlesize': 32,
      'legend.fontsize': 16,
      'xtick.labelsize': 24,
      'ytick.labelsize': 24
}
plt.rcParams.update(params)

plt.rc('text', usetex=False)  # You might want usetex=True to get DejaVu Sans
plt.rc('font', **{'family': 'sans-serif', 'serif': ['DejaVu Sans']})
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams.update({'font.size': 32})
plt.tick_params(top=False, right=False, labelsize=24)

mpl.rcParams["svg.fonttype"] = "none"

if len(sys.argv) != 4:
    raise ArgumentError("""should run ./plot_runs_separate.py
                        path/to/env_config save/dir path/to/agent_config
                        """)

env_json = sys.argv[1]
DIR = sys.argv[2]
agent_json = sys.argv[3]


def get_y_bounds(env, per_env_tuning):
    """
    Get the bounds for the y-axis plots on `env` given that `per_env_tuning`
    determines whether we are tuning per environment or across environments.
    """
    if per_env_tuning:
        if "mountaincar" in env.lower():
            return (-1000, -50)
        elif "acrobot" in env.lower():
            return (-1000, -50)
        elif "pendulum" in env.lower():
            return (-1000, 1000)
    else:
        if "mountaincar" in env.lower():
            return (-1000, -50)
        elif "acrobot" in env.lower():
            return (-1000, -50)
        elif "pendulum" in env.lower():
            return (-1000, 950)

    if "breakout" in env.lower():
        return (0, 25)


# Load configuration files
with open(env_json, "r") as infile:
    env_config = json.load(infile)

with open(agent_json, "r") as infile:
    agent_config = json.load(infile)
agent = agent_config["agent_name"]

ENV = env_config["env_name"]

# Uncomment the next lines if using ICML data
if agent == "GreedyAC":
    agent = "cem"
elif agent == "GreedyACSoftmax":
    agent = "cem_softmax"

if ENV == "Pendulum-v0":
    env = "PendulumFixed-v0"
else:
    env = ENV

if ENV == "MountainCarContinuous-v0":
    env_config["env_name"] = "MountainCar-v0"
    env_config["continuous"] = True
    ENV = env_config["env_name"]

# Script
if DIR:
    data_file = f"./results/{DIR}/{env}_{agent}results/data.pkl"
else:
    data_file = f"./results/{env}_{agent}results/data.pkl"
with open(data_file, "rb") as infile:
    data = pickle.load(infile)

# Find best hypers
# #################################
# For new runs
# #################################
best_hp = hypers.best(data)[0]
per_env_tuning = True

# Expand data to ensure episodic environments have the same number of data
# points per run
if "pendulum" not in ENV.lower():
    data = runs.expand_episodes(data, best_hp)
low_x = 0
if "pendulum" not in ENV.lower():
    high_x = np.cumsum(
        data["experiment_data"][best_hp]["runs"][0]["train_episode_steps"]
    )[-1]
else:
    high_x = len(
        data["experiment_data"][best_hp]["runs"][0]["train_episode_steps"]
    )

# Go through and get the list of hyperparameter indices ordered by AUC
num_runs = list(range(len(data["experiment_data"][best_hp]["runs"])))
auc = hypers.get_performance(data, best_hp, repeat=False).mean(axis=-1)
order = np.argsort(auc)

# Figure out the number of rows and columns for the subplots
num_plots = len(num_runs)
COLS = 4
ROWS = max(1, ceil(num_plots / COLS))
fig = plt.figure(figsize=(7 * COLS, 4.8 * ROWS), constrained_layout=True)
spec = fig.add_gridspec(ROWS, COLS)

# Plot
low_y, high_y = get_y_bounds(ENV, per_env_tuning)
returns = []
for i, run_num in enumerate(order):
    run = data["experiment_data"][best_hp]["runs"][run_num]

    # Figure out which row and column of the subplots we are on
    y = i // COLS
    x = i - y * COLS
    ax = fig.add_subplot(spec[y, x])

    if "pendulum" not in ENV.lower():
        # If an episodic environment, ignore the last episode, since it will be
        # cut off. We actually are cutting off too much here, but the
        # alternative is to iterate over the entire data set twice, since we
        # need to find the maximum steps for the last episode.
        cutoff = env_config["steps_per_episode"]
        performance = run["train_episode_rewards"][:-cutoff]
    else:
        performance = run["train_episode_rewards"]

    ax.plot(
        performance,
        label=f"Run {i}",
        linewidth=2.5,
        color="#007bff",
    )

    # Only set x ticks for bottom row
    if y == ROWS-1:
        ax.set_xticks([low_x, high_x])
    else:
        ax.set_xticks([])

    # Only set y ticks for leftmost column
    if x == 0:
        ax.set_yticks(get_y_bounds(ENV, per_env_tuning))
    else:
        ax.set_yticks([])

    # Set axis title and bounds
    ax.set_title(f"Run {i}")
    ax.set_xlim(low_x, high_x)
    ax.set_ylim(low_y-10, high_y+10)

    # Adjust axis spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

    returns.append(performance)

# Calculate returns and stderr of returns
returns = np.array(returns)
mean = returns.mean(axis=0)
stderr = np.std(returns, axis=0, ddof=1)
stderr /= np.sqrt(returns.shape[0])

ax = fig.add_subplot(spec[:, COLS-1])
ax.fill_between(
    np.arange(mean.shape[-1]),
    mean-stderr,
    mean+stderr,
    alpha=0.1,
    color="#161c1e",
)
ax.plot(mean, label="Mean", linewidth=3.0, color="#161c1e")

# Set title and axes limits
ax.set_title("Mean")
ax.set_xlim(low_x, high_x)
ax.set_ylim(low_y-10, high_y+10)
ax.set_yticks(get_y_bounds(ENV, per_env_tuning))
ax.set_xticks([low_x, high_x])

# Adjust axis spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

# Add the figure title
fig.suptitle(ENV)

fig.savefig(
    f"{os.path.expanduser('~')}/{ENV}_{agent}_runs.png",
    bbox_inches="tight",
)
