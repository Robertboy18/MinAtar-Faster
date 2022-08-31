import pickle
import seaborn as sns
import os
import matplotlib.pyplot as plt
import numpy as np
import hypers
import json
import sys
import plot_utils as plot
import matplotlib as mpl
mpl.rcParams["font.size"] = 24
mpl.rcParams["svg.fonttype"] = "none"


# Place environment name with type of environment in type_map so that we know
# how to plot/evaluate. This terrible code-style is due to legacy code which
# needs to be fixed badly.
CONTINUING = "continuing"
EPISODIC = "episodic"
type_map = {
        "MinAtarBreakout": EPISODIC,
        "MinAtarFreeway": EPISODIC,
        "LunarLanderContinuous-v2": EPISODIC,
        "Bimodal3Env": CONTINUING,
        "Bimodal2DEnv": CONTINUING,
        "Bimodal1DEnv": CONTINUING,
        "BipedalWalker-v3": EPISODIC,
        "Hopper-v2": EPISODIC,
        "PuddleWorld-v1": EPISODIC,
        "MountainCar-v0": EPISODIC,
        "MountainCarContinuous-v0": EPISODIC,
        "PendulumFixed-v0": CONTINUING,
        "Pendulum-v0": CONTINUING,
        "Acrobot-v1": EPISODIC,
        "Pendulum-v1": CONTINUING,
        "Walker2d": EPISODIC,
        "Swimmer-v2": EPISODIC
        }

if len(sys.argv) < 4:
    raise ArgumentError("""invalid arguments, call ./plot_mse
                        path/to/env_config dir/with/data.pkl
                        path/to/agent_config(s)
                        """)
env_json = sys.argv[1]
DIR = sys.argv[2]
agent_json = sys.argv[3:]

# Load configuration files
with open(env_json, "r") as infile:
    env_config = json.load(infile)
agent_configs = []
for j in agent_json:
    with open(j, "r") as infile:
        agent_configs.append(json.load(infile))

ENV = env_config["env_name"]
ENV_TYPE = type_map[ENV]
PERFORMANCE_METRIC_TYPE = "train"
DATA_FILE = "data.pkl"


# Script
DATA_FILES = []
for config in agent_configs:
    agent = config["agent_name"]
    if DIR:
        DATA_FILES.append(f"./results/{DIR}/{ENV}_{agent}results")
    else:
        DATA_FILES.append(f"./results/{ENV}_{agent}results")

DATA = []
for f in DATA_FILES:
    print(f"Opening file: {f}")
    with open(os.path.join(f, DATA_FILE), "rb") as infile:
        DATA.append(pickle.load(infile))

# Find best hypers
BEST_IND = []
for agent in DATA:
    best_hp = hypers.best(agent)[0]
    BEST_IND.append(best_hp)

# Generate labels for plots
labels = []
for ag in DATA:
    labels.append([ag["experiment"]["agent"]["agent_name"]])

CMAP = "tab10"
colours = list(sns.color_palette(CMAP, 8).as_hex())
colours = list(map(lambda x: [x], colours))
plt.rcParams["axes.prop_cycle"] = mpl.cycler(color=sns.color_palette(CMAP))

# Plot the mean + standard error
print("=== Plotting mean with standard error")
PLOT_TYPE = "train"
SOLVED = 0
TYPE = "online" if PLOT_TYPE == "train" else "offline"
best_ind = list(map(lambda x: [x], BEST_IND))

plot_labels = list(map(lambda x: x[0], labels))  # Adjust labels for plot
fig, ax = plot.mean_with_stderr(
    DATA,
    PLOT_TYPE,
    best_ind,
    [5000]*len(best_ind),
    plot_labels,
    env_type="episodic",
    figsize=(16, 16),
    colours=colours,
)
ax.set_title(ENV)

fig.savefig(
    f"{os.path.expanduser('~')}/{ENV}.png",
    bbox_inches="tight",
)
