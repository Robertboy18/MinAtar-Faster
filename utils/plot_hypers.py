import pickle
import functools
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy
import json
import sys
import seaborn as sns
import plot_utils as plot
import matplotlib as mpl
import experiment_utils as exp
import hypers


# Place environment name with type of environment in type_map so that we know
# how to plot/evaluate. This terrible code-style is due to legacy code which
# needs to be fixed badly.
CONTINUING = "continuing"
EPISODIC = "episodic"
type_map = {
        "MinAtarBreakout": EPISODIC,
        "MinAtarFreeway": EPISODIC,
        "PendulumFixed-v0": EPISODIC,
        "Acrobot-v1": EPISODIC,
        "BipedalWalker-v3": EPISODIC,
        "LunarLanderContinuous-v2": EPISODIC,
        "Bimodal1DEnv": CONTINUING,
        "Hopper-v2": EPISODIC,
        "PuddleWorld-v1": EPISODIC,
        "MountainCar-v0": EPISODIC,
        "MountainCarContinuous-v0": EPISODIC,
        "Pendulum-v0": CONTINUING,
        "Pendulum-v1": CONTINUING,
        "Walker2d": EPISODIC,
        "Swimmer-v2": EPISODIC
}

if len(sys.argv) < 5:
    print("invalid number of inputs:")
    print(f"\t{sys.argv[0]} env_json hyper agent_json")

env_json = sys.argv[1]
DIR = sys.argv[2]
HYPER = sys.argv[3]
agent_json = sys.argv[4:]

# Load configuration files
with open(env_json, "r") as infile:
    env_config = json.load(infile)
if "gamma" not in env_config:
    env_config["gamma"] = -1

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
    with open(os.path.join(f, DATA_FILE), "rb") as infile:
        DATA.append(pickle.load(infile))

# Generate labels for plots
labels = []
for ag in DATA:
    labels.append([ag["experiment"]["agent"]["agent_name"]])
colours = [["#003f5c"], ["#bc5090"], ["#ffa600"], ["#ff6361"], ["#58cfa1"]]

# Plot the hyperparameter sensitivities
all_fig, all_ax = plot.hyper_sensitivity(DATA, HYPER)

# Adjust axis spines
all_ax.spines['top'].set_visible(False)
all_ax.spines['right'].set_visible(False)
all_ax.spines['bottom'].set_linewidth(2)
all_ax.spines['left'].set_linewidth(2)

# Set title and legend
all_ax.set_title(HYPER + " " + os.path.basename(env_json).rstrip(".json"))
all_ax.legend()

all_fig.savefig(
    f"{os.path.expanduser('~')}/{ENV}_{HYPER}.png",
    bbox_inches="tight",
)
exit(0)
