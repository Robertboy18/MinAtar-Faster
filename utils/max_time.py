#!/usr/bin/env python3

# This script looks through all runs of an experiment, over all hyper settings.
# It will return the runtime from the longest running experiment.

import numpy as np
import pickle
import sys
import os

if len(sys.argv) != 2:
    print(f"{sys.argv[0]}: checks the maximum runtime over all runs for an "
          "experiment")
    print("usage:")
    print(f"\t{sys.argv[0]} path/to/dir/containing/data.pkl")

f = sys.argv[1]
with open(f, "rb") as infile:
    data = pickle.load(infile)

time = []
for hyper in data["experiment_data"]:
    for run in data["experiment_data"][hyper]["runs"]:
        total = run["train_time"] + run["eval_time"]
        time.append(total)

print("Maximum run time:", np.max(time) / 3600)
