#!/usr/bin/env python3

import sys
import pickle
import os
import json
import utils.experiment_utils as exp
import click


def add_dicts(data, newfiles):
    """
    add_dicts adds the data dictionaries in newfiles to the existing
    dictionary data. This function assumes that the hyperparameter
    indices between data and those found in each file in newfiles are
    consistent.
    """
    set_experiment_val = False
    if data is None:
        set_experiment_val = True
        data = {
                "experiment_data": {},
                "experiment": {},
                }
    # Add data from all other dictionaries
    for file in newfiles:
        with open(file, "rb") as in_file:
            # Read in the new dictionary
            try:
                in_data = pickle.load(in_file)
            except EOFError:
                print(file)
                continue

            if set_experiment_val:
                data["experiment"] = in_data["experiment"]

            # Add experiment data to running dictionary
            for key in in_data["experiment_data"]:
                # Check if key exists
                if key in data["experiment_data"]:
                    if "learned_params" in \
                            data["experiment_data"][key]["runs"][0]:
                        del data["experiment_data"][key]["runs"][0][
                            "learned_params"]
                    # continue
                    # Append data if existing
                    data["experiment_data"][key]["runs"].extend(
                        in_data["experiment_data"][key]["runs"])

                else:
                    # Key doesn't exist - add data to dictionary
                    data["experiment_data"][key] = \
                            in_data["experiment_data"][key]

    return data


@click.command(help="combine a number of data files in a single " +
               "directory into a single data file called data.pkl")
@click.argument("directory", required=True, type=click.Path(exists=True))
def main(directory):
    data = None
    if os.path.exists(os.path.join(directory, "data.pkl")):
        print("remove data.pkl from directory first")

    files = os.listdir(directory)
    if "data.pkl" in files:
        files.remove("data.pkl")
    filenames = list(map(lambda x: os.path.join(directory, x), files))
    data = add_dicts(data, filenames)

    with open(os.path.join(directory, "data.pkl"), "wb") as outfile:
        pickle.dump(data, outfile)


if __name__ == "__main__":
    main()
