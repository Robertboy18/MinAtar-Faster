import numpy as np
from collections.abc import Iterable
from copy import deepcopy
from pprint import pprint
try:
    from utils.runs import expand_episodes
except ModuleNotFoundError:
    from runs import expand_episodes


CONTINIUING = "continuing"
EPISODIC = "episodic"
TRAIN = "train"
EVAL = "eval"


def sweeps(parameters, index):
    """
    Gets the parameters for the hyperparameter sweep defined by the index.

    Each hyperparameter setting has a specific index number, and this function
    will get the appropriate parameters for the argument index. In addition,
    this the indices will wrap around, so if there are a total of 10 different
    hyperparameter settings, then the indices 0 and 10 will return the same
    hyperparameter settings. This is useful for performing loops.

    For example, if you had 10 hyperparameter settings and you wanted to do
    10 runs, the you could just call this for indices in range(0, 10*10). If
    you only wanted to do runs for hyperparameter setting i, then you would
    use indices in range(i, 10, 10*10)

    Parameters
    ----------
    parameters : dict
        The dictionary of parameters, as found in the agent's json
        configuration file
    index : int
        The index of the hyperparameters configuration to return

    Returns
    -------
    dict, int
        The dictionary of hyperparameters to use for the agent and the total
        number of combinations of hyperparameters (highest possible unique
        index)
    """
    # If the algorithm is a batch algorithm, ensure the batch size if less
    # than the replay buffer size
    if "batch_size" in parameters and "replay_capacity" in parameters:
        batches = np.array(parameters["batch_size"])
        replays = np.array(parameters["replay_capacity"])
        legal_settings = []

        # Calculate the legal combinations of batch sizes and replay capacities
        for batch in batches:
            legal = np.where(replays >= batch)[0]
            legal_settings.extend(list(zip([batch] *
                                           len(legal), replays[legal])))

        # Replace the configs batch/replay combos with the legal ones
        parameters["batch/replay"] = legal_settings
        replaced_hps = ["batch_size", "replay_capacity"]
    else:
        replaced_hps = []

    # Get the hyperparameters corresponding to the argument index
    out_params = {}
    accum = 1
    for key in parameters:
        if key in replaced_hps:
            # Ignore the HPs that have been sanitized and replaced by a new
            # set of HPs
            continue

        num = len(parameters[key])
        if key == "batch/replay":
            # Batch/replay must be treated differently
            batch_replay_combo = parameters[key][(index // accum) % num]
            out_params["batch_size"] = batch_replay_combo[0]
            out_params["replay_capacity"] = batch_replay_combo[1]
            accum *= num
            continue

        out_params[key] = parameters[key][(index // accum) % num]
        accum *= num

    return (out_params, accum)


def total(parameters):
    """
    Similar to sweeps but only returns the total number of
    hyperparameter combinations. This number is the total number of distinct
    hyperparameter settings. If this function returns k, then there are k
    distinct hyperparameter settings, and indices 0 and k refer to the same
    distinct hyperparameter setting.

    Parameters
    ----------
    parameters : dict
        The dictionary of parameters, as found in the agent's json
        configuration file

    Returns
    -------
    int
        The number of distinct hyperparameter settings
    """
    return sweeps(parameters, 0)[1]


def satisfies(data, f):
    """
    Similar to hold_constant. Returns all hyperparameter settings
    that result in f evaluating to True.

    For each run, the hyperparameter dictionary for that run is inputted to f.
    If f returns True, then those hypers are kept.

    Parameters
    ----------
    data : dict
        The data dictionary generate from running an experiment
    f : f(dict) -> bool
        A function mapping hyperparameter settings (in a dictionary) to a
        boolean value

    Returns
    -------
    tuple of list[int], dict
        The list of hyperparameter settings satisfying the constraints
        defined by constant_hypers and a dictionary of new hyperparameters
        which satisfy these constraints
    """
    indices = []

    # Generate a new hyperparameter configuration based on the old
    # configuration
    new_hypers = deepcopy(data["experiment"]["agent"]["parameters"])
    # Clear the hyper configuration
    for key in new_hypers:
        if isinstance(new_hypers[key], list):
            new_hypers[key] = set()

    for index in data["experiment_data"]:
        hypers = data["experiment_data"][index]["agent_hyperparams"]
        if not f(hypers):
            continue

        # Track the hyper indices and the full hyper settings
        indices.append(index)
        for key in new_hypers:
            if key not in data["experiment_data"][index]["agent_hyperparams"]:
                # print(f"{key} not in agent hyperparameters, ignoring...")
                continue

            if isinstance(new_hypers[key], set):
                agent_val = data["experiment_data"][index][
                    "agent_hyperparams"][key]

                # Convert lists to a hashable type
                if isinstance(agent_val, list):
                    agent_val = tuple(agent_val)

                new_hypers[key].add(agent_val)
            else:
                if key in new_hypers:
                    value = new_hypers[key]
                    raise IndexError("clobbering existing hyper " +
                                     f"{key} with value {value} with " +
                                     f"new value {agent_val}")
                new_hypers[key] = agent_val

    # Convert each set in new_hypers to a list
    for key in new_hypers:
        if isinstance(new_hypers[key], set):
            new_hypers[key] = sorted(list(new_hypers[key]))

    return indices, new_hypers


def hold_constant(data, constant_hypers):
    """
    Returns the hyperparameter settings indices and hyperparameter values
    of the hyperparameter settings satisfying the constraints constant_hypers.

    Returns the hyperparameter settings indices in the data that
    satisfy the constraints as well as a new dictionary of hypers which satisfy
    the constraints. The indices returned are the hyper indices of the original
    data and not the indices into the new hyperparameter configuration
    returned.

    Parameters
    ----------
    data: dict
        The data dictionary generated from an experiment

    constant_hypers: dict[string]any
        A dictionary mapping hyperparameters to a value that they should be
        equal to.

    Returns
    -------
    tuple of list[int], dict
        The list of hyperparameter settings satisfying the constraints
        defined by constant_hypers and a dictionary of new hyperparameters
        which satisfy these constraints

    Example
    -------
    >>> data = ...
    >>> contraints = {"stepsize": 0.8}
    >>> hold_constant(data, constraints)
    (
        [0, 1, 6, 7],
        {
            "stepsize": [0.8],
            "decay":    [0.0, 0.5],
            "epsilon":  [0.0, 0.1],
        }
    )
    """
    indices = []

    # Generate a new hyperparameter configuration based on the old
    # configuration
    new_hypers = deepcopy(data["experiment"]["agent"]["parameters"])
    # Clear the hyper configuration
    for key in new_hypers:
        if isinstance(new_hypers[key], list):
            new_hypers[key] = set()

    # Go through each hyperparameter index, checking if it satisfies the
    # constraints
    for index in data["experiment_data"]:
        # Assume we hyperparameter satisfies the constraints
        constraint_satisfied = True

        # Check to see if the agent hyperparameter satisfies the constraints
        for hyper in constant_hypers:
            constant_val = constant_hypers[hyper]

            # Ensure the constrained hyper exists in the data
            if hyper not in data["experiment_data"][index][
               "agent_hyperparams"]:
                raise IndexError(f"no such hyper {hyper} in agent hypers")

            agent_val = data["experiment_data"][index]["agent_hyperparams"][
                hyper]

            if agent_val != constant_val:
                # Hyperparameter does not satisfy the constraints
                constraint_satisfied = False
                break

        # If the constraint is satisfied, then we will store the hypers
        if constraint_satisfied:
            indices.append(index)

            # Add the hypers to the configuration
            for key in new_hypers:
                if isinstance(new_hypers[key], set):
                    agent_val = data["experiment_data"][index][
                        "agent_hyperparams"][key]

                    if isinstance(agent_val, list):
                        agent_val = tuple(agent_val)

                    new_hypers[key].add(agent_val)
                else:
                    if key in new_hypers:
                        value = new_hypers[key]
                        raise IndexError("clobbering existing hyper " +
                                         f"{key} with value {value} with " +
                                         f"new value {agent_val}")
                    new_hypers[key] = agent_val

    # Convert each set in new_hypers to a list
    for key in new_hypers:
        if isinstance(new_hypers[key], set):
            new_hypers[key] = sorted(list(new_hypers[key]))

    return indices, new_hypers


def renumber(data, hypers):
    """
    Renumbers the hyperparameters in data to reflect the hyperparameter map
    hypers. If any hyperparameter settings exist in data that do not exist in
    hypers, then those data are discarded.

    Note that each hyperparameter listed in hypers must also be listed in data
    and vice versa, but the specific hyperparameter values need not be the
    same. For example if "decay" ∈ data[hypers], then it also must be in hypers
    and vice versa. If 0.9 ∈ data[hypers][decay], then it need *not* be in
    hypers[decay].

    This function does not mutate the input data, but rather returns a copy of
    the input data, appropriately mutated.

    Parameters
    ----------
    data : dict
        The data dictionary generated from running the experiment
    hypers : dict
        The new dictionary of hyperparameter values

    Returns
    -------
    dict
        The modified data dictionary

    Examples
    --------
    >>> data = ...
    >>> contraints = {"stepsize": 0.8}
    >>> new_hypers = hold_constant(data, constraints)[1]
    >>> new_data = renumber(data, new_hypers)
    """
    data = deepcopy(data)
    # Ensure each hyperparameter is in both hypers and data; hypers need not
    # list every hyperparameter *value* that is listed in data, but it needs to
    # have the same hyperparameters. E.g. if "decay" exists in data then it
    # should also exist in hypers, but if 0.9 ∈ data[hypers][decay], this value
    # need not exist in hypers.
    for key in data["experiment"]["agent"]["parameters"]:
        if key not in hypers:
            raise ValueError("data and hypers should have all the same " +
                             f"hyperparameters but {key} ∈ data but ∉ hypers")

    # Ensure each hyperparameter listed in hypers is also listed in data. If it
    # isn't then it isn't clear which value of this hyperparamter the data in
    # data should map to. E.g. if "decay" = [0.1, 0.2] ∈ hypers but ∉ data,
    # which value should we set for the data in data when renumbering? 0.1 or
    # 0.2?
    for key in hypers:
        if key not in data["experiment"]["agent"]["parameters"]:
            raise ValueError("data and hypers should have all the same " +
                             f"hyperparameters but {key} ∈ hypers but ∉ data")

    new_data = {}
    new_data["experiment"] = data["experiment"]
    new_data["experiment"]["agent"]["parameters"] = hypers
    new_data["experiment_data"] = {}

    total_hypers = total(hypers)

    for i in range(total_hypers):
        setting = sweeps(hypers, i)[0]

        for j in data["experiment_data"]:
            agent_hypers = data["experiment_data"][j]["agent_hyperparams"]
            setting_in_data = True

            # For each hyperparameter value in setting, ensure that the
            # corresponding agent hyperparameter is equal. If not, ignore that
            # hyperparameter setting.
            for key in setting:
                # If the hyper setting is iterable, then check each value in
                # the iterable to ensure it is equal to the corresponding
                # value in the agent hyperparameters
                if isinstance(setting[key], Iterable):
                    if len(setting[key]) != len(agent_hypers[key]):
                        setting_in_data = False
                        break
                    for k in range(len(setting[key])):
                        if setting[key][k] != agent_hypers[key][k]:
                            setting_in_data = False
                            break

                # Non-iterable data
                elif setting[key] != agent_hypers[key]:
                    setting_in_data = False
                    break

            if setting_in_data:
                new_data["experiment_data"][i] = data["experiment_data"][j]

    return new_data


def get_performance(data, hyper, type_=TRAIN, repeat=True):
    """
    Returns the data for each run of key, optionally adjusting the runs'
    data so that each run has the same number of data points. This is
    accomplished by repeating each episode's performance by the number of
    timesteps the episode took to complete

    Parameters
    ----------
    data : dict
        The data dictionary
    hyper : int
        The hyperparameter index to get the run data of
    repeat : bool
        Whether or not to repeat the runs data

    Returns
    -------
    np.array
        The array of performance data
    """
    if type_ not in (TRAIN, EVAL):
        raise ValueError(f"unknown type {type_}")

    key = type_ + "_episode_rewards"

    if repeat:
        data = expand_episodes(data, hyper, type_)

    run_data = []
    for run in data["experiment_data"][hyper]["runs"]:
        run_data.append(run[key])

    return np.array(run_data)


def best(data, perf=TRAIN):
    """
    Returns the hyperparameter index of the hyper setting which resulted in the
    highest AUC of the learning curve. AUC is calculated by computing the AUC
    for each run, then taking the average over all runs.

    Parameters
    ----------
    data : dict
        The data dictionary
    perf : str
        The type of performance to evaluate, train or eval

    Returns
    -------
    np.array[int], np.float32
        The hyper settings that resulted in the maximum return as well as the
        maximum return
    """
    max_hyper = int(np.max(list(data["experiment_data"].keys())))
    hypers = [np.finfo(np.float64).min] * (max_hyper + 1)
    for hyper in data["experiment_data"]:
        hyper_data = []
        for run in data["experiment_data"][hyper]["runs"]:
            hyper_data.append(run[f"{perf}_episode_rewards"].mean())

        hyper_data = np.array(hyper_data)
        hypers[hyper] = hyper_data.mean()

    return np.argmax(hypers), np.max(hypers)


def get(data, ind):
    """
    Gets the hyperparameters for hyperparameter settings index ind

    data : dict
        The Python data dictionary generated from running main.py
    ind : int
        Gets the returns of the agent trained with this hyperparameter
        settings index

    Returns
    -------
    dict
        The dictionary of hyperparameters
    """
    return data["experiment_data"][ind]["agent_hyperparams"]


def which(data, hypers, equal_keys=False):
    """
    Get the hyperparameter index at which all agent hyperparameters are
    equal to those specified by hypers.

    Parameters
    ----------
    data : dict
        The data dictionary that resulted from running an experiment
    hypers : dict[string]any
        A dictionary of hyperparameters to the values that those
        hyperparameters should take on
    equal_keys : bool, optional
        Whether or not all keys must be shared between the sets of agent
        hyperparameters and the argument hypers. By default False.

    Returns
    -------
    int, None
        The hyperparameter index at which the agent had hyperparameters equal
        to those specified in hypers.

    Examples
    --------
    >>> data = ... # Some data from an experiment
    >>> hypers = {"critic_lr": 0.01, "actor_lr": 1.0}
    >>> ind = which(data, hypers)
    >>> print(ind in data["experiment_data"])
        True
    """
    for ind in data["experiment_data"]:
        is_equal = True
        agent_hypers = data["experiment_data"][ind]["agent_hyperparams"]

        # Ensure that all keys in each dictionary are equal
        if equal_keys and set(agent_hypers.keys()) != set(hypers.keys()):
            continue

        # For the current set of agent hyperparameters (index ind), check to
        # see if all hyperparameters used by the agent are equal to those
        # specified by hypers. If not, then break and check the next set of
        # agent hyperparameters.
        for h in hypers:
            if h in agent_hypers and hypers[h] != agent_hypers[h]:
                is_equal = False
                break

        if is_equal:
            return ind

    # No agent hyperparameters were found that coincided with the argument
    # hypers
    return None
