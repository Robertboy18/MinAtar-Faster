#!/usr/bin/env python3

# Import modules
import gym
from copy import deepcopy
from env.PendulumEnv import PendulumEnv
from env.Acrobot import AcrobotEnv
from env.Gridworld import GridworldEnv
import env.MinAtar as MinAtar
import numpy as np


class Environment:
    """
    Environment is a wrapper around concrete implementations of environments
    which logs data.
    """
    def __init__(self, config, seed, monitor=False, monitor_after=0):
        """
        Constructor

        Parameters
        ----------
        config : dict
            The environment configuration file
        seed : int
            The seed to use for all random number generators
        monitor : bool
            Whether or not to render the scenes as the agent learns, by
            default False
        monitor_after : int
            If monitor is True, how many timesteps should pass before
            the scene is rendered, by default 0.
        """

        self.steps = 0
        self.episodes = 0

        # Whether to render the environment, and when to. Useful for debugging.
        self.monitor = monitor
        self.steps_until_monitor = monitor_after

        # Set up the wrapped environment
        self.env_name = config["env_name"]
        self.env = _env_factory(config)
        self.env.seed(seed=seed)
        self.steps_per_episode = config["steps_per_episode"]

        # Log environment info
        if "info" in dir(self.env):
            self.info = self.env.info
        else:
            self.info = {}

    @property
    def action_space(self):
        """
        Gets the action space of the Gym environment

        Returns
        -------
        gym.spaces.Space
            The action space
        """
        return self.env.action_space

    @property
    def observation_space(self):
        """
        Gets the observation space of the Gym environment

        Returns
        -------
        gym.spaces.Space
            The observation space
        """
        return self.env.observation_space

    def seed(self, seed):
        """
        Seeds the environment with a random seed

        Parameters
        ----------
        seed : int
            The random seed to seed the environment with
        """
        self.env.seed(seed)

    def reset(self):
        """
        Resets the environment by resetting the step counter to 0 and resetting
        the wrapped environment. This function also increments the total
        episode count.

        Returns
        -------
        2-tuple of array_like, dict
            The new starting state and an info dictionary
        """
        self.steps = 0
        self.episodes += 1

        state = self.env.reset()

        return state, {"orig_state": state}

    def render(self):
        """
        Renders the current frame
        """
        self.env.render()

    def step(self, action):
        """
        Takes a single environmental step

        Parameters
        ----------
        action : array_like of float
            The action array. The number of elements in this array should be
            the same as the action dimension.

        Returns
        -------
        float, array_like of float, bool, dict
            The reward and next state as well as a flag specifying if the
            current episode has been completed and an info dictionary
        """
        if self.monitor and self.steps_until_monitor < 0:
            self.render()
        elif self.monitor:
            self.steps_until_monitor -= (
                1 if self.steps_until_monitor >= 0 else 0
            )

        self.steps += 1

        # Get the next state, reward, and done flag
        state, reward, done, info = self.env.step(action)
        info["orig_state"] = state

        # If the episode completes, return the goal reward
        if done:
            info["steps_exceeded"] = False
            return state, reward, done, info

        # If the maximum time-step was reached
        if self.steps >= self.steps_per_episode > 0:
            done = True
            info["steps_exceeded"] = True

        return state, reward, done, info


def _env_factory(config):
    """
    Instantiates and returns an environment given an environment configuration
    file.

    Parameters
    ----------
    config : dict
        The environment config

    Returns
    -------
    gym.Env
        The environment to train on
    """
    name = config["env_name"]
    seed = config["seed"]
    env = None

    if name == "Pendulum-v0":
        env = PendulumEnv(seed=seed, continuous_action=config["continuous"])

    elif name == "Gridworld":
        env = GridworldEnv(config["rows"], config["cols"])
        env.seed(seed)

    elif name == "Acrobot-v1":
        env = AcrobotEnv(seed=seed, continuous_action=config["continuous"])

    # If using MinAtar environments, we need a wrapper to permute the batch
    # dimensions to be consistent with PyTorch.
    elif "minatar" in name.lower():
        if "/" in name:
            raise ValueError(f"specify environment as MinAtar{name} rather " +
                             "than MinAtar/{name}")

        minimal_actions = config.get("use_minimal_action_set", True)
        stripped_name = name[7:].lower()  # Strip off "MinAtar"

        env = MinAtar.BatchFirst(
            MinAtar.GymEnv(
                stripped_name,
                use_minimal_action_set=minimal_actions,
            )
        )

    # Otherwise use a gym environment
    else:
        env = gym.make(name).env
        env.seed(seed)

    return env
