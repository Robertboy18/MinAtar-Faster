# Import modules
import numpy as np
from agent.baseAgent import BaseAgent
from PyFixedReps import TileCoder
import time
import warnings
import inspect


class ESarsa(BaseAgent):
    """
    Class Esarsa implements the Expected Sarsa(λ) algorithm
    """
    def __init__(self, decay, lr, gamma, epsilon,
                 action_space, bins, num_tilings, env, seed=None,
                 trace_type="replacing", policy_type="εgreedy",
                 include_bias=True):
        super().__init__()
        self.batch = False

        # Set the agent's policy sampler
        if seed is None:
            seed = int(time())
        self.random = np.random.default_rng(seed=int(seed))
        self.seed = seed

        # Needed so that when evaluating offline, we don't explore
        self.is_training = True

        # Tile Coder
        self.include_bias = include_bias
        input_ranges = list(zip(env.observation_space.low,
                                env.observation_space.high))
        dims = env.observation_space.shape[0]
        params = {
                    "dims": dims,
                    "tiles": bins,
                    "tilings": num_tilings,
                    "input_ranges": input_ranges,
                    "scale_output": False,
                }
        self.tiler = TileCoder(params)
        state_features = self.tiler.features() + self.include_bias

        # The weight parameters
        self.actions = action_space.n
        self.weights = np.zeros((self.actions, state_features))

        # Set learning rates and other scaling factors
        if decay < 0.0:
            raise ValueError("cannot have trace decay rate < 0")
        self.decay = decay
        self.lr = lr / (num_tilings + self.include_bias)
        self.gamma = gamma
        self.epsilon = epsilon
        print(self.lr)

        if policy_type not in ("εgreedy"):
            raise ValueError("policy_type must be one of 'εgreedy'")
        self.policy_type = policy_type

        # Eligibility traces
        if trace_type not in ("accumulating", "replacing"):
            raise ValueError("trace_type must be one of 'accumulating', " +
                             "'replacing'")
        self.use_trace = decay > 0.0
        if self.use_trace:
            self.trace = np.zeros_like(self.weights)
            self.trace_type = trace_type

        source = inspect.getsource(inspect.getmodule(inspect.currentframe()))
        self.info = {"source": source}

    def sample_action(self, state):
        """
        Samples an action from the actor

        Parameters
        ----------
        state : np.array
            The state feature vector, not one hot encoded

        Returns
        -------
        np.array of float
            The action to take
        """
        return self._sample_action(state)

    def _sample_action(self, state):
        """
        Samples an action

        Parameters
        ----------
        state : np.array
            The state feature vector, not one hot encoded

        Returns
        -------
        np.array of float
            The action to take
        """
        # Take random action with probability ε and only if in training mode
        if self.policy_type == "εgreedy" and self.epsilon != 0 and \
           self.is_training:
            if self.random.uniform() < self.epsilon:
                action = self.random.choice(self.actions)
                return action

        state = self._tiler_indices(state)
        action_vals = self.weights[:, state].sum(axis=1)

        if self.policy_type == "εgreedy":
            # Choose maximum action
            max_actions = np.where(action_vals == np.max(action_vals))[0]
            if len(max_actions) > 1:
                return max_actions[self.random.choice(len(max_actions))]
            else:
                return max_actions[0]

        else:
            raise ValueError(f"unknown policy type {self.policy_type}")

    def _get_probs(self, state):
        """
        Gets the probability of taking each action in state `state`

        Parameters
        ----------
        state : np.array
            The state observation, not tile-coded

        Returns
        -------
        np.array[float]
            The probabilities of taking each action in state `state`
        """
        state = self._tiler_indices(state)
        if self.policy_type == "εgreedy":
            probs = np.zeros(self.actions)
            probs += self.epsilon / self.actions

            action_vals = self.weights[:, state].sum(axis=1)
            max_actions = np.where(action_vals == np.max(action_vals))[0]
            probs[max_actions] += (1 - self.epsilon) / len(max_actions)
        else:
            raise ValueError(f"unknown policy type {self.policy_type}")

        return probs

    def _tiler_indices(self, state):
        if self.include_bias:
            return np.concatenate(
                [
                    np.zeros((1,), dtype=np.int32),
                    self.tiler.get_indices(state) + 1,
                ]
            )

        return self.tiler.get_indices(state)

    def update(self, state, action, reward, next_state, done_mask):
        """
        Takes a single update step

        Parameters
        ----------
        state : np.array or array_like of np.array
            The state feature vector
        action : np.array of float or array_like of np.array
            The action taken
        reward : float or array_like of float
            The reward seen by the agent after taking the action
        next_state : np.array or array_like of np.array
            The feature vector of the next state transitioned to after the
            agent took the argument action
        done_mask : bool or array_like of bool
            False if the agent reached the goal, True if the agent did not
            reach the goal yet the episode ended (e.g. max number of steps
            reached)
            Note: this parameter is not used; it is only kept so that the
            interface BaseAgent is consistent and can be used for both
            Soft Actor-Critic and Linear-Gaussian Actor-Critic
        """
        state = self._tiler_indices(state)

        δ = reward
        δ -= self.weights[action, state].sum()

        # Update the trace
        if self.use_trace:
            if self.trace_type == "accumulating":
                self.trace[action, state] += 1
            elif self.trace_type == "replacing":
                self.trace[action, state] = 1
            else:
                raise ValueError(f"unknown trace type {self.trace_type}")

        # Adjust δ if we are in an intra-episode timestep
        episode_done = not done_mask
        if not episode_done:
            probs = self._get_probs(next_state)
            next_state = self._tiler_indices(next_state)

            next_q = self.gamma * self.weights[:, next_state].sum(axis=1)
            𝔼_next_q = probs @ next_q
            δ += 𝔼_next_q

        # Update the weights
        if self.use_trace:
            self.weights += (self.lr * δ * self.trace)

            # Decay the trace
            self.trace *= (self.decay * self.gamma)
        else:
            self.weights[action, state] += (self.lr * δ)

        return

    def reset(self):
        """
        Resets the agent between episodes
        """
        self.trace = np.zeros_like(self.weights)
        self.first_call = True

    def eval(self):
        """
        Sets the agent into offline evaluation mode, where the agent will not
        explore
        """
        self.is_training = False

    def train(self):
        """
        Sets the agent to online training mode, where the agent will explore
        """
        self.is_training = True

    def get_parameters(self):
        pass
