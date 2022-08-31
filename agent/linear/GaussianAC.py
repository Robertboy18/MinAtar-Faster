#!/usr/bin/env python3

# Import modules
import numpy as np
from agent.baseAgent import BaseAgent
from time import time
from PyFixedReps import TileCoder
from env.Bimodal import Bimodal1DEnv


class GaussianAC(BaseAgent):
    """
    Class GaussianAC implements Linear-Gaussian Actor-Critic with eligibility
    trace, as outlined in "Model-Free Reinforcement Learning with Continuous
    Action in Practice", which can be found at:

    https://hal.inria.fr/hal-00764281/document

    The major difference is that this algorithm uses the discounted setting
    instead of the average reward setting as used in the above paper. This
    linear actor critic support multi-dimensional actions as well.
    """
    def __init__(self, decay, actor_lr_scale, critic_lr,
                 gamma, accumulate_trace, action_space, bins, num_tilings,
                 env, use_critic_trace, use_actor_trace, scaled=False,
                 clip_stddev=1000, seed=None, trace_type="replacing"):
        """
        Constructor

        Parameters
        ----------
        decay : float
            The eligibility decay rate, lambda
        actor_lr : float
            The learning rate for the actor
        critic_lr : float
            The learning rate for the critic
        state_features : int
            The size of the state feature vectors
        gamma : float
            The environmental discount factor
        accumulate_trace : bool
            Whether or not to accumulate the eligibility traces or not, which
            may be desirable if the task is continuing. If it is, then the
            eligibility trace vectors will be accumulated and not reset between
            "episodes" when calling the reset() method.
        scaled : bool, optional
            Whether the actor learning rate should be scaled by sigma^2 for
            learning stability, by default False
        clip_stddev : float, optional
            The value at which the standard deviation is clipped in order to
            prevent numerical overflow, by default 1000. If <= 0, then
            no clipping is done.
        seed : int
            The seed to use for the normal distribution sampler, by default
            None. If set to None, uses the integer value of the Unix time.
        """
        super().__init__()
        self.batch = False

        # Set the agent's policy sampler
        if seed is None:
            seed = int(time())
        self.random = np.random.default_rng(seed=int(seed))
        self.seed = seed

        # Save whether or not the task is continuing
        self.accumulate_trace = accumulate_trace

        # Needed so that when evaluating offline, we don't explore
        self.is_training = True

        # Determine standard deviation clipping
        self.clip_stddev = clip_stddev > 0
        self.clip_threshold = np.log(clip_stddev)

        # Tile Coder
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
        state_features = self.tiler.features() + 1

        # The weight parameters
        self.action_dims = action_space.high.shape[0]
        self.sigma_weights = np.zeros((self.action_dims, state_features))
        self.mu_weights = np.zeros((self.action_dims, state_features))
        self.actor_weights = np.zeros(state_features * 2)
        self.critic_weights = np.zeros(state_features)

        # Set learning rates and other scaling factors
        self.scaled = scaled
        self.decay = decay
        self.critic_lr = critic_lr / (num_tilings + 1)
        self.actor_lr = actor_lr_scale * self.critic_lr
        self.gamma = gamma

        # Eligibility traces
        self.use_actor_trace = use_actor_trace
        if trace_type not in ("replacing", "accumulating"):
            raise ValueError("trace_type must be one of 'accumulating', " +
                             "'replacing'")
        self.trace_type = trace_type

        if self.use_actor_trace:
            self.mu_trace = np.zeros_like(self.mu_weights)
            self.sigma_trace = np.zeros_like(self.sigma_weights)

        self.use_critic_trace = use_critic_trace
        if self.use_critic_trace:
            self.critic_trace = np.zeros(state_features)

        if isinstance(env.env, Bimodal1DEnv):
            self.info = {
                "actor": {"mean": [], "stddev": []},
            }
            self.store_dist = True
        else:
            self.store_dist = False

        source = inspect.getsource(inspect.getmodule(inspect.currentframe()))
        self.info = {"source": source}

    def get_mean(self, state):
        """
        Gets the mean of the parameterized normal distribution

        Parameters
        ----------
        state : np.array
            The indices of the nonzero features in the one-hot encoded state
            feature vector

        Returns
        -------
        float
            The mean of the normal distribution
        """
        return self.mu_weights[:, state].sum(axis=1)

    def get_stddev(self, state):
        """
        Gets the standard deviation of the parameterized normal distribution

        Parameters
        ----------
        state : np.array
            The indices of the nonzero features in the one-hot encoded state
            feature vector

        Returns
        -------
        float
            The standard deviation of the normal distribution
        """
        # Return un-clipped standard deviation if no clipping
        if not self.clip_stddev:
            return np.exp(self.sigma_weights[:, state].sum(axis=1))

        # Clip the standard deviation to prevent numerical overflow
        log_std = np.clip(self.sigma_weights[:, state].sum(axis=1),
                          -self.clip_threshold, self.clip_threshold)
        return np.exp(log_std)

    def sample_action(self, state):
        """
        Samples an action from the actor

        Parameters
        ----------
        state : np.array
            The observation, not tile coded

        Returns
        -------
        np.array of float
            The action to take
        """
        # state = np.concatenate(
        #     [
        #         np.ones((1,), dtype=np.int32),
        #         self.tiler.encode(state),
        #     ]
        # )
        state = np.concatenate(
            [
                np.zeros((1,), dtype=np.int32),
                self.tiler.get_indices(state) + 1,
            ]
        )
        mean = self.get_mean(state)

        # If in offline evaluation mode, return the mean action
        if not self.is_training:
            return np.array(mean)

        stddev = self.get_stddev(state)

        # Sample action from a normal distribution
        action = self.random.normal(loc=mean, scale=stddev)
        return action

    def get_actor_grad(self, state, action):
        """
        Gets the gradient of the actor's parameters

        Parameters
        ----------
        state : np.array
            The indices of the nonzero features in the one-hot encoded state
            feature vector
        action : np.array of float
            The action taken

        Returns
        -------
        np.array
            The gradient vector of the actor's weights, in the form
            [grad_mu_weights^T, grad_sigma_weights^T]^T
        """
        std = self.get_stddev(state)
        mean = self.get_mean(state)

        grad_mu = np.zeros_like(self.mu_weights)
        grad_sigma = np.zeros_like(self.sigma_weights)

        if action.shape[0] != 1:
            # Repeat state along rows to match number of action dims
            n = action.shape[0]
            state = np.expand_dims(state, 0)
            state = state.repeat(n, axis=0)

            scale_mu = (1 / (std ** 2)) * (action - mean)
            scale_sigma = ((((action - mean) / std)**2) - 1)

            # Reshape scales so we can use broadcasted multiplication
            scale_mu = np.expand_dims(scale_mu, axis=1)
            scale_sigma = np.expand_dims(scale_sigma, axis=1)

            # grad_mu = scale_mu * state
            # grad_sigma = scale_sigma * state

        else:
            scale_mu = (1 / (std ** 2)) * (action - mean)
            scale_sigma = ((((action - mean) / std)**2) - 1)

        grad_mu[:, state] = scale_mu
        grad_sigma[:, state] = scale_sigma

        return grad_mu, grad_sigma

    def update(self, state, action, reward, next_state, done_mask):
        """
        Takes a single update step

        Parameters
        ----------
        state : np.array or array_like of np.array
            The state feature vector, not tile coded
        action : np.array of float or array_like of np.array
            The action taken
        reward : float or array_like of float
            The reward seen by the agent after taking the action
        next_state : np.array or array_like of np.array
            The feature vector of the next state transitioned to after the
            agent took the argument action, not tile coded
        done_mask : bool or array_like of bool
            False if the agent reached the goal, True if the agent did not
            reach the goal yet the episode ended (e.g. max number of steps
            reached)
            Note: this parameter is not used; it is only kept so that the
            interface BaseAgent is consistent and can be used for both
            Soft Actor-Critic and Linear-Gaussian Actor-Critic
        """
        # state = np.concatenate(
        #     [
        #         np.ones((1,), dtype=np.int32),
        #         self.tiler.encode(state)
        #     ]
        # )
        # next_state = np.concatenate(
        #     [
        #         np.ones((1,), dtype=np.int32),
        #         self.tiler.encode(next_state)
        #     ]
        # )
        state = np.concatenate(
            [
                np.zeros((1,), dtype=np.int32),
                self.tiler.get_indices(state) + 1,
            ]
        )
        next_state = np.concatenate(
            [
                np.zeros((1,), dtype=np.int32),
                self.tiler.get_indices(next_state) + 1,
            ]
        )

        # Calculate TD error
        v = self.critic_weights[state].sum()
        next_v = self.critic_weights[next_state].sum()
        target = reward + self.gamma * next_v * done_mask
        delta = target - v

        # Critic update
        if self.use_critic_trace:
            # Update critic eligibility trace
            self.critic_trace *= (self.gamma * self.decay)
            # self.critic_trace = (self.gamma * self.decay *
            #                      self.critic_trace) + state
            if self.trace_type == "accumulating":
                self.critic_trace[state] += 1
            elif self.trace_type == "replacing":
                self.critic_trace[state] = 1
            else:
                raise ValueError("unkown trace type {self.trace_type}")
            # Update critic
            self.critic_weights += (self.critic_lr * delta * self.critic_trace)
        else:
            grad = np.zeros_like(self.critic_weights)
            grad[state] = 1
            self.critic_weights += (self.critic_lr * delta * grad)

        # Actor update
        mu_grad, sigma_grad = self.get_actor_grad(state, action)
        if self.use_actor_trace:
            # Update actor eligibility traces
            self.mu_trace *= (self.gamma * self.decay)
            self.sigma_trace *= (self.gamma * self.decay)
            if self.trace_type == "accumulating":
                self.mu_trace[:, state] += mu_grad
                self.sigma_trace[:, state] += sigma_grad
            else:
                self.mu_trace[:, state] = mu_grad[:, state]
                self.sigma_trace[:, state] = sigma_grad[:, state]

            # Update actor weights
            lr = self.actor_lr
            lr *= 1 if not self.scaled else (self.get_stddev(state) ** 2)
            self.mu_weights += (lr * delta * self.mu_trace)
            self.sigma_weights += (lr * delta * self.sigma_trace)
        else:
            lr = self.actor_lr
            lr *= 1 if not self.scaled else (self.get_stddev(state) ** 2)
            self.mu_weights += (lr * delta * mu_grad)
            self.sigma_trace = (lr * delta * sigma_grad)

        # In order to be consistent across all children of BaseAgent, we
        # return all transitions with the shape B x N, where N is the number
        # of state, action, or reward dimensions and B is the batch size = 1
        reward = np.array([reward])

        return np.expand_dims(state, axis=0), np.expand_dims(action, axis=0), \
            np.expand_dims(reward, axis=0), np.expand_dims(next_state, axis=0)

    def reset(self):
        """
        Resets the agent between episodes
        """
        if self.accumulate_trace:
            return
        if self.use_actor_trace:
            self.mu_trace = np.zeros_like(self.mu_trace)
            self.sigma_trace = np.zeros_like(self.sigma_trace)
        if self.use_critic_trace:
            self.critic_trace = np.zeros_like(self.critic_trace)

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
        """
        Gets all learned agent parameters such that training can be resumed.

        Gets all parameters of the agent such that, if given the
        hyperparameters of the agent, training is resumable from this exact
        point. This include the learned average reward, the learned entropy,
        and other such learned values if applicable. This does not only apply
        to the weights of the agent, but *all* values that have been learned
        or calculated during training such that, given these values, training
        can be resumed from this exact point.

        For example, in the GaussianAC class, we must save not only the actor
        and critic weights, but also the accumulated eligibility traces.

        Returns
        -------
        dict of str to array_like
            The agent's weights
        """
        pass


if __name__ == "__main__":
    a = GaussianAC(0.9, 0.1, 0.1, 0.5, 3, False)
    print(a.actor_weights, a.critic_weights)
    state = np.array([1, 2, 1])
    action = a.sample_action(state)
    a.update(state, action, 1, np.array([1, 2, 2]), 0.9)
    print(a.actor_weights, a.critic_weights)
    state = np.array([1, 2, 2])
    action = a.sample_action(state)
    a.update(state, action, 1, np.array([3, 1, 2]), 0.9)
    print(a.actor_weights, a.critic_weights)
