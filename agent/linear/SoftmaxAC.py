# Import modules
import numpy as np
from agent.baseAgent import BaseAgent
from PyFixedReps import TileCoder
import time
from scipy import special
import inspect


class SoftmaxAC(BaseAgent):
    """
    Class SoftmaxAC implements a Linear-Softmax Actor-Critic with eligibility
    traces. The algorithm works in the discounted setting, rather than in the
    average reward setting and is similar to the algorithm outlined in the
    Policy Gradient chapter in the RL Book.
    """
    def __init__(self, decay, actor_lr, critic_lr, gamma,
                 accumulate_trace, action_space, bins, num_tilings, env,
                 use_critic_trace, use_actor_trace, temperature, seed=None,
                 trace_type="replacing"):
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

        # Set the agent's policy sampler
        if seed is None:
            seed = int(time())
        self._random = np.random.default_rng(seed=int(seed))
        self._seed = seed

        # Needed so that when evaluating offline, we don't explore
        self._is_training = True

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
        self._tiler = TileCoder(params)
        state_features = self._tiler.features() + 1

        # The weight parameters
        self._action_n = action_space.n
        self._avail_actions = np.array(range(self._action_n))
        self._size = state_features
        self._actor_weights = np.zeros((self._action_n, state_features))
        self._critic_weights = np.zeros(state_features)  # State value critic

        # Set learning rates and other scaling factors
        self._critic_α = critic_lr / (num_tilings + 1)
        self._actor_α = actor_lr / (num_tilings + 1)
        self._γ = gamma
        if temperature < 0:
            raise ValueError("cannot use temperature < 0")
        self._τ = temperature

        # Eligibility traces
        if trace_type not in ("accumulating", "replacing"):
            raise ValueError("trace_type must be one of accumulating', " +
                             "'replacing'")
        if decay < 0:
            raise ValueError("cannot use decay < 0")
        elif decay >= 1:
            raise ValueError("cannot use decay >= 1")
        elif decay == 0:
            use_actor_trace = use_critic_trace = False
        else:
            self._λ = decay

        self._trace_type = trace_type
        self.use_actor_trace = use_actor_trace
        if self.use_actor_trace:
            self._actor_trace = np.zeros((self._action_n, state_features))
        self.use_critic_trace = use_critic_trace
        if self.use_critic_trace:
            self._critic_trace = np.zeros(state_features)

        source = inspect.getsource(inspect.getmodule(inspect.currentframe()))
        self.info = {"source": source}

    def _get_logits(self, state):
        """
        Gets the logits of the policy in state

        Parameters
        ----------
        state : np.array
            The indices of the nonzero features in the tile coded state
            representation

        Returns
        -------
        np.array of float
            The logits of each action
        """
        if self._τ == 0:
            raise ValueError("cannot compute logits when τ = 0")

        logits = self._actor_weights[:, state].sum(axis=1)
        logits -= np.max(logits)  # For numerical stability
        return logits / self._τ

    def _get_probs(self, state_ind):
        if self._τ == 0:
            q_values = self._actor_weights[:, state_ind].sum(axis=-1)

            max_value = np.max(q_values)
            max_actions = np.where(q_values == max_value)[0]

            probs = np.zeros(self._action_n)
            probs[max_actions] = 1 / len(max_actions)
            return probs

        logits = self._get_logits(state_ind)
        logits -= logits.max()  # Subtract max because SciPy breaks things
        pi = special.softmax(logits)
        return pi

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
        state = np.concatenate(
            [
                np.zeros((1,), dtype=np.int32),
                self._tiler.get_indices(state) + 1,
            ]
        )
        probs = self._get_probs(state)

        # If in offline evaluation mode, return the action of maximum
        # probability
        if not self._is_training:
            actions = np.where(probs == np.max(probs))[0]
            if len(actions) == 1:
                return actions[0]
            else:
                return self._random.choice(actions)

        return self._random.choice(self._action_n, p=probs)

    def _actor_grad(self, state, action):
        """
        Returns the gradient of the actor's performance in `state`
        evaluated at the action `action`

        Parameters
        ----------
        state : np.ndarray
            The state observation, not tile coded
        action : int
            The action to evaluate the gradient on
        """
        π = self._get_probs(state)
        π = np.reshape(π, (self._actor_weights.shape[0], 1))
        features = np.zeros_like(self._actor_weights)
        features[action, state] = 1

        grad = features
        grad[:, state] -= π
        return grad

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
        state = np.concatenate(
            [
                np.zeros((1,), dtype=np.int32),
                self._tiler.get_indices(state) + 1,
            ]
        )
        next_state = np.concatenate(
            [
                np.zeros((1,), dtype=np.int32),
                self._tiler.get_indices(next_state) + 1,
            ]
        )

        # Calculate TD error
        target = reward + done_mask * self._γ * \
            self._critic_weights[next_state].sum()
        estimate = self._critic_weights[state].sum()
        delta = target - estimate

        # Critic update
        if self.use_critic_trace:
            # Update critic eligibility trace
            self._critic_trace *= (self._γ * self._λ)
            if self._trace_type == "accumulating":
                self._critic_trace[state] += 1
            elif self._trace_type == "replacing":
                self._critic_trace[state] = 1
            else:
                raise ValueError(f"unknown trace type {self._trace_type}")

            # Update critic
            self._critic_weights += (self._critic_α * delta *
                                     self._critic_trace)
        else:
            grad = np.zeros_like(self._critic_weights)
            grad[state] = 1
            self._critic_weights += (self._critic_α * delta * grad)

        # Actor update
        actor_grad = self._actor_grad(state, action)
        if self.use_actor_trace:
            # Update actor eligibility traces
            self._actor_trace *= (self._γ * self._λ)
            self._actor_trace += actor_grad

            # Update actor weights
            self._actor_weights += (self._actor_α * delta * self._actor_trace)
        else:
            self._actor_weights += (self._actor_α * delta * actor_grad)

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
        if self.use_actor_trace:
            self._actor_trace = np.zeros_like(self._actor_trace)
        if self.use_critic_trace:
            self._critic_trace = np.zeros(self._size)

    def eval(self):
        """
        Sets the agent into offline evaluation mode, where the agent will not
        explore
        """
        self._is_training = False

    def train(self):
        """
        Sets the agent to online training mode, where the agent will explore
        """
        self._is_training = True

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

        For example, in the SoftmaxAC class, we must save not only the actor
        and critic weights, but also the accumulated eligibility traces.

        Returns
        -------
        dict of str to array_like
            The agent's weights
        """
        pass


if __name__ == "__main__":
    a = SoftmaxAC(0.9, 0.1, 0.1, 0.5, 3, False)
    print(a.actor_weights, a.critic_weights)
    state = np.array([1, 2, 1])
    action = a.sample_action(state)
    a.update(state, action, 1, np.array([1, 2, 2]), 0.9)
    print(a.actor_weights, a.critic_weights)
    state = np.array([1, 2, 2])
    action = a.sample_action(state)
    a.update(state, action, 1, np.array([3, 1, 2]), 0.9)
    print(a.actor_weights, a.critic_weights)
