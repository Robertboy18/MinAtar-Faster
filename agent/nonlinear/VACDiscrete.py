#!/usr/bin/env python3

# Import modules
import torch
import time
from gym.spaces import Box, Discrete
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from agent.baseAgent import BaseAgent
import agent.nonlinear.nn_utils as nn_utils
from agent.nonlinear.policy_utils import GaussianPolicy, SoftmaxPolicy
from agent.nonlinear.value_function_utils import QMLP
from utils.experience_replay import TorchBuffer as ExperienceReplay


class VACDiscrete(BaseAgent):
    """
    VACDiscrete implements the Vanilla Actor-Critic agent.

    VACDiscrete works only with discrete actions and uses MLP function
    approximators.
    """
    def __init__(self, num_inputs, action_space, gamma, tau, alpha, policy,
                 target_update_interval, critic_lr, actor_lr_scale,
                 actor_hidden_dim, critic_hidden_dim,
                 replay_capacity, seed, batch_size, betas, cuda=False,
                 clip_stddev=1000, init=None, activation="relu"):
        """
        Constructor

        Parameters
        ----------
        num_inputs : int
            The number of input features
        action_space : gym.spaces.Space
            The action space from the gym environment
        gamma : float
            The discount factor
        tau : float
            The weight of the weighted average, which performs the soft update
            to the target critic network's parameters toward the critic
            network's parameters, that is: target_parameters =
            ((1 - τ) * target_parameters) + (τ * source_parameters)
        alpha : float
            The entropy regularization temperature. See equation (1) in paper.
        policy : str
            The type of policy, currently, only support "softmax"
        target_update_interval : int
            The number of updates to perform before the target critic network
            is updated toward the critic network
        critic_lr : float
            The critic learning rate
        actor_lr : float
            The actor learning rate
        actor_hidden_dim : int
            The number of hidden units in the actor's neural network
        critic_hidden_dim : int
            The number of hidden units in the critic's neural network
        replay_capacity : int
            The number of transitions stored in the replay buffer
        seed : int
            The random seed so that random samples of batches are repeatable
        batch_size : int
            The number of elements in a batch for the batch update
        cuda : bool, optional
            Whether or not cuda should be used for training, by default False.
            Note that if True, cuda is only utilized if available.
        clip_stddev : float, optional
            The value at which the standard deviation is clipped in order to
            prevent numerical overflow, by default 1000. If <= 0, then
            no clipping is done.
        init : str
            The initialization scheme to use for the weights, one of
            'xavier_uniform', 'xavier_normal', 'uniform', 'normal',
            'orthogonal', by default None. If None, leaves the default
            PyTorch initialization.

        Raises
        ------
        ValueError
            If the batch size is larger than the replay buffer
        """
        super().__init__()
        self.batch = True

        # Ensure batch size < replay capacity
        if batch_size > replay_capacity:
            raise ValueError("cannot have a batch larger than replay " +
                             "buffer capacity")

        # Set the seed for all random number generators, this includes
        # everything used by PyTorch, including setting the initial weights
        # of networks. PyTorch prefers seeds with many non-zero binary units
        self.torch_rng = torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.is_training = True
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.discrete_action = isinstance(action_space, Discrete)
        self.state_dims = num_inputs

        self.device = torch.device("cuda:0" if cuda and
                                   torch.cuda.is_available() else "cpu")

        if isinstance(action_space, Box):
            raise ValueError("VACDiscrete can only be used with " +
                             "discrete actions")
        elif isinstance(action_space, Discrete):
            self.action_dims = 1
            # Keep a replay buffer
            self.replay = ExperienceReplay(replay_capacity, seed, num_inputs,
                                           1, self.device)
        self.batch_size = batch_size

        # Set the interval between timesteps when the target network should be
        # updated and keep a running total of update number
        self.target_update_interval = target_update_interval
        self.update_number = 0

        # Create the critic Q function
        if isinstance(action_space, Box):
            action_shape = action_space.shape[0]
        elif isinstance(action_space, Discrete):
            action_shape = 1

        self.critic = QMLP(num_inputs, action_shape,
                           critic_hidden_dim, init, activation).to(
                               device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr,
                                 betas=betas)

        self.critic_target = QMLP(num_inputs, action_shape,
                                  critic_hidden_dim, init, activation).to(
                                      self.device)
        nn_utils.hard_update(self.critic_target, self.critic)

        self.policy_type = policy.lower()
        actor_lr = actor_lr_scale * critic_lr
        if self.policy_type == "softmax":
            self.num_actions = action_space.n
            self.policy = SoftmaxPolicy(num_inputs, self.num_actions,
                                        actor_hidden_dim, activation,
                                        action_space, init).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=actor_lr,
                                     betas=betas)

        else:
            raise NotImplementedError(f"policy type {policy} not implemented")

    def sample_action(self, state):
        """
        Samples an action from the agent

        Parameters
        ----------
        state : np.array
            The state feature vector

        Returns
        -------
        array_like of float
            The action to take
        """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if self.is_training:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)

        act = action.detach().cpu().numpy()[0]
        if not self.discrete_action:
            return act
        else:
            return int(act[0])

    def sample_action_(self, state, size):
        """
        sample_action_ is like sample_action, except the rng for
        action selection in the environment is not affected by running
        this function.
        """
        if len(state.shape) > 1 or state.shape[0] > 1:
            raise ValueError("sample_action_ takes a single state")
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            if self.is_training:
                mean, log_std = self.policy.forward(state)

        if not self.is_training:
            return mean.detach().cpu().numpy()[0]

        mean = mean.detach().cpu().numpy()[0]
        std = np.exp(log_std.detach().cpu().numpy()[0])
        return self.rng.normal(mean, std, size=size)

    def update(self, state, action, reward, next_state, done_mask):
        """
        Takes a single update step, which may be a number of offline
        batch updates

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
        """
        if self.discrete_action:
            action = np.array([action])
        # Keep transition in replay buffer
        self.replay.push(state, action, reward, next_state, done_mask)

        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, \
            mask_batch = self.replay.sample(batch_size=self.batch_size)

        # When updating Q functions, we don't want to backprop through the
        # policy and target network parameters
        with torch.no_grad():
            next_state_action, _, _ = \
                self.policy.sample(next_state_batch)
            qf_next_value = self.critic_target(next_state_batch,
                                               next_state_action)

            q_target = reward_batch + mask_batch * self.gamma * qf_next_value

        # Two Q-functions to reduce positive bias in policy improvement
        q_prediction = self.critic(state_batch, action_batch)
        # print(torch.cat([reward_batch, action_batch, mask_batch], dim=1))
        # print(q_prediction)

        # Calculate the losses on each critic
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        q_loss = F.mse_loss(q_prediction, q_target)

        # Update the critic
        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

        # Calculate the actor loss using Eqn(5) in FKL/RKL paper
        # No need to use a baseline in this setting
        state_batch = state_batch.repeat_interleave(self.num_actions, dim=0)
        actions = torch.tensor([n for n in range(self.num_actions)])
        actions = actions.repeat(self.batch_size)
        actions = actions.unsqueeze(-1)

        q = self.critic(state_batch, actions)
        log_prob = self.policy.log_prob(state_batch, actions)
        prob = log_prob.exp()

        policy_loss = prob * (q - log_prob * self.alpha)
        policy_loss = policy_loss.reshape([self.batch_size, self.num_actions])
        policy_loss = -policy_loss.sum(dim=1).mean()

        # Update the actor
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Update target network
        self.update_number += 1
        if self.update_number % self.target_update_interval == 0:
            self.update_number = 0
            nn_utils.soft_update(self.critic_target, self.critic, self.tau)

    def reset(self):
        """
        Resets the agent between episodes
        """
        pass

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

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None,
                   critic_path=None):
        """
        Saves the models so that after training, they can be used.

        Parameters
        ----------
        env_name : str
            The name of the environment that was used to train the models
        suffix : str, optional
            The suffix to the filename, by default ""
        actor_path : str, optional
            The path to the file to save the actor network as, by default None
        critic_path : str, optional
            The path to the file to save the critic network as, by default None
        """
        pass

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        """
        Loads in a pre-trained actor and a pre-trained critic to resume
        training.

        Parameters
        ----------
        actor_path : str
            The path to the file which contains the actor
        critic_path : str
            The path to the file which contains the critic
        """
        pass

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

        For example, in the LinearAC class, we must save not only the actor
        and critic weights, but also the accumulated eligibility traces.

        Returns
        -------
        dict of str to float, torch.Tensor
            The agent's weights
        """
        pass
