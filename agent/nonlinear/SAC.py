# Import modules
import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from agent.baseAgent import BaseAgent
import agent.nonlinear.nn_utils as nn_utils
from agent.nonlinear.policy.MLP import SquashedGaussian
from agent.nonlinear.value_function.MLP import DoubleQ, Q
from utils.experience_replay import TorchBuffer as ExperienceReplay


class SAC(BaseAgent):
    """
    SAC implements the Soft Actor-Critic agent found in the paper
    https://arxiv.org/pdf/1812.05905.pdf.

    SAC works only with continuous action spaces and uses MLP function
    approximators.
    """
    def __init__(self, gamma, tau, alpha, policy,
                 target_update_interval, critic_lr, actor_lr_scale, alpha_lr,
                 actor_hidden_dim, critic_hidden_dim, replay_capacity, seed,
                 batch_size, betas, env, reparameterized=True, soft_q=True,
                 double_q=True, automatic_entropy_tuning=False, cuda=False,
                 clip_stddev=1000, init=None, activation="relu"):
        """
        Constructor

        Parameters
        ----------
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
            The type of policy, currently, only support "gaussian"
        target_update_interval : int
            The number of updates to perform before the target critic network
            is updated toward the critic network
        critic_lr : float
            The critic learning rate
        actor_lr : float
            The actor learning rate
        alpha_lr : float
            The learning rate for the entropy parameter, if using an automatic
            entropy tuning algorithm (see automatic_entropy_tuning) parameter
            below
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
        automatic_entropy_tuning : bool, optional
            Whether the agent should automatically tune its entropy
            hyperparmeter alpha, by default False
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
        soft_q : bool
            Whether or not to learn soft Q functions, by default True. The
            original SAC uses soft Q functions since we learn an
            entropy-regularized policy. When learning an entropy regularized
            policy, guaranteed policy improvement (in the ideal case) only
            exists with respect to soft action values.
        reparameterized: bool
            Whether to use the reparameterization trick to learn the policy or
            to use the log-likelihood trick. The original SAC uses the
            reparameterization trick.
        double_q : bool
            Whether or not to use two Q value functions or not. The original
            SAC uses two Q value functions.

        Raises
        ------
        ValueError
            If the batch size is larger than the replay buffer
        """
        super().__init__()

        # Ensure batch size < replay capacity
        if batch_size > replay_capacity:
            raise ValueError("cannot have a batch larger than replay " +
                             "buffer capacity")

        action_space = env.action_space
        obs_space = env.observation_space
        obs_dim = obs_space.shape
        # Ensure we are working with vector observations
        if len(obs_dim) != 1:
            raise ValueError(
                f"""SAC works only with vector observations, but got
                observation with shape {obs_dim}."""
            )

        # Set the seed for all random number generators, this includes
        # everything used by PyTorch, including setting the initial weights
        # of networks. PyTorch prefers seeds with many non-zero binary units
        self._torch_rng = torch.manual_seed(seed)
        self._rng = np.random.default_rng(seed)

        # Random hypers and fields
        self._is_training = True  # Whether in training or evaluation mode
        self._gamma = gamma  # Discount factor
        self._tau = tau  # Polyak averaging constant for target networks
        self._alpha = alpha  # Entropy scale
        self._reparameterized = reparameterized  # Whether to use reparam trick
        self._soft_q = soft_q  # Whether to use soft Q functions or nor
        self._double_q = double_q  # Whether or not to use a double Q critic

        self._device = torch.device("cuda:0" if cuda and
                                    torch.cuda.is_available() else "cpu")

        # Experience replay buffer
        self._batch_size = batch_size
        self._replay = ExperienceReplay(replay_capacity, seed, obs_space.shape,
                                        action_space.shape[0], self._device)

        # Set the interval between timesteps when the target network should be
        # updated and keep a running total of update number
        self._target_update_interval = target_update_interval
        self._update_number = 0

        # Automatic entropy tuning
        self._automatic_entropy_tuning = automatic_entropy_tuning
        assert not self._automatic_entropy_tuning

        # Set up the critic and target critic
        self._init_critics(
            obs_space,
            action_space,
            critic_hidden_dim,
            init,
            activation,
            critic_lr,
            betas,
        )

        # Set up the policy
        self._policy_type = policy.lower()
        actor_lr = actor_lr_scale * critic_lr
        self._init_policy(
            obs_space,
            action_space,
            actor_hidden_dim,
            init,
            activation,
            actor_lr,
            betas,
            clip_stddev,
        )

        # Set up auto entropy tuning
        if self._automatic_entropy_tuning is True:
            self._target_entropy = -torch.prod(
                torch.Tensor(action_space.shape).to(self._device)
            ).item()
            self._log_alpha = torch.zeros(
                1,
                requires_grad=True,
                device=self._device,
            )
            self._alpha_optim = Adam([self._log_alpha], lr=alpha_lr)

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
        state = torch.FloatTensor(state).to(self._device).unsqueeze(0)
        if self._is_training:
            action, _, _, _ = self._policy.rsample(state)
        else:
            _, _, action, _ = self._policy.rsample(state)

        return action.detach().cpu().numpy()[0]  # size (1, action_dims)

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
        # Keep transition in replay buffer
        self._replay.push(state, action, reward, next_state, done_mask)

        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, \
            mask_batch = self._replay.sample(batch_size=self._batch_size)

        self._update_critic(state_batch, action_batch, reward_batch,
                            next_state_batch, mask_batch)

        self._update_actor(state_batch, action_batch, reward_batch,
                           next_state_batch, mask_batch)

    def _update_actor(self, state_batch, action_batch, reward_batch,
                      next_state_batch, mask_batch):
        """
        Update the actor given a batch of transitions sampled from a replay
        buffer.
        """
        # Calculate the actor loss
        if self._reparameterized:
            # Reparameterization trick
            pi, log_pi, _, _ = self._policy.rsample(state_batch)
            q = self._get_q(state_batch, pi)

            policy_loss = ((self._alpha * log_pi) - q).mean()

        else:
            # Log likelihood trick
            with torch.no_grad():
                # Context manager ensures that we don't backprop through the q
                # function when minimizing the policy loss
                pi, log_pi, _, x_t = self._policy.sample(state_batch)
                q = self._get_q(state_batch, pi)

            # Compute the policy loss, grad_log_pi will be the only
            # differentiated value
            grad_log_pi = self._policy.log_prob(state_batch, x_t)
            policy_loss = grad_log_pi * (self._alpha * log_pi - q)
            policy_loss = policy_loss.mean()

        # Update the actor
        self._policy_optim.zero_grad()
        policy_loss.backward()
        self._policy_optim.step()

        # Tune the entropy if appropriate
        if self._automatic_entropy_tuning:
            alpha_loss = -(self._log_alpha *
                           (log_pi + self._target_entropy).detach()).mean()

            self._alpha_optim.zero_grad()
            alpha_loss.backward()
            self._alpha_optim.step()

            self._alpha = self._log_alpha.exp()

    def _update_critic(self, state_batch, action_batch, reward_batch,
                       next_state_batch, mask_batch):
        """
        Update the critic(s) given a batch of transitions sampled from a replay
        buffer.
        """
        if self._double_q:
            self._update_double_critic(
                state_batch,
                action_batch,
                reward_batch,
                next_state_batch,
                mask_batch,
            )
        else:
            self._update_single_critic(
                state_batch,
                action_batch,
                reward_batch,
                next_state_batch,
                mask_batch,
            )

    def _update_single_critic(self, state_batch, action_batch, reward_batch,
                              next_state_batch, mask_batch):
        """
        Update the critic using a batch of transitions when using a single Q
        critic.
        """
        if self._double_q:
            raise ValueError("cannot call _update_single_critic when using " +
                             "a double Q critic")

        # When updating Q functions, we don't want to backprop through the
        # policy and target network parameters
        with torch.no_grad():
            # Sample an action in the next state for the SARSA update
            next_state_action, next_state_log_pi, _, _ = \
                self._policy.sample(next_state_batch)

            # Calculate the Q value of the next action in the next state
            q_next = self._critic_target(next_state_batch, next_state_action)
            if self._soft_q:
                q_next -= self._alpha * next_state_log_pi

            # Calculate the target for the SARSA update
            q_target = reward_batch + mask_batch * self._gamma * q_next

        # Calculate the Q value of each action in each respective state
        q = self._critic(state_batch, action_batch)

        # Calculate the loss between the target and estimate Q values
        q_loss = F.mse_loss(q, q_target)

        # Update the critic
        self._critic_optim.zero_grad()
        q_loss.backward()
        self._critic_optim.step()

        # Increment the running total of updates and update the critic target
        # if needed
        self._update_number += 1
        if self._update_number % self._target_update_interval == 0:
            self._update_number = 0
            nn_utils.soft_update(self._critic_target, self._critic, self._tau)

    def _update_double_critic(self, state_batch, action_batch, reward_batch,
                              next_state_batch, mask_batch):
        """
        Update the critic using a batch of transitions when using a double Q
        critic.
        """

        if not self._double_q:
            raise ValueError("cannot call _update_single_critic when using " +
                             "a double Q critic")

        # When updating Q functions, we don't want to backprop through the
        # policy and target network parameters
        with torch.no_grad():
            # Sample an action in the next state for the SARSA update
            next_state_action, next_state_log_pi, _, _ = \
                self._policy.sample(next_state_batch)

            # Calculate the action values for the next state
            next_q1, next_q2 = self._critic_target(next_state_batch,
                                                   next_state_action)

            # Double Q: target uses the minimum of the two computed action
            # values
            min_next_q = torch.min(next_q1, next_q2)

            # If using soft action value functions, then adjust the target
            if self._soft_q:
                min_next_q -= self._alpha * next_state_log_pi

            # Calculate the target for the action value function update
            q_target = reward_batch + mask_batch * self._gamma * min_next_q

        # Calculate the two Q values of each action in each respective state
        q1, q2 = self._critic(state_batch, action_batch)

        # Calculate the losses on each critic
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        q1_loss = F.mse_loss(q1, q_target)

        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        q2_loss = F.mse_loss(q2, q_target)
        q_loss = q1_loss + q2_loss

        # Update the critic
        self._critic_optim.zero_grad()
        q_loss.backward()
        self._critic_optim.step()

        # Increment the running total of updates and update the critic target
        # if needed
        self._update_number += 1
        if self._update_number % self._target_update_interval == 0:
            self._update_number = 0
            nn_utils.soft_update(self._critic_target, self._critic, self._tau)

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
        self._is_training = False

    def train(self):
        """
        Sets the agent to online training mode, where the agent will explore
        """
        self._is_training = True

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

    def _init_critics(self, obs_space, action_space, critic_hidden_dim, init,
                      activation, critic_lr, betas):
        """
        Initializes the critic(s)
        """
        num_inputs = obs_space.shape[0]
        if self._double_q:
            critic_type = DoubleQ
        else:
            critic_type = Q

        self._critic = critic_type(num_inputs, action_space.shape[0],
                                   critic_hidden_dim, init,
                                   activation).to(device=self._device)
        self._critic_optim = Adam(self._critic.parameters(), lr=critic_lr,
                                  betas=betas)

        self._critic_target = critic_type(num_inputs, action_space.shape[0],
                                          critic_hidden_dim, init,
                                          activation).to(self._device)

        nn_utils.hard_update(self._critic_target, self._critic)

    def _init_policy(self, obs_space, action_space, actor_hidden_dim, init,
                     activation,  actor_lr, betas, clip_stddev):
        """
        Initializes the policy
        """
        num_inputs = obs_space.shape[0]
        if self._policy_type == "squashedgaussian":
            self._policy = SquashedGaussian(num_inputs, action_space.shape[0],
                                            actor_hidden_dim, activation,
                                            action_space, clip_stddev,
                                            init).to(self._device)
            self._policy_optim = Adam(self._policy.parameters(), lr=actor_lr,
                                      betas=betas)

        else:
            raise NotImplementedError(f"policy {self._policy_type} unknown")

    def _get_q(self, state_batch, action_batch):
        """
        Gets the Q values for `action_batch` actions in `state_batch` states
        from the critic, rather than the target critic.

        Parameters
        ----------
        state_batch : torch.Tensor
            The batch of states to calculate the action values in. Of the form
            (batch_size, state_dims).
        action_batch : torch.Tensor
            The batch of actions to calculate the action values of in each
            state. Of the form (batch_size, action_dims).
        """
        if self._double_q:
            q1, q2 = self._critic(state_batch, action_batch)
            q = torch.min(q1, q2)
        else:
            q = self._critic(state_batch, action_batch)

        return q
