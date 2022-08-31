# Import modules
import os
from gym.spaces import Box
import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from agent.baseAgent import BaseAgent
import agent.nonlinear.nn_utils as nn_utils
from agent.nonlinear.policy.CNN import Softmax
from agent.nonlinear.value_function.CNN import DoubleDiscreteQ as Q
from utils.experience_replay import TorchBuffer as ExperienceReplay


class SACDiscrete(BaseAgent):
    """
    SACDiscrete implements a discrete-action Soft Actor-Critic agent with CNN
    function approximation.

    SACDiscrete works only with discrete action spaces.
    """
    def __init__(self, gamma, tau, alpha, policy, env,
                 target_update_interval, critic_lr, actor_lr_scale, alpha_lr,
                 hidden_dim, kernel_sizes, channels, replay_capacity, seed,
                 batch_size, betas, cuda=False,
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

        Raises
        ------
        ValueError
            If the batch size is larger than the replay buffer
        """
        self.step = 0

        # Do some random error checking
        action_space = env.action_space
        obs_space = env.observation_space

        if isinstance(action_space, Box):
            raise ValueError("SACDiscrete can only be used with " +
                             "discrete actions")

        # Ensure batch size < replay capacity
        if batch_size > replay_capacity:
            raise ValueError("cannot have a batch larger than replay " +
                             "buffer capacity")

        super().__init__()
        self.batch = True

        # Set the seed for all random number generators, this includes
        # everything used by PyTorch, including setting the initial weights
        # of networks. PyTorch prefers seeds with many non-zero binary units
        self.torch_rng = torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.is_training = True
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.device = torch.device("cuda:0" if cuda and
                                   torch.cuda.is_available() else "cpu")

        # Keep a replay buffer
        obs_dim = obs_space.shape
        self.replay = ExperienceReplay(replay_capacity, seed, obs_dim,
                                       1, self.device)
        self.batch_size = batch_size

        # Set the interval between timesteps when the target network should be
        # updated and keep a running total of update number
        self.target_update_interval = target_update_interval
        self.update_number = 0

        self.num_actions = action_space.n
        self.critic = Q(obs_dim, self.num_actions, channels, kernel_sizes,
                        hidden_dim, init, activation).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr,
                                 betas=betas)

        self.critic_target = Q(obs_dim, self.num_actions, channels,
                               kernel_sizes, hidden_dim, init,
                               activation).to(self.device)
        nn_utils.hard_update(self.critic_target, self.critic)

        self.policy_type = policy.lower()
        if self.policy_type == "softmax":
            self.policy = Softmax(obs_dim, channels, kernel_sizes, hidden_dim,
                                  init, activation,
                                  action_space).to(self.device)

            actor_lr = actor_lr_scale * critic_lr
            self.policy_optim = Adam(self.policy.parameters(), lr=actor_lr,
                                     betas=betas)

        else:
            raise NotImplementedError(f"policy type {policy.lower()} not " +
                                      "available")

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
            raise ValueError("cannot sample actions in eval mode yet")

        act = action.detach().cpu().numpy()[0]
        return int(act[0])

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
        self.step += 1

        # Adjust action to ensure it can be sent to the experience replay
        # buffer properly
        action = np.array([action])

        # Keep transition in replay buffer
        self.replay.push(state, action, reward, next_state, done_mask)

        if self.step % 4 != 0:
            return

        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, \
            mask_batch = self.replay.sample(batch_size=self.batch_size)

        # For rewards, actions, and masks, we know they are scalars, so
        # squeeze the final dimension
        reward_batch = reward_batch.squeeze()
        mask_batch = mask_batch.squeeze()
        action_batch = action_batch.squeeze().long()

        # When updating Q functions, we don't want to backprop through the
        # policy and target network parameters when computing the target for
        # the update
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = \
                    self.policy.sample(next_state_batch, log_prob=True)
            next_state_log_pi = next_state_log_pi.squeeze(-1)

            next_q1, next_q2 = self.critic_target(next_state_batch)

            next_q1 = next_q1[np.arange(self.batch_size),
                              next_state_action.squeeze()]
            next_q2 = next_q2[np.arange(self.batch_size),
                              next_state_action.squeeze()]

            min_soft_q = torch.min(next_q1, next_q2) \
                - self.alpha * next_state_log_pi
            target = reward_batch + mask_batch * self.gamma * min_soft_q

        # Get the value of the current state
        q1, q2 = self.critic(state_batch)
        q1 = q1[np.arange(self.batch_size), action_batch]
        q2 = q2[np.arange(self.batch_size), action_batch]

        # Calculate the losses on each critic
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        q1_loss = F.mse_loss(q1, target)

        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        q2_loss = F.mse_loss(q2, target)
        q_loss = q1_loss + q2_loss

        # Update the critic
        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

        # Calculate the actor loss using Eqn(5) in FKL/RKL paper
        # Repeat the state for each action
        # state_batch = state_batch.repeat_interleave(self.num_actions, dim=0)
        # actions = torch.tensor([n for n in range(self.num_actions)])
        # actions = actions.repeat(self.batch_size)
        # actions = actions.long()

        q1, q2 = self.critic(state_batch)
        q1 = q1.flatten()
        q2 = q2.flatten()
        # q1 = q1[np.arange(state_batch.shape[0]), actions]
        # q2 = q2[np.arange(state_batch.shape[0]), actions]
        min_q = torch.min(q1, q2)

        log_prob = self.policy.all_log_prob(state_batch).squeeze().flatten()
        prob = log_prob.exp()
        policy_loss = prob * (min_q - log_prob * self.alpha)
        policy_loss = policy_loss.reshape([self.batch_size, self.num_actions])
        policy_loss = -policy_loss.sum(dim=1).mean()

        # Update the actor
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Increment the running total of updates and update the critic target
        # if needed
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
        self.policy.eval()
        self.critic.eval()
        self.is_training = False

    def train(self):
        """
        Sets the agent to online training mode, where the agent will explore
        """
        self.policy.train()
        self.critic.train()
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
#         if not os.path.exists('models/'):
#             os.makedirs('models/')
#
#         if actor_path is None:
#             actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
#         if critic_path is None:
#             critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
#         print('Saving models to {} and {}'.format(actor_path, critic_path))
#         torch.save(self.policy.state_dict(), actor_path)
#         torch.save(self.critic.state_dict(), critic_path)

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
#         parameters = {}
#         parameters["actor_weights"] = self.policy.state_dict()
#         parameters["actor_optimizer"] = self.policy_optim.state_dict()
#         parameters["critic_weights"] = self.critic.state_dict()
#         parameters["critic_optimizer"] = self.critic_optim.state_dict()
#         parameters["critic_target"] = self.critic_target.state_dict()
#         parameters["entropy"] = self.alpha
#
#         if self.automatic_entropy_tuning:
#             parameters["log_entropy"] = self.log_alpha
#             parameters["entropy_optimizer"] = self.alpha_optim.state_dict()
#             parameters["target_entropy"] = self.target_entropy
#
#         return parameters


if __name__ == "__main__":
    import gym
    a = gym.make("MountainCarContinuous-v0")
    actions = a.action_space
    s = SAC(num_inputs=5, action_space=actions, gamma=0.9, tau=0.8,
            alpha=0.2, policy="Gaussian", target_update_interval=10,
            critic_lr=0.01, actor_lr=0.01, alpha_lr=0.01, actor_hidden_dim=200,
            critic_hidden_dim=200, replay_capacity=50, seed=0, batch_size=10,
            automatic_entropy_tuning=False, cuda=False)
