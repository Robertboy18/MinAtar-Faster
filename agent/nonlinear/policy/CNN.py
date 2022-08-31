# Import modules
import agent.nonlinear.nn_utils as nn_utils
import numpy as np
import time
import torch
from torch.distributions import Normal, Independent
import torch.nn as nn
import torch.nn.functional as F


# Global variables
EPSILON = 1e-6


# Class definitions
class Softmax(nn.Module):
    """
    Softmax implements a softmax policy in each state, parameterized
    using an CNN to predict logits.
    """
    def __init__(self, input_dim, channels, kernel_sizes,
                 hidden_sizes,  init, activation, action_space):
        """
        Constructor

        Parameters
        ----------
        input_dim : tuple[int, int, int]
            Dimensionality of state features, which should be (channels,
            height, width)
        channels : array-like[int]
            The number of channels in each hidden convolutional layer
        kernel_sizes : array-like[int]
            The number of channels in each consecutive convolutional layer
        hidden_sizes : array-like[int]
            The number of units in each consecutive fully connected layer
        init : str
            The initialization scheme to use for the weights, one of
            'xavier_uniform', 'xavier_normal', 'uniform', 'normal',
            'orthogonal', by default None. If None, leaves the default
            PyTorch initialization.
        activation : indexable[str] or str
            The activation function to use; each element should be one of
            'relu', 'tanh'
        action_space : gym.Spaces.Discrete
            The action space
        """
        super(Softmax, self).__init__()

        self.num_actions = action_space.n
        self.conv, self.linear = nn_utils._construct_conv_linear(
            input_dim,
            self.num_actions,
            channels,
            kernel_sizes,
            hidden_sizes,
            init,
            activation,
            False,
        )

    def forward(self, state):
        """
        Performs the forward pass through the network, predicting a logit for
        each action in `state`.

        Parameters
        ----------
        state : torch.Tensor[float] or np.array[float]
            The state that the action was taken in

        Returns
        -------
        torch.Tensor
            The logit for each action in `state` with shape `(batch,
            num_actions)`
        """
        if isinstance(state, np.ndarray):
            x = torch.tensor(state)

        x = self.conv(state)
        return self.linear(torch.flatten(x, start_dim=1))

    def sample(self, state, num_samples=1, log_prob=False):
        """
        Returns actions sampled from the policy in `state`

        Parameters
        ----------
        state : torch.Tensor
            The states to sample the actions in
        num_samples : int, optional
            The number of actions to sampler per state
        log_prob : bool, optional
            Whether or not to return the log probability of each action in
            each state in `state`, by default `False`

        Returns
        -------
        torch.Tensor
            A sample of `num_samples` actions in each state, with shape
            `(num_samples, batch, action_dims = 1)`
        """
        logits = self.forward(state)

        probs = F.softmax(logits, dim=1)

        policy = torch.distributions.Categorical(probs)
        actions = policy.sample((num_samples,))

        log_prob_val = None
        if log_prob:
            log_prob_val = F.log_softmax(logits, dim=1)
            log_prob_val = torch.gather(log_prob_val, dim=1, index=actions)

        if num_samples == 1:
            actions = actions.squeeze(0)
            if log_prob:
                log_prob_val = log_prob_val.squeeze(0)

        actions = actions.unsqueeze(-1)
        if log_prob:
            log_prob_val = log_prob_val.unsqueeze(-1)

        return actions.long(), log_prob_val, None

    def all_log_prob(self, states):
        """
        Returns the log probability of taking each action in `states`.
        """
        logits = self.forward(states)
        log_probs = F.log_softmax(logits, dim=1)

        return log_probs

    def log_prob(self, states, actions):
        """
        Returns the log probability of taking `actions` in `states`.
        """
        logits = self.forward(states)
        log_probs = F.log_softmax(logits, dim=1)
        if actions.shape[0] == log_probs.shape[0] and len(actions.shape) == 1:
            actions = actions.unsqueeze(-1)
        log_probs = torch.gather(log_probs, dim=1, index=actions.long())

        return log_probs
