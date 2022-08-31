#!/usr/bin/env python3

# Import modules
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import agent.nonlinear.nn_utils as nn_utils


class Q(nn.Module):
    """
    Class Q implements an action-value network using a CNN function
    approximator. The network has a single output, which is the action value
    for the input action in the input state.

    The action value is compute by first convolving the state observation, the
    concatenating the flattened state convolution with the action and using
    this as input to the fully connected layers. A single action value is
    outputted for the input action.
    """
    def __init__(self, input_dim, action_dim, channels, kernel_sizes,
                 hidden_sizes, init, activation):
        """
        Constructor

        Parameters
        ----------
        input_dim : tuple[int, int, int]
            Dimensionality of state features, which should be (channels,
            height, width)
        action_dim : int
            Dimensionality of the action vector
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
        """
        super(Q, self).__init__()

        self.conv, self.linear = nn_utils._construct_conv_linear(
            input_dim,
            action_dim,
            channels,
            kernel_sizes,
            hidden_sizes,
            init,
            activation,
            True,
        )

    def forward(self, state, action):
        """
        Performs the forward pass through the network, predicting the
        action-value for `action` in `state`.

        Parameters
        ----------
        state : torch.Tensor[float]
            The state that the action was taken in
        action : torch.Tensor[float] or np.ndarray[float]
            The action to get the value of

        Returns
        -------
        torch.Tensor
            The action value prediction
        """
        if isinstance(state, np.ndarray):
            x = torch.tensor(state)

        x = self.conv(state)

        x = torch.flatten(x)
        x = torch.cat([x, action])
        return self.linear(x)


class DiscreteQ(nn.Module):
    """
    Class DiscreteQ implements an action-value network using a CNN function
    approximator. The network outputs one action value for each available
    action.
    """
    def __init__(self, input_dim, num_actions, channels, kernel_sizes,
                 hidden_sizes, init, activation):
        """
        Constructor

        Parameters
        ----------
        input_dim : tuple[int, int, int]
            Dimensionality of state features, which should be (channels,
            height, width)
        num_actions : int
            The number of available actions in the environment
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
        """
        super(DiscreteQ, self).__init__()

        self.conv, self.linear = nn_utils._construct_conv_linear(
            input_dim,
            num_actions,
            channels,
            kernel_sizes,
            hidden_sizes,
            init,
            activation,
            False,
        )

    def forward(self, state):
        """
        Performs the forward pass through the network, predicting an action
        value for each action in `state`.

        Parameters
        ----------
        state : torch.Tensor[float] or np.array[float]
            The state that the action was taken in

        Returns
        -------
        torch.Tensor
            The action value prediction for each action in `state`
        """
        if isinstance(state, np.ndarray):
            x = torch.tensor(x)

        x = self.conv(state)
        return self.linear(torch.flatten(x, start_dim=1))


class DoubleDiscreteQ(nn.Module):
    """
    Class DoubleDiscreteQ implements a double action-value network
    using a CNN function approximator.
    The network outputs two action values for each available action.
    """
    def __init__(self, input_dim, num_actions, channels, kernel_sizes,
                 hidden_sizes, init, activation):
        """
        Constructor

        Parameters
        ----------
        input_dim : tuple[int, int, int]
            Dimensionality of state features, which should be (channels,
            height, width)
        num_actions : int
            The number of available actions in the environment
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
        """
        super(DoubleDiscreteQ, self).__init__()

        self.conv1, self.linear1 = nn_utils._construct_conv_linear(
            input_dim,
            num_actions,
            channels,
            kernel_sizes,
            hidden_sizes,
            init,
            activation,
            False,
        )

        self.conv2, self.linear2 = nn_utils._construct_conv_linear(
            input_dim,
            num_actions,
            channels,
            kernel_sizes,
            hidden_sizes,
            init,
            activation,
            False,
        )

    def forward(self, state):
        """
        Performs the forward pass through the network, predicting an action
        value for each action in `state`.

        Parameters
        ----------
        state : torch.Tensor[float] or np.array[float]
            The state that the action was taken in

        Returns
        -------
        torch.Tensor
            The action value prediction for each action in `state`
        """
        if isinstance(state, np.ndarray):
            x = torch.tensor(x)

        x1 = self.conv1(state)
        q1 = self.linear1(torch.flatten(x1, start_dim=1))

        x2 = self.conv2(state)
        q2 = self.linear2(torch.flatten(x2, start_dim=1))

        return q1, q2
