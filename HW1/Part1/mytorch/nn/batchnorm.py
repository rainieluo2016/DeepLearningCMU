from mytorch.tensor import Tensor
import numpy as np
from mytorch.nn.module import Module

class BatchNorm1d(Module):
    """Batch Normalization Layer

    Args:
        num_features (int): # dims in input and output
        eps (float): value added to denominator for numerical stability
                     (not important for now)
        momentum (float): value used for running mean and var computation

    Inherits from:
        Module (mytorch.nn.module.Module)
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features

        self.eps = Tensor(np.array([eps]))
        self.momentum = Tensor(np.array([momentum]))  # 1 - alpha

        # To make the final output affine
        self.gamma = Tensor(np.ones((self.num_features,)), requires_grad=True, is_parameter=True)
        self.beta = Tensor(np.zeros((self.num_features,)), requires_grad=True, is_parameter=True)

        # Running mean and var
        self.running_mean = Tensor(np.zeros(self.num_features,), requires_grad=False, is_parameter=False)
        self.running_var = Tensor(np.ones(self.num_features,), requires_grad=False, is_parameter=False)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Args:
            x (Tensor): (batch_size, num_features)
        Returns:
            Tensor: (batch_size, num_features)
        """
        m = x.shape[0]
        sample_size = Tensor(np.array([m]))
        if self.is_train:
            # mean = np.mean(x.data, axis=1).reshape(-1, 1)
            # print('mean', Tensor(mean), mean.shape)
            # print('x - mean', x.data - mean)
            # var = np.sum(np.square(x.data - mean), axis=1).reshape(-1, 1) / m
            # unbiased_var = var * m / (m - 1)
            # print('var', var, var.shape)
            # print('unbiased var', unbiased_var, unbiased_var.shape)
            # norm = (x.data - mean) / np.sqrt(var + self.eps.data)
            # print('denominator', np.sqrt(var + self.eps.data))
            # print('norm', norm, norm.shape)
            # print('gamma', self.gamma.data, 'beta', self.beta.data)
            # out = self.gamma.data * norm + self.beta.data
            # print('out', out, out.shape)
            # self.running_mean.data = (1 - self.momentum.data) * self.running_mean.data + self.momentum.data * mean
            # self.running_var.data = (1 - self.momentum.data) * self.running_var.data + self.momentum.data * unbiased_var
            mean = x.sum(axis=0, keepdims=True) / sample_size
            # print(x.shape, mean.shape)
            centered_x = x - mean
            var = (centered_x * centered_x).sum(axis=0, keepdims=True) / sample_size
            # print(centered_x.shape, var.shape)
            unbiased_var = var * Tensor(np.array([m / (m - 1)]))
            norm = centered_x / (var + self.eps).sqrt()
            out = self.gamma * norm + self.beta
            # self.running_mean.data = (1 - self.momentum.data) * self.running_mean.data + self.momentum.data * mean.data
            # self.running_var.data = (1 - self.momentum.data) * self.running_var.data + self.momentum.data * unbiased_var.data
            self.running_mean.data = (1 - self.momentum.data) * self.running_mean.data + self.momentum.data * mean.data
            self.running_var.data = (1 - self.momentum.data) * self.running_var.data + self.momentum.data * unbiased_var.data
        else:
            norm = (x - self.running_mean) / (self.running_var + self.eps).sqrt()
            out = self.gamma * norm + self.beta
            # print('hi')
        return out

