import numpy as np
from mytorch import tensor
from mytorch.tensor import Tensor
from mytorch.nn.module import Module
from mytorch.nn.activations import Tanh, ReLU, Sigmoid
from mytorch.nn.util import PackedSequence, pack_sequence, unpack_sequence
from mytorch.nn.rnn import TimeIterator


class GRUUnit(Module):
    '''
    This class defines a single GRU Unit block.
    
    NOTE: *args is placed just to ignore the nonlinearity parameter that it recevies from GRU module as GRUs have fixed set of activation functions that are called unlike RNNs. Given we will be using the same rnn.TimeIterator class to construct GRU we need to ignore the nonlinearity parameter. 

    Args:
        input_size (int): # features in each timestep
        hidden_size (int): # features generated by the GRUUnit at each timestep
    '''

    def __init__(self, input_size, hidden_size, *args, **kwargs):
        super(GRUUnit, self).__init__()

        # Initializing parameters
        self.weight_ir = Tensor(np.random.randn(hidden_size, input_size), requires_grad=True, is_parameter=True)
        self.bias_ir = Tensor(np.zeros(hidden_size), requires_grad=True, is_parameter=True)

        self.weight_iz = Tensor(np.random.randn(hidden_size, input_size), requires_grad=True, is_parameter=True)
        self.bias_iz = Tensor(np.zeros(hidden_size), requires_grad=True, is_parameter=True)

        self.weight_in = Tensor(np.random.randn(hidden_size, input_size), requires_grad=True, is_parameter=True)
        self.bias_in = Tensor(np.zeros(hidden_size), requires_grad=True, is_parameter=True)

        self.weight_hr = Tensor(np.random.randn(hidden_size, hidden_size), requires_grad=True, is_parameter=True)
        self.bias_hr = Tensor(np.zeros(hidden_size), requires_grad=True, is_parameter=True)

        self.weight_hz = Tensor(np.random.randn(hidden_size, hidden_size), requires_grad=True, is_parameter=True)
        self.bias_hz = Tensor(np.zeros(hidden_size), requires_grad=True, is_parameter=True)

        self.weight_hn = Tensor(np.random.randn(hidden_size, hidden_size), requires_grad=True, is_parameter=True)
        self.bias_hn = Tensor(np.zeros(hidden_size), requires_grad=True, is_parameter=True)

        self.hidden_size = hidden_size

    def __call__(self, input, hidden=None):
        return self.forward(input, hidden)

    def forward(self, input, hidden=None):
        '''
        Args:
            input (Tensor): (effective_batch_size,input_size)
            hidden (Tensor,None): (effective_batch_size,hidden_size)
        Return:
            Tensor: (effective_batch_size,hidden_size)
        '''

        # TODO: INSTRUCTIONS
        # Perform matrix operations to construct the intermediary value from input and hidden tensors
        # Remeber to handle the case when hidden = None. Construct a tensor of appropriate size, filled with 0s to use as the hidden.
        tanh = Tanh()
        sigmoid = Sigmoid()
        input_reset = input @ self.weight_ir.T()
        if hidden is None:
            hidden = Tensor(np.zeros((input.shape[0], self.weight_ir.shape[0])))
        hidden_reset = hidden @ self.weight_hr.T()
        r_t = sigmoid(input_reset + self.bias_ir.reshape(1, -1) + hidden_reset + self.bias_hr.reshape(1, -1))
        input_update = input @ self.weight_iz.T()
        hidden_update = hidden @ self.weight_hz.T()
        z_t = sigmoid(input_update + self.bias_iz.reshape(1, -1) + hidden_update + self.bias_hz.reshape(1, -1))
        n_t = tanh(input @ self.weight_in.T() + self.bias_in.reshape(1, -1) + r_t * (hidden @ self.weight_hn.T() + self.bias_hn.reshape(1, -1)))
        h_t = (tensor.Tensor(1) - z_t) * n_t + z_t * hidden
        return h_t


class GRU(TimeIterator):
    '''
    Child class for TimeIterator which appropriately initializes the parent class to construct an GRU.
    Args:
        input_size (int): # features in each timestep
        hidden_size (int): # features generated by the GRUUnit at each timestep
    '''

    def __init__(self, input_size, hidden_size):
        super(GRU, self).__init__(GRUUnit, input_size, hidden_size)