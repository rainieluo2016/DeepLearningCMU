import numpy as np

import mytorch.tensor as tensor
from mytorch.autograd_engine import Function


class Transpose(Function):
    @staticmethod
    def forward(ctx, a):
        if not len(a.shape) == 2:
            raise Exception("Arg for Transpose must be 2D tensor: {}".format(a.shape))
        # print('before transpose', a)
        requires_grad = a.requires_grad
        b = tensor.Tensor(a.data.T, requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        # print('after transpose', b)
        return b

    @staticmethod
    def backward(ctx, grad_output):
        # print('transpose back', tensor.Tensor(grad_output.data.T))
        return tensor.Tensor(grad_output.data.T)


class Reshape(Function):
    @staticmethod
    def forward(ctx, a, shape):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Reshape must be tensor: {}".format(type(a).__name__))
        ctx.shape = a.shape
        requires_grad = a.requires_grad
        c = tensor.Tensor(a.data.reshape(shape), requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        return tensor.Tensor(grad_output.data.reshape(ctx.shape)), None


class Log(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Log must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.log(a.data), requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return tensor.Tensor(grad_output.data / a.data)


class Exp(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Log must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.exp(a.data), requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return tensor.Tensor(np.exp(a.data) * grad_output.data)


"""EXAMPLE: This represents an Op:Add node to the comp graph.

See `Tensor.__add__()` and `autograd_engine.Function.apply()`
to understand how this class is used.

Inherits from:
    Function (autograd_engine.Function)
"""


class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # # Check that args have same shape
        # if a.data.shape != b.data.shape:
        #     raise Exception("Both args must have same sizes: {}, {}".format(a.shape, b.shape))

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)
        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data + b.data, requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        grad_a = np.ones(a.data.shape) * grad_output.data
        grad_b = np.ones(b.data.shape) * grad_output.data
        if grad_a.shape != a.data.shape:
            grad_a = unbroadcast(grad_a, a.data.shape)
        if grad_b.shape != b.data.shape:
            grad_b = unbroadcast(grad_b, b.data.shape)
        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)


class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # Check that args have same shape
        # if a.data.shape != b.data.shape:
        #     raise Exception("Both args must have same sizes: {}, {}".format(a.shape, b.shape))

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data - b.data, requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        grad_a = np.ones(a.shape) * grad_output.data
        grad_b = np.ones(b.shape) * grad_output.data * -1
        if grad_a.shape != a.data.shape:
            grad_a = unbroadcast(grad_a, a.data.shape)
        if grad_b.shape != b.data.shape:
            grad_b = unbroadcast(grad_b, b.data.shape)
        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)


class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # Check that args have same shape
        # if a.data.shape != b.data.shape:
        #     raise Exception("Both args must have same sizes: {}, {}".format(a.shape, b.shape))

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data * b.data, requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        grad_a = b.data * grad_output.data
        grad_b = a.data * grad_output.data
        if grad_a.shape != a.data.shape:
            grad_a = unbroadcast(grad_a, a.data.shape)
        if grad_b.shape != b.data.shape:
            grad_b = unbroadcast(grad_b, b.data.shape)
        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)


class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))

        # Check that args have same shape
        # if a.data.shape != b.data.shape:
        #     raise Exception("Both args must have same sizes: {}, {}".format(a.shape, b.shape))

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data / b.data, requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        grad_a = grad_output.data / b.data
        grad_b = -1 * a.data * grad_output.data / (b.data * b.data)
        if grad_a.shape != a.data.shape:
            grad_a = unbroadcast(grad_a, a.data.shape)
        if grad_b.shape != b.data.shape:
            grad_b = unbroadcast(grad_b, b.data.shape)
        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)


class Matmul(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))
        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(np.matmul(a.data, b.data), requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        grad_a = np.dot(grad_output.data, b.T().data)
        grad_b = np.dot(a.T().data, grad_output.data)
        # the order of gradients returned should match the order of the arguments
        return tensor.Tensor(grad_a), tensor.Tensor(grad_b)


class ReLU(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Log must be tensor: {}".format(type(a).__name__))

        requires_grad = a.requires_grad
        c = tensor.Tensor(np.where(a.data > 0, a.data, 0), requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        ctx.save_for_backward(c)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        c = ctx.saved_tensors[0]
        return tensor.Tensor(np.where(c.data > 0, 1, 0)) * grad_output


# class Sum(Function):
#     @staticmethod
#     def forward(ctx, a):
#         if not type(a).__name__ == 'Tensor':
#             raise Exception("Arg for Log must be tensor: {}".format(type(a).__name__))
#         ctx.save_for_backward(a)
#         c = tensor.Tensor(np.sum(a.data, axis=1).reshape(-1, 1), requires_grad=False,
#                           is_leaf=True)
#         return c
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         a = ctx.saved_tensors[0]
#         N, D = a.data.shape
#         return tensor.Tensor(1/N * np.ones((N,D))) * grad_output

class Sum(Function):
    @staticmethod
    def forward(ctx, a, axis, keepdims):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Only log of tensor is supported")
        ctx.axis = axis
        ctx.shape = a.shape
        if axis is not None:
            ctx.len = a.shape[axis]
        ctx.keepdims = keepdims
        requires_grad = a.requires_grad
        c = tensor.Tensor(a.data.sum(axis=axis, keepdims=keepdims), requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        grad = np.broadcast_to(grad_output.data, ctx.shape)
        assert grad.shape == ctx.shape
        # Take note that gradient tensors SHOULD NEVER have requires_grad = True.
        return tensor.Tensor(grad), None, None


class Square(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Log must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        c = tensor.Tensor(np.sqrt(a.data), requires_grad=False,
                          is_leaf=True)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return tensor.Tensor(-1 / np.square(a)) * grad_output


def unbroadcast(grad, shape, to_keep=0):
    while len(grad.shape) != len(shape):
        grad = grad.sum(axis=0)
    for i in range(len(shape) - to_keep):
        if grad.shape[i] != shape[i]:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


# def unbroadcast(in_array, to_shape):
#     print(in_array.shape, to_shape)
#     current_shape = in_array.shape
#     if len(current_shape) == len(to_shape):
#         for i in range(len(current_shape)):
#             if current_shape[i] != to_shape[i] and to_shape[i] == 1:
#                 return np.sum(in_array, axis=i)
#     else:
#         print('here')
#         for i in range(len(current_shape)):
#             print(i, current_shape[i])
#             if current_shape[i] != to_shape[0]:
#                 print(np.sum(in_array, axis=i))
#                 return np.sum(in_array, axis=i)

class XELoss(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(type(a).__name__, type(b).__name__))
        # Save inputs to access later in backward pass.

        batch_size, num_classes = a.shape
        b = to_one_hot(b, num_classes)
        # print('pred - target', a - b)
        alpha = tensor.Tensor(np.array([1e-6]))
        y_hat = a.exp() / a.exp().sum(axis=1, keepdims=True)
        # print('softmax y_hat', y_hat)
        ctx.save_for_backward(y_hat, b)
        # y_hat = np.exp(predicted.data) / np.sum(np.exp(predicted.data), axis=1).reshape(-1, 1)
        # print('y_hat', y_hat)
        log_y_hat = alpha + (y_hat - alpha).log()
        # print('log_y_hat', log_y_hat)
        # loss = -np.sum(np.array([log_y_hat[i, y_i] for i, y_i in enumerate(target.data)])) / batch_size
        loss = tensor.Tensor(np.array([-1 / batch_size])) * (b * log_y_hat).sum(axis=None)
        # print('loss', loss)
        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(float(loss.data), requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors
        # batch_size, num_classes = a.shape
        # b = to_one_hot(b, num_classes)
        # the order of gradients returned should match the order of the arguments
        return (a - b) * grad_output


def get_conv1d_output_size(input_size, kernel_size, stride):
    """Gets the size of a Conv1d output.

    Notes:
        - This formula should NOT add to the comp graph.
        - Yes, Conv2d would use a different formula,
        - But no, you don't need to account for Conv2d here.

        - If you want, you can modify and use this function in HW2P2.
            - You could add in Conv1d/Conv2d handling, account for padding, dilation, etc.
            - In that case refer to the torch docs for the full formulas.

    Args:
        input_size (int): Size of the input to the layer
        kernel_size (int): Size of the kernel
        stride (int): Stride of the convolution

    Returns:
        int: size of the output as an int (not a Tensor or np.array)
    """
    # TODO: implement the formula in the writeup. One-liner; don't overthink
    return int(np.floor((input_size - kernel_size) / stride) + 1)


class Conv1d(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride):
        """The forward/backward of a Conv1d Layer in the comp graph.

        Notes:
            - Make sure to implement the vectorized version of the pseudocode
            - See Lec 10 slides # TODO: FINISH LOCATION OF PSEUDOCODE
            - No, you won't need to implement Conv2d for this homework.

        Args:
            x (Tensor): (batch_size, in_channel, input_size) input data
            weight (Tensor): (out_channel, in_channel, kernel_size)
            bias (Tensor): (out_channel,)
            stride (int): Stride of the convolution

        Returns:
            Tensor: (batch_size, out_channel, output_size) output data
        """
        # For your convenience: ints for each size
        batch_size, in_channel, input_size = x.shape
        out_channel, _, kernel_size = weight.shape
        # TODO: Save relevant variables for backward pass
        ctx.save_for_backward(x, weight, bias, tensor.Tensor(stride))

        # TODO: Get output size by finishing & calling get_conv1d_output_size()
        output_size = get_conv1d_output_size(input_size, kernel_size, stride)

        # TODO: Initialize output with correct size
        out = np.zeros((batch_size, out_channel, output_size))

        # TODO: Calculate the Conv1d output.
        # Remember that we're working with np.arrays; no new operations needed.
        for b in range(batch_size):
            for i in range(0, input_size - kernel_size + 1, stride):
                segment = x.data[b, :, i:i+kernel_size]
                out[b, :, int(i / stride)] = np.sum(np.sum(segment * weight.data, axis=1), axis=1) + bias.data

        # TODO: Put output into tensor with correct settings and return
        out_tensor = tensor.Tensor(out, x.requires_grad)
        return out_tensor

    @staticmethod
    def backward(ctx, grad_output):
        # TODO: Finish Conv1d backward pass. It's surprisingly similar to the forward pass.
        x, weight, bias, stride = ctx.saved_tensors
        stride = int(stride.data.item())
        batch_size, in_channel, input_size = x.shape
        out_channel, _, kernel_size = weight.shape
        grad = grad_output.data
        upsampled_grad = np.zeros((batch_size, out_channel, input_size - kernel_size + 1))
        for i in range(grad.shape[0]):
            upsampled_grad[i] = np.stack([grad[i]] + [grad[i]*0] * (stride - 1), axis=-1).reshape(grad[i].shape[0], -1)[:, :upsampled_grad.shape[-1]]
        padded_grad = np.zeros((batch_size, out_channel, input_size + kernel_size - 1))
        for i in range(upsampled_grad.shape[0]):
            padded_grad[i] = np.pad(upsampled_grad[i], ((0,0), (kernel_size - 1, kernel_size - 1)), 'constant', constant_values=0)
        dW = np.zeros((out_channel, in_channel, kernel_size))
        dX = np.zeros((batch_size, in_channel, input_size))
        db = np.ones((out_channel, ))
        for b in range(batch_size):
            for ic in range(in_channel):
                for i in range(input_size):
                    w = np.flip(weight.data[:, ic, :], 1)
                    g = padded_grad[b, :, i:i+kernel_size]
                    dX[b, ic, i] = np.sum(w * g)

        for b in range(batch_size):
            for oc in range(out_channel):
                for i in range(kernel_size):
                    y = x.data[b, :, i:i + upsampled_grad[0].shape[1]]
                    g = upsampled_grad[b, oc, :].reshape(1, -1)
                    dW[oc, :, i] += np.sum(y * g, axis=1)
        db = np.sum(np.sum(grad, axis=0), axis=1)
        return tensor.Tensor(dX), tensor.Tensor(dW), tensor.Tensor(db)


class Sigmoid(Function):
    @staticmethod
    def forward(ctx, a):
        b_data = np.divide(1.0, np.add(1.0, np.exp(-a.data)))
        ctx.out = b_data[:]
        b = tensor.Tensor(b_data, requires_grad=a.requires_grad)
        b.is_leaf = not b.requires_grad
        return b

    @staticmethod
    def backward(ctx, grad_output):
        b = ctx.out
        grad = grad_output.data * b * (1 - b)
        return tensor.Tensor(grad)


class Tanh(Function):
    @staticmethod
    def forward(ctx, a):
        b = tensor.Tensor(np.tanh(a.data), requires_grad=a.requires_grad)
        ctx.out = b.data[:]
        b.is_leaf = not b.requires_grad
        return b

    @staticmethod
    def backward(ctx, grad_output):
        out = ctx.out
        grad = grad_output.data * (1 - out ** 2)
        return tensor.Tensor(grad)


def cross_entropy(predicted, target):
    """Calculates Cross Entropy Loss (XELoss) between logits and true labels.
    For MNIST, don't call this function directly; use nn.loss.CrossEntropyLoss instead.

    Args:
        predicted (Tensor): (batch_size, num_classes) logits
        target (Tensor): (batch_size,) true labels

    Returns:
        Tensor: the loss as a float, in a tensor of shape ()
    """
    batch_size, num_classes = predicted.shape

    # Tip: You can implement XELoss all here, without creating a new subclass of Function.
    #      However, if you'd prefer to implement a Function subclass you're free to.
    #      Just be sure that nn.loss.CrossEntropyLoss calls it properly.

    # Tip 2: Remember to divide the loss by batch_size; this is equivalent
    #        to reduction='mean' in PyTorch's nn.CrossEntropyLoss
    target = to_one_hot(target, num_classes)
    exp_x = predicted.exp()
    alpha = tensor.Tensor(np.amin(exp_x.data))
    log_y_hat = predicted - (alpha + (predicted - alpha).exp().sum(axis=1, keepdims=True).log())
    loss = (tensor.Tensor(np.array([-1 / batch_size])) * target * log_y_hat).sum(axis=None, keepdims=False)
    return loss


def to_one_hot(arr, num_classes):
    """(Freebie) Converts a tensor of classes to one-hot, useful in XELoss

    Example:
    >>> to_one_hot(Tensor(np.array([1, 2, 0, 0])), 3)
    [[0, 1, 0],
     [0, 0, 1],
     [1, 0, 0],
     [1, 0, 0]]
     
    Args:
        arr (Tensor): Condensed tensor of label indices
        num_classes (int): Number of possible classes in dataset
                           For instance, MNIST would have `num_classes==10`
    Returns:
        Tensor: one-hot tensor
    """
    arr = arr.data.astype(int)
    a = np.zeros((arr.shape[0], num_classes))
    a[np.arange(len(a)), arr] = 1
    return tensor.Tensor(a, requires_grad=True)
