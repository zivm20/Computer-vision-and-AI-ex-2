# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

pass


# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
from CV7062610.layers import *
from CV7062610.fast_layers import *


def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward_fast(x, w, b,retCupy=True)
    out, relu_cache = relu_forward_fast(a,retCupy=False)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward_fast(dout, relu_cache,retCupy=True)
    dx, dw, db = affine_backward_fast(da, fc_cache,retCupy=False)
    return dx, dw, db


def conv_relu_forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param,retCupy=True)
    out, relu_cache = relu_forward_fast(a,retCupy=False)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward_fast(dout, relu_cache,retCupy=True)
    dx, dw, db = conv_backward_fast(da, conv_cache,retCupy=False)
    return dx, dw, db

def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param,retCupy=True)
    s, relu_cache = relu_forward_fast(a,retCupy=True)
    out, pool_cache = max_pool_forward_fast(s, pool_param,retCupy=False)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache,retCupy=True)
    da = relu_backward_fast(ds, relu_cache,retCupy=True)
    dx, dw, db = conv_backward_fast(da, conv_cache,retCupy=False)
    return dx, dw, db







