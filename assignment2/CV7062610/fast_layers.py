from __future__ import print_function
import numpy as np

try:
    from CV7062610.im2col_cython import col2im_cython, im2col_cython
    from CV7062610.im2col_cython import col2im_6d_cython
except ImportError:
    print("""=========== You can safely ignore the message below if you are NOT working on ConvolutionalNetworks.ipynb ===========""")
    print("\tYou will need to compile a Cython extension for a portion of this assignment.")
    print("\tThe instructions to do this will be given in a section of the notebook below.")
    print("\tThere will be an option for Colab users and another for Jupyter (local) users.")

from CV7062610.im2col import *

import cupy as cp
from numba import jit, prange



    
        

def cov(x_pad, w, b,stride,out):
    
    
    for o_H in range(out.shape[2]):
        for o_W in range(out.shape[3]): 
            out[:,:,o_H,o_W] = cp.einsum('nchw, fchw -> nf',x_pad[:,:,stride*o_H:stride*o_H+w.shape[2],stride*o_W:stride*o_W+w.shape[3]],w)
    
    for F in range(out.shape[1]):
        out[:,F,:,:] = out[:,F,:,:] + b[F] 
    return out
        
        
            
            

def conv_forward_strides(x, w, b, conv_param):
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param["stride"], conv_param["pad"]

    # Check dimensions
    # assert (W + 2 * pad - WW) % stride == 0, 'width does not work'
    # assert (H + 2 * pad - HH) % stride == 0, 'height does not work'

    # Pad the input
    p = pad
    x_padded = np.pad(cp.asnumpy(x), ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")

    # Figure out output dimensions
    H += 2 * pad
    W += 2 * pad
    out_h = (H - HH) // stride + 1
    out_w = (W - WW) // stride + 1

    # Perform an im2col operation by picking clever strides
    
    out = cp.zeros((N,F,out_h,out_w))
    out = cov(cp.array(x_padded), cp.array(w), cp.array(b), stride,out)
    out = cp.asnumpy(out)

    

    cache = (x_padded, w, b, conv_param)#, x_cols)
    return out, cache




import tensorflow as tf
import numpy as np




import numpy as np
from numba import jit, prange


def conv_backward_strides(dout,cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x_pad,w,b,conv_param = cache
    pad = conv_param["pad"]
    stride = conv_param["stride"]
    x = x_pad 
    if pad>0:
        x=x[:,:,pad:-pad,pad:-pad]

    dw = np.zeros_like(w)
    dx = np.zeros_like(x)
    db = np.zeros_like(b)

    
    for F in prange(w.shape[0]):
        for N in range(dout.shape[0]):
            
            for H in range(dout.shape[2]):
                for W in range(dout.shape[3]):
                    for C in range(w.shape[1]):
                        for HH in range( w.shape[2]):
                            for WW in range(w.shape[3]):
                            
                                dw[F,C,HH,WW] = dw[F,C,HH,WW] + dout[N,F,H,W]*x_pad[N,C,H*stride + HH,W*stride + WW]
                                
                                if HH + H*stride - pad >= 0 and HH + H*stride  - pad < x.shape[2] and WW + W*stride - pad >= 0 and WW + W*stride - pad < x.shape[3]:
                                    dx[N,C,H*stride+HH - pad,W*stride+WW - pad] = dx[N,C,H*stride+HH - pad,W*stride+WW - pad] + dout[N,F,H,W]*w[F,C, HH, WW]
                    db[F] = db[F]+dout[N,F,H,W]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


    



conv_forward_fast = conv_forward_strides
conv_backward_fast = conv_backward_strides


def max_pool_forward_fast(x, pool_param):
    """
    A fast implementation of the forward pass for a max pooling layer.

    This chooses between the reshape method and the im2col method. If the pooling
    regions are square and tile the input image, then we can use the reshape
    method which is very fast. Otherwise we fall back on the im2col method, which
    is not much faster than the naive method.
    """
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param["pool_height"], pool_param["pool_width"]
    stride = pool_param["stride"]

    same_size = pool_height == pool_width == stride
    tiles = H % pool_height == 0 and W % pool_width == 0
    if same_size and tiles:
        out, reshape_cache = max_pool_forward_reshape(x, pool_param)
        cache = ("reshape", reshape_cache)
    else:
        out, im2col_cache = max_pool_forward_im2col(x, pool_param)
        cache = ("im2col", im2col_cache)
    return out, cache


def max_pool_backward_fast(dout, cache):
    """
    A fast implementation of the backward pass for a max pooling layer.

    This switches between the reshape method an the im2col method depending on
    which method was used to generate the cache.
    """
    method, real_cache = cache
    if method == "reshape":
        return max_pool_backward_reshape(dout, real_cache)
    elif method == "im2col":
        return max_pool_backward_im2col(dout, real_cache)
    else:
        raise ValueError('Unrecognized method "%s"' % method)

