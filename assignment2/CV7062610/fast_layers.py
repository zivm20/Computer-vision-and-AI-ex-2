from __future__ import print_function


try:
    from CV7062610.im2col_cython import col2im_cython, im2col_cython
    from CV7062610.im2col_cython import col2im_6d_cython
except ImportError:
    print("""=========== You can safely ignore the message below if you are NOT working on ConvolutionalNetworks.ipynb ===========""")
    print("\tYou will need to compile a Cython extension for a portion of this assignment.")
    print("\tThe instructions to do this will be given in a section of the notebook below.")
    print("\tThere will be an option for Colab users and another for Jupyter (local) users.")

from CV7062610.im2col import *

import numpy as cp
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
    x_padded = cp.pad(cp.array(x), ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")

    # Figure out output dimensions
    H += 2 * pad
    W += 2 * pad
    out_h = (H - HH) // stride + 1
    out_w = (W - WW) // stride + 1
    
    out = cp.zeros((N,F,out_h,out_w))
    out = cov(cp.array(x_padded), cp.array(w), cp.array(b), stride,out)
    
    #for using cupy
    #out = cp.asnumpy(out)

    

    cache = (x_padded, w, b, conv_param)#, x_cols)
    return out, cache


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

    x_pad,w,b,conv_param =cache
    x_pad = cp.array(x_pad)
    w = cp.array(w)
    b = cp.array(b)
    
    pad = conv_param["pad"]
    stride = conv_param["stride"]
    
    x = x_pad 
    if pad>0:
        x=x[:,:,pad:-pad,pad:-pad]

    dw = cp.zeros_like(w)
    dx = cp.zeros_like(x)
    
    for HH in range(dw.shape[2]):
        for WW in range(dw.shape[3]): 
            dw[:,:,HH,WW] = cp.einsum('nchw, nfhw -> fc',x_pad[:,:,HH:HH+dout.shape[2]*stride:stride,WW:WW+dout.shape[3]*stride:stride],dout)

    
    dx_pad = cp.pad(dx, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    
    for H in range(0,dout.shape[2]):
        for W in range(0,dout.shape[3]):
            dx_pad[:,:,H*stride:H*stride+w.shape[2], W*stride:W*stride+w.shape[3]] += cp.einsum('nf, fchw -> nchw',dout[:,:,H,W],w)

    if pad>0:
        dx=dx_pad[:,:,pad:-pad,pad:-pad]
    else:
        dx = dx_pad
    
    db = cp.sum(dout, axis=(0, 2, 3))

    
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
    out = cp.zeros((N,C,1+(H-pool_height)//stride,1+(W-pool_width)//stride))
    x = cp.array(x)
    
    for o_H in range(out.shape[2]):
        for o_W in range(out.shape[3]):
            out[:,:,o_H,o_W] = cp.max(cp.max(x[:,:,o_H*stride:o_H*stride + pool_height, o_W*stride:o_W*stride + pool_width],axis=2),axis=2)
            
            

    
    #out = cp.asnumpy(out)
    #x = cp.asnumpy(x)
    return out, (x,pool_param)


def max_pool_backward_fast(dout, cache):
    """
    A fast implementation of the backward pass for a max pooling layer.

    This switches between the reshape method an the im2col method depending on
    which method was used to generate the cache.
    """
    x, pool_param = cache
    x = cp.array(x)
    dx = cp.zeros_like(x)
    pH = pool_param['pool_height']
    pW = pool_param['pool_width']
    stride = pool_param['stride']
    
    for H in range(dout.shape[2]):
        for W in range(dout.shape[3]):
            pool = x[:,:,H*stride:H*stride + pH, W*stride:W*stride + pW]
                    
            mask = (pool == cp.max(cp.max(pool, axis=3, keepdims=True),axis=2,keepdims=True))
            dx[:,:,H*stride:H*stride+pH,W*stride:W*stride+pW]+= mask*dout[:,:,H,W][:,:,cp.newaxis,cp.newaxis]

    return dx

