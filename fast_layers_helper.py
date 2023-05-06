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

try:
    import cupy as cp
    asnumpy = cp.asnumpy
    usingNumpy = False
    print("running on GPU mode")
except ImportError:
    import numpy as cp
    asnumpy = cp.array
    usingNumpy = True

from numpy import ndarray





def cov(x_pad, w, b,stride,out):
    for o_H in range(out.shape[2]):
        for o_W in range(out.shape[3]): 
            out[:,:,o_H,o_W] = cp.einsum('nchw, fchw -> nf',x_pad[:,:,stride*o_H:stride*o_H+w.shape[2],stride*o_W:stride*o_W+w.shape[3]],w)
    for F in range(out.shape[1]):
        out[:,F,:,:] = out[:,F,:,:] + b[F] 
    return out
        
def conv_forward_fast(x, w, b, conv_param,force_numpy=False,force_cupy=False):
    if force_numpy and not usingNumpy:
        return conv_forward_fast_(asnumpy(x), asnumpy(w), asnumpy(b), conv_param)
    elif force_cupy and not usingNumpy:
        return conv_forward_fast_(cp.array(x), cp.array(w), cp.array(b), conv_param)
    else:
        return conv_forward_fast_(x, w, b, conv_param)

def conv_forward_fast_(x, w, b, conv_param):
    
    retNumpy = isinstance(x,ndarray)
    if retNumpy and not usingNumpy:
        x = cp.array(x)
        w = cp.array(w)
        b = cp.array(b)
    
    
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param["stride"], conv_param["pad"]

    # Check dimensions
    # assert (W + 2 * pad - WW) % stride == 0, 'width does not work'
    # assert (H + 2 * pad - HH) % stride == 0, 'height does not work'

    # Pad the input
    p = pad
    x_padded = cp.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")

    # Figure out output dimensions
    H += 2 * pad
    W += 2 * pad
    out_h = (H - HH) // stride + 1
    out_w = (W - WW) // stride + 1
    
    out = cp.zeros((N,F,out_h,out_w))
    out = cov(x_padded, w, b, stride,out)
    
    if (retNumpy and not usingNumpy):
        out = asnumpy(out)
        x_padded = asnumpy(x_padded)
        w = asnumpy(w)
        b = asnumpy(b)
    
    cache = (x_padded, w, b, conv_param)
    return out, cache


def conv_backward_fast(dout,cache,force_numpy=False,force_cupy=False):
    if force_numpy and not usingNumpy:
        return conv_backward_fast_(asnumpy(dout),cache)
    elif force_cupy and not usingNumpy:
        return conv_backward_fast_(cp.array(dout),cache)
    else:
        return conv_backward_fast_(dout,cache)

def conv_backward_fast_(dout,cache):
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
    retNumpy = isinstance(dout,ndarray)
    if retNumpy and not usingNumpy:
        x_pad = cp.array(x_pad)
        w = cp.array(w)
        b = cp.array(b)
        dout = cp.array(dout) 
    
    
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

    if (retNumpy and not usingNumpy):
        dx = asnumpy(dx)
        dw = asnumpy(dw)
        db = asnumpy(db)

    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


    




def max_pool_forward_fast(x, pool_param,force_numpy=False,force_cupy=False):
    if force_numpy and not usingNumpy:
        return max_pool_forward_fast_(asnumpy(x), pool_param)
    elif force_cupy and not usingNumpy:
        return max_pool_forward_fast_(cp.array(x), pool_param)
    else:
        return max_pool_forward_fast_(x, pool_param)

def max_pool_forward_fast_(x, pool_param):
    """
    A fast implementation of the forward pass for a max pooling layer.

    This chooses between the reshape method and the im2col method. If the pooling
    regions are square and tile the input image, then we can use the reshape
    method which is very fast. Otherwise we fall back on the im2col method, which
    is not much faster than the naive method.
    """
    retNumpy = isinstance(x,ndarray)
    if retNumpy and not usingNumpy:
        x = cp.array(x)
    
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param["pool_height"], pool_param["pool_width"]
    stride = pool_param["stride"]
    out = cp.zeros((N,C,1+(H-pool_height)//stride,1+(W-pool_width)//stride))
    
    
    for o_H in range(out.shape[2]):
        for o_W in range(out.shape[3]):
            out[:,:,o_H,o_W] = cp.max(cp.max(x[:,:,o_H*stride:o_H*stride + pool_height, o_W*stride:o_W*stride + pool_width],axis=2),axis=2)
            
            
    if (retNumpy and not usingNumpy):
        out = cp.asnumpy(out)
        x = cp.asnumpy(x)


    return out, (x,pool_param)





def max_pool_backward_fast(dout, cache,force_numpy=False,force_cupy=False):
    if force_numpy and not usingNumpy:
        return max_pool_backward_fast_(asnumpy(dout), cache)
    elif force_cupy and not usingNumpy:
        return max_pool_backward_fast_(cp.array(dout), cache)
    else:
        return max_pool_backward_fast_(dout, cache)

def max_pool_backward_fast_(dout, cache):
    """
    A fast implementation of the backward pass for a max pooling layer.

    This switches between the reshape method an the im2col method depending on
    which method was used to generate the cache.
    """
    x, pool_param = cache
    retNumpy = isinstance(dout,ndarray)
    if retNumpy and not usingNumpy:
        x = cp.array(x)
        dout = cp.array(dout)
    
    
    dx = cp.zeros_like(x)
    pH = pool_param['pool_height']
    pW = pool_param['pool_width']
    stride = pool_param['stride']
    
    for H in range(dout.shape[2]):
        for W in range(dout.shape[3]):
            pool = x[:,:,H*stride:H*stride + pH, W*stride:W*stride + pW]
            
            mask = (pool == cp.max(cp.max(pool, axis=3, keepdims=True),axis=2,keepdims=True))
            dx[:,:,H*stride:H*stride+pH,W*stride:W*stride+pW] += mask*dout[:,:,H,W][:,:,cp.newaxis,cp.newaxis]
    
    if (retNumpy and not usingNumpy):
        dx = asnumpy(dx)

    return dx









def affine_forward_fast(x, w, b,force_numpy=False,force_cupy=False):
    if force_numpy and not usingNumpy:
        return affine_forward_fast_(asnumpy(x), asnumpy(w), asnumpy(b))
    elif force_cupy and not usingNumpy:
        return affine_forward_fast_(cp.array(x), cp.array(w), cp.array(b))
    else:
        return affine_forward_fast_(x, w, b)

def affine_forward_fast_(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    retNumpy = isinstance(x,ndarray)
    if retNumpy and not usingNumpy:
        x = cp.array(x)
        w = cp.array(w)
        b = cp.array(b)
        


    #reshape x into the (N,D) and then compute the output using matrix multiplication    
    out = (cp.reshape(x,(x.shape[0],w.shape[0])) @ w) + b
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    if (retNumpy and not usingNumpy):
        x = asnumpy(x)
        w = asnumpy(w)
        b = asnumpy(b)
        out = asnumpy

    cache = (x, w, b)
    return out, cache

#@njit(parallel=True)
def affine_backward_fast(dout, cache,force_numpy=False,force_cupy=False):
    if force_numpy and not usingNumpy:
        return affine_backward_fast_(asnumpy(dout), cache)
    elif force_cupy and not usingNumpy:
        return affine_backward_fast_(cp.array(dout), cache)
    else:
        return affine_backward_fast_(dout, cache)

def affine_backward_fast_(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache

    retNumpy = isinstance(dout,ndarray)
    if retNumpy and not usingNumpy:
        x = cp.array(x)
        w = cp.array(w)
        
        dout = cp.array(x)


    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # out = f(x,w,b) = x@w + b -> z = x@w -> out = z + b
    # dout/db = 1
    db = cp.ones(dout.shape[0]) @ dout#/dout.shape[0]

    # dout/dw = dz/dw * dout/dz, dout/dz = 1 
    # dz/dw = x
    dw = cp.reshape(x,(x.shape[0],w.shape[0])).T@dout#/dout.shape[0]

    # dout/dx = dz/dx * dout/dz, dout/dz = 1 
    # dz/dx = w
    dx = cp.reshape(dout@w.T,x.shape)#/dout.shape[1]
    
    if (retNumpy and not usingNumpy):
        dx = asnumpy(dx)
        dw = asnumpy(dw)
        db = asnumpy(db)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db

#@njit(parallel=True)
def relu_forward_fast(x,force_numpy=False,force_cupy=False):
    if force_numpy and not usingNumpy:
        return relu_forward_fast_(asnumpy(x))
    elif force_cupy and not usingNumpy:
        return relu_forward_fast_(cp.array(x))
    else:
        return relu_forward_fast_(x)

def relu_forward_fast_(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    
    retNumpy = isinstance(x,ndarray)
    if retNumpy and not usingNumpy:
        x = cp.array(x)

    out = cp.maximum(x,0)

    if (retNumpy and not usingNumpy):
        x = asnumpy(x)
        out = asnumpy(out)
       
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache

#@njit(parallel=True)
def relu_backward_fast(dout, cache,force_numpy=False,force_cupy=False):
    if force_numpy and not usingNumpy:
        return relu_backward_fast_(asnumpy(dout), cache)
    elif force_cupy and not usingNumpy:
        return relu_backward_fast_(cp.array(dout), cache)
    else:
        return relu_backward_fast_(dout, cache)

def relu_backward_fast_(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    retNumpy = isinstance(dout,ndarray)
    if retNumpy and not usingNumpy:
        x = cp.array(x)
        dout = cp.array(x)
    
    dx = dout * (x>=0)
    
    
    if (retNumpy and not usingNumpy):
        dx = asnumpy(dx)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx
