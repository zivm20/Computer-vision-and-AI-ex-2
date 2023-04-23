from builtins import range
import numpy as np
from numba import njit

@njit(parallel=True)
def affine_forward(x, w, b):
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

    #reshape x into the (N,D) and then compute the output using matrix multiplication
    out = (np.reshape(x,(x.shape[0],w.shape[0])) @ w) + b
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache

@njit(parallel=True)
def affine_backward(dout, cache):
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
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # out = f(x,w,b) = x@w + b -> z = x@w -> out = z + b
    # dout/db = 1
    db = np.ones(dout.shape[0]) @ dout#/dout.shape[0]

    # dout/dw = dz/dw * dout/dz, dout/dz = 1 
    # dz/dw = x
    dw = np.reshape(x,(x.shape[0],w.shape[0])).T@dout#/dout.shape[0]

    # dout/dx = dz/dx * dout/dz, dout/dz = 1 
    # dz/dx = w
    dx = np.reshape(dout@w.T,x.shape)#/dout.shape[1]
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db

@njit(parallel=True)
def relu_forward(x):
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

    out = np.maximum(x,0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache

@njit(parallel=True)
def relu_backward(dout, cache):
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
    
    dx = dout * (x>=0)
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


@njit(parallel=True)
def sum_axis0(x):
    return np.ones(x.shape[0])@x

@njit(parallel=True)
def bachnorm_foward_train_(x:np.ndarray,gamma,beta,train_mean,train_var,eps,momentum):
    curr_mean = sum_axis0(x)/x.shape[0]
    curr_var = sum_axis0(((x-curr_mean)**2))/x.shape[0]

    train_mean = momentum * train_mean + (1-momentum)*curr_mean
    train_var = momentum * train_var + (1-momentum)*curr_var
    
    std = np.sqrt(curr_var+eps)
    x_center = x-curr_mean
    x_norm = x_center/std
    out = gamma*x_norm+beta

    cache = (x_norm,x_center,std,gamma)

    return out, cache, train_mean, train_var



@njit(parallel=True)
def bachnorm_foward_test_(x:np.ndarray,gamma,beta,train_mean,train_var,eps,momentum):
    std = np.sqrt(train_var+eps)
    x_center = x-train_mean
    x_norm = x_center/std
    out = gamma*x_norm+beta

    cache = (x_norm,x_center,std,gamma)

    return out, cache

def bachnorm_foward(x:np.ndarray,gamma,beta,bn_params):
    mode=bn_params.get('mode',None)
    train_mean=bn_params.get('train_mean',np.zeros(x.shape[1]))
    train_var=bn_params.get('train_var',np.zeros(x.shape[1]))
    eps=bn_params.get('eps',1e-5)
    momentum=bn_params.get('momentum',0.9)
    if mode == 'train':
        out, cache, train_mean, train_var = bachnorm_foward_train_(x,gamma,beta,train_mean,train_var,eps,momentum)
        bn_params['train_mean'] = train_mean
        bn_params['train_var'] = train_var
    elif mode == 'check':
        curr_mean = sum_axis0(x)/x.shape[0]
        curr_var = sum_axis0(((x-curr_mean)**2))/x.shape[0]

        train_mean = curr_mean
        train_var = curr_var

        std = np.sqrt(train_var+eps)
        x_center = x-train_mean
        x_norm = x_center/std
        out = gamma*x_norm+beta

        cache = (x_norm,x_center,std,gamma)
    else:
        out, cache = bachnorm_foward_test_(x,gamma,beta,train_mean,train_var,eps,momentum)
        
    return out, cache

@njit(parallel=True)
def bachnorm_backward(dout,cache):
    x_norm, x_centered, std, gamma = cache
    dgamma = sum_axis0(dout*x_norm)
    dbeta = sum_axis0(dout)
    dx_norm = dout*gamma

    dx = (1/dout.shape[0]) * (1/std)*(dout.shape[0]*dx_norm - sum_axis0(dx_norm) - x_norm*sum_axis0(dx_norm*x_norm))
    return dx, dgamma, dbeta

def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x_pad, w, b, conv_param)
    """
    out = None
    x_pad = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    pad = conv_param["pad"]
    stride = conv_param["stride"]
    out = np.zeros((x.shape[0],w.shape[0],1+(x.shape[2]+2*pad-w.shape[2])//stride,1+(x.shape[3]+2*pad-w.shape[3])//stride))
    x_pad = np.copy(x)
    x_pad = np.pad(x_pad,((0,0),(0,0),(pad,pad),(pad,pad)),'constant', constant_values=0)
    for n in range(out.shape[0]):
        for f in range(out.shape[1]):
            for o_H in range(out.shape[2]):
                for o_W in range(out.shape[3]): 
                    for c in range(x_pad.shape[1]): 
                        for hh in range(0,w.shape[2]):
                            for ww in range(0,w.shape[3]):
                                
                                #o_W*stride+(w_W-1)//2 
                                out[n,f,o_H,o_W] += x_pad[n,c,stride*o_H+hh,stride*o_W+ww] * w[f,c,hh,ww]
                    out[n,f,o_H,o_W]+= b[f]
                        

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x_pad, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
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

    
    
    
    for F in range(w.shape[0]):
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


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    pH = pool_param["pool_height"]
    pW = pool_param["pool_width"]
    stride = pool_param["stride"]
    out = np.zeros((x.shape[0],x.shape[1],1+(x.shape[2]-pH)//stride,1+(x.shape[3]-pW)//stride))

    for n in range(out.shape[0]):
        for c in range(out.shape[1]):
            for h in range(out.shape[2]):
                for w in range(out.shape[3]):
                    out[n,c,h,w] = np.max(x[n,c,h*stride:h*stride+pH,w*stride:w*stride+pW])

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    x, pool_param = cache
    dx = np.zeros_like(x)
    pH = pool_param['pool_height']
    pW = pool_param['pool_width']
    stride = pool_param['stride']
    
    for n in range(dout.shape[0]):
        for c in range(dout.shape[1]):
            for h in range(dout.shape[2]):
                for w in range(dout.shape[3]):
                    pool = x[n,c,h*stride:h*stride+pH,w*stride:w*stride+pW]
                    mask = pool==np.max(pool)
                    
                    dx[n,c,h*stride:h*stride+pH,w*stride:w*stride+pW] += mask * dout[n,c,h,w]
                    
    """
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    dx2 = np.zeros((N, C, H, W))
    H_out = 1 + (H - pool_height) / stride
    W_out = 1 + (W - pool_width) / stride

    for i in range(0, N):
        x_data = x[i]

        xx, yy = -1, -1
        for j in range(0, H-pool_height+1, stride):
            yy += 1
            for k in range(0, W-pool_width+1, stride):
                xx += 1
                x_rf = x_data[:, j:j+pool_height, k:k+pool_width]
                for l in range(0, C):
                    x_pool = x_rf[l]
                    mask = x_pool == np.max(x_pool)
                    dx2[i, l, j:j+pool_height, k:k+pool_width] += dout[i, l, yy, xx] * mask

            xx = -1
    
    print(dx-dx2)
    """
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx

@njit(parallel=True)
def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    """
    
    target = np.eye(x.shape[1])[y]
    s = np.exp(x)
    probs = (s.T*(1/np.sum(s,axis=1))).T
    
    
    
    loss = -np.sum(np.log(probs)*target) / x.shape[0]
    dx = (probs.copy()-target)/x.shape[0]
   
    return loss, dx
