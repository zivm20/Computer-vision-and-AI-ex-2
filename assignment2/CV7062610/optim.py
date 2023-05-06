import numpy as np

"""
This file implements various first-order update rules that are commonly used
for training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning
    rate, momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not
perform well; however the default values of the other hyperparameters should
work well for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""

def adam(w, dw, config=None):
    """
    adam gradient decent.
    
    config format:
    - learning_rate: Scalar learning rate.
    - epsilon: Scalar epsilon
    - b1: Exponential  decay rate 1, should be 0 < b1 < 1
    - b2: Moving average param 2, should be 0 < b2 < 1
    """
    if config is None:
        config = {}
    config.setdefault("b1", 0.9)
    config.setdefault("b2", 0.999)
    config.setdefault("m",0)
    config.setdefault("v",0)
    
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("epsilon", 1e-12)
    
    lr = config["learning_rate"]
    eps = config["epsilon"]
    m = config["m"]
    v = config["v"]
    b1 = config["b1"]
    b2 = config["b2"]
    


    m = b1 * m + (1 - b1) * dw
    v = b2 * v + (1 - b2) * (dw**2)

    #bias correct m and v
    m_hat = m / (1 - b1)
    v_hat = v / (1 - b2)

    #update weights
    next_w = w - lr * m_hat / (np.sqrt(v_hat) + eps)

    config["m"] = m
    config["v"] = v
    config["b1"] = b1*b1
    config["b2"] = b2*b2


    return next_w,config


    

def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    lr = config["learning_rate"]
    ###########################################################################
    # TODO: Implement the SGD update formula.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    next_w = w - lr*dw

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config