from builtins import object
import numpy as np

from CV7062610.layers import *
from CV7062610.fast_layers import *
from CV7062610.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.layers = []
        self.layers.append({"type":"conv","filter_num":num_filters,"H":filter_size,"W":filter_size,"scale":weight_scale,"pad":(filter_size - 1) // 2,"stride":1})
        self.layers.append({"type":"relu"})
        self.layers.append({"type":"max_pool","filter_num":num_filters,"H":filter_size,"W":filter_size,"scale":weight_scale,"pad":(filter_size - 1) // 2,"stride":1})
        self.layers.append({"type":"affine","scale":weight_scale,"n_classes":hidden_dim})
        self.layers.append({"type":"relu"})
        self.layers.append({"type":"affine","scale":weight_scale,"n_classes":num_classes})
        self.layers.append({"type":"softmax"})

        
        outShape = np.array([input_dim[0],input_dim[1],input_dim[2]])
        

        pNum = 0
        for layer in self.layers:
            layer.setdefault("pad", 0)
            layer.setdefault("H",1)
            layer.setdefault("W",1)
            layer.setdefault("stride",1)
            layer.setdefault("filter_num",1)
            layer.setdefault("scale",weight_scale)
            layer.setdefault("n_classes",1)
            
            pad = layer["pad"]
            H = layer["H"]
            W = layer["W"]
            stride = layer["stride"]
            filter_num = layer["filter_num"]
            scale = layer["scale"]
            n_classes = layer["n_classes"]
            
            if layer["type"] == "conv" and len(outShape)==3:
                self.params['W'+str(pNum)] = np.random.normal(0,scale,(filter_num,outShape[0],H,W))
                self.params['b'+str(pNum)] = np.zeros(filter_num)
                pNum+=1
            elif layer["type"] == "affine":
                self.params['W'+str(pNum)] = np.random.normal(0,scale,(np.prod(outShape),n_classes))
                self.params['b'+str(pNum)] = np.zeros(n_classes)
                pNum+=1
                    
            if layer["tyoe"] == "conv" or layer["type"]=="max_pool":
                outShape[2] = (outShape[2] + 2*pad - H) // stride + 1
                outShape[3] = (outShape[3] + 2*pad - W) // stride + 1
            elif layer["type"] == "affine":
                outShape[0] = n_classes
                outShape[1] = 1
                outShape[2] = 1


        
    

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)