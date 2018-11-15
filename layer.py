#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 22:56:07 2018

@author: ben
"""

import numpy as np
from enum import Enum
from activation_function import RELU
#from scipy.linalg.blas import get_blas_funcs

class layertype(Enum):
    """ 
    Define the tyep of layer
    """
    INPUT = 0
    INSIDE = 1
    OUTPUT = -1

class Layer():
    """ 
    Define a class for a single layer of an artificial neuron network
    weights W and shift b are stored in the same matrix as self.parameters=[b, W]
    """

    def __init__(self, nb_neurons, nb_inputs, activation_function, layer_type):
        """
        Inputs:
            nb_neurons: number of neurons in the layer
            nb_inputs: number of neurons in the previous layer, or
                number of input variables (for input layer)
            activation_function: object derived from class activation_function
            layer_type: object layertype to define the type of layer
        """
        self.parameters = np.zeros([nb_neurons, nb_inputs+1])
        self.f = activation_function
        self.layer_type = layer_type
        
        #self.dgemv = get_blas_funcs('gemv')
    
    def initialize_weights(self):
        self.parameters = np.random.randn(self.nb_neurons()*(self.nb_neurons_prev()+1))\
        .reshape(self.parameters.shape)
        
    def nb_neurons(self):
        return self.parameters.shape[0]
    
    def nb_neurons_prev(self):
        return self.parameters.shape[1] - 1
        
    def compute_forward(self, a):
        x = np.concatenate((np.ones([1,1]), a), axis=0)
        return self.parameters.dot(x)
        #return dgemv(1.0, self.parameters, x) #slower on my machine
    
    #def compute_forward(self, a):
    #    b = self.parameters[:,:1]
    #    W = self.parameters[:,1:]
    #    return self.dgemv(1.0, W, a, 1.0, b)
    
    #def backpropagate(self, d):


if __name__ == "__main__":
    # run unit tests
    lay = Layer(200, 250, RELU(), layertype.INSIDE)
    lay.initialize_weights()
    x = np.random.randn(lay.nb_neurons_prev()).reshape((-1,1))
    y = lay.compute_forward(x)
    print(y.shape)