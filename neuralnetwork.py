#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 15:49:30 2018

@author: ben
"""
import numpy as np

#from sklearn.datasets import load_iris
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.model_selection import train_test_split

from activation_function import RELU, sigmoid, linear
from layer import Layer, layertype

class neuralnetwork():
    """
    define a class for a forward-feed artificial neural network
    """
    
    def __init__(self, activation_function, architecture):
        """
        Inputs:
            architecture = list containing the number of neurons in each layer
        """
        self.activation_function = activation_function
        self.layers = []
        # input layer
        nn = architecture[0]
        f = self.activation_function[0]
        self.layers.append(Layer(nn, nn, f, layertype.INPUT))
        nn_prev = nn
        # inside layers
        for nn, f in zip(architecture[1:-1], self.activation_function[1:-1]):
            self.layers.append(Layer(nn, nn_prev, f, layertype.INSIDE))
            nn_prev = nn
        # output layer
        nn = architecture[-1]
        f = self.activation_function[-1]
        self.layers.append(Layer(nn, nn_prev, f, layertype.OUTPUT))
    
    
    def initialize_weights(self):
        for ll in self.layers:
            ll.initialize_weights()
    
    def load_data(self, x, y):
        self.x = x
        self.y = y
    

    def compute_fwd(self, x):
        a = x
        self.A = [a]
        for ll in self.layers:
            a = ll.compute_forward(a)
            self.A.append(a)
        return a
    
    def compute_bprop(self, misfit):
        d = misfit
        for ll, a in zip(self.layers[:0:-1], self.A[-2::-1]):
            d = ll.compute_dadb(a) * d
            ll.grad_parameters[:,:1] += d
            ll.grad_parameters[:,1:] += d.dot(a.T)
            d = ll.W().T.dot(d)
    
    def compute_loss(self):
        cost = 0.0
        for xx, yy in zip(self.x, self.y):
            cost += 0.5*np.sum((self.compute_fwd(xx.reshape((-1,1)))-yy.reshape((-1,1)))**2)
        return cost
    
    def compute_derivative(self):
        for ll in self.layers:
            ll.reset_grad()
        cost = 0.0
        for xx, yy in zip(self.x, self.y):
            misfit = self.compute_fwd(xx.reshape((-1,1)))-yy.reshape((-1,1))
            cost += 0.5*np.sum((misfit)**2)
            self.compute_bprop(misfit)
        return cost

if __name__ == "__main__":   
    """
    # load and prepare the iris dataset
    iris = load_iris()
    x = iris.data
    y_ = iris.target.reshape(-1, 1) # Convert data to a single column
    # One Hot encode the class labels
    encoder = OneHotEncoder(sparse=False)
    y = encoder.fit_transform(y_)
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20, stratify=y)
    """
    
    # assemble ANN
    architecture = [2,4,10,4,2]
    #f = [RELU()]*len(architecture)
    f = [linear(), RELU(), RELU(), sigmoid(), sigmoid()]
    ANN = neuralnetwork(f, architecture)
    for ll in ANN.layers:
        print(ll.layer_type, ll.nb_neurons_prev(), ll.nb_neurons())
    ANN.initialize_weights()
    #ANN.load_data(train_x[:1,:], train_y[:1,:])
    ANN.load_data(np.array([[1.0, 1.0],[-1.0,1.0],[-1.0,-1.0]]), 
                  np.array([[1.0, 0.0],[0.0,1.0],[1.0,0.0]]))
    cost = ANN.compute_loss()
    print('cost=',cost)
    cost2 = ANN.compute_derivative()
    print('cost2=',cost2)
    
    # check derivative with finite-difference
    EPS = [1e-2, 1e-3, 1e-4]
    layernb = 3
    nbsh = ANN.layers[layernb].parameters.shape
    H = np.random.randn(np.prod(nbsh)).reshape(nbsh)
    param = ANN.layers[layernb].parameters.copy()
    cost0 = ANN.compute_loss()
    grad_direc = (ANN.layers[layernb].grad_parameters*H).sum()
    print('grad_direc=%.4e' % grad_direc)
    for eps in EPS:
        ANN.layers[layernb].parameters = param + eps*H
        cost1 = ANN.compute_loss()
        fd = (cost1-cost0)/eps
        print('eps=%.2e, fd=%.4e, err=%.2e' % (eps, fd, np.abs(fd-grad_direc)/np.abs(grad_direc)))