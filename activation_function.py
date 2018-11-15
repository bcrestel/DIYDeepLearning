#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 23:01:57 2018

@author: ben
"""
import numpy as np
from abc import ABC, abstractclassmethod

class activation_function(ABC):
    """
    Abstract base class for the activation function 
    of the layer of an artificial neural network
    Used to define an interface
    """
    
    @abstractclassmethod
    def evaluate(self, x):
        """
        Evaluate function at x
        """
        pass
    
    def evaluate_firstderivative(self, x):
        """
        Evaluate derivative of function at x
        """
        raise NotImplementedError


class linear(activation_function):
    
    def evaluate(self, x):
        return x
    
    def evaluate_firstderivative(self, x):
        return 1.0
    

class RELU(activation_function):
    """
    Implements a classical REctified Linear Unit
    """
    
    def evaluate(self, x):
        """
        About 10 times faster than vectorizing max(0,x)
        """
        return (x>0)*x
    
    def evaluate_firstderivative(self, x):
        return np.array(x>0, dtype=float)