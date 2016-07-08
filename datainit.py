# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 21:11:11 2016

@author: Michael
"""

import pandas as pd
import numpy as np
import os

def get(hidden_layers, output_layer):
    # load training data
    train = pd.read_csv(os.getcwd() + '\\train.csv')
    test = pd.read_csv(os.getcwd() + '\\test.csv')
    
    # Set X and y values
    X = np.matrix(train.ix[:40000,'pixel0':])
    y = np.matrix(train.ix[:40000,'label'])
    X_cv = np.matrix(train.ix[40000:,'pixel0':])
    y_cv = np.matrix(train.ix[40000:, 'label'])
    X_test = np.matrix(test.ix[:,'pixel0':])
    
    # set layer values
    input_layer = np.size(X, 1)
    hidden_nodes = hidden_layers + 23
    
    # create weights info
    layers = []
    layers.append(input_layer)
    for i in range(hidden_layers):
        layers.append(hidden_nodes)

    layers.append(output_layer)
    
    weights_info = [(layers[i+1], layers[i]) for i in range(len(layers) - 1)]
    weights = [(layers[i+1], layers[i] + 1) for i in range(len(layers) - 1)]
    
    # create thetas
    orig_thetas = [] 
    for i in range(len(weights_info)):
        orig_thetas.append(np.matrix(np.random.randn(*weights_info[i])))
        orig_thetas[i] = orig_thetas[i] / np.sqrt(X.shape[1])
        orig_thetas[i] = np.insert(orig_thetas[i], 0, np.random.randn(), 1)
   
    return X, y, orig_thetas, weights, test, X_cv, y_cv, X_test