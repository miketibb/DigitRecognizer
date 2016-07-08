# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 20:55:38 2016

@author: Michael
"""

import numpy as np
from datainit import get
from scipy.optimize import minimize
from functions import cost, gradient, pack_thetas, forward_propagate

class Workstation(object):
    def __init__(self, X=None, y=None, lamb=5.0, hidden_layers=2, output_layer=10):
        X, y, orig_thetas, weights, test, X_cv, y_cv, X_test = get(hidden_layers, output_layer)
        self.X = X
        self.y = y
        self.orig_thetas = orig_thetas
        self.test = test
        self.all_thetas = 0
        self.lamb = lamb
        self.weights = weights
        self.X_cv = X_cv
        self.y_cv = y_cv
        self.X_test = X_test
        print('Workstation initialized')
        
        
    def find_min(self):
        self.all_thetas = pack_thetas(self.orig_thetas)
        fmin = minimize(fun=cost, x0=self.all_thetas, args=(self.weights, self.X, self.y, self.lamb), method='TNC', jac=gradient,
                        options = {'maxiter':500})
        self.all_thetas = np.matrix(fmin.x)
        self.all_thetas = self.all_thetas.T
        
        
    def predict_all(self):
        print('Predicting...')
        accuracy = forward_propagate(self.all_thetas, self.weights, self.X_cv, self.y_cv)
        print(accuracy)
        
        
if __name__ == '__main__':
    ws = Workstation()
    ws.find_min()
    ws.predict_all()