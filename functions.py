# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 22:57:29 2016

@author: Michael
"""

from sklearn.preprocessing import OneHotEncoder
import numpy as np

def cost(all_thetas, weights, X, y, lamb):
    thetas = unpack_thetas(all_thetas, weights)
    
    # add column of 1's
    X = X/255
    a1 = np.insert(X, 0, 1, 1)
    
    # create a binary index matrix of y data and initialize activation layers
    encoder = OneHotEncoder(sparse=False)
    y_matrix = encoder.fit_transform(y.T)
    act_layers = activation_layers(a1, thetas)
    
    # cost function created in seperate parts
    first = np.multiply(-y_matrix, np.log(act_layers[-1]))
    second = np.multiply(1 - y_matrix, np.log(1 - act_layers[-1]))
    
    # regularization
    reg_1 = lamb/(2 * len(X))
    reg_2 = 0
    for i in range(len(thetas)):
        reg_2 += np.power(thetas[i][...,1:], 2).sum()
    
    J = 1/len(X) * (first - second).sum() + (reg_1 * reg_2)
    print('Current Cost')
    print(J)
    print('*' * 20)
    return J


def gradient(all_thetas, weights, X, y, lamb):
    thetas = unpack_thetas(all_thetas, weights)
    # add column of 1's
    X = X/255
    a1 = np.insert(X, 0, 1, 1)
    
    # create a binary index matrix of y data and activation layers
    encoder = OneHotEncoder(sparse=False)
    y_matrix = encoder.fit_transform(y.T)      
    act_layers = activation_layers(a1, thetas)
    
    
    #slice out first column in all thetas except theta1 
    theta_delta = []
    for i in range(len(thetas)):
        theta_delta.append(thetas[i][:, 1:])
    
    d = []
    Delta = []
    theta_grad = []
    for i in range(len(thetas)):
        if i == 0:
            d.insert(0, act_layers[-(i+1)] - y_matrix) #Work backwards through act_layers
            
        else:
            d.insert(0, np.multiply(d[0] * theta_delta[-i],np.multiply(
                act_layers[-(i+1)][:, 1:], (1-act_layers[-(i+1)][:, 1:]))))

        # Create Deltas
        Delta.insert(0, d[0].T * act_layers[-(i+2)])
        
        #Create Theta_grad
        theta_grad.insert(0, Delta[0] / len(y_matrix))
     
    # place a column of 0's in the first column of all thetas
    for i in range(len(thetas)):
        theta_delta[i] = lamb/len(y_matrix) * theta_delta[i]
        theta_grad[i] += np.insert(theta_delta[i], 0, 0, 1)    
    gradient_theta = pack_thetas(theta_grad)
    print(act_layers[-1])
    return gradient_theta


def forward_propagate(all_thetas, weights, X, y):
    thetas = unpack_thetas(all_thetas, weights)
    # add column of 1's
    X = X/255
    a1 = np.insert(X, 0, 1, 1)
    act_layers = activation_layers(a1, thetas)
            
    predict = np.argmax(act_layers[-1], axis=1)
    print(predict[:10])
    print(y[:10].T)
    correct = [1 if a==b else 0 for (a,b) in zip(predict, y.T)]
    accuracy = (sum(map(int, correct))/ float(len(correct)))
    return 'accuracy = {0}%'.format(accuracy * 100)
    
#    np.savetxt('digit_sigmoid.csv', 
#           np.c_[range(1, len(predict)+1), predict], 
#           delimiter=',', 
#           header = 'ImageId,Label', 
#           comments = '', 
#           fmt='%d')   

   
def sigmoid(z):
    return 1/(1 + np.exp(-z))
    
    
def activation_layers(a1, thetas):
    act_layers = []
    act_layers.append(a1)
    for i in range(len(thetas)):
        act_layers.append(sigmoid(act_layers[i] * thetas[i].T))
        if i != (len(thetas) - 1):
            act_layers[i+1] = np.insert(act_layers[i + 1], 0, 1, 1)
    return act_layers
            
            
def pack_thetas(thetas):
    new_thetas = np.matrix(np.ravel(thetas[0])).T
    
    for i in range(1, len(thetas)):
        new_thetas = np.concatenate((new_thetas, np.matrix(np.ravel(thetas[i])).T), axis=0)
    return new_thetas
       
    
def unpack_thetas(all_new_thetas, weights):
    theta_temp = []
    temp = 0
    wght_totals = [l * m for l,m in weights]
    for i in range(len(weights)):
        theta_temp.append(np.reshape(all_new_thetas[temp:temp + wght_totals[i]], weights[i]))
        temp += wght_totals[i]
    return theta_temp
