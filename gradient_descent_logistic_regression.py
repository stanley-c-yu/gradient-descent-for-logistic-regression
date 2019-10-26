# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:39:54 2019

@author: stany
"""
import numpy as np 
import statsmodels.api as sm
from sklearn import metrics


class GradientDescentLogisticRegression:  
    '''
    Calculates the Cross-Entropy Objective and Gradient Descent for
    Logistic Regression. 
    ''' 
    
    def __init__(self,X,y,learning_rate,max_iter,print_theta=False):
        self.X = X 
        self.X1 = np.matrix(sm.add_constant(X)) #Adds a column of ones to the front of X  
        self.m,self.n = self.X1.shape
        self.y = y.as_matrix().reshape((self.m,1))
        self.theta = np.ones((self.n, 1)) #initialize "random" theta as a column of ones
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.print_theta = print_theta 
        
    def pred(self, theta, x): 
        prediction = np.matmul(np.transpose(theta), np.transpose(x))
        return 1/(1+np.exp(-prediction))
        
    def loss_function(self,theta): 
        '''
        Calculate the Cross Entropy Objective, our Loss Func. for Log. Reg.
        '''
#         zed = np.matmul(self.X1,theta)
#         print(zed)
#         predictions = 1/(1 + np.exp(-zed))
#         ceo = -1 * (np.matmul(np.transpose(self.y),np.log(predictions)) + np.matmul(np.transpose(1-self.y),np.log(np.absolute(1-predictions))))
        total = 0
        for i in range(self.m): 
            pred = self.pred(theta, self.X1[i,:])
            total += self.y[i]*np.log(pred) + (1-self.y[i]*np.log(1-pred))
        return -total
#         #print(np.log(np.absolute(1-h))) #can't take the log of a negative number.  An appropriate solution?  
#         return(ceo)
        
    def calculate_f1_score(self,theta): 
        '''
        Calculates the F1 Score using metrics from SKLearn.  
        '''
        zed = np.matmul(self.X1,theta)
        predictions = 1/(1 + np.exp(-zed))
        f1_score = metrics.f1_score(self.y, np.around(predictions), labels = None, pos_label=1, average='binary', sample_weight=None)
        return(f1_score) #F1 Score doesn't seem to vary when changing iterations or learning rate.  Why?  
    
    def gradient_descent(self): 
        '''
        Calculates gradient descent for Logistic Regression. 
        '''
        print_theta = self.print_theta
        alpha = self.learning_rate #alpha represents learning_rate
        max_iter = self.max_iter
        theta = self.theta 
        for i in range(0, max_iter):
            loss = self.loss_function(theta)
            zed = np.matmul(self.X1,theta)
            predictions = 1/(1+ (1/np.exp(zed)))
            step = alpha*(np.matmul(np.transpose(self.X1),(np.around(predictions)-self.y)))
            theta = theta-step 
        f1_score = self.calculate_f1_score(theta)
        if print_theta==True:
            print("Theta: ", theta)
        print("\nCross Entropy: %f" % (loss), "\nAlpha = %s" % alpha, "\nIterations: %s" % max_iter, "\nF1 Score: ", f1_score)
        return(None)