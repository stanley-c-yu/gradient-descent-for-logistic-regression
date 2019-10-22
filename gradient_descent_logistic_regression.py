# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:39:54 2019

@author: stany
"""
import numpy as np 
import statsmodels.api as sm


class GradientDescentLogisticRegression:  
    '''
    Calculates the Cross-Entropy Objective and Gradient Descent for
    Logistic Regression. 
    ''' 
    
    def __init__(self,X,y,learning_rate,max_iter):
        self.X = X 
        self.X1 = np.matrix(sm.add_constant(X)) #Adds a column of ones to the front of X  
        self.y = y 
        self.m_y = np.shape(self.y)
        self.y = np.reshape(self.y, (self.m_y[0],1))
        self.m,self.n = np.shape(self.X1)
        self.theta = np.ones((self.n, 1)) #initialize "random" theta as a column of ones
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        
    def calculate_zed(self): 
        '''
        Gets medieval on his ass. 
        
        Also calculates zed.  
        '''
        theta = self.theta
        zed = np.matmul(self.X1,theta) 
        return(zed)  
        
    def calculate_predictions(self): 
        '''
        Calculates the predictions using the LogReg hypothesis formula.  
        '''
        zed = self.calculate_zed() 
        denominator_second_term = np.exp(-zed)
        denominator = 1 + denominator_second_term 
        predictions = 1/denominator
        #predictions = np.transpose(predictions)
        return(predictions) 
        
    def cross_entropy_objective(self): 
        '''
        Calculate the Cross Entropy Objective, our Loss Func. for Log. Reg.
        '''
        y = self.y
        m_y = np.shape(y)
        ones = np.ones(m_y)
        predictions = self.calculate_predictions()
        log_predictions = np.log(predictions)
        ones_minus_y = np.subtract(ones, y)
        loss = np.dot(np.transpose(y), log_predictions) + np.dot(np.transpose(ones_minus_y),log_predictions) ##
        loss = -loss
        return(loss)
        
    def gradient_descent(self):
        '''
        Calculates gradient descent for logistic regression.
        '''
        X1 = self.X1
        y = self.y
        learning_rate = self.learning_rate 
        max_iter = self.max_iter
        theta = self.theta 
        for i in range(0, max_iter):
            loss = self.cross_entropy_objective()
            predictions = self.calculate_predictions()
            innermost_term = (predictions-y)
            inner_term = np.matmul(np.transpose(X1),innermost_term) 
            step = learning_rate*inner_term 
            #return(np.shape(predictions))
            theta = theta-step 
        print("Theta: \n", theta, " \nCross Entropy: %f" % (loss), "\nAlpha = %s" % learning_rate, "\nIterations: %s" % max_iter)
        return(None)