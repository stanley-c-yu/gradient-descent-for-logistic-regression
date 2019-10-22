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
        self.y = y 
        self.m_y = np.shape(self.y)
        self.y = np.reshape(self.y, (self.m_y[0],1))
        self.m,self.n = np.shape(self.X1)
        self.theta = np.ones((self.n, 1)) #initialize "random" theta as a column of ones
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.print_theta = print_theta 
        
    def calculate_zed(self): 
        '''
        Calculate zed.  
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
        #print(predictions)
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
        loss = np.dot(np.transpose(y), log_predictions) + np.dot(np.transpose(ones_minus_y),log_predictions)
        loss = -loss
        return(loss)
        
    def calculate_f1_score(self): 
        '''
        Calculates the F1 Score using metrics from SKLearn.  
        '''
        y = self.y
        predictions = self.calculate_predictions()  
        f1_score = metrics.f1_score(y, predictions.round(), labels = None, pos_label=1, average='binary', sample_weight=None) 
        return(f1_score)
    
    def gradient_descent(self): 
        '''
        Calculates gradient descent for Logistic Regression. 
        '''
        print_theta = self.print_theta
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
            theta = theta-step 
        f1_score = self.calculate_f1_score()
        if print_theta==True:
            print("Theta: ", theta)
        print("\nCross Entropy: %f" % (loss), "\nAlpha = %s" % learning_rate, "\nIterations: %s" % max_iter, "\nF1 Score: ", f1_score)
        #if max_iter==100: 
        #    print("F1 Score: ", f1_score)
        return(None)