import pandas as pd
import numpy as np 
import statsmodels.api as sm 

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

##Import the spambase dataset and adjust as necessary 
spambase = pd.read_csv('spambase.data',header=None)
spambase.rename(columns={0:"word_freq_make", 1:"word_freq_address", 2:"word_freq_all", 3:"word_freq_3d", 4:"word_freq_our", 
                    5:"word_freq_over", 6:"word_freq_remove", 7:"word_freq_internet", 8:"word_freq_order", 9:"word_freq_mail",
                    10:"word_freq_receive", 11:"word_freq_will", 12:"word_freq_people", 13:"word_freq_report", 14:"word_freq_addresses",
                    15:"word_freq_free", 16:"word_freq_business", 17:"word_freq_email", 18:"word_freq_you", 19:"word_freq_credit", 
                    20:"word_freq_your", 21:"word_freq_font", 22:"word_freq_000", 23:"word_freq_money", 24:"word_freq_hp", 
                    25:"word_freq_hpl", 26:"word_freq_george", 27:"word_freq_650", 28:"word_freq_lab", 29:"word_freq_labs", 
                    30:"word_freq_telnet", 31:"word_freq_857", 32:"word_freq_data", 33:"word_freq_415", 34:"word_freq_85", 
                    35:"word_freq_technology", 36:"word_freq_1999", 37:"word_freq_parts", 38:"word_freq_pm", 39:"word_freq_direct", 
                    40:"word_freq_cs", 41:"word_freq_meeting", 42:"word_freq_original", 43:"word_freq_project", 44:"word_freq_re",
                    45:"word_freq_edu", 46:"word_freq_table", 47:"word_freq_conference", 48:"char_freq_;", 49:"char_freq_(", 
                    50:"char_freq_[", 51:"char_freq_!", 52:"char_freq_$", 53:"char_freq_#", 54:"capital_run_length_average", 
                    55:"capital_run_length_longest", 56:"capital_run_length_total", 57:"is_spam"},inplace=True)
#inplace: Makes changes in original Data Frame if True.


##Split spambase into feature and response sets 
SB_features = spambase.iloc[:, 0:57]
SB_response = spambase.iloc[:, 57]
#SB_response2 = spambase[['is_spam']] this will only select the is_spam column
#spambase.drop(['is_spam'],axis=1)  this will select everything but the is_spam column by dropping 

##Split SB_features and SB_response into training and testing sets (75% and 25% respectively)
SBf_train, SBf_test, SBr_train, SBr_test = train_test_split(
        SB_features, SB_response, test_size=0.25, train_size=0.75, 
        random_state = 0, stratify=SB_response
        )

##Standardize the dataset by first using preprocessing to compute the mean and standard deviation for future scaling
##Then scale the data sets 
SBf_train = preprocessing.StandardScaler().fit_transform(SBf_train.values)
SBf_test = preprocessing.StandardScaler().fit_transform(SBf_test.values)

class logistic_regression: 
    
    def __init__(self, X, y_actual, alpha, max_iter):
        self.X = X 
        self.y_actual = y_actual 
        self.alpha = alpha 
        self.max_iter = max_iter
#        self.threshold = threshold
        
    
    def sigmoid(self, z): 
    	return 1/(1+np.exp(-z))
    
    def predictor(self, theta, X): 
    	predictions = np.matmul(X, theta) 
    	sigmoidal_prediction = self.sigmoid(predictions) 
    	return(sigmoidal_prediction)
    
    def loss(self, h,y):
        h = h + 1e-9
        h = np.array(h,dtype=np.complex128) 
        y = np.array(y,dtype=np.complex128)
        h = h.flatten()
        y = y.flatten()
        return (-((y*np.log(h))-((1-y)*np.log(1-h)))).mean()
    
    def gradient_descent(self): 
        X1 = np.matrix(sm.add_constant(self.X))
        m,n = X1.shape
        y_actual = self.y_actual.as_matrix().reshape((m,1))
        theta = np.ones((n,1))
        predictions = None
        for i in range(0,self.max_iter): 
            predictions = self.predictor(theta,X1)
            gradient = np.matmul(np.transpose(X1),
    			(predictions-y_actual))
            step = self.alpha*gradient
            theta = theta - step
        f1 = metrics.f1_score(y_actual,np.around(predictions),labels=None,
    		pos_label=1,average='binary',sample_weight=None)
        ceo = self.loss(predictions,y_actual)
        print("\nCross Entropy: %f" % (ceo), "\nAlpha = %s" % self.alpha,
    	"\nIterations: %s" % self.max_iter, "\nF1 Score: ", f1)
        return(theta) 
        
    def classify(self, threshold=0.5):
        X1 = np.matrix(sm.add_constant(self.X))
        theta = self.gradient_descent()
        return [1 if i >= threshold else 0 for i in self.predictor(theta,X1)]

#gradient_descent(SBf_test,SBr_test,0.1,10)
#print(classify(SBf_test,SBr_test,0.1,10,0.5))
#gradient_descent(SBf_test,SBr_test,0.1,50)
#print(classify(SBf_test,SBr_test,0.1,10,0.5))
#gradient_descent(SBf_test,SBr_test,0.1,100)
#print(classify(SBf_test,SBr_test,0.1,100,0.5))

log_reg = logistic_regression(SBf_test,SBr_test,0.1,100)
#log_reg.gradient_descent()
print(log_reg.classify())

