# gradient-descent-for-logistic-regression
An implementation of gradient descent for binary logistic regression.  

The implementation uses the Spambase dataset provided by UC Irvine's Machine Learning Repository.  Spambase contains over 5000 data points regarding SPAM (i.e., "junk") and HAM emails, which are characterized by 57 feature columns and a 58th response column denoting emails as either 1 (for SPAM) or 0 (for HAM).  

Data is stored in the included "spambase.data" file, which is unlabeled, however the names are listed in order in the accompanying "spambase.names" files.  Further details regarding the data are include in the "spambase.documentaion" file.  

The "gradient_descent_logistic_regression.py" file first reads in and pre-processes the data into standardized training and testing sets.  It then trains a model using the training set before testing.  Model quality is gauged by comparing F1 Scores and Cross Entropy over varying learning rates ("alpha values"), iterations, and thresholds.  
