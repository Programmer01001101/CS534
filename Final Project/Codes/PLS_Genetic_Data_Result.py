# This file tests PLS's performance on genetic data
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from PLS_Implementation import PLS
from simpleLogisticRegression import LR



def read_file(path):
    df = pd.read_csv(path,delimiter = ',')
    return df

# Imports and Preprocess data
expression_train = read_file("expression_data_train.csv")
expression_test = read_file("expression_test.csv")
X = expression_train.iloc[:,0:24368].values
y = expression_train[['Y_label']].values
X_test = expression_test.iloc[:,0:24368].values
y_test = expression_test[['Y_label']].values
y = np.squeeze(y)
y_test = np.squeeze(y_test)

# Run PLS and fits it to the train data
pls = PLS(3)
pls.fit(X, y)
x_train = pls.transform(X)
x_test = pls.transform(X_test)

# Run Logistic regression on the results of PLS from previous step
LogisticRegression = LR()
LogisticRegression.fit(x_train, y)
y_pred = LogisticRegression.predict(x_test)

# Display scores
print("Accuracy of PLS on Clinical Data" , accuracy_score(y_test, y_pred))
print("F1 Score of PLS on Clinical Data" , f1_score(y_test, y_pred))




