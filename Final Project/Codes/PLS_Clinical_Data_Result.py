# This file tests PLS 's preformance on clinical data

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from PLS_Implementation import PLS
from simpleLogisticRegression import LR


# Imports and preprocess data
train = pd.read_csv('clinical_train.csv')
x_train = np.array(train.drop(['PATIENT_ID', 'Y_label'], axis=1))
y_train = np.array(train['Y_label'])
test = pd.read_csv('clinical_test.csv')
x_test = np.array(test.drop(['PATIENT_ID', 'Y_label'], axis=1))
y_test = np.array(test['Y_label'])


# Run PLS and fits it to the train data
pls = PLS(17)
pls.fit(x_train, y_train)

x_train = pls.transform(x_train)
x_test = pls.transform(x_test)


# Run logistic regression on the result of PLS
LogisticRegression = LR()
LogisticRegression.fit(x_train, y_train)
y_pred = LogisticRegression.predict(x_test)

# Prints the scores
print("Accuracy of PLS on Clinical Data" , accuracy_score(y_test, y_pred))
print("F1 Score of PLS on Clinical Data" , f1_score(y_test, y_pred))



