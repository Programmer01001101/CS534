
# The program runs relief-f on clinical data.
# It performs Relief-F on clinical data by selecting different number of features and found out that we achieve maximum accuracy when all features are selected
# It suggests that feature selection is unnecessary for clinical data

import pandas as pd
import numpy as np
import ReliefImplementation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# imports and preprocess data
train = pd.read_csv('clinical_train.csv')
x_train = np.array(train.drop(['PATIENT_ID', 'Y_label'], axis = 1))
y_train = np.array(train['Y_label'])
test = pd.read_csv('clinical_test.csv')
x_test = np.array(test.drop(['PATIENT_ID', 'Y_label'], axis = 1))
y_test = np.array(test['Y_label'])

# placeholder for accuracy and f1score
accuracyScore = []
f1Score=[]

# fits train data to relief-f
fs = ReliefImplementation.ReliefF(neighbors=10)
rfc1=RandomForestClassifier(class_weight= {0:1 , 1:2} , random_state=0, max_features='auto', n_estimators= 200, max_depth=3, criterion='entropy')
fs.fit(x_train, y_train)

# Find the best number of features
for numfeatures in range(1,17):
    rfc1.fit(fs.transform(x_train,numfeatures), y_train)
    pred=rfc1.predict(fs.transform(x_test,numfeatures))
    accuracyScore.append(accuracy_score(y_test, pred))
    f1Score.append(f1_score(y_test,pred))



# Plot number of features vs. accuracy/fscore
plt.plot(range(1,17),accuracyScore)
plt.plot(range(1,17),f1Score)
plt.xlabel("NumFeatures")
plt.ylabel("Scores")
plt.legend(["Accuracy", "F-Score"])
plt.show()


