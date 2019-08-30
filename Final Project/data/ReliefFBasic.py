import pandas as pd
import numpy as np
# from ReliefF import ReliefF
from ReliefSource import ReliefF
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

train = pd.read_csv('clinical_train.csv')
x_train = np.array(train.drop(['PATIENT_ID', 'Y_label'], axis = 1))
y_train = np.array(train['Y_label'])
test = pd.read_csv('clinical_test.csv')
x_test = np.array(test.drop(['PATIENT_ID', 'Y_label'], axis = 1))
y_test = np.array(test['Y_label'])
accuracyScore = []
f1Score=[]

for numfeature in range(1,17):

    fs = ReliefF(n_neighbors=100, n_features_to_keep=numfeature)
    x_train_subset = fs.fit_transform(x_train, y_train)

    pls2 = PLSRegression(n_components=1)
    pls2.fit(x_train_subset, y_train)



    x_test_subset = fs.transform(x_test)

    Y_pred = pls2.predict(x_test_subset)
    pred = []
    for ele in Y_pred:
        if ele > 0.5:
            pred.append(1)
        else:
            pred.append(0)

    accuracyScore.append( accuracy_score(y_test,pred))
    f1Score.append(f1_score(y_test, pred))

print(accuracyScore)
print(f1Score)
plt.plot(range(1,17),accuracyScore)
plt.plot(range(1,17),f1Score)

plt.show()


