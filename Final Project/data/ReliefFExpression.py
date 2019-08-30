import numpy as np
import pandas as pd
import ReliefSource
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt



def preprocessing(df):
    df = df.T
    df = df.drop(['Entrez_Gene_Id'], axis = 0)
    df = df.dropna()
    return df

def read_file(path):
    df = pd.read_csv(path,delimiter = '\t')
    return df

raw_data = read_file("data_expression_median.txt")
df = preprocessing(raw_data)
trainData = pd.read_csv("clinical_train.csv")
testData = pd.read_csv('clinical_test.csv')
mergedTest = pd.merge(left=df,right=testData, left_on=df.index, right_on='PATIENT_ID')
mergedTrain = pd.merge(left=df, right=trainData, left_on=df.index, right_on='PATIENT_ID')


Xtrain = mergedTrain.iloc[:, 0:24368].values
ytrain = np.squeeze(mergedTrain[['Y_label']].values)
Xtest = mergedTest.iloc[:,0:24368].values
ytest = np.squeeze(mergedTest[['Y_label']].values)
accuracyScore = []
fScore = []

fs = ReliefSource.ReliefF(n_neighbors=100)
rfc1=RandomForestClassifier(class_weight= {0:1 , 1:2} , random_state=0, max_features='auto', n_estimators= 200, max_depth=3, criterion='entropy')
fs.fit(Xtrain, ytrain)

for numfeatures in range(1,30000,300):
    rfc1.fit(fs.transform(Xtrain,numfeatures), ytrain)
    pred=rfc1.predict(fs.transform(Xtest,numfeatures))
    accuracyScore.append(accuracy_score(ytest, pred))
    fScore.append(f1_score(ytest,pred))

#
# for numfeatures in range(1,30000,300):
#     fs = ReliefF(n_neighbors=100, n_features_to_keep=numfeatures)
#     Xtrain_subset = fs.fit_transform(Xtrain, ytrain)
#     rfc1=RandomForestClassifier(class_weight= {0:1 , 1:2} , random_state=0, max_features='auto', n_estimators= 200, max_depth=3, criterion='entropy')
#     rfc1.fit(Xtrain_subset, ytrain)
#     features = rfc1.top_features
#     Xtest_subset  = fs.transform(Xtest)
#     pred=rfc1.predict(Xtest_subset)
#     accuracyScore.append(accuracy_score(ytest, pred))
#     fScore.append(f1_score(ytest,pred))
#     print(numfeatures)
#
# print(accuracyScore)
# print(fScore)
# Acc = [0.44285714285714284, 0.6464285714285715, 0.6678571428571428, 0.6678571428571428, 0.6428571428571429, 0.675, 0.6607142857142857, 0.6714285714285714, 0.6535714285714286, 0.6714285714285714, 0.6678571428571428, 0.6607142857142857, 0.6678571428571428, 0.6607142857142857, 0.675, 0.6464285714285715, 0.6607142857142857, 0.6464285714285715, 0.6642857142857143, 0.6678571428571428, 0.6607142857142857, 0.6678571428571428, 0.6428571428571429, 0.6535714285714286, 0.6464285714285715, 0.6464285714285715, 0.6607142857142857, 0.6642857142857143, 0.6464285714285715, 0.6428571428571429, 0.6464285714285715, 0.6357142857142857, 0.6428571428571429, 0.6428571428571429, 0.6321428571428571, 0.6607142857142857, 0.6392857142857142, 0.6714285714285714, 0.6571428571428571, 0.6535714285714286, 0.6392857142857142, 0.6285714285714286, 0.6357142857142857, 0.6571428571428571, 0.6642857142857143, 0.6321428571428571, 0.65, 0.65, 0.625, 0.6607142857142857, 0.6392857142857142, 0.6714285714285714, 0.6357142857142857, 0.6571428571428571, 0.6392857142857142, 0.6357142857142857, 0.6321428571428571, 0.6357142857142857, 0.6571428571428571, 0.65, 0.6678571428571428, 0.6464285714285715, 0.6464285714285715, 0.6285714285714286, 0.6357142857142857, 0.6392857142857142, 0.6428571428571429, 0.6285714285714286, 0.6392857142857142, 0.6392857142857142, 0.6392857142857142, 0.6285714285714286, 0.6357142857142857, 0.6392857142857142, 0.6392857142857142, 0.6392857142857142, 0.6392857142857142, 0.6392857142857142, 0.6392857142857142, 0.6285714285714286, 0.6321428571428571, 0.6357142857142857, 0.6428571428571429, 0.6428571428571429, 0.6428571428571429, 0.6428571428571429, 0.6428571428571429, 0.6428571428571429, 0.6428571428571429, 0.6428571428571429, 0.6428571428571429, 0.6428571428571429, 0.6428571428571429, 0.6428571428571429, 0.6428571428571429, 0.6428571428571429, 0.6428571428571429, 0.6428571428571429, 0.6428571428571429, 0.6428571428571429]
# i=1
# index = 0
# cur= 0
# for ele in Acc:
#     if ele > cur :
#         cur = ele
#         index = i
#     i+=300
#
# print(index)

# plt.plot(range(1,30000,300),Acc)
# plt.plot(range(1,30000,300),Fsc)
plt.legend(["Accuracy" , "Fscore"])
plt.xlabel("numFeatures")
plt.ylabel("scores")
plt.show()