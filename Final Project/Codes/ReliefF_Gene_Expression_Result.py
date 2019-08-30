# This file tested relief-f algorithm on gene expression data
# It first finds the best number of features and then finds the model's accuracy based on the best number of features selected.

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import ReliefImplementation


def read_file(path):
    df = pd.read_csv(path,delimiter = ',')
    return df

# Read data and preprocess data
expression_train = read_file("expression_data_train.csv")
expression_test = read_file("expression_test.csv")
X = expression_train.iloc[:,0:24368].values
y = expression_train[['Y_label']].values
X_test = expression_test.iloc[:,0:24368].values
y_test = expression_test[['Y_label']].values
y = np.squeeze(y)
y_test = np.squeeze(y_test)


# Placeholder for accuracy and fscore
accuracyScore = []
fScore = []

# Run relief-f on the trainning data
fs = ReliefImplementation.ReliefF(neighbors=10)
rfc1=RandomForestClassifier(class_weight= {0:1 , 1:2} , random_state=0, max_features='auto', n_estimators= 200, max_depth=3, criterion='entropy')
fs.fit(X, y)

# Find the best number of features
for numfeatures in range(1,30000,300):
    rfc1.fit(fs.transform(X,numfeatures), y)
    pred=rfc1.predict(fs.transform(X_test,numfeatures))
    accuracyScore.append(accuracy_score(y_test, pred))
    fScore.append(f1_score(y_test,pred))

# Plots number of features vs. accuracy/f-score
plt.plot(range(1,30000,300),accuracyScore)
plt.plot(range(1,30000,300),fScore)
plt.legend(["Accuracy" , "Fscore"])
plt.xlabel("numFeatures")
plt.ylabel("scores")
plt.show()


# The best number feature selected is 1500, fits the model with the best 1500 features
rfc1.fit(fs.transform(X,1500), y)
pred=rfc1.predict(fs.transform(X_test,1500))

# Display scores
print("Accuracy for ReliefF on Gene Expression data: ",accuracy_score(y_test,pred))
print("F1 for ReliefF on Gene Expression  data: ",f1_score(y_test,pred))
print("Precison for ReliefF on Gene Expression  data: ",precision_score(y_test,pred))
print("Recall for ReliefF on Gene Expression  data: ",recall_score(y_test,pred))

# Plot ROC AUC Curve
# Compute ROC curve and ROC area for each class
pred = rfc1.predict_proba(fs.transform(X_test,1500))
pred = pred[:,1]

fpr,tpr,threshold = roc_curve(y_test, pred)
roc_auc = auc(fpr,tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()