
# This File preprocess gene mutation data. It runs our baseline model on gene mutation data and see the data's performance.
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier


def read_file(path):
    df = pd.read_csv(path,delimiter = '\t')
    return df

def preprocess(df):
    df = df.fillna("")
    return df


raw_data = read_file("data_mutations_consequence.txt")
df=preprocess(raw_data)

d = defaultdict(preprocessing.LabelEncoder)
# Encoding the variable
fit = df.iloc[:,1:].apply(lambda x: d[x.name].fit_transform(x))

# Inverse the encoded
fit.apply(lambda x: d[x.name].inverse_transform(x))

# Using the dictionary to label future data
dat = df.iloc[:,1:].apply(lambda x: d[x.name].transform(x))
dat2 = df.iloc[:,0:1]
df = pd.concat([dat2, dat], axis=1)
df.rename(columns={'\t': 'Key'}, inplace=True)

clic_data = read_file("data_clinical_patient.txt")
clic_data = clic_data.fillna("")
clic_data = clic_data.iloc[4:]
osstatus = clic_data.iloc[:,13]
osstatus = pd.DataFrame(data=osstatus)

from collections import defaultdict
from sklearn import preprocessing

d = defaultdict(preprocessing.LabelEncoder)
# Encoding the variable
fit = osstatus.apply(lambda x: d[x.name].fit_transform(x))

# Inverse the encoded
fit.apply(lambda x: d[x.name].inverse_transform(x))

# Using the dictionary to label future data
osstatus = osstatus.apply(lambda x: d[x.name].transform(x))

pd2 = pd.concat([clic_data.iloc[:,0], osstatus], axis=1)

cat_df = pd.merge(left = df, right = pd2,left_on = 'Key',right_on='#Patient Identifier')
cat_df = cat_df[cat_df['Overall Survival Status']!= 0]

cat_df = cat_df.drop(['#Patient Identifier'], axis = 1)
cat_df = cat_df.drop(['Key'], axis = 1)

y = np.array(cat_df['Overall Survival Status'])
X = np.array(cat_df.drop(['Overall Survival Status'], axis = 1))
y[y == 2] = 0


def plot_AUC_ROC(X, y):
    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
    clf.fit(X_train, y_train)
    y_score = clf.predict(X_test)

    # Plot ROC AUC Curve
    # Compute ROC curve and ROC area for each class
    fpr, tpr, threshold = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

plot_AUC_ROC(X, y)
