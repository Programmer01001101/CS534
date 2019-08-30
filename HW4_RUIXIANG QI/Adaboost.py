a# THIS ASSIGNMENT IS WRITTEN BY MYSELF WITHOUT CONSULTING OTHER SOURCES. -- Ruixiang Qi



from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


filename = "ionosphere.data"



def main():
    numEstimator = 300

    # Question 1a
    # results0 , results5 and results are Adaboost with boosting with stumps,
    # 5-node trees, and 10-node trees respectively. 
    results0 =  crossValidation(filename, numEstimator,2,1)
    results5 = crossValidation(filename, numEstimator,5,1)
    results10 = crossValidation(filename, numEstimator,10,1)

    plt.rcParams.update({'font.size': 22})
    plt.figure(0,figsize=(20,20))
    plt.xlim(0, 300)
    plt.ylim(0,0.3)
    plt.plot(range(1,numEstimator+1),results0)
    plt.plot(range(1,numEstimator+1), results5)
    plt.plot(range(1,numEstimator+1), results10)
    plt.title("1.a. Adaboost - weak learner complexity")
    plt.xlabel("weak learners")
    plt.ylabel("Prediction error")
    plt.legend(["max splits = 1","max splits =5","max splits = 10"])
    plt.savefig("hw4_1a.png")

    # Question 1b
    # results5Shrinkage is Adaboost with 5-node trees and shrinkage of 0.9
    results5Shrinkage = crossValidation(filename, numEstimator, 5, 0.9)

    plt.figure(1,figsize=(20, 20))
    plt.xlim(0, 300)
    plt.ylim(0, 0.3)
    plt.plot(range(1, numEstimator + 1), results5)
    plt.plot(range(1, numEstimator + 1), results5Shrinkage)
    plt.title("1.b. Adaboost - shrinkage")
    plt.xlabel("weak learners")
    plt.ylabel("Prediction error")
    plt.legend(["max splits = 5", "max splits =5, shrinkage"])
    plt.savefig("hw4_1b.png")
    plt.show()

# The following method implements Adaboost, it returns a matrix that contains predicted results of all estimators
# For example, if numEstimator is 300, the method returns a matrix with 300 rows, each row conresponds to the
# prediction of adboost with some number of estimator
def Adaboost(numEstimater,numTreeNodes,traindata,traintarget,testdata,shrinkage):
    numData = traindata.shape[0]
    weights = np.array([1/numData for ele in range(numData)])
    # Accumulate result
    results = np.matrix(np.zeros(testdata.shape[0])).T
    # Store predicted labels of all number of estimators
    allPredictedLabels = []

    for n in range(numEstimater):

        # Create a weak leaner and fit it to training data
        clf = DecisionTreeClassifier(max_leaf_nodes = numTreeNodes)
        clf.fit(traindata,traintarget,sample_weight= weights)
        predictedLabel = np.array(clf.predict(traindata))

        # Compute weighted error
        I = []
        for ele in range(len(predictedLabel)):
            if predictedLabel[ele] != traintarget[ele]:
                I.append(1)
            else:
                I.append(0)
        I = np.array(I)
        error = sum(weights * I) /sum(weights)
        #update weights , add shrinkage if needed
        am = np.log1p((1-error)/error)* shrinkage
        weights = np.array(weights* np.exp(am*I))
        results += am* np.matrix(clf.predict(testdata)).T
        # predicted label for current number of estimator
        predictedLabel = []
        for ele in results:
            if ele > 0:
                predictedLabel.append(1)
            else :
                predictedLabel.append(-1)
        # Append the predicted label of current number of estimators to the final matrix
        allPredictedLabels.append(predictedLabel)
    return allPredictedLabels


# The following method is nested cross validation with 5 folds.
# It retuns an array that contains errors for different number of estimators
def crossValidation(filename , numEstimator, numTreeNodes,shrinkage):
    data = preprocess(filename)
    splitSize = int(data.shape[0]/5)
    currentIndex = 0
    # Initilize final result
    allErrors = []
    for i in range(numEstimator):
        allErrors.append(0)
    #Outer Loop
    for outerloopNum in range(5):
        testColumns = [i for i in range(currentIndex,currentIndex+splitSize)]
        trainColumns = [i for i in range(data.shape[0]) if i not in testColumns]
        testData = data[testColumns,:]
        trainData = data[trainColumns,:]

        currentIndex +=splitSize
        innerLoopIndex = 0
        bestxTrain = []
        bestyTrain = []
        xtest = testData[:, 0:34]
        ytest = testData[:, 34]
        bestinnerError = float('inf')
        # Inner loop
        for innerLoopNum in range(4):
            validateColumns = [i for i in range(innerLoopIndex,innerLoopIndex+splitSize)]
            innerTrainColumns = [i for i in range(trainData.shape[0]) if i not in validateColumns]
            validateData = trainData[validateColumns, :]
            innerTrainData = trainData[innerTrainColumns, :]
            innerLoopIndex += splitSize

            xtrain = innerTrainData[:,0:34]
            ytrain = innerTrainData[:, 34]
            xvalid = validateData [:,0:34]
            yvalid = validateData [:,34]

            currentError = 0
            allPredictedLabel = Adaboost(numEstimater=numEstimator, numTreeNodes= numTreeNodes, traindata = xtrain, traintarget= ytrain, testdata=xvalid,shrinkage=shrinkage)
            for predictedLabel in allPredictedLabel:
                numcorrect = 0
                for eleIndex in range(len(predictedLabel)):
                    if predictedLabel[eleIndex] == yvalid[eleIndex]:
                        numcorrect += 1
                currentError = 1 - numcorrect / len(yvalid)

            if currentError < bestinnerError:
                bestinnerError=currentError
                bestxTrain= xtrain
                bestyTrain=ytrain

        allPredictedLabel = Adaboost(numEstimater=numEstimator, numTreeNodes= numTreeNodes, traindata = bestxTrain, traintarget= bestyTrain , testdata= xtest,shrinkage=shrinkage)
        errorIndex=0
        for predictedLabel in allPredictedLabel:
            numcorrect = 0
            for eleIndex in range(len(predictedLabel)):
                if predictedLabel[eleIndex] == ytest[eleIndex]:
                   numcorrect +=1
            error = 1 - numcorrect / len(ytest)
            allErrors[errorIndex] += error
            errorIndex +=1
    return np.divide(allErrors,5)





# A method that preprocess the raw data
def preprocess(filename):
    rawData = np.genfromtxt(filename,delimiter=",",dtype=str)
    data = []
    target= []
    for ele in rawData:
        data.append((ele[0:34]))
        target.append(ele[34:35])
    data = np.array([[float(e) for e in ele] for ele in data])
    lb = preprocessing.LabelBinarizer(neg_label= -1 , pos_label= 1)
    lb.fit(['g','b'])
    target = lb.fit_transform(np.array(target))
    data = np.append(data,target, 1)
    np.random.shuffle(data)
    return data

if __name__ == "__main__":
        main()