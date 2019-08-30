# THIS ASSIGNMENT IS WRITTEN BY MYSELF WITHOUT CONSULTING OTHER SOURCES. -- Ruixiang Qi
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
import numpy as np

import matplotlib.pyplot as plt

filename = "ionosphere.data"


def main():

    numTrees = 500

    # 2.a. feature sampling
    # result0,result1 and result 2 are random forests with m = sqrt(p) , m=1
    # and m = p respectively.
    results0 = crossValidation(filename, numTrees,'sqrt',False)
    results1 = crossValidation(filename, numTrees, 1, False)
    results2 = crossValidation(filename, numTrees, None, False)
   
    plt.rcParams.update({'font.size': 22})
    plt.figure(0, figsize=(20, 20))
    plt.xlim(0, 300)
    plt.ylim(0, 0.3)
    plt.plot(range(1, numTrees + 1), results0)
    plt.plot(range(1, numTrees + 1), results1)
    plt.plot(range(1, numTrees + 1), results2)
    plt.title("2.a. Random Forests - feature sampling")
    plt.xlabel("trees")
    plt.ylabel("Prediction error")
    plt.legend(["m = sqrt(p)", "m = 1", "m = p" , 'none','auto'])
    plt.savefig("hw4_2a.png")

    # 2.b. depth control
    #  resultsWithControl is randomForest with depth control, 
    # with min number of samples per terminal node set to 10.
    plt.figure(1, figsize=(20, 20))
    resultsWithControl = crossValidation(filename, numTrees, 'sqrt', True)

    plt.plot(range(1, numTrees + 1), results0)
    plt.plot(range(1, numTrees + 1), resultsWithControl)

    plt.title("2.b. Random Forests - depth control")
    plt.xlabel("trees")
    plt.ylabel("Prediction error")
    plt.xlim(0, 300)
    plt.ylim(0, 0.3)
    plt.legend(["m = sqrt(p)", "m = sqrt(p) - depth control"])
    plt.savefig("hw4_2b.png")

    plt.show()

# Preprocess the code
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





# The Random Forest Algorithm Method, it returns a matrix that contains predicted results of all numbers of trees
# For example, if numTree is 500, the method returns a matrix with 500 rows, each row conresponds to the
# prediction of random forest with some number of trees
def randomForest(numTrees, traindata, xtest, m, depthcontrol):
    results = []
    allPredictedLabels = []
    for i in range(numTrees):
        randomSample = selectBootstrapSamples(traindata)
        xtrain = randomSample[:, 0:34]
        ytrain = randomSample[:, 34]

        # Grow a random-forest tree
        if depthcontrol:
            clf = DecisionTreeClassifier(max_features=m, min_samples_leaf=10)
            # clf = DecisionTreeClassifier(max_features=m, max_depth =5)
        else:
            clf = DecisionTreeClassifier(max_features=m)
        clf.fit(xtrain, ytrain)

        # Prediction by the current tree
        predictedLabel = np.array(clf.predict(xtest))
        results.append(predictedLabel)
        # Majority votes of current amount of trees
        allPredictedLabels.append(majorityvote(results))
    return np.array(allPredictedLabels)



# Select random samples from traindata
def selectBootstrapSamples(traindata):
    Z = int(len(traindata)/2)
    randomIndex = np.random.choice(len(traindata), Z)
    return (traindata[randomIndex])


# A method that returns the majority vote of each row in a matrix, returns an array
def majorityvote(predictionMatrix):
    x = np.array(predictionMatrix).T
    x = np.ndarray.tolist(x)
    result = []
    for prediction in x:
        vote1 =0
        vote2 =0
        for ele in prediction:
            if ele == 1:
                vote1 +=1
            else:
                vote2 +=1
        if vote1>=vote2:
            result.append(1)
        else:
            result.append(-1)
    return result

# The following method is nested cross validation with 5 folds.
# It retuns an array that contains errors for different number of estimators
def crossValidation(filename , numTrees,m,depthcontrol):
    data = preprocess(filename)
    splitSize = int(data.shape[0]/5)
    currentIndex = 0
    # Initilize final result
    allErrors = []
    for i in range(numTrees):
        allErrors.append(0)
    #Outer Loop
    for outerLoopNum in range(5):
        testColumns = [i for i in range(currentIndex,currentIndex+splitSize)]
        trainColumns = [i for i in range(data.shape[0]) if i not in testColumns]
        testData = data[testColumns,:]
        trainData = data[trainColumns,:]

        currentIndex +=splitSize
        innerLoopIndex = 0
        bestTrainData = []
        xtest = testData[:, 0:34]
        ytest = testData[:, 34]
        bestinnerError = float('inf')
        # Inner loop
        for innerLoppNum in range(4):
            validateColumns = [i for i in range(innerLoopIndex,innerLoopIndex+splitSize)]
            innerTrainColumns = [i for i in range(trainData.shape[0]) if i not in validateColumns]
            validateData = trainData[validateColumns, :]
            innerTrainData = trainData[innerTrainColumns, :]
            innerLoopIndex += splitSize

            xvalid = validateData [:,0:34]
            yvalid = validateData [:,34]

            currentError = 0
            allPredictedLabel = randomForest(numTrees = numTrees, traindata = innerTrainData , xtest=xvalid, m = m , depthcontrol = depthcontrol)
            for predictedLabel in allPredictedLabel:
                numcorrect = 0
                for eleIndex in range(len(predictedLabel)):
                    if predictedLabel[eleIndex] == yvalid[eleIndex]:
                        numcorrect += 1
                currentError = 1 - numcorrect / len(yvalid)
            # Find best trainning data in validation sets
            if currentError < bestinnerError:
                bestinnerError = currentError
                bestTrainData = innerTrainData
        allPredictedLabel = randomForest(numTrees = numTrees, traindata = bestTrainData , xtest=xtest, m = m , depthcontrol= depthcontrol)
        errorIndex=0
        # Calculate sum of errors in five folds
        for predictedLabel in allPredictedLabel:
            numcorrect = 0
            for eleIndex in range(len(predictedLabel)):
                if predictedLabel[eleIndex] == ytest[eleIndex]:
                    numcorrect += 1
            error = 1 - numcorrect / len(ytest)
            allErrors[errorIndex] += error
            errorIndex += 1
    return np.divide(allErrors,5)









if __name__ == "__main__":
     main()