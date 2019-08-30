# This file implements relief-f from scratch
import numpy as np
class ReliefF(object):

    def __init__(self, neighbors=10):
        self.featureScores = None
        self.bestFeatures = None
        self.neighbors = neighbors

    # this method constructs a matrix that contians pariwise distance between all samples
    #  it returns the best n features (n is neighbor's value)
    def distanceMatrix(self,X):
        dMatrix = []
        for entries in range(X.shape[0]):
            dArray = []
            for entries2 in range(X.shape[0]):
                dArray.append(np.linalg.norm(X[entries]-X[entries2]))
            dMatrix.append(dArray)
        dMatrix=np.matrix(dMatrix)
        return np.argsort(dMatrix,axis=1)[:,range(1,self.neighbors+1)]

    # This method fits relief-f to train data and updates the bestfeatures
    def fit(self, X, y):
        # Initialize feature scores
        self.featureScores = np.zeros(X.shape[1])

        #Construct a matrix that contains all the distances
        dMatrix = self.distanceMatrix(X)

        #Iterater through the whole dataset(could also randomly select sample from the dataset)
        for entries in range(X.shape[0]):
            #Find nearest neighbors
            nearestIndices = np.array(dMatrix[entries])[0]
            numIndices = len(nearestIndices)
            sameLabel = np.zeros(numIndices)
            sameFeature = []
            # Calculate rewards/penalties
            attributerange = np.max(X[entries])-np.min(X[entries])
            for i in range(numIndices):
                sameLabel[i] = np.absolute(y[entries]-y[nearestIndices[i]])/attributerange
                sameFeature.append(np.absolute(X[entries] - X[nearestIndices[i]]) / attributerange)
            sameFeature = np.matrix(sameFeature)
            # Update feature scores
            self.featureScores += np.ravel(np.dot(sameFeature.T, sameLabel))
        self.bestFeatures = np.argsort(self.featureScores)[::-1]

    # Select the best features from a data (test set)
    def transform(self, X, numFeatures):
        return X[:, self.bestFeatures[:numFeatures]]
