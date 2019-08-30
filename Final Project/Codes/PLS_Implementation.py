# This file implements Partial Least Squares from scratch

import numpy as np
class PLS(object):
    def __init__(self, numComponents):
        self.numComponents = numComponents

    # this method fits the model to train data
    def fit(self,X,Y):
        #Normalize X and Y
        X = np.matrix(X)
        Y = np.matrix(Y).reshape(-1, 1)
        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)
        self.xWeights = []
        self.xLoading = []
        # Iteration starts
        for k in range(self.numComponents):
            # Obtain the leftmost and rightmost singular vector from SVD
            U, _, VT = np.linalg.svd(np.matmul(X.T, Y))
            xWeight, yWeight =  U[:, 0],VT.T[:, 0].T
            xScore = np.matmul(X, xWeight)
            # Calculate the loadings
            x_loadings = np.matmul(X.T, xScore) / np.matmul(xScore.T, xScore)
            y_loadings = np.matmul(Y.T, xScore)/ np.matmul(xScore.T, xScore)
            # Updates X and Y
            X = X - np.matmul(xScore, x_loadings.T)
            Y = Y - np.matmul(xScore, y_loadings.T)

            # Append the extracted feature after each iteration
            self.xWeights.append(np.matrix.tolist(xWeight.ravel())[0])
            self.xLoading.append(np.matrix.tolist(x_loadings.ravel())[0])

        # Use weights and loadings to find the transform matrix
        self.xWeights = np.matrix(self.xWeights)
        self.xLoading = np.matrix(self.xLoading)
        self.xTransfrom = np.matmul(self.xWeights, np.matmul(self.xLoading.T, self.xWeights))
        return self

    # this method transforms input data and returns the transformed data
    def transform(self, X):
        X = X - X.mean(axis=0)
        return np.matmul(X, self.xTransfrom.T)
