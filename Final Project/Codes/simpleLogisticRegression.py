import numpy as np

# In this program, I implement a simple logistic regression without regularization from scra.
# This is used for classification in the Partial Least Squares Algorithm
class LR(object):
    def __init__(self):

        self.W = None
        self.b = None

    def fit(self, X, Y):
        # Initilize weights and bias and learning rate
        numAttribute = X.shape[1]
        numSample = len(Y)
        self.W = np.zeros((numAttribute,1))
        self.b= np.zeros((1,1))
        lr = 0.05

        for epoch in range(1000):
            for index in range(numSample):
                currentX = np.array(np.matrix.tolist(X[index])[0]).reshape(1,numAttribute)
                Z = np.matmul(currentX,self.W) + self.b
                z = self.sigmoid(Z.item(0)) - Y[index]
                w = currentX.T * z / numSample
                self.W = self.W - lr *w
                self.b = self.b - lr * np.sum(z)

    def predict(self,X):
        result = []
        for index in range(X.shape[0]):
            currentX = np.array(np.matrix.tolist(X[index])[0])
            Z = np.matmul(currentX, self.W) + self.b
            if self.sigmoid(Z)>0.5:
                result.append(1)
            else:
                result.append(0)
        return result

    #Sigmoid function
    def sigmoid(self,X):
        return 1/(1+np.e**(-X))
    # Loss for logistic regression
    def loss(self,y,y_hat):
        return -np.mean(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))




