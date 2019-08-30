import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from KMeans import bestKmeans
from sklearn.metrics.pairwise import euclidean_distances


data=scipy.io.loadmat('HW5.mat')['X']
# Calculate sum of squares of each data point to its centroid
def calculateWk(clusters):
    wk = 0
    for cluster in clusters:
        cluster=np.array(cluster)
        cluster = np.array(cluster).reshape(32,int(len(cluster.ravel())/32))
        wk += np.sum(euclidean_distances(cluster, cluster)**2/2/len(cluster))
    return wk

# Generate reference dataset using algorithm introduced in the paper
def generateReferenceDataset(data):
    _, _, VT = np.linalg.svd(data)
    X = np.matmul(data, np.transpose(VT))
    numCol = len(data[0])
    numRow = len(data)
    Z_prime = np.matrix([np.random.uniform(np.min(X[:,i]),np.max(X[:,i]),numRow) for i in range(numCol)]).T
    Z = np.matmul(Z_prime,VT)
    return Z

# Main Method for gap statistic
def gapStatistic(data, B, K):
    # Placeholders
    dataDispersion,meanReferencesDispersions,sd = np.zeros(K),np.zeros(K),np.zeros(K)

    # Try different number of k
    for index in range(K):
        clusters,_,_ = bestKmeans(data,index+1,1)
        dataDispersion[index] = np.log(calculateWk(clusters))
        # Contains all references for current K
        references = np.zeros(B)

        for i in range(B):
            clusters, _, _ = bestKmeans(generateReferenceDataset(data), index + 1, 1)
            dataDispersion[index] = np.log(calculateWk(clusters))

        meanReferencesDispersions[index] = np.mean(references)
        sd[index] =  np.sqrt(1 / B * sum(np.square(references - meanReferencesDispersions[index])))
    redC = sd * np.sqrt(1 + 1 / B)
    result = np.array(meanReferencesDispersions) - np.array(dataDispersion)
    return result,redC , np.argmax(result)+1


result, redC , bestK = gapStatistic(data,200,21)

# Plot graphs
plt.plot(range(1,22),result)
s1 = plt.scatter(range(1,22), result, marker='_' ,s = 300)
redCir = [k for k in range(1,21) if result[k-1] >= result[k] - redC[k]]
redCirInd = [ele - 1 for ele in redCir]
s2 = plt.scatter(redCir, result[redCirInd], marker='o', facecolors='none', edgecolor='r')
s3 = plt.scatter(bestK, result[bestK-1], marker='*', facecolor='yellow', s=200, edgecolor='black')

plt.xlabel("K")
plt.ylabel("Gap")
plt.xlim(0,21)
plt.title("Gap Statistic - Selecting K")

plt.legend((s1,s2,s3),('Gap','K|Gap(K)>=Gap(K+1)-S _k+1','K*=16'))
plt.show()
