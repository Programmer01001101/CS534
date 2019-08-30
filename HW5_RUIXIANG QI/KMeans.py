import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

import warnings
warnings.filterwarnings("ignore")


# Main Method


def bestKmeans(data, k, numKMeansPerformed):

    # Run KMeansClustering for many times to find the cluster with least objective function value
    ObjectiveFunctionValue = float("inf")
    finalResult = []
    finalCluster = []
    finalCentroids = []
    for iteration in range(numKMeansPerformed):
        clusters, result, Centroids, ObjectiveValue = KMeansClustering(k,data)
        if ObjectiveValue < ObjectiveFunctionValue:
            finalResult = result
            finalCluster = clusters
            finalCentroids = Centroids

    return finalCluster, finalResult,finalCentroids


def KMeansClustering(k,data):


    # Initialize the initial clusters
    indices = np.random.choice(len(data), k)
    Centroids = [x for x in data[indices]]
    oldCentroids, Clusters = findCentroids(Centroids, data)
    differentCentroid =True
    # Iteration Starts
    while differentCentroid:
        Centroids, Clusters = findCentroids(oldCentroids,data)
        differentCentroid = differentCentroids(Centroids,oldCentroids)
        oldCentroids = Centroids

    # Calculate Final Objective Function Value
    ObjectiveValue = 0
    for currentdata in data:
        for clusterIndex in range(k):
            ObjectiveValue += np.linalg.norm(Centroids[clusterIndex] - currentdata)



    result = assignLabel(Clusters)
    result = [ele[1] for ele in result]
    return Clusters,result,Centroids,ObjectiveValue


# Find new Centroids based on old Centroids, returns new centroids and new clusters
def findCentroids(Centroids,data):
    Clusters = [[] for ele in Centroids]
    for currentdata in data:
        dist = float("inf")
        cluster = -1
        # Classify each data point to the cluster that it is closest to
        for clusterIndex in range(len(Clusters)):
            if np.linalg.norm(Centroids[clusterIndex] - currentdata) < dist:
                dist = np.linalg.norm(Centroids[clusterIndex] - currentdata)
                cluster = clusterIndex
        Clusters[cluster].append(currentdata)
    Centroids = []
    for cluster in Clusters:
        Centroids.append(np.mean([x for x in cluster], axis=0))
    return tuple(Centroids), tuple(Clusters)


# Find if the centroids have changed
def differentCentroids(oldCentroids, Centroids):
    for index in range(len(oldCentroids)):
        if not np.array_equal(np.array(oldCentroids) , np.array(Centroids)):
            return True
    return False

# Assign labels to elements in clusters
def assignLabel(Clusters):
    resultList = []
    for clusterIndex in range(len(Clusters)):
        for ele in Clusters[clusterIndex]:
            resultList.append(tuple([ele,clusterIndex]))
    return resultList



def main():
    # Create Toy Dataset
    data, y = make_blobs(n_samples=500, centers=3, n_features=2,
                         random_state=3)
    # Number of Clusters
    k = 3
    numKMeansPerformed = 5
    clusters, result, centroids = bestKmeans(data, k, numKMeansPerformed)
    data = []
    for ele in clusters:
        data += [x for x in ele]
    index = 0
    while index < k:
        plt.scatter([data[i][0] for i in range(len(data)) if result[i] == index],
                    [data[i][1] for i in range(len(data)) if result[i] == index])
        index += 1
    for ele in centroids:
        plt.plot(ele[0], ele[1], marker='s', markersize=5, color="black")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("K-means (K=3)")
    plt.show()

if __name__ == '__main__':
     main()