# James Wynne - Data Minning Assignment 2 - 200901150

import numpy as np
import matplotlib.pyplot as plt
import math
import random

# read data files
animals = open("animals", "r").read()
fruits = open("fruits", "r").read()
countries = open("countries", "r").read()
veggies = open("veggies", "r").read()

animals = animals.splitlines()
fruits = fruits.splitlines()
countries = countries.splitlines()
veggies = veggies.splitlines()


# create array of data. Each instance has 300 features, the number of instances in each file varies
data = np.zeros((0, 300))
# create an array of class labels
CLASSLABLES = np.array([0]*len(animals))
CLASSLABLES = np.append(CLASSLABLES, np.array([1]*len(fruits)))
CLASSLABLES = np.append(CLASSLABLES, np.array([2]*len(countries)))
CLASSLABLES = np.append(CLASSLABLES, np.array([3]*len(veggies)))


def formatData(data):
    # create a numpy array (num instances x features=300)
    featureVectors = np.zeros((len(data), 300))
    for i in range(len(data)):
        instance = data[i].split(" ")
        for j in range(0, 300):
            featureVectors[i][j] = instance[j + 1]
    return featureVectors

for item in [animals, fruits, countries, veggies]:
    data = np.concatenate((data, formatData(item)), axis=0)


NUM_INSTANCES = data.shape[0]
NUM_CLASSES = 4


# an array of the min/max values in each dimension to be used for intialising k
mins = data.min(0)
maxs = data.max(0)
dataRange = maxs - mins


# distance measuring functions -------------
def euclidean(instance, mean):
    distance = np.linalg.norm(instance - mean)
    return distance

def manhattan(instance, mean):
    distance = np.linalg.norm(instance - mean, 1)
    return distance

def cosineSimilarity(instance, mean):
    similarity = np.dot(instance, mean) / (np.linalg.norm(instance) * np.linalg.norm(mean))
    return 1 - similarity


# clustering algorithm -------------------
def kMeansClustering(data, k, distMeasure=euclidean):
    # pick instances in the data to be the initial means
    means = np.zeros((k, 300))
    instancesToBeMeans = []
    for i in range(k):
        picked = False
        choice = 0
        while(not picked):
            choice = random.randint(0, NUM_INSTANCES - 1)
            if choice not in instancesToBeMeans:
                instancesToBeMeans.append(choice)
                picked = True
        means[i] = data[choice]


    # loop between assigning instances to nearest center & moving center to the mean
    noChange = False
    numIter = 0
    assignedCluster = np.zeros((NUM_INSTANCES), dtype=int)

    while(not noChange and numIter < 100):
        numIter += 1
        # assign instances to nearest mean
        assignedCluster_B = assignToMean(k, means, distMeasure)
        # move center to mean
        means = newMean(assignedCluster_B, k, means)
        # check for any change in assignments
        if np.array_equal(assignedCluster, assignedCluster_B ): noChange = True
        assignedCluster = assignedCluster_B

    # evaluate clustering
    clusterTable = makeClusterTable(assignedCluster, k)
    contingencyTable = calculateContingencyTable(clusterTable)
    # calculate precision recall and F-measure
    PRF = calculatePRF(contingencyTable)
    return PRF


# funtions for clustering algorithm -------------------
def assignToMean(k, means, distMeasure):
    assign = np.zeros((NUM_INSTANCES), dtype=int)
    for i in range(NUM_INSTANCES):
        distToMean = np.zeros(k)
        for j in range(k):
            distToMean[j] = distMeasure(data[i], means[j])
        assign[i] = np.argmin(distToMean)
    return assign

def newMean(assignedCluster, k, means):
    sumPoints = np.zeros((k, 300))
    numPoints = np.zeros(k, dtype=int)
    for i in range(NUM_INSTANCES):
        sumPoints[assignedCluster[i]] += data[i]
        numPoints[assignedCluster[i]] += 1
    for i in range(k):
        # at this point, if nothing is assigned to a particular mean should I randomly move it?
        if numPoints[i] != 0:
            means[i] = sumPoints[i] / numPoints[i]
    return means

def numPairs(n):
    return math.factorial(n)/(math.factorial(2) * math.factorial(n-2))

def makeClusterTable(assignedCluster, k):
    clusterTable = np.zeros((k, NUM_CLASSES), dtype=int)
    for i in range(NUM_INSTANCES):
        clusterTable[assignedCluster[i]][CLASSLABLES[i]] += 1
    return clusterTable

def calculateCombinations(x):
    if len(x) <= 1:
        return 0
    elif len(x) == 2:
        return x[0] * x[1]
    else:
        total = 0
        for i in range(1, len(x)):
            total += x[0] * x[i]
        return total + calculateCombinations(x[1:])

def calculateContingencyTable(clusterTable):
    contingencyTable = {'TP':0, 'FN':0,'FP':0,'TN':0}
    # TP + FP  is sum the possible pair combinations within each cluster
    TP_FP = 0
    for col in np.sum(clusterTable, axis=1):
        if col >= 2:
            TP_FP += numPairs(col)
    # calculate TPs
    for i in range(clusterTable.shape[0]):
        for j in range(clusterTable.shape[1]):
            if clusterTable[i][j] > 1:
                contingencyTable['TP'] += numPairs(clusterTable[i][j])
    contingencyTable['FP'] = TP_FP - contingencyTable['TP']
    for i in range(clusterTable.shape[1]):
        contingencyTable['FN'] += calculateCombinations(clusterTable[...,i])
    TN_FN = calculateCombinations(np.sum(clusterTable, axis=1))
    contingencyTable['TN'] = TN_FN - contingencyTable['FN']
    return contingencyTable

def calculatePRF(contingencyTable):
    precision = contingencyTable['TP']/(contingencyTable['TP'] + contingencyTable['FP'])
    recall = contingencyTable['TP']/(contingencyTable['TP'] + contingencyTable['FN'])
    fscore = (2 * precision * recall)/(precision + recall)
    return precision, recall, fscore

def l2normalize(data):
    normalizedDate = np.zeros(data.shape)
    for i in range(data.shape[0]):
        normalizedDate[i] = data[i] / np.linalg.norm(data[i])
    return normalizedDate




# run experiment --------------------------
def runClustering(data, distMeasure, numRuns):
    results = np.zeros((10,3))
    # run clustering for k = (1,10)
    for i in range(1, 11):
        runs = np.zeros((numRuns, 3))
        # run each clustering multiple times and average results
        for j in range(numRuns):
            runs[j] = kMeansClustering(data, i, distMeasure)
        results[i - 1] = np.mean(runs, axis=0)
    return results


def plotResults(results, figureNumber, title):

    plt.figure(figureNumber)
    plt.title(title)
    plt.xlim((0.1,10.9))

    xAxis = list(range(1,11))
    w = 0.2
    # ax = plt.subplot(111)
    plt.bar([t - w for t in xAxis], results[...,0], width=w, color='b', align='center', label="Precision")
    plt.bar(xAxis, results[...,1], width=w, color='r', align='center', label="Recall")
    plt.bar([t + w for t in xAxis], results[...,2], width=w, color='g', align='center', label="F-Score")
    # plt.bar(range(1,11), results[...,0], 'ro', label="Precision")
    # plt.bar(range(1,11), results[...,1], 'b--', label="Recall")
    # plt.bar(range(1,11), results[...,2], 'g^', label="F-Score")
    plt.xlabel('number of clusters')
    plt.legend()

NUM_RUNS = 10

# (2) Unnormalized data with Euclidean distance
print("")
print('Clustering with unnormalized data, Euclidean distance')
results_2 = runClustering(data, euclidean, NUM_RUNS)
plotResults(results_2, 1, '(2) Unnormalized data with Euclidean distance')

# (3) L2 normalized data, Euclidean distance
print("")
print('Clustering with L2 normalized data, Euclidean distance')
results_3 = runClustering(l2normalize(data), euclidean, NUM_RUNS)
plotResults(results_3, 2, '(3) L2 normalized data, Euclidean distance')

# (4) Unnormalized data, Manhattan distance
print("")
print('Clustering with unnormalized data, Manhattan distance')
results_4 = runClustering(data, manhattan, NUM_RUNS)
plotResults(results_4, 3, '(4) Unnormalized data, Manhattan distance')

# (5) L2 normalized data, Manhattan distance
print("")
print('Clustering with L2 normalized data, Manhattan distance')
results_5 = runClustering(l2normalize(data), manhattan, NUM_RUNS)
plotResults(results_5, 4, '(5) L2 normalized data, Manhattan distance')

# (6) Unnormalized data, cosine similarity
print("")
print('Clustering with unnormalized data, cosine similarity')
results_6 = runClustering(data, cosineSimilarity, NUM_RUNS)
plotResults(results_6, 5, '(6) - Unnormalized data, cosine similarity')

plt.show()
