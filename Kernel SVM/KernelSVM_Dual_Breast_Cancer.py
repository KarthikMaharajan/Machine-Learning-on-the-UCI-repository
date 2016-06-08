import matplotlib.pyplot as plt
import csv
import random as rnd
from random import shuffle
import numpy
import sys
from numpy import *
from numpy.linalg import inv
from scipy.spatial.distance import pdist, squareform
import scipy
from sklearn import preprocessing
numpy.set_printoptions(threshold = 'nan')

digitsData = []
partitionsMap = {}
testPartition = []
trainPartition = []
trainPartitionLabels = []
testPartitionLabels = []
kernelMatrix = []
gaussianVariance = 10
error = 0
alpha = []
K = 2
Y = []

kFoldDict = []

def main():
	dataImport()

def dataImport():
	global digitsData
	global trainPartition
	global testPartition
	global trainPartitionLabels
	global testPartitionLabels
	global kernelMatrix
	global kFoldDict
	global error

	kFoldDict.append([])
	dataFile = open('Data/Breast-Cancer-train.csv', 'rb')
	digitsData = list(csv.reader(dataFile, delimiter = ','))

	#converting read features, labels to integers
	for i in range(0,len(digitsData)):
		digitsData[i] = list(map(int, digitsData[i]))

	#partition data for K-Fold cross validation
	random.shuffle(digitsData)
	partitionData()

	for key,value in partitionsMap.iteritems():
		trainPartition = []
		trainPartitionLabels = []
		testPartitionLabels = []
		testPartition = value
		testPartitionKey = key
		#train on remaining partitions
		for key,value in partitionsMap.iteritems():
			if(key != testPartitionKey):
				for row in value:
					trainPartition.append(row)
		#extract and construct trainLabels and testLabels
		p=0
		for row in trainPartition:
			trainPartitionLabels.append([])
			trainPartitionLabels[p] = row[-1]
			p = p+1
		p=0
		for row in testPartition:
			testPartitionLabels.append([])
			testPartitionLabels[p] = row[-1]
			p = p+1
		#train on trainPartition
		computeKernelMatrix()
		for lambdaVal in frange(1,5,0.5):
			error = 0
			kernelSVM(lambdaVal)
			validate()
			kFoldDict[0].append(error)
			print "Error for ",lambdaVal," ",error, " %:",(float(error)/float(len(testPartition)))
	test(1)

def test(minlambdaVal):
	global testPartition
	global testPartitionLabels
	global error 

	testFile = open('Data/Breast-Cancer-test.csv', 'rb')
	testData = list(csv.reader(testFile, delimiter = ','))

	#converting read features, labels to integers
	for i in range(0,len(testData)):
		testData[i] = list(map(int, testData[i]))

	error = 0
	kernelSVM(minlambdaVal)
	testMatrix = numpy.delete(testPartition,9,1)

	for i in range(0,len(testMatrix)):
		validateVector(testMatrix[i],testPartitionLabels[i][0])
	print "Test error ",error, " ",(float(error)/float(len(testMatrix)))

def validate():
	global testPartition
	global testPartitionLabels
	global trainPartition

	trainPartition = numpy.array(trainPartition)

	testPartition = numpy.array(testPartition)
	testMatrix = testPartition
	testMatrix = numpy.delete(testMatrix,9,1)
	for i in range(0,len(testMatrix)):
		validateVector(testMatrix[i],testPartitionLabels[i][0])

def validateVector(testVector,label):
	#validate with testPartition and testPartitionLabels on calculated alpha
	global testPartition
	global testPartitionLabels
	global alpha
	global trainPartition
	global Y
	global trainPartitionLabels
	global error
	
	calculatedLabels = numpy.zeros((1,K))

	trainMatrix = numpy.delete(trainPartition,9,1)
	testMatrix = numpy.delete(testPartition,9,1)

	for i in range(0,len(trainMatrix)):
		calculatedLabels = calculatedLabels+numpy.dot((numpy.dot(alpha[i],trainPartitionLabels[i][0])),computeKernelFunction(testVector,trainMatrix[i]))

	maxValue = calculatedLabels[0][0]
	maxLabel = 2
	for i in range(1,K):
		if(calculatedLabels[0][i] > maxValue):
			maxValue = calculatedLabels[0][i]
			maxLabel = 4
	if (label != maxLabel):
		error = error+1

def computeKernelMatrix():
	global kernelMatrix
	global trainPartition
	kernelMatrix = []
	
	trainMatrix = numpy.array(trainPartition)
	#removing labels from trainmatrix

	trainMatrix = numpy.delete(trainMatrix,9,1)

	#compute Kernel Matrix for trainPartition elements
	for i in range (0,len(trainMatrix)):
		kernelMatrix.append([])
		for j in range (0,len(trainMatrix)):
			kernelMatrix[i].append(computeKernelFunction(trainMatrix[i],trainMatrix[j]))

def computeKernelFunction(featureVectorOne,featureVectorTwo):
	
	sum = pow((numpy.linalg.norm(numpy.subtract(featureVectorOne,featureVectorTwo))),2)
	expTerm = -(sum/2*gaussianVariance)
	expValue = exp(expTerm)
	return expValue

def kernelSVM(lambdaVal):
	global trainPartitionLabels
	global testPartitionLabels
	global kernelMatrix
	global alpha
	global Y
	Y = []

	kernelMatrix = numpy.array(kernelMatrix)
	trainPartitionLabels = numpy.array(trainPartitionLabels)
	trainPartitionLabels = trainPartitionLabels.reshape((-1,1))
	
	testPartitionLabels = numpy.array(testPartitionLabels)
	testPartitionLabels = testPartitionLabels.reshape((-1,1))

	#Computing trainLables Y Matrix
	for i in range(0,len(trainPartitionLabels)):
		Y.append([])
		for j in range(0,K):
			if(trainPartitionLabels[i][0] == j):
				Y[i].append(1)
			else:
				Y[i].append(0)
	Y = numpy.array(Y)

	calculateAlpha(lambdaVal)

def calculateAlpha(lambdaVal):
	global trainPartitionLabels
	global testPartitionLabels
	global trainPartition
	global K
	global kernelMatrix
	global alpha
	global Y

	identityMatrix = numpy.identity(len(trainPartition))
	alpha = numpy.add(kernelMatrix,lambdaVal*identityMatrix)
	alpha = numpy.linalg.inv(alpha)
	alpha = numpy.dot(alpha,Y)

def partitionData():
	global digitsData

	mapIndex = 1;
	size = 174;
	counter = 2;
	firstPartition = []
	for i in range(0,174):
		firstPartition.append([])
		firstPartition[i] = digitsData[i]
	partitionsMap[mapIndex] = firstPartition;
	mapIndex = mapIndex+1;
	counter = counter-1;

	while (counter>0):
		k = 0;
		partition = []
		for j in range(0, 175):
			partition.append([])
			partition[k] = digitsData[i+j];
			k = k+1;
		partitionsMap[mapIndex] = partition
		mapIndex = mapIndex+1;
		i = i + 175;
		counter = counter-1;

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

if __name__ == "__main__": main()

