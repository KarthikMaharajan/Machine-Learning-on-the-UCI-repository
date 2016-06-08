import matplotlib.pyplot as plt
import csv
import random as rnd
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
gaussianVariance = 0.0001
alpha = []
Z = []
C = 2
theta = []
A = []
K = 10
Y = []
epsilon = 0.000001
error = 0


def main():
	dataImport()

def dataImport():
	global digitsData
	global trainPartition
	global testPartition
	global trainPartitionLabels
	global testPartitionLabels
	global kernelMatrix

	dataFile = open('Data/optdigits.tra', 'rb')
	digitsData = list(csv.reader(dataFile, delimiter = ','))

	#converting read features, labels to integers
	for i in range(0,len(digitsData)):
		digitsData[i] = list(map(int, digitsData[i]))

	#partition data for K-Fold cross validation
	partitionData()

	for key,value in partitionsMap.iteritems():
		trainPartition = []
		trainPartitionLables = []
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
		kernelSVM()
		validate()
		break

def validate():
	global testPartition
	global testPartitionLabels
	global trainPartition

	trainPartition = numpy.array(trainPartition)

	testPartition = numpy.array(testPartition)
	testMatrix = testPartition
	testMatrix = numpy.delete(testMatrix,64,1)
	testMatrix = preprocessing.scale(testMatrix)
	for i in range(0,len(testMatrix)):
		validateVector(testMatrix[i],testPartitionLabels[i][0])
	print "error",error


def validateVector(testVector,label):
	#validate with testPartition and testPartitionLabels on calculated alpha
	global testPartition
	global testPartitionLabels
	global alpha
	global trainPartition
	global error

	trainMatrix = numpy.delete(trainPartition,64,1)

	#transposing alpha
	alphaTran = numpy.transpose(alpha)
	for i in range(0,K):
		for j in range(0,len(trainMatrix)):
			alphaTran[i][j] = alphaTran[i][j]*(computeKernelFunction(trainMatrix[j],testVector))
			if(j != 0):
				alphaTran[i][0] = alphaTran[i][0] + alphaTran[i][j]

	maxValue = alphaTran[0][0]
	maxLabel = 0
	for i in range(1,K):
		print i,alphaTran[i][0],maxValue
		if(alphaTran[i][0] > maxValue):
			maxValue = alphaTran[i][0]
			maxLabel = i
	if (label != maxLabel):
		error = error+1

def computeKernelMatrix():
	global kernelMatrix
	global trainPartition
	
	trainMatrix = numpy.array(trainPartition)
	#removing labels from trainmatrix

	trainMatrix = numpy.delete(trainMatrix,64,1)
	trainMatrix = preprocessing.scale(trainMatrix)
	
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

def kernelSVM():
	global trainPartitionLabels
	global testPartitionLabels
	global kernelMatrix
	global alpha

	kernelMatrix = numpy.array(kernelMatrix)
	trainPartitionLabels = numpy.array(trainPartitionLabels)
	trainPartitionLabels = trainPartitionLabels.reshape((-1,1))
	
	testPartitionLabels = numpy.array(testPartitionLabels)
	testPartitionLabels = testPartitionLabels.reshape((-1,1))

	initialiseVectors()

	alphaNew = alpha
	delta = 100
	calculateAlpha()
	calculateZ()
	
	while(delta > 0.5):
		calculateAlpha()
		calculateZ()
		delta = numpy.linalg.norm(numpy.subtract(alpha,alphaNew))
		alphaNew = alpha
		print "delta", delta

def calculateZ():
	global A
	global theta
	global K
	global kernelMatrix
	global alpha
	global Z

	alphaTerm = numpy.zeros((len(trainPartition), 1))

	for i in range(0, K):
		for j in range(0, K):
			if i != j:
				for k in range(0, len(trainPartition)):
					alphaTerm[k] = alphaTerm[k] + alpha[k][i] - alpha[k][j] 

	Z = numpy.dot(K*(K-1), (A + theta))
	Z = Z - numpy.dot(kernelMatrix, alphaTerm)

def calculateAlpha():
	global trainPartitionLabels
	global testPartitionLabels
	global Z
	global trainPartition
	global K
	global kernelMatrix
	global alpha
	alphaSum = numpy.zeros((len(trainPartition),1))

	#Computing ZTran vector
	for i in range(0,len(trainPartition)):
		if Z[i] == 0:
			Z[i] = epsilon;
		Z[i] = 1/Z[i]
	Ztran = Z.reshape((1,-1)) #of dimension 1xN
	
	#Computing alpha matrix - updating weights for one class at a time
	for k in range(0,K):
		#calculating Y_k
		Y = []
		Y.append([])
		for i in range(0,len(trainPartitionLabels)):
			if(trainPartitionLabels[i] == k):
				Y[0].append(1)
			else:
				Y[0].append(0)
		Y = numpy.array(Y)
		Y = Y.reshape((-1,1))
		
		#calculating alpha_k
		mul = numpy.dot(Y,Ztran)
		mul = numpy.dot(mul,kernelMatrix)
		mul = numpy.dot(mul,kernelMatrix)
		mul = (C*(k-1)/2)*mul
		firstTerm = numpy.add(kernelMatrix,mul)

		invFirstTerm = inv(firstTerm)

		mulFirst = C/2*Y
		mulFirst = numpy.dot(mulFirst,Ztran)
		mulFirst = numpy.dot(mulFirst,kernelMatrix)

		for i in range(0,len(trainPartition)):
			for j in range(0,K):
				if(j != k):
					alphaSum[i] = alphaSum[i] + alpha[i][j]

		alphaSumTran = numpy.transpose(alphaSum)
		sumThree = numpy.add(Z,theta)
		mulSecond = (k-1)*sumThree
		mulSecond = numpy.add(mulSecond,numpy.dot(alphaSumTran,kernelMatrix))
		secondTerm = numpy.dot(mulFirst,mulSecond)

		finalAlpha = numpy.dot(invFirstTerm,secondTerm)
		for i in range(0,len(trainPartition)):
			temp = alpha[i][k]
			alpha[i][k] = finalAlpha[i][0]

def initialiseVectors():
	global Z
	global alpha
	global theta
	global A
	global K

	#initialising aplha(NxK) to 1
	alpha = numpy.ones((len(trainPartition),K))
	Z = numpy.ones((len(trainPartition),1))
	theta = numpy.ones((len(trainPartition),1))
	A = numpy.ones((len(trainPartition),1))

def partitionData():
	global digitsData

	mapIndex = 1;
	size = 382;
	counter = 10;
	firstPartition = []
	for i in range(0,385):
		firstPartition.append([])
		firstPartition[i] = digitsData[i]
	partitionsMap[mapIndex] = firstPartition;
	mapIndex = mapIndex+1;
	counter = counter-1;

	while (counter>0):
		k = 0;
		partition = []
		for j in range(0, 382):
			partition.append([])
			partition[k] = digitsData[i+j];
			k = k+1;
		partitionsMap[mapIndex] = partition
		mapIndex = mapIndex+1;
		i = i + 382;
		counter = counter-1;

if __name__ == "__main__": main()

