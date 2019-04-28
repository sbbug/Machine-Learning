
#KNN实现手写体图片识别

from __future__ import print_function
from numpy import *
import operator
from os import listdir
from collections import Counter
from functions import threshold

def createDataSet():

    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']

    return group,labels

def classify0(inX,dataSet,labels,k):

    #距离计算  dataSet里的属性shape是(行，列)的一个元组
    dataSetSize = dataSet.shape[0]

    diffMat = tile(inX,(dataSetSize,1))-dataSet

    #取平方
    sqDiffMat = diffMat**2
    #将矩阵每一行相加
    sqDistances = sqDiffMat.sum(axis=1)

    #开方
    distances = sqDistances**0.5

    #排序
    sortedDistIndicies = distances.argsort()

    print("=================")
    print(sortedDistIndicies)



    classCount = {}
    for i in range(k):

        voteIlabel = labels[sortedDistIndicies[i]]
        print(voteIlabel)
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1


    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)

    print("=================")
    print(sortedClassCount)

    return sortedClassCount[0][0]


def test1():

    group,labels = createDataSet()
    print(str(group))
    print(str(labels))
    print(classify0([0.1,0.1],group,labels,3))


def file2matrix(filename):

    fr = open(filename)

    numberOfLines = len(fr.readlines())

    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    fr = open(filename)

    index = 0
    for line in fr.readlines():

        line = line.strip()

        listFromLine = line.split('\t')

        returnMat[index,:] = listFromLine[0:3]

        classLabelVector.append(int(listFromLine[-1]))

        index+=1

    return returnMat,classLabelVector

def autoNorm(dataSet):

    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)

    ranges = maxVals-minVals

    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]

    normDataSet = dataSet-title(minVals,(m,1))

    normDataSet = normDataSet / title(ranges,(m,1))

    return normDataSet,ranges,minVals

def datingClassTest():

    #测试范围，一部分作为测试，一部分作为样本
    hoRatio = 0.1
    #从文件中读取数据
    datingDataMat,datingLabels = file2matrix('./datingTestSet2.txt')
    #归一化数据
    normMat,ranges,minVals = autoNorm(datingDataMat)

    m = normMat.shape[0]

    numTestVecs = int(m * hoRatio)

    print('numTestVecs=',numTestVecs)

    errorCount = 0.0

    for i in range(numTestVecs):

        classifierResult = classify0(normMat[i:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)

        print("the classifier came back with:%d,the real answer is:%d"%(classifierResult,datingLabels[i]))

        if (classifierResult != datingLabels[i]):errorCount+=1

    print("the total error rate is:%f" %(errorCount/float(numTestVecs)))

#将图像文本数据转换为向量
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)

    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):

            returnVect[0,32*i+j] = int(lineStr[j])

    return returnVect

def handwritingClassTest():

    #1.导入数据
    hwLabels = []
    trainingFileList = listdir('./data/trainingDigits')

    m = len(trainingFileList)
    trainingMat = zeros((m,1024))

    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)

        #将32*32矩阵转换为为1*1024的矩阵
        trainingMat[i,:] = img2vector('./data/trainingDigits/'+fileNameStr)

    #2.导入测试数据

    # testFileList = listdir('./data/testDigits')
    # errorCount = 0.0
    #
    # mTest = len(testFileList)
    #
    # for i in range(mTest):
    #     fileNameStr = testFileList[i]
    #     fileStr = fileNameStr.split('.')[0]
    #     classNumStr = int(fileStr.split('_')[0])
    #     vectorUnderTest = img2vector('./data/testDigits/'+fileNameStr)
    #     classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
    #     print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
    #
    #     if(classifierResult != classNumStr): errorCount+=1.0
    #
    # print("\nthe total number od errors is "+str(errorCount))
    # print("\nthe total error rate is"+str(errorCount/float(mTest)))

    #当个测试

    vectorUnderTest = threshold("F:\\6.png")
    classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 6)
    print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, 6))

    #从文件读取做单个测试
    # vectorUnderTest = img2vector('./data/testDigits/9_66.txt')
    # classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
    # print("the classifier came back with: "+str(classifierResult))

if __name__ == '__main__':

    handwritingClassTest()