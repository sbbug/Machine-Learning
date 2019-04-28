# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 12:09:57 2018

@author: sun
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
# 计算均值
def meanX(dataX):
    return np.mean(dataX,axis=0)


# 使用主成分分析获取投影矩阵
def pca(XMat,k):
    
    print(XMat.shape)
    
    average = meanX(XMat)
    
    m,n = np.shape(XMat)

    avgs = np.tile(average,(m,1))
    
    # 对样本进行中心化
    data_adjust = XMat-avgs
    
    # 计算样本协方差矩阵
    convX = np.cov(data_adjust.T)
    print(convX.shape)
    # 计算特征值与特征向量
    featValue,featVec = np.linalg.eig(convX)
    
    # 对特征值从大到小排序
    index = np.argsort(-featValue)
    
    
    if k>n:
        print("k必须低于特征值个数")
        return
    
    else:
        
        # 选择前k个特征向量
        selectVec = np.matrix(featVec.T[index[:k]])
        
        print(selectVec.shape)
        
        # 将数据进行投影
        # finalData = XMat*selectVec.T
        
        # reconData = (finalData*selectVec) + average
       
    return selectVec.T  # 返回投影矩阵

# 根据投影矩阵将数据进行投影
def getPro(W,X):

    return  X*W   

# 数据集载入函数

def loadData(datafile):

    return np.array(pd.read_csv(datafile,sep=",",header=-1)).astype(np.float)    
    
    
def process(datafile):    
     
    # 获取样本数据矩阵(xi,yi)
    XMat = loadData(datafile)
    
    # 获取y值
    YVec = XMat[:,60]
    
    # 提取举证的x值
    XMat = np.delete(XMat,60,axis=1)
   
    return XMat,YVec

# 计算向量距离
def distance(vec1,vec2):

    return np.linalg.norm(vec1-vec2)
    
    
def findMin(train_x,vec):
    
    min_dist = 1000.0
    
    index = -1
    
    for i in range(train_x.shape[0]):
          
          dis = distance(train_x[i],vec)
               
          if dis < min_dist:
              min_dist = dis
              index = i
    
    return index



if __name__ == "__main__":
    
    K = 30
    
    starttime = datetime.datetime.now()
    
    datafile1 = "C:\\Users\\sun\\Desktop\\ML_homework\\data\\splice-train.txt"
    
    datafile2 = "C:\\Users\\sun\\Desktop\\ML_homework\\data\\splice-test.txt"
    
    # 获取处理后的训练数据
    train_x,train_y = process(datafile1)
    
    # 获取处理后的测试数据
    test_x,test_y = process(datafile2)
    
    # 获取投影矩阵
    W = pca(train_x,K)

    # 分别将训练数据与测试数据降维
    train_x = getPro(W,train_x)
    
    test_x = getPro(W,test_x)
    
    print(train_x.shape)
    print(test_x.shape)
    
  
    #正确数量
    n = 0
    
    #测试样本数目
    N = test_x.shape[0]
    
    for i in range(N):
        
        print("test=====enter"+str(i))
        
        index = findMin(train_x,test_x[i])
        
        if train_y[index]==test_y[i]:
            
            n=n+1
        
    
    print("positive======="+str(n))
    
    print("K========="+str(K))
    
    print("准确率====="+str(round(float(n/N),6)))

    endtime = datetime.datetime.now()
    
    print ("用时"+str((endtime - starttime).seconds))
