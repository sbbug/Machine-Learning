'''
输入:输入一副带有单个数字的图片(32*32)

输出:该图片二值化后的向量
'''

import cv2 as cv
from numpy import *


#全局阈值
def threshold(filename):

    img = cv.imread(filename, cv.IMREAD_COLOR)
    for i in range(32):
        for j in range(32):
            print(img[i][j], end=" ")
        print('')
    imgVec = zeros((1,1024))
    print(imgVec)
    # 首先变为灰度图
    gray = cv.cvtColor(img , cv.COLOR_BGR2GRAY)
    for i in range(32):
        for j in range(32):
            print(gray[i][j], end=" ")
        print('')
    # cv.THRESH_BINARY |cv.THRESH_OTSU 根据THRESH_OTSU阈值进行二值化  cv.THRESH_BINARY_INV(黑白调换)
    ret , binary = cv.threshold( gray , 0, 255 , cv.THRESH_BINARY |cv.THRESH_OTSU)
    #上面的0 为阈值 ，当cv.THRESH_OTSU 不设置则 0 生效
    #ret 阈值 ， binary二值化图像

    print("阈值：", ret)
    for i in range(32):
        for j in range(32):
            print(binary[i][j],end=" ")
        print('')
    #将图像数据0-1化
    for i in range(32):
        for j in range(32):
            if binary[i][j]==255:
                binary[i][j]=0
            else:
                binary[i][j] = 1
    #将二维图像数据转换为一维向量
    n=0
    for i in range(32):
        for j in range(32):
            imgVec[0][n] = binary[i][j]
            n=n+1
            print(binary[i][j],end=" ")
        print('')

    return imgVec




