"""
============================
SVR 선박성능학습 (잘됨)
============================
"""
import sys
import shipData as sd
import graph as graph
import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt


# 데이터 관련 입력 파라미터 설정
trainCount = 8000
testCount = 300
callSign = "3ffb8"
startTrainDate = '2016-01-01'
endTrainDate = '2016-06-01'
startTestDate = '2016-06-01'
endTestDate = '2016-08-01'


# svm 파라미터 설정
svmKernel = "rbf"
svmgamma = "auto"
svmC = 5000
svmEpsilon = 10
svmDegree = 3

def SVRProcess(svmC, svmDegree, svmEpsilon, svmgamma, svmKernel, trainFeatures, trainLabel, testFeatures):
    # SVR 학습 및 예측
    clf = svm.SVR(C=svmC, cache_size=2000, coef0=0.0, degree=svmDegree, epsilon=svmEpsilon, gamma=svmgamma, kernel=svmKernel, max_iter=-1, shrinking=True, tol=0.001, verbose=False)
    clf = Pipeline([('norm', StandardScaler()), ('clf', clf)])
    clf.fit(trainFeatures, trainLabel)
    y_hat = clf.predict(testFeatures)
    return y_hat


def fit_func(x, a, b):
    return a * pow(x, b)

def doSVR(callSign):
    
    # 학습 데이터 셋
    trainData = sd.shipDataQuery(callSign, startTrainDate, endTrainDate, trainCount)
    # 테스트 데이터 셋
    testData = sd.shipDataQuery(callSign, startTestDate, endTestDate, testCount)
    if trainData == 0 or testData == 0:
        return

    preticeData = SVRProcess(svmC, svmDegree, svmEpsilon, svmgamma, svmKernel, trainData[0], trainData[1], testData[0])
    print("r2 : ",r2_score(preticeData, testData[1]))
    print("rmse : ",sqrt(mean_squared_error(preticeData, testData[1])))
    graph.saveGraphSVM(callSign, 0, testData[0], testData[1], preticeData, startTrainDate, endTrainDate, trainCount, startTestDate, endTestDate, testCount, svmKernel, svmC, svmgamma, svmEpsilon)

doSVR(callSign)