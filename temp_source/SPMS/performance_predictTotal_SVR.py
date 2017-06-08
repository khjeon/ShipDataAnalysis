# -*- coding: utf-8 -*-
import pymssql
import savefig as gs
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from numpy  import array
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from multiprocessing import Process, Queue

# 그래프 한글 출력을 위한 폰트 설정
font_location = "C:/Windows/Fonts/malgunsl.ttf"
font_name = fm.FontProperties(fname=font_location).get_name()
matplotlib.rc('font', family=font_name)  # 한글 폰트 설정

# 데이터 관련 입력 파라미터 설정
# shipCallsigns = ["3FFB8", "3EWB4", "3ETU2"]
shipCallsigns = ["3EEH2", "3EEK8", "3EIG9", "3ENU2", "3EPV8", "3EQZ9", "3ETG3", "3ETU2", "3EWB4", "3EXW2", "3EZL9", "3FFB8", "3FFH3", "3FQI", "3FQI8", "3FUG9", "3FUP4", "3FXC5", "H3OR", "V7AH6", "V7AQ5", "V7BB5", "V7BV5", "V7VX2", "V7YY5", "V7ZU6"]
trainCount = 8000
testCount = 300

# svm 파라미터 설정
svmKernel = "rbf"
svmgamma = "auto"
svmC = 4000
svmEpsilon = 0.01
svmDegree = 3

# 기타 설정
saveTime = datetime.datetime.now().strftime('%Y%m%d_%H%M')


def shipDateQuery(callsign, trainMonth, gapDate, testMonth):
    # 해당선박의 학습, 예측 기간을 자동으로 불러온다. 저장된 마지막 날짜 기준에서 ..
    date = []
    conn = pymssql.connect(server='183.103.118.217:21000', user='sa', password='@120bal@', database='SHIP_DB_EARTH')
    queryDate = conn.cursor()
    queryDate.execute("SELECT TOP 1 [TIME_STAMP]  FROM [SHIP_DB_EARTH].[dbo].[SAILING_DATA] WHERE CALLSIGN ='"+callsign+"' AND AT_SEA = 1 AND PRIMARY_DATA_CHECK = 1 AND ERROR_MEFOFLOW_DATA = 1 order by TIME_STAMP desc ")
    
    for raw in queryDate:
        testEndDate = raw[0]

    testBeginDate = testEndDate - datetime.timedelta(days = testMonth * 30)
    trainEndDate =  testEndDate - datetime.timedelta(days = gapDate * 30 + testMonth * 30)
    trainbegineDate = testEndDate - datetime.timedelta(days = trainMonth * 30 + gapDate * 30 + testMonth * 30)
    date.append(trainbegineDate.strftime('%Y-%m-%d'))
    date.append(trainEndDate.strftime('%Y-%m-%d'))
    date.append(testBeginDate.strftime('%Y-%m-%d'))
    date.append(testEndDate.strftime('%Y-%m-%d'))
    conn.close()
    return date


# train 데이터 set 만들기
def shipDataQuery(callSign, beginDate, EndDate, dataSplitCount):
    conn = pymssql.connect(server='183.103.118.217:21000', user='sa', password='@120bal@', database='SHIP_DB_EARTH')
    queryDataSet = conn.cursor()
    queryDataSet.execute("SELECT [SPEED_VG], [SHAFT_REV], [DRAFT_FORE], [DRAFT_AFT], [BHP_BY_FOC], [REL_WIND_SPEED], [REL_WIND_DIR], [CURRNET_SPEED_REL], [HTSGW], [TIME_STAMP], [SLIP]  FROM [SHIP_DB_EARTH].[dbo].[SAILING_DATA] WHERE CALLSIGN ='"+callSign+"' AND TIME_STAMP > '"  + beginDate + "' AND TIME_STAMP <  '"+ EndDate +"' AND AT_SEA = 1 AND PRIMARY_DATA_CHECK = 1 AND ERROR_MEFOFLOW_DATA = 1 AND ERROR_SHAFTREV_DATA = 1")

    features = []
    label = []
    dataSet = []

    # features와 label 설정
    for item in queryDataSet:
        draft_mid = (item[2] + item[3]) / 2
        features.append([item[0],item[1], item[2], item[3], draft_mid, item[5], item[6], item[7], item[10]])
        label.append(item[4])
    rawDataCount = len(features)
    dataCount =  dataSplitCount

    if len(features) == 0:
        print("error : " + callSign + " : " + str(beginDate) + " ~ " + str(EndDate))
        return 0

    dataSizeRatio = 1 - ( dataSplitCount /  len(features) )
    if  dataSplitCount == 9999 or dataSizeRatio < 0:
        dataCount =  rawDataCount
        dataSizeRatio = 0

    x_first, X_sec, y_first, y_sec = train_test_split(features, label, test_size=dataSizeRatio, random_state=10)
    dataSet.append(np.array(x_first))
    dataSet.append(np.array(y_first))
    dataSet.append(rawDataCount)
    dataSet.append(dataCount)
    conn.close()
    return dataSet
    

def SVRProcess(svmC, svmDegree, svmEpsilon, svmgamma, svmKernel, trainFeatures, trainLabel, testFeatures):

    # SVR 학습 및 예측
    clf = svm.SVR(C=svmC, cache_size=2000, coef0=0.0, degree=svmDegree, epsilon=svmEpsilon, gamma=svmgamma, kernel=svmKernel, max_iter=-1, shrinking=True, tol=0.001, verbose=False)
    clf = Pipeline([('norm', StandardScaler()), ('clf', clf)])
    clf.fit(trainFeatures, trainLabel)
    y_hat = clf.predict(testFeatures)
    return y_hat

def saveGraph(callSign, graphMode, testFeatures, testLabel, y_hat, trainBeginDate, trainEndDate, trainCount, testBeginDate, testEndDate, testCount, svmKernel, svmC, svmgamma, svmEpsilon):
    # 그래프 그리기
    if graphMode == 0:
        garo = 25
    if graphMode == 1:
        garo = testCount/12
    plt.figure(figsize=(garo,6), dpi=70)
    plt.title(callSign.upper() + " / TRAIN_DATE : " + trainBeginDate + " ~ " + trainEndDate +"(" + str(trainCount) +")" + " / PREDICT_DATE : " + testBeginDate + " ~ " + testEndDate + "(" +str(testCount) +")" + " / " + "Kernel: " + svmKernel +", "+ "C: " + str(svmC) +", " +  "Gamma: " + str(svmgamma) +", "+ "Epsilon: " + str(svmEpsilon) , fontsize=17)
    plt.plot(y_hat, 'b-', label ="PREDICT", linewidth=2)
    plt.plot(testLabel, 'r-' , label ="TEST")
    plt.xlabel("TIME")
    plt.ylabel("POWER(KW)")
    yrange = max(testLabel) * 1.1 - min(testLabel) * 0.9
    plt.ylim(min(testLabel) * 0.9,max(testLabel) * 1.1)
    plt.legend(loc='upper left', frameon=True)
    gap =  np.mean(testLabel)-np.mean(y_hat) 
    gap =  np.mean(testLabel)-np.mean(y_hat) 
    plt.text(testCount/10-7, max(testLabel)*1.05, "TEST(Mean): " + str(round(np.mean(testLabel)))+ "Kw / PREDICT(Mean): " + str(round(np.mean(y_hat)))+ "Kw / 성능차이(Mean): " + str(round(gap))+ "Kw",va='top', ha='left', fontsize=14,   bbox={'facecolor':'red', 'alpha':0.5, 'pad':5})
    if  gap < -50 and gap > -300:
        plt.text(testCount/10-7, max(testLabel) * 0.97,'*Predict가 왜 더 높을까??*.', va='top', ha='left',fontsize=12,  color='red')
    if  gap > 300 or gap < -300:
        plt.text(testCount/10-7, max(testLabel) * 0.97,'*예측이 보완이 필요함(오차가 큼)*.',va='top', ha='left', fontsize=12,  color='red')

    # 그래프 y right 축
    plt.twinx()
    plt.ylabel("SPEED(Kn) & DRAFT(M)")
    plt.tick_params(axis="y")
    plt.plot(testFeatures[:,2], 'g-', label="DRAFT")
    plt.plot(testFeatures[:,0], 'm-', label="SOG")
    plt.legend(loc='upper right', frameon=True)
    plt.ylim(3,40)
    plt.tight_layout()
    gs.save("img/"+saveTime+"/"+callSign.upper(), ext="png", close=False, verbose=True)


def doSVR(callSign):
    # 메인 컨트롤러
    date = shipDateQuery(callSign, 3, 1, 1)
    trainData = shipDataQuery(callSign, date[0], date[1], trainCount)
    testData = shipDataQuery(callSign, date[2], date[3], testCount)
    if trainData == 0 or testData == 0:
        return
    preticeData = SVRProcess(svmC, svmDegree, svmEpsilon, svmgamma, svmKernel, trainData[0], trainData[1], testData[0])
    saveGraph(callSign, 0, testData[0], testData[1], preticeData, date[0], date[1], trainCount, date[2], date[3], testCount, svmKernel, svmC, svmgamma, svmEpsilon)




# 멀티프로세싱
if __name__=='__main__':
    procs = []
    result = []
    result_proc =[]
    for callSign in shipCallsigns:
        procs.append(Process(target=doSVR(callSign)))
    
    for p in procs:
        p.start()
    
    for p in procs:
        p.join()
    