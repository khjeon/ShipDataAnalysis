"""
============================
linear regeression ANN 선박성능학습 (잘됨)
============================

@author : lab021 / 이상봉
"""
import os
import tensorflow as tf
import math
import numpy as np
import shipData as sd
import graph as graph
import savefig as gs
import matplotlib.pyplot as plt
import datetime
import csv
import pandas as pd
from math import sqrt
from sklearn import datasets, metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from pykalman import KalmanFilter
from scipy.optimize import curve_fit
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn import ensemble

import pymssql
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 데이터 부르고 stat 추가할것!!

def DataQuery(callSign, startDate, endDate, averaging, dataCount, QueryData, isShuffle, features, label):
    # # 읽어본 데이터 배열 만들기
    _dataCount = dataCount * (averaging == 0 and 1 or averaging * 6)  
    _data = sd.shipDataQuery(callSign, startDate, endDate, dataCount, QueryData, isShuffle)
    _data = _data.dropna(axis=0)
    _data = _data[(_data['SPEED_VG'] > 3) & (_data['SPEED_VG'] < 20) \
    & (_data['SPEED_LW'] > 3) & (_data['SPEED_LW'] < 20) \
    & (_data['SLIP'] > -50) & (_data['SLIP'] < 50) & (_data['DRAFT_FORE'] > 10) & (_data['DRAFT_FORE'] < 30) \
    & (_data['DRAFT_AFT'] > 10) & (_data['DRAFT_AFT'] < 30) & (_data['REL_WIND_DIR'] >= 0) & (_data['REL_WIND_DIR'] <= 360) \
    & (_data['REL_WIND_SPEED'] > -200) & (_data['REL_WIND_SPEED'] < 200) & (_data['RUDDER_ANGLE'] > -5) & (_data['RUDDER_ANGLE'] < 5) \
    & (_data['BHP_BY_FOC'] > 1000) & (_data['BHP_BY_FOC'] < 30000) & (_data['SST'] > 0) & (_data['SST'] < 40) &  (_data['UGRD'] > -10) & (_data['UGRD'] < 10) &  (_data['VGRD'] > -10) & (_data['VGRD'] < 10)  ]
    # data = data.loc[:,['SPEED_VG', 'SPEED_LW', 'SLIP', 'DRAFT_FORE', 'DRAFT_AFT', 'REL_WIND_DIR', 'REL_WIND_SPEED', 'RUDDER_ANGLE']]
    if  averaging > 0 :
        _data = _data.resample(rule=str(averaging)+'min', on='TIME_STAMP').mean()
    if  averaging == 0 :
        _data = _data.drop('TIME_STAMP', axis=1)
    _data = _data.dropna(axis=0)
    xData = np.array(_data.loc[:,features])
    yData = np.array(_data.loc[:,label])
    return xData, yData


def BasicData(callSign):
    conn = pymssql.connect(server='218.39.195.13:21000', user='sa', password='@120bal@', database='SHIP_DB_EARTH')
    item1 =  'seatrialSpeedToPowerAtBallast'
    item2 =  'seatrialSpeedToPowerAtScant'
    ballst = "SELECT * FROM [SHIP_DB].[dbo].[SHIP_COEFFICIENT] WHERE CALLSIGN ='"+callSign+"' AND CURVE_FIT_ITEM ='" + item1 + "'"
    laden = "SELECT * FROM [SHIP_DB].[dbo].[SHIP_COEFFICIENT] WHERE CALLSIGN ='"+callSign+"' AND CURVE_FIT_ITEM ='" + item2 + "'"
    ballst = pd.read_sql(ballst,conn)
    laden = pd.read_sql(laden,conn)
    ballastB = ballst.at[0,'B']
    ballastA = ballst.at[0,'A']
    ladenB = laden.at[0,'B']
    ladenA = laden.at[0,'A']
    result = []
    result.append(ballastB)
    result.append(ballastA)
    result.append(ladenB)
    result.append(ladenA)
    return result
   

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#데이터 관련 입력 파라미터 설정
averaging = 10 # 시간 이동평균 min 시간 설정
trainCount = 8000000
testCount = 9800000
isTrainShuffle = 0 # 0 : timeseries, 1 : 셔플
isTestShuffle = 0 # 0 : timeseries, 1 : 셔플
kalmanfilter = 0 # 0 : 칼만필터 smooth 미적용, 1 : 적용 
kalmansmooth = 1.5
normalization = 1 # 0 : 정규화미적용 , 1: 적용
solver = 0 # 0 : ann, 1 : gbt

callSign = "3ffb8"
# Features = ['SPEED_VG','DRAFT_FORE', 'DRAFT_AFT', 'REL_WIND_DIR', 'REL_WIND_SPEED', 'RUDDER_ANGLE', 'SHIP_HEADING', 'UGRD', 'VGRD','SST']
Features = ['SPEED_VG','SLIP']
Label = ['BHP_BY_FOC']
startTrainDate = '2016-01-01'
endTrainDate = '2016-06-01'
startTestDate = '2016-06-02'
endTestDate = '2016-11-28'
saveTime = datetime.datetime.now().strftime('%Y%m%d_%H%M')
queryData = "[TIME_STAMP], [SHAFT_POWER], [BHP_BY_FOC], [SPEED_LW],\
    [REL_WIND_DIR], [REL_WIND_SPEED], [SPEED_VG], [SHIP_HEADING], [SHAFT_REV], [DRAFT_FORE], [DRAFT_AFT], [WATER_DEPTH], [RUDDER_ANGLE], [SST],\
    [SLIP],[UGRD], [VGRD]"

# # 읽어본 데이터 배열 만들기
trainCount = trainCount * (averaging == 0 and 1 or averaging * 6)  
testCount = testCount * (averaging == 0 and 1 or averaging * 6)  
# x_train_data = sd.shipDataQuery(callSign, startTrainDate, endTrainDate, trainCount, Features, isTrainShuffle)
# y_train_data = sd.shipDataQuery(callSign, startTrainDate, endTrainDate, trainCount, Label, isTrainShuffle)
# x_test_data = sd.shipDataQuery(callSign, startTestDate, endTestDate, testCount, Features, isTestShuffle)
# y_test_data = sd.shipDataQuery(callSign, startTestDate, endTestDate, testCount, Label, isTestShuffle)


x_train_data, y_train_data = DataQuery(callSign, startTrainDate, endTrainDate, averaging, trainCount, queryData, isTrainShuffle, Features,  Label)
x_test_data, y_test_data = DataQuery(callSign, startTestDate, endTestDate, averaging, testCount, queryData, isTrainShuffle, Features,  Label)


# 데이터 정규화

if normalization == 1 :
    robust_scaler1 = RobustScaler()
    x_train_data = robust_scaler1.fit_transform(x_train_data)
    x_test_data = robust_scaler1.transform(x_test_data)

    robust_scaler2 = RobustScaler()
    y_train_data = robust_scaler2.fit_transform(y_train_data)
    y_test_data = robust_scaler2.transform(y_test_data)

print("xtrain",x_train_data[:3])
print("y_train_data",y_train_data[0:3])
print("x_test_data",x_test_data[0:3])
print("y_test_data",y_test_data[0:3])



print("Normalization Complete")

if  solver == 0 :
    NHIDDEN = 512
    STDEV = 0.5
    KMIX = 8 # number of mixtures
    NOUT = KMIX * 3 # pi, mu, stdev
    config = tf.ConfigProto(device_count = {'gpu': 0})

    oneDivSqrtTwoPI = 1 / math.sqrt(2*math.pi) # normalisation factor for gaussian, not needed.

    
    # ANN network 정의
    X = tf.placeholder(tf.float64, shape = [None,x_train_data.shape[1]], name = 'features')
    Y = tf.placeholder(tf.float64, shape = [None, 1], name = 'label')

    W = {
        "l1": tf.Variable(tf.random_normal([x_train_data.shape[1],NHIDDEN], stddev=STDEV, dtype=tf.float64)),
        "l2": tf.Variable(tf.random_normal([NHIDDEN,NOUT], stddev=STDEV, dtype=tf.float64))
    }
    b = {
        "l1": tf.Variable(tf.random_normal([1,NHIDDEN], stddev=STDEV, dtype=tf.float64)),
        "l2": tf.Variable(tf.random_normal([1,NOUT], stddev=STDEV, dtype=tf.float64))
    }


    
    def mlp(_x, _W, _b):
        l1 = tf.nn.relu(tf.matmul(_x, _W['l1']) + _b['l1'])
        return tf.matmul(l1, _W['l2']) + _b['l2']
    
    def get_mixture_coef(output):
        checknan = tf.placeholder(dtype=tf.float64, shape=[None,output.shape[0]])
        checknan = output
        out_pi = tf.placeholder(dtype=tf.float64, shape=[None,KMIX])
       
        out_sigma = tf.placeholder(dtype=tf.float64, shape=[None,KMIX])
        out_mu = tf.placeholder(dtype=tf.float64, shape=[None,KMIX])
        out_pi, out_sigma, out_mu = tf.split(output,3,1)
        max_pi = tf.reduce_max(out_pi, 1, keep_dims=True)
        out_pi = tf.subtract(out_pi, max_pi)
        out_pi = tf.exp(out_pi)
        
        normalize_pi = tf.reciprocal(tf.reduce_sum(out_pi, 1, keep_dims=True))
        out_pi = tf.multiply(normalize_pi, out_pi)
        out_sigma = tf.exp(out_sigma)
       
        return out_pi, out_sigma, out_mu, checknan

    def tf_normal(y, mu, sigma):
        result = tf.subtract(y, mu)
        result = tf.multiply(result,tf.reciprocal(sigma))
        result = -tf.square(result)/2
        return tf.multiply(tf.exp(result),tf.reciprocal(sigma))*oneDivSqrtTwoPI

    def get_lossfunc(out_pi, out_sigma, out_mu, y):
        result = tf_normal(y, out_mu, out_sigma)
        result = tf.multiply(result, out_pi)
        result = tf.reduce_sum(result, 1, keep_dims=True)
        result = -tf.log(result)
        result = tf.reduce_mean(result)
        return result

    def get_pi_idx(x, pdf):
        N = pdf.size
        accumulate = 0
        result = np.argmax(pdf)
        return result
        # for i in range(0, N):
        #     accumulate += pdf[i]
        #     if (accumulate >= x):
        #         return i
        # # print ('error with sampling ensemble')
        # return -1


    def generate_ensemble(out_pi, out_mu, out_sigma, M = 1):
        NTEST = out_pi.shape[0]
        print("NTEST : ",NTEST)
        result = np.random.rand(NTEST, M) # initially random [0, 1]
        rn = np.random.randn(NTEST, M) # normal random matrix (0.0, 1.0)
        mu = 0
        std = 0
        idx = 0

    # transforms result into random ensembles
        for j in range(0, M):
            for i in range(0, NTEST):
                idx = get_pi_idx(result[i, j], out_pi[i])
                mu = out_mu[i, idx]
                std = out_sigma[i, idx]
                # print("mu : ", mu)
                # print("std : ", std)
                # result[i, j] = mu + rn[i, j]*std
                result[i, j] = mu
        return result


    if normalization == 1:
        learning_rate = 1e-4
        training_epochs = 50000

    if normalization == 0:
        learning_rate = 1
        training_epochs = 50000



    hypothesis = mlp(X, W, b)
    out_pi, out_sigma, out_mu, checknan = get_mixture_coef(hypothesis)

    
    lossfunc = get_lossfunc(out_pi, out_sigma, out_mu, Y)
    train = tf.train.AdamOptimizer(learning_rate).minimize(lossfunc)

    batch_size = 0
    dropout_rate = 0
    cost_history = []


    # 그래프 실행
    init = tf.global_variables_initializer()
    with tf.Session(config=config) as sess:
        sess.run(init)

        for step in range(training_epochs+1):

            sess.run(train, feed_dict={X: x_train_data, Y: y_train_data})
            if  step > 0:
                print ("error : ", step, sess.run(lossfunc, feed_dict={X: x_train_data, Y: y_train_data}))
                # print ("tensor", step, sess.run(checknan, feed_dict={X: x_train_data, Y: y_train_data}))
                cost_history = np.append(cost_history,sess.run(lossfunc,feed_dict={X: x_train_data, Y: y_train_data}))


        plt.plot(range(len(cost_history)),cost_history)
        plt.axis([0,len(cost_history),np.min(cost_history),np.max(cost_history)])
        plt.show()
        
        # 학습 결과 확인
        with tf.name_scope("accuracy"):
            if normalization == 1:
                x_train_data[:,1] =  robust_scaler1.transform(np.ones(x_train_data.shape[1])*3)[1]
                # x_train_data[:,4:8] =  robust_scaler1.transform(np.ones(x_train_data.shape[1])*0)[0]
            
            
            # if normalization == 0:
            #     x_train_data[:,0] =  3
            #     # x_train_data[:,4:7] = 0
            
            out_pi_test, out_sigma_test, out_mu_test, checknan_test = sess.run(get_mixture_coef(hypothesis), feed_dict={X: x_test_data})
            y_test_pred = np.array(np.transpose(generate_ensemble(out_pi_test, out_mu_test, out_sigma_test)).reshape(-1),dtype='float64')
            out_pi_train, out_sigma_train, out_mu_train, checknan_train = sess.run(get_mixture_coef(hypothesis), feed_dict={X: x_train_data})
            y_train_pred = np.array(np.transpose(generate_ensemble(out_pi_train, out_mu_train, out_sigma_train)).reshape(-1),dtype='float64')
            y_test_data = np.array(np.transpose(y_test_data).reshape(-1),dtype='float64')
            

            if normalization == 1 :
                x_train_data = robust_scaler1.inverse_transform(x_train_data)
                x_test_data = robust_scaler1.inverse_transform(x_test_data)
                y_test_pred = robust_scaler2.inverse_transform(y_test_pred)
                y_train_pred = robust_scaler2.inverse_transform(y_train_pred)
                y_train_data = robust_scaler2.inverse_transform(y_train_data)
                y_test_data = robust_scaler2.inverse_transform(y_test_data)
         
            random = np.random.permutation(len(x_test_data))[:300]
            graph.saveGraphANN(callSign, 1, x_test_data[random], y_test_data[random], y_test_pred[random] ,startTrainDate, endTrainDate, trainCount, startTestDate, endTestDate, testCount, learning_rate, training_epochs, batch_size, dropout_rate, averaging)

r2 = r2_score(y_test_pred, y_test_data)
rmse = sqrt(mean_squared_error(y_test_pred, y_test_data))
print("r2 : ",r2)
print("rmse : ",rmse)    

# 시각화
def fit_func(x, a, b):
    return a * pow(x, b)


basic = BasicData(callSign)



x_trainCurve = np.array(x_train_data[:,0],dtype='float64')
y_trainCurve = np.array(y_train_data[:,0],dtype='float64')
param_firstData = curve_fit(fit_func, x_trainCurve, y_trainCurve)
param_firstData2 = curve_fit(fit_func, x_trainCurve, y_train_pred)

x_range = range(5, 18)
x = np.array(x_range)  
ballast = eval(str(basic[0])+ '*x**' +str(basic[1]))
laden = eval(str(basic[2])+ '*x**' +str(basic[3]))
y_trainPlot = eval(str(param_firstData[0][0])+'*x**'+str(param_firstData[0][1]))
y_predPlot = eval(str(param_firstData2[0][0])+'*x**'+str(param_firstData2[0][1]))
fig = plt.figure(figsize=(16,8))


plt.plot(x_trainCurve, y_trainCurve, 'ro')
plt.plot(x, y_trainPlot, color="red")
plt.plot(x_trainCurve, y_train_pred, 'bx')
plt.plot(x, y_predPlot, color="blue")

plt.plot(x, ballast, color="black")
plt.plot(x, laden, color="green")

plt.title(callSign +" / " + startTrainDate + " ~ " + endTrainDate + " / r: " + str(round(r2,2)) + " / rmse:" + str(round(rmse,2)) + " / y = "+str(round(param_firstData2[0][0],2)) + "x^"+ str(round(param_firstData2[0][1],2)), color ='blue')
plt.xlabel('SPEED_VG', color='red')
plt.ylabel('SHAFT_POWER', color='red')
plt.axis([11,17,5000,15000])
plt.grid(True)
gs.save("./result/img/ann/"+saveTime+"/"+callSign.upper()+"2", ext="png", close=False, verbose=True)
plt.show()