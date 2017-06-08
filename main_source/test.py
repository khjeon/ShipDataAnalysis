"""
============================
linear regeression ANN 선박성능학습 (잘됨)
============================

@author : lab021 / 이상봉
"""
import os
import tensorflow as tf
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
averaging = 4 # 시간 이동평균 min 시간 설정
trainCount = 80000000
testCount = 120
isTrainShuffle = 0 # 0 : timeseries, 1 : 셔플
isTestShuffle = 1 # 0 : timeseries, 1 : 셔플
kalmanfilter = 1 # 0 : 칼만필터 smooth 미적용, 1 : 적용 
kalmansmooth = 2.0
normalization = 1 # 0 : 정규화미적용 , 1: 적용
solver = 0 # 0 : ann, 1 : gbt


callSign = "3ewb4"
Features = "[TIME_STAMP] , [SPEED_VG], [SHAFT_REV]"
# Features = "[TIME_STAMP], [SLIP], [SPEED_VG], [DRAFT_FORE], [DRAFT_AFT], [REL_WIND_DIR], [REL_WIND_SPEED], [RUDDER_ANGLE], [SHIP_HEADING]"
Label = "[TIME_STAMP],[BHP_BY_FOC]"
startTrainDate = '2016-05-18'
endTrainDate = '2016-10-17'
startTestDate = '2016-10-18'
endTestDate = '2017-02-26'
saveTime = datetime.datetime.now().strftime('%Y%m%d_%H%M')


# # 읽어본 데이터 배열 만들기
trainCount = trainCount * (averaging == 0 and 1 or averaging * 6)  
testCount = testCount * (averaging == 0 and 1 or averaging * 6)  
x_train_data_raw = sd.shipDataQuery(callSign, startTrainDate, endTrainDate, trainCount, Features, isTrainShuffle)
y_train_data_raw = sd.shipDataQuery(callSign, startTrainDate, endTrainDate, trainCount, Label, isTrainShuffle)
x_test_data_raw = sd.shipDataQuery(callSign, startTestDate, endTestDate, testCount, Features, isTestShuffle)
y_test_data_raw = sd.shipDataQuery(callSign, startTestDate, endTestDate, testCount, Label, isTestShuffle)


x_train_data = x_train_data_raw.drop('TIME_STAMP', axis=1)
y_train_data = y_train_data_raw.drop('TIME_STAMP', axis=1)
x_test_data = x_test_data_raw.drop('TIME_STAMP', axis=1)
y_test_data = y_test_data_raw.drop('TIME_STAMP', axis=1)

x_train_data = np.array(x_train_data)
y_train_data = np.array(y_train_data)
x_test_data = np.array(x_test_data)
y_test_data = np.array(y_test_data)


print("Data Loading Complete")
# 칼만 필터 observation_covariance = 1(값을 올리면 smooth를 강하게), transition_covariance = 1(값을 올리면 원래 센서 값 형태를 강하게) 
x_train_data_init=[]
x_test_data_init=[]

for step in range(x_train_data.shape[1]):
    x_train_data_init = np.append(x_train_data_init, x_train_data[0][step])
    x_test_data_init = np.append(x_test_data_init, x_test_data[0][step])

if  kalmanfilter == 1 :
    kf = KalmanFilter(initial_state_mean=x_train_data_init, n_dim_obs=x_train_data.shape[1], observation_covariance=np.eye(x_train_data.shape[1])*kalmansmooth, transition_covariance=np.eye(x_train_data.shape[1]))
    x_train_data_kalman = kf.smooth(x_train_data)[0]

    kf = KalmanFilter(initial_state_mean=y_train_data[0], n_dim_obs=1, observation_covariance=np.eye(1)*kalmansmooth, transition_covariance=np.eye(1) )
    y_train_data_kalman = kf.smooth(y_train_data)[0]

    kf = KalmanFilter(initial_state_mean=x_test_data_init, n_dim_obs=x_test_data.shape[1], observation_covariance=np.eye(x_test_data.shape[1])*kalmansmooth, transition_covariance=np.eye(x_test_data.shape[1]))
    x_test_data_kalman = kf.smooth(x_test_data)[0]

    kf = KalmanFilter(initial_state_mean=y_test_data[0], n_dim_obs=1, observation_covariance=np.eye(1)*kalmansmooth, transition_covariance=np.eye(1) )
    y_test_data_kalman = kf.smooth(y_test_data)[0]

    garo = int(trainCount / 15) 
    x = np.array(range(0, x_train_data.shape[0])) 
    fig = plt.figure(figsize=(garo,7))
    plt.plot(x, x_train_data[:,1], 'b-')
    plt.plot(x, x_train_data_kalman[:,1], 'r-')
    # plt.show()

    print(x_train_data_raw)

    x_train_data = pd.DataFrame({"TIME_STAMP" : x_train_data_raw['TIME_STAMP'], "SPEED_VG" : x_train_data_kalman[:,0], "SHAFT_REV" : x_train_data_kalman[:,1]})
    y_train_data = pd.DataFrame({"TIME_STAMP" : y_train_data_raw['TIME_STAMP'], "BHP" : y_train_data_kalman[:,0]})
    x_test_data = pd.DataFrame({"TIME_STAMP" : x_test_data_raw['TIME_STAMP'], "SPEED_VG" : x_test_data_kalman[:,0], "SHAFT_REV" : x_test_data_kalman[:,1]})
    y_test_data = pd.DataFrame({"TIME_STAMP" : y_test_data_raw['TIME_STAMP'], "BHP" : y_test_data_kalman[:,0]})



if  averaging > 0 :
    x_train_data = x_train_data.resample(rule=str(averaging)+'min', on='TIME_STAMP').mean()
    y_train_data = y_train_data.resample(rule=str(averaging)+'min', on='TIME_STAMP').mean()
    x_test_data = x_test_data.resample(rule=str(averaging)+'min', on='TIME_STAMP').mean()
    y_test_data = y_test_data.resample(rule=str(averaging)+'min', on='TIME_STAMP').mean()
    x_train_data = x_train_data.dropna(axis=0)
    y_train_data = y_train_data.dropna(axis=0)
    x_test_data = x_test_data.dropna(axis=0)
    y_test_data = y_test_data.dropna(axis=0)

if  averaging == 0 :
    x_train_data = x_train_data.drop('TIME_STAMP', axis=1)
    y_train_data = y_train_data.drop('TIME_STAMP', axis=1)
    x_test_data = x_test_data.drop('TIME_STAMP', axis=1)
    y_test_data = y_test_data.drop('TIME_STAMP', axis=1)

x_train_data = np.array(x_train_data)
y_train_data = np.array(y_train_data)
x_test_data = np.array(x_test_data)
y_test_data = np.array(y_test_data)

# print("Kalmanfilter Complete")

# 데이터 정규화

if normalization == 1 :
    robust_scaler1 = RobustScaler()
    x_train_data = robust_scaler1.fit_transform(x_train_data)
    x_test_data = robust_scaler1.transform(x_test_data)

    robust_scaler2 = RobustScaler()
    y_train_data = robust_scaler2.fit_transform(y_train_data)
    y_test_data = robust_scaler2.transform(y_test_data)

# print("Normalization Complete")

if  solver == 0 :
    NHIDDEN = 24
    STDEV = 0.5
    config = tf.ConfigProto(device_count = {'gpu': 0})


    # ANN network 정의
    X = tf.placeholder(tf.float32, shape = [None,x_train_data.shape[1]], name = 'features')
    Y = tf.placeholder(tf.float32, shape = [None, 1], name = 'label')

    W = {
        "l1": tf.Variable(tf.random_normal([x_train_data.shape[1],NHIDDEN], stddev=STDEV, dtype=tf.float32)),
        "l2": tf.Variable(tf.random_normal([NHIDDEN,1], stddev=STDEV, dtype=tf.float32))
    }
    b = {
        "l1": tf.Variable(tf.random_normal([1,NHIDDEN], stddev=STDEV, dtype=tf.float32)),
        "l2": tf.Variable(tf.random_normal([1,1], stddev=STDEV, dtype=tf.float32))
    }

    def mlp(_x, _W, _b):
        l1 = tf.nn.relu(tf.matmul(_x, _W['l1']) + _b['l1'])
        return tf.matmul(l1, _W['l2']) + _b['l2']

    hypothesis = mlp(X, W, b)
    cost = tf.reduce_mean(tf.square(hypothesis - Y))

    if normalization == 1:
        learning_rate = 1e-4
        training_epochs = 50000

    if normalization == 0:
        learning_rate = 1e-4
        training_epochs = 50000



    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(cost)
    batch_size = 0
    dropout_rate = 0
    cost_history = []

    # 그래프 실행
    init = tf.global_variables_initializer()
    with tf.Session(config=config) as sess:
        sess.run(init)

        for step in range(training_epochs+1):

            sess.run(train, feed_dict={X: x_train_data, Y: y_train_data})
            if step % 10 == 0 and step > 500:
                print ("tensor", step, sess.run(cost, feed_dict={X: x_train_data, Y: y_train_data}))
                cost_history = np.append(cost_history,sess.run(cost,feed_dict={X: x_train_data, Y: y_train_data}))


        plt.plot(range(len(cost_history)),cost_history)
        plt.axis([0,len(cost_history),np.min(cost_history),np.max(cost_history)])
        plt.show()
        
        # 학습 결과 확인
        with tf.name_scope("accuracy"):
            if normalization == 1:
                x_train_data[:,0] =  robust_scaler1.transform(np.ones(x_train_data.shape[1])*0)[0]
                # x_train_data[:,4:7] =  robust_scaler1.transform(np.ones(x_train_data.shape[1])*0)[0]
            
            
            if normalization == 0:
                x_train_data[:,0] =  3
                # x_train_data[:,4:7] = 0
            
            y_test_pred = np.array(np.transpose(sess.run(hypothesis, feed_dict={X: x_test_data})).reshape(-1),dtype='float64')
            y_train_pred = np.array(np.transpose(sess.run(hypothesis, feed_dict={X: x_train_data})).reshape(-1),dtype='float64')

            if normalization == 1 :
                x_train_data = robust_scaler1.inverse_transform(x_train_data)
                x_test_data = robust_scaler1.inverse_transform(x_test_data)
                y_test_pred = robust_scaler2.inverse_transform(y_test_pred)
                y_train_pred = robust_scaler2.inverse_transform(y_train_pred)
                y_train_data = robust_scaler2.inverse_transform(y_train_data)
                y_test_data = robust_scaler2.inverse_transform(y_test_data)
            
            graph.saveGraphANN(callSign, 1, x_test_data, y_test_data, y_test_pred, startTrainDate, endTrainDate, trainCount, startTestDate, endTestDate, testCount, learning_rate, training_epochs, batch_size, dropout_rate, averaging)



if  solver == 1 :

    # Fit regression model
    learning_rate = 0.1
    n_estimators = 50000
    max_depth = 4
    min_samples_split = 2
    max_features = 0.5

    params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_split': min_samples_split,
            'learning_rate': learning_rate, 'loss': 'ls', 'max_features' : max_features }
    clf = ensemble.GradientBoostingRegressor(**params)
    clf.fit(x_train_data, y_train_data)
    

    if normalization == 1:
        x_train_data[:,0] =  robust_scaler1.transform(np.ones(x_train_data.shape[1])*3)[0]
        # x_train_data[:,4:7] =  robust_scaler1.transform(np.ones(x_train_data.shape[1])*0)[0]
            
            
    if normalization == 0:
        x_train_data[:,0] =  3
        # x_train_data[:,4:7] = 0

    y_test_pred = clf.predict(x_test_data)
    y_train_pred = clf.predict(x_train_data)

    if normalization == 1 :
        x_train_data = robust_scaler1.inverse_transform(x_train_data)
        x_test_data = robust_scaler1.inverse_transform(x_test_data)
        y_train_pred = robust_scaler2.inverse_transform(y_train_pred)
        y_test_pred = robust_scaler2.inverse_transform(y_test_pred)
        y_train_data = robust_scaler2.inverse_transform(y_train_data)
        y_test_data = robust_scaler2.inverse_transform(y_test_data)
    
    graph.saveGraphGBT(callSign, 1, x_test_data, y_test_data, y_test_pred, startTrainDate, endTrainDate, trainCount, startTestDate, endTestDate, testCount, learning_rate, n_estimators, max_depth, min_samples_split, averaging)
    
r2 = r2_score(y_test_pred, y_test_data)
rmse = sqrt(mean_squared_error(y_test_pred, y_test_data))
print("r2 : ",r2)
print("rmse : ",rmse)    



# 시각화
def fit_func(x, a, b):
    return a * pow(x, b)


basic = BasicData(callSign)



x_trainCurve = np.array(x_train_data[:,1],dtype='float64')
y_trainCurve = np.array(y_train_data[:,0],dtype='float64')
param_firstData = curve_fit(fit_func, x_trainCurve, y_trainCurve)
param_firstData2 = curve_fit(fit_func, x_trainCurve, y_train_pred)

x_range = range(6, 18)
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
plt.axis([6,18,0,15000])
plt.grid(True)
plt.show()