"""
============================
linear regeression ANN 선박성능학습 (잘됨)
============================

@author : lab021
"""
#%%
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


# 데이터 관련 입력 파라미터 설정
trainCount = 80000
testCount = 250
isShuffle = 1 # 0 : timeseries, 1 : 셔플
callSign = "3ffb8"
Features = "[TIME_STAMP], [SLIP], [SPEED_VG]"
Label = "[TIME_STAMP],[BHP_BY_FOC]"
startTrainDate = '2016-01-01'
endTrainDate = '2016-06-01'
startTestDate = '2016-07-02'
endTestDate = '2016-11-01'

# np.set_printoptions(formatter={'float': lambda x: "%.2f" % (x,)})
saveTime = datetime.datetime.now().strftime('%Y%m%d_%H%M')
# testData = sd.shipDataQuery(callSign, startTestDate, endTestDate, testCount)

# # 읽어본 데이터 배열 만들기
x_train_data = sd.shipDataQuery(callSign, startTrainDate, endTrainDate, trainCount, Features, isShuffle)
y_train_data = sd.shipDataQuery(callSign, startTrainDate, endTrainDate, trainCount, Label, isShuffle)
x_test_data = sd.shipDataQuery(callSign, startTestDate, endTestDate, testCount, Features, isShuffle)
y_test_data = sd.shipDataQuery(callSign, startTestDate, endTestDate, testCount, Label, isShuffle)

x_train_data = x_train_data.drop('TIME_STAMP', 1)
y_train_data = y_train_data.drop('TIME_STAMP', 1)
x_test_data = x_test_data.drop('TIME_STAMP', 1)
y_test_data = y_test_data.drop('TIME_STAMP', 1)

x_train_data = np.array(x_train_data)
y_train_data = np.array(y_train_data)
x_test_data = np.array(x_test_data)
y_test_data = np.array(y_test_data)

# 칼만 필터 observation_covariance = 1(값을 올리면 smooth를 강하게), transition_covariance = 1(값을 올리면 원래 센서 값 형태를 강하게) 
x_train_data_init=[]
x_test_data_init=[]

for step in range(x_train_data.shape[1]):
    x_train_data_init = np.append(x_train_data_init, x_train_data[0][step])
    x_test_data_init = np.append(x_test_data_init, x_test_data[0][step])


kf = KalmanFilter(initial_state_mean=x_train_data_init, n_dim_obs=x_train_data.shape[1], observation_covariance=np.eye(x_train_data.shape[1])*1.5, transition_covariance=np.eye(x_train_data.shape[1]))
x_train_data_kalman = kf.smooth(x_train_data)[0]


kf = KalmanFilter(initial_state_mean=y_train_data[0], n_dim_obs=1, observation_covariance=np.eye(1)*2, transition_covariance=np.eye(1) )
y_train_data_kalman = kf.smooth(y_train_data)[0]

kf = KalmanFilter(initial_state_mean=x_test_data_init, n_dim_obs=x_test_data.shape[1], observation_covariance=np.eye(x_test_data.shape[1])*1.5, transition_covariance=np.eye(x_test_data.shape[1]))
x_test_data_kalman = kf.smooth(x_test_data)[0]

kf = KalmanFilter(initial_state_mean=y_test_data[0], n_dim_obs=1, observation_covariance=np.eye(1)*2, transition_covariance=np.eye(1) )
y_test_data_kalman = kf.smooth(y_test_data)[0]

# garo = int(trainCount / 15) 
# x = np.array(range(0, x_train_data.shape[0])) 
# fig = plt.figure(figsize=(garo,7))
# plt.plot(x, x_train_data[:,1], 'b-')
# plt.plot(x, x_train_data_kalman[:,1], 'r-')
# plt.show()

x_train_data = x_train_data_kalman
y_train_data = y_train_data_kalman
x_test_data = x_test_data_kalman
y_test_data = y_test_data_kalman

# 데이터 정규화
robust_scaler1 = RobustScaler()
x_train_data = robust_scaler1.fit_transform(x_train_data)
x_test_data = robust_scaler1.transform(x_test_data)

robust_scaler2 = RobustScaler()
y_train_data = robust_scaler2.fit_transform(y_train_data)
y_test_data = robust_scaler2.transform(y_test_data)

# ANN network 정의
X = tf.placeholder(tf.float32, shape = [None,x_train_data.shape[1]], name = 'features')
Y = tf.placeholder(tf.float32, shape = [None, 1], name = 'label')

NHIDDEN = 20

W = {
    "l1": tf.Variable(tf.random_normal([x_train_data.shape[1],NHIDDEN], stddev=1.0, dtype=tf.float32)),
    "l2": tf.Variable(tf.random_normal([NHIDDEN,1], stddev=1.0, dtype=tf.float32))
}
b = {
    "l1": tf.Variable(tf.random_normal([1,NHIDDEN], stddev=1.0, dtype=tf.float32)),
    "l2": tf.Variable(tf.random_normal([1,1], stddev=1.0, dtype=tf.float32))
}

def mlp(_x, _W, _b):
    l1 = tf.nn.relu(tf.matmul(_x, _W['l1']) + _b['l1'])
    return tf.matmul(l1, _W['l2']) + _b['l2']

hypothesis = mlp(X, W, b)
cost = tf.reduce_mean(tf.square(hypothesis - Y))

learning_rate = 0.01
training_epochs = 5000
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(cost)
batch_size = 0
dropout_rate = 0
cost_history = []

# 그래프 실행
init = tf.global_variables_initializer()
with tf.Session() as sess:
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
       
        # x_train_data[:,0] =  robust_scaler1.transform(np.ones(x_train_data.shape[1])*2)[0]
        # x_train_data[:,4:7] =  robust_scaler1.transform(np.ones(x_train_data.shape[1])*0)[0]
        
        # x_train_data[:,0] =  3
        # x_train_data[:,2] =8
        # x_train_data[:,3:6] = 0
        
        y_test_pred = np.array(np.transpose(sess.run(hypothesis, feed_dict={X: x_test_data})).reshape(-1),dtype='float64')
        y_train_pred = np.array(np.transpose(sess.run(hypothesis, feed_dict={X: x_train_data})).reshape(-1),dtype='float64')


        r2 = r2_score(y_test_pred, y_test_data)
        rmse = sqrt(mean_squared_error(y_test_pred, y_test_data))
        print("r2 : ",r2)
        print("rmse : ",rmse)

        # x_train_data = robust_scaler1.inverse_transform(x_train_data)
        # x_test_data = robust_scaler1.inverse_transform(x_test_data)

        # y_test_pred = robust_scaler2.inverse_transform(y_test_pred)
        # y_train_pred = robust_scaler2.inverse_transform(y_train_pred)
        # y_train_data = robust_scaler2.inverse_transform(y_train_data)
        # y_test_data = robust_scaler2.inverse_transform(y_test_data)
        graph.saveGraphANN(callSign, 1, x_test_data, y_test_data, y_test_pred, startTrainDate, endTrainDate, trainCount, startTestDate, endTestDate, testCount, learning_rate, training_epochs, batch_size, dropout_rate)

# 시각화
def fit_func(x, a, b):
    return a * pow(x, b)


x_trainCurve = np.array(x_train_data[:,1],dtype='float64')
y_trainCurve = np.array(y_train_data[:,0],dtype='float64')


param_firstData = curve_fit(fit_func, x_trainCurve, y_trainCurve)
param_firstData2 = curve_fit(fit_func, x_trainCurve, y_train_pred)

x_range = range(5, 18)
x = np.array(x_range)  

y_trainPlot = eval(str(param_firstData[0][0])+'*x**'+str(param_firstData[0][1]))
y_predPlot = eval(str(param_firstData2[0][0])+'*x**'+str(param_firstData2[0][1]))

fig = plt.figure(figsize=(16,8))

plt.plot(x_trainCurve, y_trainCurve, 'bx')
plt.plot(x, y_trainPlot, color="blue")

plt.plot(x_trainCurve, y_train_pred, 'ro')
plt.plot(x, y_predPlot, color="red")

plt.title(callSign +" / " + startTrainDate + " ~ " + endTrainDate + " / r: " + str(round(r2,2)) + " / rmse:" + str(round(rmse,2)) + " / y = "+str(round(param_firstData2[0][0],2)) + "x^"+ str(round(param_firstData2[0][1],2)), color ='blue')
plt.xlabel('SPEED_VG', color='red')
plt.ylabel('SHAFT_POWER', color='red')
plt.axis([0,18,0,25000])
plt.grid(True)
plt.show()