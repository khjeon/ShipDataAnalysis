#%%
from pykalman import KalmanFilter
import tensorflow as tf
import numpy as np
import graph as graph
import savefig as gs
import matplotlib.pyplot as plt
import datetime
import csv
from math import sqrt
from sklearn import datasets, metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math
from tensorflow.python import debug as tf_debug
import pymssql

config = tf.ConfigProto(
        device_count = {'gpu': 0}
    )


# train 데이터 set 만들기
def shipDataQuery(callSign, beginDate, EndDate, dataSplitCount):
    conn = pymssql.connect(server='218.39.195.13:21000', user='sa', password='@120bal@', database='SHIP_DB_EARTH')
    queryDataSet = conn.cursor()
    queryDataSet.execute("SELECT [SPEED_VG], [SHAFT_REV], [DRAFT_FORE], [DRAFT_AFT], [BHP_BY_FOC], [REL_WIND_SPEED], [REL_WIND_DIR], [CURRNET_SPEED_REL], [HTSGW], [SLIP], [TIME_STAMP]  FROM [SHIP_DB_EARTH].[dbo].[SAILING_DATA] WHERE CALLSIGN ='"+callSign+"' AND TIME_STAMP > '"  + beginDate + "' AND TIME_STAMP <  '"+ EndDate +"' AND SPEED_VG > 9 AND AT_SEA = 1 AND PRIMARY_DATA_CHECK = 1 AND ERROR_MEFOFLOW_DATA = 1 AND ERROR_SHAFTREV_DATA = 1 ORDER BY TIME_STAMP")

    features = []
    label = []
    dataSet = []

    # features와 label 설정
    for item in queryDataSet:
        draft_mid = (item[2] - item[3]) 
        features.append([item[0],item[1],item[2] , item[3], draft_mid, item[5], item[6]])
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

        
    
    x_first = features[:dataSplitCount]
    y_first = label[:dataSplitCount]
    dataSet.append(np.array(x_first))
    dataSet.append(np.array(y_first))
    dataSet.append(rawDataCount)
    dataSet.append(dataCount)
    conn.close()
    return dataSet



# 데이터 관련 입력 파라미터 설정
trainCount = 800
testCount = 200
callSign = "3ffb8"
startTrainDate = '2016-01-01'
endTrainDate = '2016-03-10'
# np.set_printoptions(formatter={'float': lambda x: "%.1f" % (x,)})
trainData = shipDataQuery(callSign, startTrainDate, endTrainDate, trainCount)

x_data1 = np.array(trainData[0], dtype=float)

print(x_data1)
print(x_data1.shape)
data =  [i[0] for i in x_data1]

kf = KalmanFilter(initial_state_mean=data[0], n_dim_obs=1, observation_covariance=1, transition_covariance=1 )
tt = kf.smooth(data)[0]

x_data = np.arange(0,trainCount)
y_data = [[i[0]] for i in x_data1]

plt.figure(figsize=(15,8))
plt.plot(x_data, y_data,  'r-', x_data, tt, 'b-')
plt.grid(True)
plt.show()