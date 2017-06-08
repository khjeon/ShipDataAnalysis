"""
============================
하이퍼파라미터 서치 함수
============================
SVM 파라미터 서치 함수 - C, gamma, epsilon 서치

"""
import math
import itertools
import optunity
import optunity.metrics
import sklearn.svm
import pymssql
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from numpy  import array
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

trainCount = 1000000
callSign = "3FFB8"
startTrainDate = '2016-01-01'
endTrainDate = '2016-06-01'

# db 연결
conn = pymssql.connect(server='183.103.118.217:21000', user='sa', password='@120bal@', database='SHIP_DB_EARTH')
trainDataSet = conn.cursor()
trainDataSet.execute("SELECT [SPEED_VG], [SHAFT_REV], [DRAFT_FORE], [DRAFT_AFT], [BHP_BY_FOC], [REL_WIND_SPEED], [REL_WIND_DIR], [CURRNET_SPEED_REL], [HTSGW], [SLIP], [TIME_STAMP]  FROM [SHIP_DB_EARTH].[dbo].[SAILING_DATA] WHERE CALLSIGN ='"+callSign+"' AND TIME_STAMP > '"  + startTrainDate + "' AND TIME_STAMP <  '"+ endTrainDate +"' AND AT_SEA = 1 AND PRIMARY_DATA_CHECK = 1 AND ERROR_MEFOFLOW_DATA = 1 AND ERROR_SHAFTREV_DATA = 1 ORDER BY TIME_STAMP")

trainFeatures = []
trainLabel = []
# db을 numpy에 넣어서 Train : feature, label로 준비
for item in trainDataSet:
    draft_mid = (item[2]+item[3])/2
    trainFeatures.append([item[0],item[1], item[2],item[3],draft_mid, item[5], item[6], item[7],  item[9]])
    trainLabel.append(item[4])

trainSize = 1 - ( trainCount /  len(trainFeatures) )
X_train, X_test, y_train, y_test = train_test_split(trainFeatures, trainLabel, test_size=trainSize, random_state=0)
print("trainRaw count : ", len(trainFeatures))
trainFeatures = array(X_train)
trainLabel = array(y_train)
conn.close()
print("train count : ", len(trainFeatures))


outer_cv = optunity.cross_validated(x=trainFeatures, y=trainLabel, num_folds=3)

def compute_mse_standard(x_train, y_train, x_test, y_test):
    """ 주어진 SVR커널에서 x-cross validation"""
    clf = svm.SVR()
    clf = Pipeline([('norm', StandardScaler()), ('clf', clf)])
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    return optunity.metrics.mse(y_test, predictions)

compute_mse_standard = outer_cv(compute_mse_standard)

def compute_mse_rbf_tuned(x_train, y_train, x_test, y_test):
    """ RBF커널에서 MSE랑 하이퍼파라미터 서치."""

    # define objective function for tuning
    @optunity.cross_validated(x=x_train, y=y_train, num_iter=2, num_folds=5)
    def tune_cv(x_train, y_train, x_test, y_test, C, gamma, epsilon):
        model = sklearn.svm.SVR(C=C, cache_size=20000, gamma=gamma, epsilon=epsilon).fit(x_train, y_train)
        predictions = model.predict(x_test)
        return optunity.metrics.mse(y_test, predictions)

    # optimize parameters
    optimal_pars, _, _ = optunity.minimize(tune_cv, 150, C=[1, 9000], gamma=[0, 50], epsilon = [0, 10])
    print("optimal hyperparameters: " + str(optimal_pars))

    tuned_model = sklearn.svm.SVR(**optimal_pars).fit(x_train, y_train)
    predictions = tuned_model.predict(x_test)
    return optunity.metrics.mse(y_test, predictions)

# wrap with outer cross-validation
compute_mse_rbf_tuned = outer_cv(compute_mse_rbf_tuned)
print(compute_mse_rbf_tuned())

