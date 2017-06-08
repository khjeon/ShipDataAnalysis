# -*- coding: utf-8 -*-
"""
============================
그레디언트 부스트 트리 regeression (속도가 빠르면 잘 되나 ANN, SVM 보다 성능이 낮음)
============================
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import shipData as sd
import graph as graph
from math import sqrt
from sklearn import datasets, metrics
from sklearn.metrics import r2_score

from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error


# 데이터 관련 입력 파라미터 설정
trainCount = 8000000
testCount = 100
callSign = "3EWB4"
startTrainDate = '2016-05-14'
endTrainDate = '2016-07-30'
startTestDate = '2016-08-08'
endTestDate = '2016-09-02'

trainData = sd.shipDataQuery(callSign, startTrainDate, endTrainDate, trainCount)
testData = sd.shipDataQuery(callSign, startTestDate, endTestDate, testCount)

X_train = trainData[0]
y_train= trainData[1]
X_test = testData[0]
y_test = testData[1]

# X_test[0] = X_test[0] + 1.0
# X_test[1] = X_test[1] + 8.0
X_test[2] = X_test[2] - 1.0

# Fit regression model
learning_rate = 0.1
n_estimators = 5000
max_depth = 5
min_samples_split = 2
max_features = 0.5

params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_split': min_samples_split,
          'learning_rate': learning_rate, 'loss': 'ls', 'max_features' : max_features }
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)
y_hat = clf.predict(X_test)
graph.saveGraphGBT(callSign, 1, testData[0], testData[1], y_hat, startTrainDate, endTrainDate, trainCount, startTestDate, endTestDate, testCount, learning_rate, n_estimators, max_depth, min_samples_split)
# mse = mean_squared_error(y_test, clf.predict(X_test))
# r2 = r2_score(y_test, clf.predict(X_test))
# print("MSE: %.4f" % mse)
# print("r2: %.4f" % r2 )

# test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

# for i, y_pred in enumerate(clf.staged_predict(X_test)):
#     test_score[i] = clf.loss_(y_test, y_pred)

# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.title('Deviance')
# plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
#          label='Training Set Deviance')
# plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
#          label='Test Set Deviance')
# plt.legend(loc='upper right')
# plt.xlabel('Boosting Iterations')
# plt.ylabel('Deviance')

# feature_importance = clf.feature_importances_
# feature_importance = 100.0 * (feature_importance / feature_importance.max())
# sorted_idx = np.argsort(feature_importance)
# pos = np.arange(sorted_idx.shape[0]) + .5
# plt.subplot(1, 2, 2)
# plt.barh(pos, feature_importance[sorted_idx], align='center')
# plt.xlabel('Relative Importance')
# plt.title('Variable Importance')
plt.show()
