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
from math import sqrt
from sklearn import datasets, metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math
from tensorflow.python import debug as tf_debug

config = tf.ConfigProto(
        device_count = {'gpu': 0}
    )


# 데이터 관련 입력 파라미터 설정
trainCount = 80000
testCount = 200
callSign = "3ffb8"
startTrainDate = '2016-01-01'
endTrainDate = '2016-03-10'
startTestDate = '2016-05-11'
endTestDate = '2016-08-17'
np.set_printoptions(formatter={'float': lambda x: "%.1f" % (x,)})
saveTime = datetime.datetime.now().strftime('%Y%m%d_%H%M')
trainData = sd.shipDataQuery(callSign, startTrainDate, endTrainDate, trainCount)
testData = sd.shipDataQuery(callSign, startTestDate, endTestDate, testCount)

x_data = np.array(trainData[0], dtype=float)
y_data = np.array(np.expand_dims(trainData[1], 1),dtype=float)
x_test_data = testData[0]
y_test_data = np.array(np.expand_dims(testData[1], 1),dtype=float)

np.set_printoptions(precision=10)

def feature_normalize(dataset):
    mu = np.mean(dataset,axis=0)
    sigma = np.std(dataset,axis=0)
    return (dataset - mu)/sigma


# x_data = feature_normalize(x_data1)
# y_data = feature_normalize(y_data1)
# x_test_data = feature_normalize(x_test_data1)
# y_test_data =  feature_normalize(y_test_data1)

NHIDDEN = 24
STDEV = 0.5
KMIX = 24 # number of mixtures
NOUT = KMIX * 3 # pi, mu, stdev

x = tf.placeholder(tf.float32, shape = [None,x_data.shape[1]], name = 'features')
y = tf.placeholder(tf.float32, shape = [None, 1], name = 'label')

W1 = tf.Variable(tf.random_normal([x_data.shape[1], NHIDDEN], stddev=STDEV, dtype=tf.float32))
b1 = tf.Variable(tf.random_normal([NHIDDEN], stddev=STDEV, dtype=tf.float32))
layer1 = tf.nn.relu(tf.matmul(x,W1) + b1)

W2 = tf.Variable(tf.random_normal([NHIDDEN, NOUT], stddev=STDEV, dtype=tf.float32))
b2 = tf.Variable(tf.random_normal([NOUT], stddev=STDEV, dtype=tf.float32))
hypothesis = tf.matmul(layer1,W2) + b2


def get_mixture_coef(output):
    outmax = tf.reduce_max(output)
    out_pi = tf.placeholder(dtype=tf.float32, shape=[None,KMIX])
    out_sigma = tf.placeholder(dtype=tf.float32, shape=[None,KMIX])
    out_mu = tf.placeholder(dtype=tf.float32, shape=[None,KMIX])
    out_pi, out_sigma, out_mu = tf.split(output,3,1)
    max_pi = tf.reduce_max(out_pi, 1, keep_dims=True)
    out_pi = tf.subtract(out_pi, max_pi)
    out_pi = tf.exp(out_pi)
    normalize_pi = tf.reciprocal(tf.reduce_sum(out_pi, 1, keep_dims=True))
    out_pi = tf.multiply(normalize_pi, out_pi)
    out_sigma = tf.exp(out_sigma)
    return out_pi, out_sigma, out_mu
  

out_pi, out_sigma, out_mu = get_mixture_coef(hypothesis)
oneDivSqrtTwoPI = 1 / math.sqrt(2*math.pi) # normalisation factor for gaussian, not needed.

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
    return tf.reduce_mean(out_sigma)

lossfunc = get_lossfunc(out_pi, out_sigma, out_mu, y)

train_op = tf.train.AdamOptimizer().minimize(lossfunc)

# output_summary = tf.summary.scalar("output", output)
merged = tf.summary.merge_all()
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())




def get_pi_idx(x, pdf):
  N = pdf.size
  accumulate = 0
  for i in range(0, N):
    accumulate += pdf[i]
    if (accumulate >= x):
      return i
  print ('error with sampling ensemble')
  return -1


def generate_ensemble(out_pi, out_mu, out_sigma, M = 10):
  NTEST = x_test_data.shape[0]
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
      result[i, j] = mu + rn[i, j]*std
  return result



##############################################################
# 이제 MDN을 실제로 돌려보자
##############################################################


sess.run(tf.global_variables_initializer())
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
# writer = tf.summary.FileWriter("./result/board/ann", sess.graph)
NEPOCH = 1000
loss = np.zeros(NEPOCH)  # store the training progress here.
for i in range(NEPOCH):
    sess.run(train_op,feed_dict={x: x_data, y: y_data})
    if(i % 100 == 0):
        print(i, " : ",sess.run(lossfunc ,feed_dict={x: x_data, y: y_data}))
    # result = sess.run(merged, feed_dict={x: x_data, y: y_data})
    # writer.add_summary(summary, step)
    loss[i] = sess.run(lossfunc, feed_dict={x: x_data, y: y_data})




out_pi_test, out_sigma_test, out_mu_test = sess.run(get_mixture_coef(hypothesis), feed_dict={x: x_test_data})
y_test_data = generate_ensemble(out_pi_test, out_mu_test, out_sigma_test)


plt.figure(figsize=(8, 8))
print("x_data :" ,x_data)
plt.plot(x_data,y_data,'ro', x_test_data,y_test_data,'bo',alpha=0.3)
plt.show()

plt.figure(figsize=(8, 8))
plt.plot(x_test_data,out_mu_test,'go', x_test_data,y_test_data,'bo',alpha=0.3)
plt.show()

#     with tf.name_scope("accuracy"):
#         correct_prediction = np.transpose(sess.run(hypothesis, feed_dict={x: test_data}))
#         r2 = r2_score(correct_prediction.reshape(-1), testData[1])
#         rmse = sqrt(mean_squared_error(correct_prediction.reshape(-1), testData[1]))
#         print("r2 : ",r2)
#         print("rmse : ",rmse)

#         # # 모델 저장
#         # # saver = tf.train.Saver()
#         # # save_path = saver.save(sess, './result/model/ANN_linear.pd')

#         # 학습결과 테스트 그래프 그리기
# graph.saveGraphANN(callSign, 0, testData[0], testData[1], correct_prediction.reshape(-1), startTrainDate, endTrainDate, trainCount, startTestDate, endTestDate, testCount, learning_rate, training_epochs, batch_size, dropout_rate)
#         # np.savetxt("./result/csv/"+callSign+ "_"+saveTime+".csv", np.transpose(result), fmt='%.2f', header="SPEED_VG, SHAFT_REV, DRAFT_FORE, DRAFT_AFT, TRIM, REL_WIND_SPEED, REL_WIND_DIR, BHP, BHP(PREDICT)", delimiter=",")



# ##############################################################
# # loss를 그려보자
# ##############################################################
# print(loss)
# plt.plot(np.arange(100, NEPOCH,1), loss[100:], 'r-')
# plt.show()
