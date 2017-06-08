"""
============================
linear regeression ANN 선박성능학습 (잘됨)
============================
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


# 데이터 관련 입력 파라미터 설정
trainCount = 80000
testCount = 250
callSign = "3ffb8"
startTrainDate = '2016-01-01'
endTrainDate = '2016-03-10'
startTestDate = '2016-05-11'
endTestDate = '2016-08-17'
np.set_printoptions(formatter={'float': lambda x: "%.1f" % (x,)})
saveTime = datetime.datetime.now().strftime('%Y%m%d_%H%M')
trainData = sd.shipDataQuery(callSign, startTrainDate, endTrainDate, trainCount)

testData = sd.shipDataQuery(callSign, startTestDate, endTestDate, testCount)


x_data = np.transpose(trainData[0])
y_data = trainData[1]
test_data = np.transpose(testData[0])
# print('ddd')
# print(test_data)
# 각 환경 평균 하기
# test_avg_windspeed = np.average(test_data[5])

# test feature 변경하여 환경 변수 조정하여 환경 변수가 선박 성능에 미치는 영향 확인하기
# wind speed 영향 제거 하기
# test_data[0] = test_data[0]  
# test_data[1] = test_data[1] - 2.0
# test_data[2][np.argwhere(test_data[2]<12)] = test_data[2][np.argwhere(test_data[2]<12)] + 10.0
# test_data[3][np.argwhere(test_data[3]<12)] = test_data[3][np.argwhere(test_data[3]<12)] + 10.0
# test_data[7] = 0.0

# test_data[5] = 0.0
# current 영향 제거 하기
# test_data[6] = 0.0

# 모델 정의
W = tf.Variable(tf.random_uniform([1, len(x_data)], -10., 10.,  dtype=tf.float64))
b = tf.Variable(tf.random_uniform([1], -1., 1., dtype=tf.float64))
x = tf.placeholder(tf.float64, shape = [len(x_data), None], name = 'features')
y = tf.placeholder(tf.float64, shape = [None], name = 'label')
hypothesis = tf.matmul(W, x) + b


with tf.name_scope("cost") as scope:
    cost = tf.reduce_mean(tf.square(hypothesis - y))
    tf.summary.scalar("cost", cost)

learning_rate = 1
training_epochs = 50000
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(cost)
batch_size = 0
dropout_rate = 0
cost_history = np.empty(shape=[1],dtype=float)

# 그래프 실행
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter("./logs/"+callSign, sess.graph) # for 0.8
    merged = tf.summary.merge_all()

    for step in range(training_epochs+1):

        sess.run(train, feed_dict={x: x_data, y: y_data})
        summary = sess.run(merged, feed_dict={x: x_data, y: y_data}) 
        if step % 50 == 0:
            writer.add_summary(summary, step)
            print (step, sess.run(cost, feed_dict={x: x_data, y: y_data}), sess.run(W, feed_dict={x: x_data, y: y_data}))
        cost_history = np.append(cost_history,sess.run(cost,feed_dict={x: x_data,y: y_data}))

    plt.plot(range(len(cost_history)),cost_history)
    plt.axis([0,training_epochs,np.min(cost_history)-10,np.max(cost_history)])
    plt.show()
    
    # 학습 결과 확인
    with tf.name_scope("accuracy"):
        correct_prediction = np.transpose(sess.run(hypothesis, feed_dict={x: test_data}))
        resulttemp = np.concatenate((test_data,[testData[1]]), axis= 0)
        result = np.concatenate((resulttemp, np.transpose(correct_prediction)), axis=0)
       
        r2 = r2_score(correct_prediction, testData[1])
        rmse = sqrt(mean_squared_error(correct_prediction, testData[1]))
        # tf.scalar_summary("accuracy", rmse)
        print("r2 : ",r2)
        print("rmse : ",rmse)

        # 모델 저장
        # saver = tf.train.Saver()
        # save_path = saver.save(sess, './result/model/ANN_linear.pd')

        # 학습결과 테스트 그래프 그리기
        graph.saveGraphANN(callSign, 0, testData[0], testData[1], correct_prediction, startTrainDate, endTrainDate, trainCount, startTestDate, endTestDate, testCount, learning_rate, training_epochs, batch_size, dropout_rate)
        # np.savetxt("./result/csv/"+callSign+ "_"+saveTime+".csv", np.transpose(result), fmt='%.2f', header="SPEED_VG, SHAFT_REV, DRAFT_FORE, DRAFT_AFT, TRIM, REL_WIND_SPEED, REL_WIND_DIR, BHP, BHP(PREDICT)", delimiter=",")
