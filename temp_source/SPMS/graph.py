"""
============================
학습 결과 검증을 위한 GRAPH 그리는 모듈
============================
"""
import savefig as gs
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 그래프 한글 출력을 위한 폰트 설정
font_location = "C:/Windows/Fonts/malgunsl.ttf"
font_name = fm.FontProperties(fname=font_location).get_name()
matplotlib.rc('font', family=font_name)  # 한글 폰트 설정
# 기타 설정
saveTime = datetime.datetime.now().strftime('%Y%m%d_%H%M')

# 그래프 그리기 함수 ANN
def saveGraphANN(callSign, graphMode, testFeatures, testLabel, y_hat, trainBeginDate, trainEndDate, trainCount, testBeginDate, testEndDate, testCount, learning_rate, training_epochs, batch_size, dropout_rate,averaging):
    if  averaging == 0:
        averaging = 1
    testCount = testCount / averaging 
    if graphMode == 0:
        garo = 25
    if graphMode == 1:
        garo = testCount/6
    plt.figure(figsize=(garo,6), dpi=70)
    plt.title(callSign.upper() + " / TRAIN_DATE : " + trainBeginDate + " ~ " + trainEndDate +"(" + str(trainCount) +")" + " / PREDICT_DATE : " + testBeginDate + " ~ " + testEndDate + "(" +str(testCount) +")" + " / " + "learning_rate: " + str(learning_rate) +", "+ "training_epochs: " + str(training_epochs) +", " +  "batch_size: " + str(batch_size) +", "+ "dropout_rate: " + str(dropout_rate) , fontsize=17)
    plt.plot(y_hat, 'b-', label ="PREDICT", linewidth=2)
    plt.plot(testLabel, 'r-' , label ="TEST")
    plt.xlabel("TIME")
    plt.ylabel("POWER(KW)")
    yrange = max(testLabel) * 1.1 - min(testLabel) * 0.9
    plt.ylim(min(testLabel) * 0.9,max(testLabel) * 1.1)
    plt.legend(loc='upper left', frameon=True)
    gap =  np.mean(testLabel)-np.mean(y_hat) 
    plt.text(testCount/10-7, max(testLabel)*1.05, "TEST(Mean): " + str(round(np.mean(testLabel)))+ "Kw / PREDICT(Mean): " + str(round(np.mean(y_hat)))+ "Kw / 성능차이(Mean): " + str(round(gap))+ "Kw",va='top', ha='left', fontsize=14,   bbox={'facecolor':'red', 'alpha':0.5, 'pad':5})
    

    # 그래프 y right 축
    plt.twinx()
    plt.ylabel("SPEED(Kn) & DRAFT(M)")
    plt.tick_params(axis="y")
    # plt.plot(testFeatures[:,1], 'g-', label="rpm")
    plt.plot(testFeatures[:,1], 'm-', label="SOG")
    # print(testFeatures)
    plt.legend(loc='upper right', frameon=True)
    plt.ylim(0,50)
    plt.tight_layout()
    gs.save("./result/img/ann/"+saveTime+"/"+callSign.upper(), ext="png", close=False, verbose=True)


# 그래프 그리기 SVM
def saveGraphSVM(callSign, graphMode, testFeatures, testLabel, y_hat, trainBeginDate, trainEndDate, trainCount, testBeginDate, testEndDate, testCount, svmKernel, svmC, svmgamma, svmEpsilon, averaging):
    if  averaging == 0:
        averaging = 1
    testCount = testCount / averaging 
    if graphMode == 0:
        garo = 25
    if graphMode == 1:
        garo = testCount/6
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
    # plt.plot(testFeatures[:,2], 'g-', label="DRAFT")
    plt.plot(testFeatures[:,0], 'm-', label="SOG")
    plt.legend(loc='upper right', frameon=True)
    plt.ylim(0,50)
    plt.tight_layout()
    gs.save("img/"+saveTime+"/"+callSign.upper(), ext="png", close=False, verbose=True)



    # 그래프 그리기 SVM
def saveGraphGBT(callSign, graphMode, testFeatures, testLabel, y_hat, trainBeginDate, trainEndDate, trainCount, testBeginDate, testEndDate, testCount, learning_rate, n_estimators, max_depth, min_samples_split, averaging):
    if  averaging == 0:
        averaging = 1
    testCount = testCount / averaging 
    if graphMode == 0:
        garo = 25
    if graphMode == 1:
        garo = testCount/6
    plt.figure(figsize=(garo,6), dpi=70)
    plt.title(callSign.upper() + " / TRAIN_DATE : " + trainBeginDate + " ~ " + trainEndDate +"(" + str(trainCount) +")" + " / PREDICT_DATE : " + testBeginDate + " ~ " + testEndDate + "(" +str(testCount) +")" + " / " + "learning_rate: " + str(learning_rate) +", "+ "n_estimators: " + str(n_estimators) +", " +  "max_depth: " + str(max_depth) +", "+ "min_samples_split: " + str(min_samples_split) , fontsize=17)
    plt.plot(y_hat, 'b-', label ="PREDICT", linewidth=2)
    plt.plot(testLabel, 'r-' , label ="TEST")
    plt.xlabel("TIME")
    plt.ylabel("POWER(KW)")
    yrange = max(testLabel) * 1.1 - min(testLabel) * 0.9
    plt.ylim(min(testLabel) * 0.9,max(testLabel) * 1.1)
    plt.legend(loc='upper left', frameon=True)
    gap =  np.mean(testLabel)-np.mean(y_hat) 
    plt.text(testCount/10-7, max(testLabel)*1.05, "TEST(Mean): " + str(round(np.mean(testLabel)))+ "Kw / PREDICT(Mean): " + str(round(np.mean(y_hat)))+ "Kw / 성능차이(Mean): " + str(round(gap))+ "Kw",va='top', ha='left', fontsize=14,   bbox={'facecolor':'red', 'alpha':0.5, 'pad':5})
   
    # 그래프 y right 축
    plt.twinx()
    plt.ylabel("SPEED(Kn) & DRAFT(M)")
    plt.tick_params(axis="y")
    # plt.plot(testFeatures[:,2], 'g-', label="DRAFT")
    plt.plot(testFeatures[:,1], 'm-', label="SOG")
    plt.legend(loc='upper right', frameon=True)
    plt.ylim(0,50)
    plt.tight_layout()
    gs.save("./img/gbt/"+saveTime+"/"+callSign.upper(), ext="png", close=False, verbose=True)
