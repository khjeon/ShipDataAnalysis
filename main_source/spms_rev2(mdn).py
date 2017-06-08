"""
============================
linear regeression ANN 선박성능학습 (잘됨)
============================

@author : lab021 / 이상봉
"""
import math
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import tensorflow as tf
import pymssql
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
import plotly
import plotly.graph_objs as go

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

import pymssql
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# train 데이터 set 만들기
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
   



# HDMN CLASS DEFINE
tf_var = tf.Variable
tf_rn = tf.random_normal
tf_ru = tf.random_uniform
tf_relu = tf.nn.relu
tf_tanh = tf.nn.tanh

def plot3_2(a1, b1, c1, a2, b2, c2, speed, powerBallast, powerLaden, GRAPHCOUNT, l1="", l2="", mark=".", col1="k", col2="r", title=""):
    
    rpm = np.zeros(len(powerBallast))
    print(a1)
    print(powerBallast)


    ballast = go.Scatter3d(
        x=rpm, y=speed, z=powerBallast, mode='markers', name = 'ballast', marker=dict(size=3, opacity=0.8)
    )

    laden = go.Scatter3d(
        x=rpm, y=speed, z=powerLaden, mode='markers', name = 'laden', marker=dict(size=3, opacity=0.8)
    )
    
    predict = go.Scatter3d(
        x=a1, y=b1, z=c1, mode='markers', name = l1, marker=dict(size=2, opacity=0.8)
    )

    train = go.Scatter3d(
        x=a2, y=b2, z=c2, mode='markers', name = l2,  marker=dict(size=2, opacity=0.8)
    )

    data = [predict, train, ballast, laden]

    plotly.offline.plot({"data":data,"layout": go.Layout(legend=dict(orientation="h"),title=title, scene = dict(xaxis=dict(title="RPM"), yaxis=dict(title="SPEED_VG"),zaxis=dict(title="POWER(KW)", range = [0,20000])))
    })
  


class hmdn_class(object):
    # CONSTRUCTOR
    def __init__(self, _name, opt, _sess=None):
        # INIT STUFFS
     
        self.name = _name
        self.sess = opt['sess']
        self.xdata = opt['xTrainData']  # TRAINING DATA FEATURE
        self.ydata = opt['yTrainData']  # TRAINING DATA LABEL
        self.xTestData = opt['xTestData']  # TRAINING DATA FEATURE
        self.yTestData = opt['yTestData']  # TRAINING DATA LABEL
        self.ballastB = opt['ballastB']
        self.ballastA = opt['ballastA']
        self.ladenB = opt['ladenB']
        self.ladenA = opt['ladenA']
        


        self.ndata = self.xdata.shape[0]  # TRAINING DATA FEATURE 종류수
        self.dimx = self.xdata.shape[1]  # TRAINING DATA FEATURE 갯수
        self.dimy = self.ydata.shape[1]  # TRAINING DATA LABEL 갯수
        self.kmix = opt['kmix']  # Mixture 갯수
        self.nhid1 = opt['nhid1']  # 은닉1층 노드 수
        self.nhid2 = opt['nhid2']  # 은닉2층 노드 수
        self.hid1actv = opt['hid1actv']  # 은닉1층 활성함수
        self.hid2actv = opt['hid2actv']  # 은닉2층 활성함수
        self.var_actv = opt['var_actv']
        self.gain_p = opt['gain_p']
        self.gain_s = opt['gain_s']
        self.gain_e = opt['gain_e']
        # PARAMETERS RELATED TO LEARNING
        self.epoch = 0
        self.nepoch = opt['nepoch']
        self.nbatch = opt['nbatch']
        self.lrs = opt['lrs']
        self.wd_rate = opt['wd_rate']
        self.poltype = opt['poltype']
        self.min_testmse = 1e10
        # SAVE PATH
        self.savename = ("./weights_%s.mat" % (self.name))
        # GET RANGE OF THE DATA
        self.ymin, self.ymax = np.min(self.ydata), np.max(self.ydata)
        # BUILD MODEL
        self.build_model()
        # OPTIMIZER
        wd_rate = self.wd_rate
        self.loss = self.hgmm_nll_out \
            + wd_rate * tf.nn.l2_loss(self.W['x_h1']) \
            + wd_rate * tf.nn.l2_loss(self.W['h1_h2']) \
            + wd_rate * tf.nn.l2_loss(self.W['h2_p']) \
            + wd_rate * tf.nn.l2_loss(self.W['h2_m']) \
            + wd_rate * tf.nn.l2_loss(self.W['h2_s']) \
            + wd_rate * tf.nn.l2_loss(self.W['h2_e']) \
            + wd_rate * tf.nn.l2_loss(self.b['x_h1']) \
            + wd_rate * tf.nn.l2_loss(self.b['h1_h2']) \
            + wd_rate * tf.nn.l2_loss(self.b['h2_p']) \
            + wd_rate * tf.nn.l2_loss(self.b['h2_m']) \
            + wd_rate * tf.nn.l2_loss(self.b['h2_s']) \
            + wd_rate * tf.nn.l2_loss(self.b['h2_e'])

        # OPTIMIZER
        self.optm = tf.train.AdamOptimizer(
            learning_rate=self.lr, beta1=0.9, beta2=0.999, epsilon=0.01).minimize(self.loss)
        # , beta1=0.9, beta2=0.999, epsilon=0.01).minimize(self.loss)
        # PRINT SOME
        print("Initializing HDMN object:")
        print("[%s] ndata: [%d], dimx: [%d], dimy: [%d], ymin: [%.2f], ymax: [%.2f], kmix: [%d]"
              % (self.name, self.ndata, self.dimx, self.dimy, self.ymin, self.ymax, self.kmix))

    # BUILD MODEL
    def build_model(self):
        # INITIALIZE WEIGHTS
        STD = 0.1
        self.W = {
            'x_h1': tf_var(tf_rn([self.dimx, self.nhid1], stddev=STD), name='W/x_h1'),
            'h1_h2': tf_var(tf_rn([self.nhid1, self.nhid2], stddev=STD), name='W/h1_h2'),
            'h2_p': tf_var(tf_rn([self.nhid2, self.kmix], stddev=STD), name='W/h2_p'),
            'h2_m': tf_var(tf_rn([self.nhid2, self.kmix * self.dimy], stddev=STD), name='W/h2_m'),
            'h2_s': tf_var(tf_rn([self.nhid2, self.kmix * self.dimy], stddev=STD), name='W/h2_s'),
            'h2_e': tf_var(tf_rn([self.nhid2, self.dimy], stddev=STD), name='W/h2_e')
        }
        self.b = {
            'x_h1': tf_var(tf_rn([self.nhid1], stddev=STD), name='b/x_h1'),
            'h1_h2': tf_var(tf_rn([self.nhid2], stddev=STD), name='b/h1_h2'),
            'h2_p': tf_var(tf_rn([self.kmix], stddev=STD), name='b/h2_p'),
            # THIS SIMPLE TRICK IS IMPORTANT
            'h2_m': tf_var(tf_rn([self.kmix * self.dimy], stddev=STD), name='b/h2_m'),
            # 'h2_m': tf_var(tf_ru([self.kmix*self.dimy], minval=self.ymin, maxval=self.ymax), name='b/h2_m'),
            'h2_s': tf_var(tf_rn([self.kmix * self.dimy], stddev=STD), name='b/h2_s'),
            'h2_e': tf_var(tf_rn([self.dimy], stddev=STD), name='b/h2_e')
        }

        # DEFINE PLACEHOLDERS
        self.x = tf.placeholder(dtype=tf.float32, shape=[
                                None, self.dimx], name="X")
        self.y = tf.placeholder(dtype=tf.float32, shape=[
                                None, self.dimy], name="Y")
        self.lr = tf.placeholder(dtype=tf.float32)
        self.kp = tf.placeholder(dtype=tf.float32)
        # CONSTRUCT MAIN GRAPH
        self.hmdn_out = self.hmdn(self.x)
        # OBJECTIVE FUNCTION
        self.hgmm_nll_out = self.hgmm_nll(self.y, self.hmdn_out)

    # HMDN FUNCTION - 2단 은닉층 이후 MDN를 통해 리턴됨.
    def hmdn(self, _x):

        if self.hid1actv is 'relu':
            _h1 = tf_relu(tf.matmul(_x, self.W['x_h1']) + self.b['x_h1'])
        elif self.hid1actv is 'tanh':
            _h1 = tf_tanh(tf.matmul(_x, self.W['x_h1']) + self.b['x_h1'])
        elif self.hid1actv is 'elu':
            _h1 = tf.nn.elu(tf.matmul(_x, self.W['x_h1']) + self.b['x_h1'])

        if self.hid2actv is 'relu':
            _h2 = tf_relu(tf.matmul(_h1, self.W['h1_h2']) + self.b['h1_h2'])
        elif self.hid2actv is 'tanh':
            _h2 = tf_tanh(tf.matmul(_h1, self.W['h1_h2']) + self.b['h1_h2'])
        elif self.hid2actv is 'elu':
            _h2 = tf.nn.elu(tf.matmul(_h1, self.W['h1_h2']) + self.b['h1_h2'])

        _h2 = tf.nn.dropout(_h2, self.kp)  # DROPOUT
        # MIXTURE WEIGHTS (PI)
        _pi_hat = tf.matmul(_h2, self.W['h2_p']) + self.b['h2_p']
        _pi_hat_a = tf.exp(
            self.gain_p * (_pi_hat - tf.reduce_max(_pi_hat, 1, keep_dims=True)))
        _pi_hat_invsum = tf.reciprocal(
            tf.reduce_sum(_pi_hat_a, 1, keep_dims=True))
        _pi = tf.multiply(_pi_hat_invsum, _pi_hat_a)
        # MITURE MU (MU)
        _mu = tf.matmul(_h2, self.W['h2_m']) + self.b['h2_m']
        # MIXTURE SIGMA (SIGMA)
        _sigma_hat = tf.matmul(_h2, self.W['h2_s']) + self.b['h2_s']
        if self.var_actv is 'sigmoid':
            _sigma = self.gain_s * tf.sigmoid(_sigma_hat)
        elif self.var_actv is 'exp':
            _sigma = self.gain_s * tf.exp(_sigma_hat)
        else:
            print("SOMETHING IS WRONG IN [HMDN]")

        # HETEROSCADESTIC NOISE (ERROR)
        _err_hat = tf.matmul(_h2, self.W['h2_e']) + self.b['h2_e']
        if self.var_actv is 'sigmoid':
            _err = self.gain_e * tf.sigmoid(_err_hat)
        elif self.var_actv is 'exp':
            _err = self.gain_e * tf.exp(_err_hat)
        else:
            print("SOMETHING IS WRONG IN [HMDN]")

        # OUTS
        _out = {
            'h1': _h1, 'h2': _h2, 'pi': _pi, 'mu': _mu, 'sigma': _sigma, 'err': _err
        }
        return _out

    # UTILITY FUNCTIONS
    def kron(self, _x, _k):
        _h = tf.shape(_x)[0]
        _w = tf.shape(_x)[1]
        return tf.reshape(tf.tile(tf.expand_dims(_x, axis=2), [1, 1, _k]), [_h, _w * _k])

    def tf_normal(self, _y, _mu, _sigma):
        _result = (_y - _mu) / _sigma
        _result = -tf.square(_result) / 2
        _result = tf.exp(_result) / (math.sqrt(2 * math.pi) * _sigma)
        return _result

    # HETEROSCADESTIC GAUSSIAN MIXTURE MODEL NEGATIVE LOG LIKELIHOOD
    def hgmm_nll(self, _y, _hmdn_out):
        _pi = _hmdn_out['pi']
        _mu = _hmdn_out['mu']
        _sigma = _hmdn_out['sigma']
        _err = _hmdn_out['err']
        _probs = self.tf_normal(
            tf.tile(_y, [1, self.kmix]), _mu, _sigma + tf.tile(_err, [1, self.kmix]))
        # KRON TO ALL Y-DIM
        _temp = tf.multiply(self.kron(_pi, self.dimy), _probs)
        _res = tf.reduce_sum(_temp, 1, keep_dims=True)
        _eps = 1e-9
        _temp = _res + _eps
        return tf.reduce_mean(-tf.log(_temp))  # NLL

    # GET MOST PROBABLE OUTPUT OF HMDN
    def hmdn_sample(self, _hmdn_out):
        _sample_pi = _hmdn_out['pi']
        _sample_mu = _hmdn_out['mu']
        _sample_sigma = _hmdn_out['sigma']
        _sample_err = _hmdn_out['err']
        _nsample = _sample_pi.shape[0]
        _dimsample = self.dimy
        _outval = np.zeros((_nsample, _dimsample))
        for i in range(_nsample):
            _currpi = _sample_pi[i]
            _maxidx = np.argmax(_currpi)

            _val = _sample_mu[i, self.dimy * _maxidx:self.dimy * (_maxidx + 1)]
            _outval[i, :] = _val
        return _outval

    # TRAIN
    def train_hmdn(self):
        # FIRST, SHUFFLE DATA
        # np.random.seed(0)
        # randpermlist = np.random.permutation(self.ndata)
        # self.xdata = self.xdata[randpermlist, :]
        # self.ydata = self.ydata[randpermlist, :]
        # THEN, SEPARATE - Train Data Set에서 95%만 짜름
        # self.ntrain = np.int(self.ndata * 0.8)

        self.ntrain = self.ndata
        self.trainx = self.xdata
        self.trainy = self.ydata
        self.testx = self.xTestData
        self.testy = self.yTestData
        self.ntest = self.testx.shape[0]

        # TRAIN CONFIGURATION
        NEPOCH = self.nepoch
        NBATCH = self.nbatch
        NITER = int(self.ntrain / NBATCH)
        PRINTEVERY = NEPOCH // 10
        PLOTEVERY = NEPOCH // 4
        GRAPHCOUNT = 0
        # TRAIN
        mses_train = np.zeros(NEPOCH)
        mses_test = np.zeros(NEPOCH)
        losses = np.zeros(NEPOCH)

        for epoch in range(NEPOCH):
            self.epoch = epoch
            # PERMUTE INDICES 해당 배열 index을 섞음.
            randpermlist = np.random.permutation(self.ntrain)
            sumloss = 0
            for i in range(NITER):
                randidx = randpermlist[i * NBATCH:(i + 1) * NBATCH]
                batchx = self.trainx[randidx, :]
                batchy = self.trainy[randidx, :]
                # FEED WHILE TRAINING
                if epoch < NEPOCH / 2:
                    feeds = {self.x: batchx, self.y: batchy,
                             self.lr: self.lrs[0], self.kp: 1.0}
                elif epoch < 3 * NEPOCH / 4:
                    feeds = {self.x: batchx, self.y: batchy,
                             self.lr: self.lrs[1], self.kp: 1.0}
                else:
                    feeds = {self.x: batchx, self.y: batchy,
                             self.lr: self.lrs[2], self.kp: 1.0}
                # OPIMIZE
                self.sess.run(self.optm, feed_dict=feeds)
                curloss = self.sess.run(self.loss, feed_dict=feeds)
                if i % 200 == 0:
                    print("%.2f%%, loss: %.6f" %
                          (((NITER * epoch + i) / (NITER * NEPOCH))*100, curloss))
                sumloss += curloss

            # AVERAGE LOSS
            avgloss = sumloss / NITER

            # COMPUTE AVERAGE PREDICTION LOSS (TEST DATA)
            feeds = {self.x: self.testx, self.kp: 1.0}
            hmdn_out_val = self.sess.run(self.hmdn_out, feed_dict=feeds)
            hmdn_sample_val = self.hmdn_sample(hmdn_out_val)
            
            pred_ydata_inv = scaler_yTestData.inverse_transform(hmdn_sample_val)
            ydata_inv = scaler_yTestData.inverse_transform(self.testy)
        
            r2_test = r2_score(pred_ydata_inv, ydata_inv)
            rmse_test = sqrt(mean_squared_error(pred_ydata_inv, ydata_inv))
            mse_test = ((ydata_inv - pred_ydata_inv) ** 2).mean(axis=None)


            feeds = {self.x: self.trainx, self.kp: 1.0}
            hmdn_out_val = self.sess.run(self.hmdn_out, feed_dict=feeds)
            hmdn_sample_val = self.hmdn_sample(hmdn_out_val)

            pred_ydata_inv = scaler_yTrainData.inverse_transform(hmdn_sample_val)
            ydata_inv = scaler_yTrainData.inverse_transform(self.trainy)

            r2_train = r2_score(pred_ydata_inv, ydata_inv)
            rmse_train = sqrt(mean_squared_error(pred_ydata_inv, ydata_inv))
            mse_train = ((ydata_inv - pred_ydata_inv) ** 2).mean(axis=None)


            # SAVE
            mses_train[epoch] = mse_train
            mses_test[epoch] = mse_test
            losses[epoch] = avgloss

            # PRINT
            if (epoch % PRINTEVERY) == 0 or (epoch + 1) == NEPOCH:
                print("[%4d/%d] avgloss: %.2f, mse_train: %.d, mse_test: %.4f, r2_train: %.2f, r2_test: %.2f, rmse_train: %d, rmse_test: %d"
                      % (epoch, NEPOCH, avgloss, mse_train, mse_test, r2_train, r2_test, rmse_train, rmse_test))
                # SAVE WEIGHT
                self.save_weights()
                # SAVE CURRENT BEST WEIGHT
                if mse_test < self.min_testmse:
                    self.min_testmse = mse_test
                    self.save_good_weights()

            if (epoch % PLOTEVERY) == 0 or (epoch + 1) == NEPOCH:
                GRAPHCOUNT = GRAPHCOUNT + 1
                # PLOT
                self.plot_recon_hmdn(GRAPHCOUNT)

        print("OPTIMIZATION FINISHED.")

        # FINAL PLOT LOSS CURVE
        plt.figure(figsize=(8, 3))
        plt.plot(losses)
        plt.title('LOSS CURVE')
        plt.figure(figsize=(8, 3))
        plt.plot(mses_train)
        plt.plot(mses_test)
        plt.legend(['Train', 'Test'])
        plt.title('MSE')

    # PLOT RECON
    def plot_recon_hmdn(self, GRAPHCOUNT):
        GRAPHCOUNT = GRAPHCOUNT
        feeds = {self.x: self.xdata, self.kp: 1.0}
        hmdn_out_val = self.sess.run(self.hmdn_out, feed_dict=feeds)
        pred_ydata = self.hmdn_sample(hmdn_out_val)

        ydata_inv2 = scaler_yTrainData.inverse_transform(self.ydata)
        pred_ydata_inv2 = scaler_yTrainData.inverse_transform(pred_ydata)
        xdata_inv1 = scaler_xTrainData.inverse_transform(self.xdata)
        mse_data = ((ydata_inv2 - pred_ydata_inv2) ** 2).mean(axis=None)
        
        r2 = r2_score(pred_ydata_inv2, ydata_inv2)
        rmse = sqrt(mean_squared_error(pred_ydata_inv2, ydata_inv2))

        speed = []
        powerBallast = []
        powerLaden = []

        speedRange = (int(np.max(xdata_inv1[:, 1]) - np.min(xdata_inv1[:, 1]))+1)*10
        print(speedRange )
        speenmin = round(np.min(xdata_inv1[:, 1]), 1)

        for i in range(speedRange):
            speed = np.append(speed, (speenmin + float(i) / 10))
            powerBallast = np.append(powerBallast, ((self.ballastB * float(speenmin + i / 10))**self.ballastA))
            powerLaden = np.append(powerLaden, ((self.ladenB * float(speenmin + i / 10))**self.ladenA))


        # PLOT IN TWO DIMENSIONAL INPUT
        dim = xdata_inv1.shape[1]
        if dim > 2:
            X = xdata_inv1
            U, S, V = np.linalg.svd(X.T, full_matrices=False)
            proj_X = np.dot(X, U[:, :2])
        else:
            proj_X = xdata_inv1
        # PLOT EACH Y DIM
        ndim = self.dimy
        if ndim > 2:
            ndim = 2
        for plot_dim in range(ndim):
            plot3_2(xdata_inv1[:, 0], xdata_inv1[:, 1], ydata_inv2[:, plot_dim], xdata_inv1[:, 0], xdata_inv1[:, 1], pred_ydata_inv2[:, plot_dim], speed, powerBallast, powerLaden, GRAPHCOUNT, l1="Training Data", l2="Predicted Data", title=("[%s] kmix: [%d], dim: [%d], mse_data: [%.d], r2: [%.2f], rmse: [%d]" % (self.name, self.kmix, dim, mse_data, r2, rmse)))
    # SAVE WEIGHTS

    def save_weights(self):
        # GET WEIGHTS
        W_x_h1 = self.sess.run(self.W['x_h1'])
        W_h1_h2 = self.sess.run(self.W['h1_h2'])
        W_h2_p = self.sess.run(self.W['h2_p'])
        W_h2_m = self.sess.run(self.W['h2_m'])
        W_h2_s = self.sess.run(self.W['h2_s'])
        W_h2_e = self.sess.run(self.W['h2_e'])
        b_x_h1 = self.sess.run(self.b['x_h1'])
        b_h1_h2 = self.sess.run(self.b['h1_h2'])
        b_h2_p = self.sess.run(self.b['h2_p'])
        b_h2_m = self.sess.run(self.b['h2_m'])
        b_h2_s = self.sess.run(self.b['h2_s'])
        b_h2_e = self.sess.run(self.b['h2_e'])
        # SAMPLE TEST (XTEST->YPRED)
        test_in = self.xdata[:10, :]
        feeds = {self.x: test_in, self.kp: 1.0}
        test_out = self.hmdn_sample(
            self.sess.run(self.hmdn_out, feed_dict=feeds))
        # OTHERS TO SAVE
        xdata, ydata = self.xdata, self.ydata
        nhid1, nhid2 = self.nhid1, self.nhid2
        kmix = self.kmix
        dimx, dimy = self.dimx, self.dimy
        epoch = self.epoch
        gain_p, gain_s, gain_e = self.gain_p, self.gain_s, self.gain_e
        hid1actv, hid2actv = self.hid1actv, self.hid2actv
        var_actv = self.var_actv
        # SAVE
        scipy.io.savemat(self.savename, mdict={'W_x_h1': W_x_h1, 'W_h1_h2': W_h1_h2, 'W_h2_p': W_h2_p,
                                               'W_h2_m': W_h2_m, 'W_h2_s': W_h2_s, 'W_h2_e': W_h2_e,
                                               'b_x_h1': b_x_h1, 'b_h1_h2': b_h1_h2, 'b_h2_p': b_h2_p,
                                               'b_h2_m': b_h2_m, 'b_h2_s': b_h2_s, 'b_h2_e': b_h2_e,
                                               'test_in': test_in, 'test_out': test_out,
                                               'xdata': xdata, 'ydata': ydata,
                                               'nhid1': nhid1, 'nhid2': nhid2, 'kmix': kmix,
                                               'gain_p': gain_p, 'gain_s': gain_s, 'gain_e': gain_e,
                                               'hid1actv': hid1actv, 'hid2actv': hid2actv,
                                               'var_actv': var_actv,
                                               'epoch': epoch
                                               })
        print("[%s] SAVED." % (self.savename))
    # SAVE WEIGHTS

    def save_good_weights(self):
        # GET WEIGHTS
        W_x_h1 = self.sess.run(self.W['x_h1'])
        W_h1_h2 = self.sess.run(self.W['h1_h2'])
        W_h2_p = self.sess.run(self.W['h2_p'])
        W_h2_m = self.sess.run(self.W['h2_m'])
        W_h2_s = self.sess.run(self.W['h2_s'])
        W_h2_e = self.sess.run(self.W['h2_e'])
        b_x_h1 = self.sess.run(self.b['x_h1'])
        b_h1_h2 = self.sess.run(self.b['h1_h2'])
        b_h2_p = self.sess.run(self.b['h2_p'])
        b_h2_m = self.sess.run(self.b['h2_m'])
        b_h2_s = self.sess.run(self.b['h2_s'])
        b_h2_e = self.sess.run(self.b['h2_e'])
        # SAMPLE TEST (XTEST->YPRED)
        test_in = self.xdata[:10, :]
        feeds = {self.x: test_in, self.kp: 1.0}
        test_out = self.hmdn_sample(
            self.sess.run(self.hmdn_out, feed_dict=feeds))
        # OTHERS TO SAVE
        xdata, ydata = self.xdata, self.ydata
        nhid1, nhid2 = self.nhid1, self.nhid2
        kmix = self.kmix
        dimx, dimy = self.dimx, self.dimy
        epoch = self.epoch
        gain_p, gain_s, gain_e = self.gain_p, self.gain_s, self.gain_e
        hid1actv, hid2actv = self.hid1actv, self.hid2actv
        var_actv = self.var_actv
        # SAVE
        savename = ("./weights_%s_good.mat" % (self.name))
        scipy.io.savemat(savename, mdict={'W_x_h1': W_x_h1, 'W_h1_h2': W_h1_h2, 'W_h2_p': W_h2_p,
                                          'W_h2_m': W_h2_m, 'W_h2_s': W_h2_s, 'W_h2_e': W_h2_e,
                                          'b_x_h1': b_x_h1, 'b_h1_h2': b_h1_h2, 'b_h2_p': b_h2_p,
                                          'b_h2_m': b_h2_m, 'b_h2_s': b_h2_s, 'b_h2_e': b_h2_e,
                                          'test_in': test_in, 'test_out': test_out,
                                          'xdata': xdata, 'ydata': ydata,
                                          'nhid1': nhid1, 'nhid2': nhid2, 'kmix': kmix,
                                          'gain_p': gain_p, 'gain_s': gain_s, 'gain_e': gain_e,
                                          'hid1actv': hid1actv, 'hid2actv': hid2actv,
                                          'var_actv': var_actv,
                                          'epoch': epoch
                                          })
        print("GOOD [%s] SAVED." % (self.savename))


def DataQuery(callSign, startDate, endDate, averaging, dataCount, QueryData, isShuffle, features, label):
    # # 읽어본 데이터 배열 만들기
    _dataCount = dataCount * (averaging == 0 and 1 or averaging * 6)  
    _data = sd.shipDataQuery(callSign, startDate, endDate, dataCount, QueryData, isShuffle)
    _data = _data.dropna(axis=0)
    _data = _data[(_data['SPEED_VG'] > 3) & (_data['SPEED_VG'] < 20) \
    & (_data['SPEED_LW'] > 3) & (_data['SPEED_LW'] < 20) & (_data['SHAFT_REV'] > 10) & (_data['SHAFT_REV'] < 100) \
    & (_data['SLIP'] > -50) & (_data['SLIP'] < 50) & (_data['DRAFT_FORE'] > 3) & (_data['DRAFT_FORE'] < 30) \
    & (_data['DRAFT_AFT'] > 3) & (_data['DRAFT_AFT'] < 30) & (_data['REL_WIND_DIR'] >= 0) & (_data['REL_WIND_DIR'] <= 360) \
    & (_data['REL_WIND_SPEED'] > -200) & (_data['REL_WIND_SPEED'] < 200) & (_data['RUDDER_ANGLE'] > -5) & (_data['RUDDER_ANGLE'] < 5) \
    & (_data['BHP_BY_FOC'] > 1000) & (_data['BHP_BY_FOC'] < 30000)]
    # data = data.loc[:,['SPEED_VG', 'SPEED_LW', 'SLIP', 'DRAFT_FORE', 'DRAFT_AFT', 'REL_WIND_DIR', 'REL_WIND_SPEED', 'RUDDER_ANGLE']]
    if  averaging > 0 :
        _data = _data.resample(rule=str(averaging)+'min', on='TIME_STAMP').mean()
    if  averaging == 0 :
        _data = _data.drop('TIME_STAMP', axis=1)
    _data = _data.dropna(axis=0)
    xData = np.array(_data.loc[:,features])
    yData = np.array(_data.loc[:,label])
    return xData, yData

def Kmfilter(data, kalmanSmooth):
    # 칼만 필터 observation_covariance = 1(값을 올리면 smooth를 강하게), transition_covariance = 1(값을 올리면 원래 센서 값 형태를 강하게) 
    _data_init=[]
    for step in range(data.shape[1]):
        _data_init = np.append(_data_init, data[0][step])
    _kf = KalmanFilter(initial_state_mean=_data_init, n_dim_obs=data.shape[1], observation_covariance=np.eye(data.shape[1])*kalmanSmooth, transition_covariance=np.eye(data.shape[1]))
    _data_kalman =_kf.smooth(data)[0]
    #칼만그래프 보여주기
    # x = np.array(range(0, 200)) 
    # fig = plt.figure(figsize=(15,7))
    # plt.plot(x, data[:200,0], 'b-')
    # plt.plot(x, data_kalman[:200,0], 'r-')
    # plt.show()
    print("Kalmanfilter Complete")
    return _data_kalman




def Data_normalization(data):
    robust_scaler = RobustScaler()
    resultData = robust_scaler.fit_transform(data)
    print("Normalization Complete")
    return resultData, robust_scaler

    



def fit_func(x, a, b):
    return a * pow(x, b)
def plot2d():
    x_trainCurve = np.array(x_train_data[:,1],dtype='float64')
    y_trainCurve = np.array(y_train_data[:,0],dtype='float64')
    param_firstData = curve_fit(fit_func, x_trainCurve, y_trainCurve)
    param_firstData2 = curve_fit(fit_func, x_trainCurve, y_train_pred)

    x_range = range(5, 18)
    x = np.array(x_range)  
    y_trainPlot = eval(str(param_firstData[0][0])+'*x**'+str(param_firstData[0][1]))
    y_predPlot = eval(str(param_firstData2[0][0])+'*x**'+str(param_firstData2[0][1]))
    fig = plt.figure(figsize=(16,8))

    plt.plot(x_trainCurve, y_trainCurve, 'ro')
    plt.plot(x, y_trainPlot, color="red")
    plt.plot(x_trainCurve, y_train_pred, 'bx')
    plt.plot(x, y_predPlot, color="blue")
    plt.title(callSign +" / " + startTrainDate + " ~ " + endTrainDate + " / r: " + str(round(r2,2)) + " / rmse:" + str(round(rmse,2)) + " / y = "+str(round(param_firstData2[0][0],2)) + "x^"+ str(round(param_firstData2[0][1],2)), color ='blue')
    plt.xlabel('SPEED_VG', color='red')
    plt.ylabel('SHAFT_POWER', color='red')
    plt.axis([0,18,0,25000])
    plt.grid(True)
    plt.show()



print("Start...")
data_opt = {
    'callSign' : "3ewb4",
    'startTrainDate' : '2016-05-18',
    'endTrainDate' : '2016-10-17',
    'trainDataCount' : 80000000,
    'isTrainDataShuffle' : 0,
    'startTestDate' : '2016-11-02',
    'endTestDate' : '2017-02-28',
    'testDataCount' : 900000,
    'isTestDataShuffle' : 0,
    'queryData' : "[TIME_STAMP], [SHAFT_POWER], [BHP_BY_FOC], [SPEED_LW],\
    [REL_WIND_DIR], [REL_WIND_SPEED], [SPEED_VG], [SHIP_HEADING], [SHAFT_REV], [DRAFT_FORE], [DRAFT_AFT], [WATER_DEPTH], [RUDDER_ANGLE], [SST],\
    [SLIP]",
    'downSampleing' : 4,
    'kalmanfilter' : 0,
    'normalization' : 1,
    'features' : ['SLIP', 'SPEED_VG'],
    'label' : ['BHP_BY_FOC']
}

xTrainData, yTrainData = DataQuery(data_opt['callSign'], data_opt['startTrainDate'], data_opt['endTrainDate'], data_opt['downSampleing'], data_opt['trainDataCount'], data_opt['queryData'], data_opt['isTrainDataShuffle'], data_opt['features'],  data_opt['label'])
xTestData, yTestData = DataQuery(data_opt['callSign'], data_opt['startTestDate'], data_opt['endTestDate'], data_opt['downSampleing'], data_opt['testDataCount'], data_opt['queryData'], data_opt['isTestDataShuffle'], data_opt['features'],  data_opt['label'])

# xTrainData = Kmfilter(xTrainData,1.5)
# yTrainData = Kmfilter(yTrainData,1.5)
# xTestData = Kmfilter(xTestData,1.5)
# yTestData = Kmfilter(yTestData,1.5)

xTrainData, scaler_xTrainData = Data_normalization(xTrainData)
yTrainData, scaler_yTrainData = Data_normalization(yTrainData)
xTestData, scaler_xTestData = Data_normalization(xTestData)
yTestData, scaler_yTestData = Data_normalization(yTestData)


xTrainData = np.array(xTrainData)
yTrainData = np.array(yTrainData)
xTestData = np.array(xTestData)
yTestData = np.array(yTestData)
basic = BasicData(data_opt['callSign'])
print("Data Loading Complete...")

analysis_opt = {
    'name' : 'spms',
    'xTrainData': xTrainData,
    'yTrainData': yTrainData,
    'xTestData' : xTestData,
    'yTestData' : yTestData,
    'kmix': 3,
    'nhid1': 256,
    'nhid2': 256,
    'hid1actv': 'relu',
    'hid2actv': 'relu',
    'var_actv': 'exp',  # exp / sigmoid
    'gain_p': 1,
    'gain_s': 5,
    'gain_e': 1,
    'nepoch': 5000,
    'nbatch': 1024,
    'lrs': [2e-6, 1e-6, 1e-6],
    'wd_rate': 1e-9,  # WEIGHT DECAY RATE
    'poltype': 'Trajectory Prediction',
    'sess': sess,
    'ballastB' : basic[0],
    'ballastA' : basic[1],
    'ladenA' : basic[2],
    'ladenB' : basic[3]
}




hmdn_obj = hmdn_class('SPMS', analysis_opt)
tf.set_random_seed(0)
sess.run(tf.global_variables_initializer())
print("VARIABLES INITIALIZED.")
hmdn_obj.train_hmdn()

