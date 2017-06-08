import math
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import tensorflow as tf
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

plt.rcParams['axes.unicode_minus'] = False
# PLOT3 IN PYTHON

def plot3_2(a1, b1, c1, a2, b2, c2, GRAPHCOUNT, l1="", l2="", mark=".", col1="k", col2="r", title=""):
    predict = go.Scatter3d(
        x=a1, y=b1, z=c1, mode='markers', name = l1, marker=dict(size=2, opacity=0.8)
    )

    train = go.Scatter3d(
        x=a2, y=b2, z=c2, mode='markers', name = l2,  marker=dict(size=2, opacity=0.8)
    )

    data = [predict, train]
    plotly.offline.plot({"data":data,"layout": go.Layout(legend=dict(orientation="h"),title=title, scene = dict(xaxis=dict(title="RPM"), yaxis=dict(title="SPEED_VG"),zaxis=dict(title="POWER(KW)")))
    })
  
# HDMN CLASS DEFINE
tf_var = tf.Variable
tf_rn = tf.random_normal
tf_ru = tf.random_uniform
tf_relu = tf.nn.relu
tf_tanh = tf.nn.tanh

class hmdn_class(object):
    # CONSTRUCTOR
    def __init__(self, _name, opt, _sess=None):
        # INIT STUFFS
        self.name = _name
        self.sess = opt['sess']
        self.xdata = opt['xdata']  # TRAINING DATA FEATURE
        self.ydata = opt['ydata']  # TRAINING DATA LABEL
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
        np.random.seed(0)
        randpermlist = np.random.permutation(self.ndata)
        self.xdata = self.xdata[randpermlist, :]
        self.ydata = self.ydata[randpermlist, :]
        # THEN, SEPARATE - Train Data Set에서 95%만 짜름
        self.ntrain = np.int(self.ndata * 0.8)
        self.trainx = self.xdata[:self.ntrain, :]
        self.trainy = self.ydata[:self.ntrain, :]
        self.testx = self.xdata[self.ntrain:, :]
        self.testy = self.ydata[self.ntrain:, :]
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
                if i % 20 == 0:
                    print("%.2f%%, loss: %.6f" %
                          (((NITER * epoch + i) / (NITER * NEPOCH))*100, curloss))
                sumloss += curloss

            # AVERAGE LOSS
            avgloss = sumloss / NITER

            # COMPUTE AVERAGE PREDICTION LOSS (TEST DATA)
            feeds = {self.x: self.testx, self.kp: 1.0}
            hmdn_out_val = self.sess.run(self.hmdn_out, feed_dict=feeds)
            hmdn_sample_val = self.hmdn_sample(hmdn_out_val)
            mse_test = ((self.testy - hmdn_sample_val) ** 2).mean(axis=None)

            feeds = {self.x: self.trainx, self.kp: 1.0}
            hmdn_out_val = self.sess.run(self.hmdn_out, feed_dict=feeds)
            hmdn_sample_val = self.hmdn_sample(hmdn_out_val)
            mse_train = ((self.trainy - hmdn_sample_val) ** 2).mean(axis=None)
            # SAVE
            mses_train[epoch] = mse_train
            mses_test[epoch] = mse_test
            losses[epoch] = avgloss

            # PRINT
            if (epoch % PRINTEVERY) == 0 or (epoch + 1) == NEPOCH:
                print("[%4d/%d] avgloss: %.4f, mse_train: %.4f, mse_test: %.4f"
                      % (epoch, NEPOCH, avgloss, mse_train, mse_test))
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

        

        ydata_inv2 = robust_scaler2.inverse_transform(self.ydata)
        pred_ydata_inv2 = robust_scaler2.inverse_transform(pred_ydata)
        xdata_inv1 = robust_scaler1.inverse_transform(self.xdata)
        mse_data = ((ydata_inv2 - pred_ydata_inv2) ** 2).mean(axis=None)
        
        r2 = r2_score(pred_ydata_inv2, ydata_inv2)
        rmse = sqrt(mean_squared_error(pred_ydata_inv2, ydata_inv2))

        
    #     # PLOT IN TWO DIMENSIONAL INPUT
    #     dim = xdata_inv1.shape[1]
    #     if dim > 2:
    #         X = xdata_inv1
    #         U, S, V = np.linalg.svd(X.T, full_matrices=False)
    #         proj_X = np.dot(X, U[:, :2])
    #     else:
    #         proj_X = xdata_inv1
    #     # PLOT EACH Y DIM
    #     ndim = self.dimy
    #     if ndim > 2:
    #         ndim = 2
    #     for plot_dim in range(ndim):
    #         plot3_2(proj_X[:, 0], proj_X[:, 1], ydata_inv2[:, plot_dim], proj_X[:, 0], proj_X[:, 1], pred_ydata_inv2[:, plot_dim],GRAPHCOUNT, r2, rmse, l1="Training Data", l2="Predicted Data", title=("[%s] kmix: [%d], dim: [%d], mse_data: [%.4f]" % (self.name, self.kmix, plot_dim, mse_data)))
    # # SAVE WEIGHTS

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
    #
    #
    #


print("HMDN CLASS READY")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#데이터 관련 입력 파라미터 설정
averaging = 4  # 시간 이동평균 min 시간 설정
trainCount = 800000
testCount = 120
isTrainShuffle = 1  # 0 : timeseries, 1 : 셔플
isTestShuffle = 1  # 0 : timeseries, 1 : 셔플
kalmanfilter = 0  # 0 : 칼만필터 smooth 미적용, 1 : 적용
kalmansmooth = 1.5
normalization = 1  # 0 : 정규화미적용 , 1: 적용
solver = 0  # 0 : ann, 1 : gbt


callSign = "3ewb4"
Features = "[TIME_STAMP], [SLIP], [SPEED_VG]"
# Features = "[TIME_STAMP], [SLIP], [SPEED_VG], [DRAFT_FORE], [DRAFT_AFT], [REL_WIND_DIR], [REL_WIND_SPEED], [RUDDER_ANGLE], [SHIP_HEADING]"
Label = "[TIME_STAMP],[BHP_BY_FOC]"
startTrainDate = '2016-06-01'
endTrainDate = '2016-12-01'
startTestDate = '2016-12-01'
endTestDate = '2017-02-01'
saveTime = datetime.datetime.now().strftime('%Y%m%d_%H%M')


# # 읽어본 데이터 배열 만들기
trainCount = trainCount * (averaging == 0 and 1 or averaging * 6)
testCount = testCount * (averaging == 0 and 1 or averaging * 6)
x_train_data = sd.shipDataQuery(
    callSign, startTrainDate, endTrainDate, trainCount, Features, isTrainShuffle)
y_train_data = sd.shipDataQuery(
    callSign, startTrainDate, endTrainDate, trainCount, Label, isTrainShuffle)
x_test_data = sd.shipDataQuery(
    callSign, startTestDate, endTestDate, testCount, Features, isTestShuffle)
y_test_data = sd.shipDataQuery(
    callSign, startTestDate, endTestDate, testCount, Label, isTestShuffle)


if averaging > 0:
    x_train_data = x_train_data.resample(
        rule=str(averaging) + 'min', on='TIME_STAMP').mean()
    y_train_data = y_train_data.resample(
        rule=str(averaging) + 'min', on='TIME_STAMP').mean()
    x_test_data = x_test_data.resample(
        rule=str(averaging) + 'min', on='TIME_STAMP').mean()
    y_test_data = y_test_data.resample(
        rule=str(averaging) + 'min', on='TIME_STAMP').mean()
    x_train_data = x_train_data.dropna(axis=0)
    y_train_data = y_train_data.dropna(axis=0)
    x_test_data = x_test_data.dropna(axis=0)
    y_test_data = y_test_data.dropna(axis=0)

if averaging == 0:
    x_train_data = x_train_data.drop('TIME_STAMP', axis=1)
    y_train_data = y_train_data.drop('TIME_STAMP', axis=1)
    x_test_data = x_test_data.drop('TIME_STAMP', axis=1)
    y_test_data = y_test_data.drop('TIME_STAMP', axis=1)

x_train_data = np.array(x_train_data)
y_train_data = np.array(y_train_data)
x_test_data = np.array(x_test_data)
y_test_data = np.array(y_test_data)

print("Data Loading Complete")
# 칼만 필터 observation_covariance = 1(값을 올리면 smooth를 강하게),
# transition_covariance = 1(값을 올리면 원래 센서 값 형태를 강하게)
x_train_data_init = []
x_test_data_init = []

for step in range(x_train_data.shape[1]):
    x_train_data_init = np.append(x_train_data_init, x_train_data[0][step])
    x_test_data_init = np.append(x_test_data_init, x_test_data[0][step])

if kalmanfilter == 1:
    kf = KalmanFilter(initial_state_mean=x_train_data_init, n_dim_obs=x_train_data.shape[1], observation_covariance=np.eye(
        x_train_data.shape[1]) * kalmansmooth, transition_covariance=np.eye(x_train_data.shape[1]))
    x_train_data_kalman = kf.smooth(x_train_data)[0]

    kf = KalmanFilter(initial_state_mean=y_train_data[0], n_dim_obs=1, observation_covariance=np.eye(
        1) * kalmansmooth, transition_covariance=np.eye(1))
    y_train_data_kalman = kf.smooth(y_train_data)[0]

    kf = KalmanFilter(initial_state_mean=x_test_data_init, n_dim_obs=x_test_data.shape[1], observation_covariance=np.eye(
        x_test_data.shape[1]) * kalmansmooth, transition_covariance=np.eye(x_test_data.shape[1]))
    x_test_data_kalman = kf.smooth(x_test_data)[0]

    kf = KalmanFilter(initial_state_mean=y_test_data[0], n_dim_obs=1, observation_covariance=np.eye(
        1) * kalmansmooth, transition_covariance=np.eye(1))
    y_test_data_kalman = kf.smooth(y_test_data)[0]

    x_train_data = x_train_data_kalman
    y_train_data = y_train_data_kalman
    x_test_data = x_test_data_kalman
    y_test_data = y_test_data_kalman

print("Kalmanfilter Complete")

# 데이터 정규화

if normalization == 1:
    robust_scaler1 = RobustScaler()
    x_train_data = robust_scaler1.fit_transform(x_train_data)
    x_test_data = robust_scaler1.transform(x_test_data)

    robust_scaler2 = RobustScaler()
    y_train_data = robust_scaler2.fit_transform(y_train_data)
    y_test_data = robust_scaler2.transform(y_test_data)

print("Normalization Complete")

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
opt = {
    'xdata': x_train_data,
    'ydata': y_train_data,
    'kmix': 3,
    'nhid1': 256,
    'nhid2': 256,
    'hid1actv': 'tanh',
    'hid2actv': 'tanh',
    'var_actv': 'sigmoid',  # exp / sigmoid
    'gain_p': 1,
    'gain_s': 5,
    'gain_e': 1,
    'nepoch': 50000,
    'nbatch': 512,
    'lrs': [2e-6, 1e-6, 1e-6],
    'wd_rate': 1e-9,  # WEIGHT DECAY RATE
    'poltype': 'Trajectory Prediction',
    'sess': sess
}

print(x_train_data.shape)
hmdn_obj = hmdn_class('TRAJ_PRED', opt)
tf.set_random_seed(0)
sess.run(tf.global_variables_initializer())
print("VARIABLES INITIALIZED.")
# TRAIN
hmdn_obj.train_hmdn()
# SAVE
hmdn_obj.save_weights()
