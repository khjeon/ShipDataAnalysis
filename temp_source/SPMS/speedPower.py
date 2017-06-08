# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 15:46:39 2016

@author: HHI
"""
import pandas as pd
import numpy as np
from numpy import cos as cos
from numpy import sin as sin
from numpy import arctan2 as atan2
from numpy import sqrt as sqrt
from numpy import pi as pi
import csv
from scipy import signal
from numpy import inf as inf
import glob
import os
from matplotlib import pyplot as plt
from sklearn import mixture

from sklearn.ensemble import GradientBoostingRegressor as gtb
from scipy.optimize import curve_fit

dtr = pi/ 180
rtd = 180/pi

kn2ms = 0.5144
ms2kn = 1/0.5144


    
# address = 'D:\Ship_data\Pan_gold_hope'

# os.chdir(address)

def func(x, a, b):
    return a * np.power(x,3) +b

def twind_cal(sog, heading, wind_speed, wind_dir):
    heading = heading * dtr
    wind_dir = wind_dir * dtr
    wind_speed = wind_speed * kn2ms
    sog = sog * kn2ms
    twind_speed = sqrt(wind_speed * wind_speed + sog * sog - 2 * sog * wind_speed * cos(wind_dir))
    
    nominator = wind_speed * sin(wind_dir + heading) - sog * sin(heading)
    denominator = wind_speed * cos(wind_dir + heading) - sog * cos(heading)
    twind_dir = atan2(nominator, denominator)
    
    wind_speed = twind_speed
    wind_dir = twind_dir - heading
    wind_x = wind_speed * cos(wind_dir)
    wind_y = wind_speed * sin(wind_dir)
    
    return twind_speed , np.mod(wind_dir ,2*pi), wind_x, wind_y
    
    
def wave_generator(wind_speed, wind_dir):
    wind_cri = np.array([0, 0.3,1.5, 3.3, 5.5, 7.9, 10.7, 13.8, 17.1, 20.7, 24.4, 28.4, 32.6, inf])
    wave_cri = np.array([0, 0  ,0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.5, 7.5, 10.0, 12.5, 16.0, inf])
    wave_h = np.zeros(len(wind_speed))
    wave_dir = np.zeros(len(wind_speed))
    
    for index in range(len(wind_cri)-1):
        find_index = np.argwhere((wind_speed >= wind_cri[index]) & (wind_speed < wind_cri[index+1]))
        if len(find_index) > 0:
            wave_h[find_index] = np.random.uniform(wave_cri[index],wave_cri[index+1], len(find_index))
            wave_dir[find_index] = wind_dir[find_index] + np.random.uniform(-20,20, len(find_index))
    
    wave_x = wave_h * cos(wave_dir * dtr)
    wave_y = wave_h * sin(wave_dir * dtr)
            
    return wave_h, np.mod(wave_dir,360), wave_x, wave_y

# data_tuple = pd.read_csv('bona.csv')
inp = np.array(np.zeros([1, 8]))
tar = np.array([])
tor = np.array([])
file = glob.glob("panbona.csv")
data_tuple = pd.read_csv(file[0])

data_center = data_tuple[['AT_SEA', 'SPEED_VG']]
#data_tuple = data_tuple.rolling(window = 1).mean()
data_tuple = data_tuple.dropna()
#,'propeller_rpm',  'speed_VS_x', 'fp_drft', 'ap_drft','rel_wind_speed_M','rel_wind_direction_M','rudder_angle', 'shaft_Power', 'seawater_Temp_M', 'shaft_torque_KNM (OPTION)', 'propeller_rpm']]
#date= pd.to_datetime(data_tuple['time_UTC'])
#data_center = pd.DataFrame(data_center.values, index = date)
#    data_center = data_center.resample('120S').asfreq()
#    data_center = data_center.interpolate()
#data_tuple = data_center.dropna()ABS_WIND_SPEED
#rpm_std = pd.rolling_std(data_tuple[11],window =5)
#data_tuple = data_tuple.rolling(window = 5).mean()
#data_tuple[11] = rpm_std
#data_tuple = data_tuple.dropna()

sog = np.array(data_tuple['SPEED_VG'])
sog_u = np.array(data_tuple['SPEED_LG'])
sog_v = np.array(data_tuple['SPEED_TG'])
foc = np.array(data_tuple['ME_FOC_HOUR'])
rpm = np.array(data_tuple['SHAFT_REV'])
slip = np.array(data_tuple['SLIP'])
power = np.array(data_tuple['BHP_BY_FOC'])
wind = np.array(data_tuple['WIND'])
win_dir = np.array(data_tuple['WDIR'])
resistance = np.array(data_tuple['WIND_RESISTANCE'])

wave_dir = np.array(data_tuple['DIRPW'])
wave_val = np.array(data_tuple['HTSGW'])

cog = np.array(data_tuple['COURSE_OVER_GROUND'])
current_dir = np.array(data_tuple['CURRENT_DIR'])
current_rel = np.subtract(current_dir,cog)
current_speed = ms2kn * np.array(data_tuple['CURRENT_VEL'])
current_u = current_speed * cos(dtr * current_rel)
current_v = current_speed * sin(dtr * current_rel)


sog_u = sog_u - current_u
sog_v = sog_v - current_v

speed = np.sqrt(sog_u* sog_u + sog_v * sog_v)

draft = 0.5 * (np.array(data_tuple['DRAFT_MID_PORT']) + np.array(data_tuple['DRAFT_MID_STBD']))

trim =  np.array(data_tuple['DRAFT_FORE']) - np.array(data_tuple['DRAFT_AFT'])

rudder = np.array(data_tuple['RUDDER_ANGLE'])

"""
inp_data component description
0 : speed, 1: draft, 2: trim, 3: wind_speed, 4: wind_direction, 5: wave_height, 6: wave_direction, 
7 : slip, 8: rudder

output: foc

"""
inp_data = np.append(speed.reshape([len(speed),1]), draft.reshape([len(speed),1]),1)
inp_data = np.append(inp_data, trim.reshape([len(speed),1]),1)
inp_data = np.append(inp_data, wind.reshape([len(speed),1]),1)
inp_data = np.append(inp_data, win_dir.reshape([len(speed),1]),1)
inp_data = np.append(inp_data, wave_val.reshape([len(speed),1]),1)
inp_data = np.append(inp_data, wave_dir.reshape([len(speed),1]),1)
inp_data = np.append(inp_data, slip.reshape([len(speed),1]),1)
inp_data = np.append(inp_data, rudder.reshape([len(speed),1]),1)

output = power

"""
Data filtering
1st: by upper and lower limit 
2st: PCA - foc Vs power

"""
#낮은 높은 곳의 필터를 검
first_filter_index = np.argwhere((np.abs(foc) < 100) | (speed > 14) | (speed < 6) | (foc > 1700)|(np.abs(rudder) > 5))

#
inp_data = np.delete(inp_data, first_filter_index, 0)
speed = np.delete(speed, first_filter_index, 0)
output = np.delete(output, first_filter_index, 0)
estimator = gtb(learning_rate = 0.1, max_depth = 4, min_samples_leaf = 2, max_features = 0.5, n_estimators=5000)
estimator.fit(inp_data, output)

test = inp_data
print(test.shape)
# print(test[:,0])
test[:,1] = 7.5
test[:,2] = -0.1
test[:,3:7] = 0
test[:,7] = 0.08
test[:,8] = 0
prediction = estimator.predict(test)

plt.scatter(test[:,0], prediction)
plt.show()




"""
clustering operation region by GMM and EM
min_max_indicator first component : min, second component : max

Fistly clust 4 component and discard maximum region
then for minimum region will be reclustered by two region
"""
component = 4
gmm = mixture.GMM(n_components=component, covariance_type='full')
ind = np.argwhere((speed < 14) &(speed > 6))
speed = speed[ind]
gmm.fit(speed)
clust_ind = gmm.predict(speed)
print("clust : " , clust_ind)
speed = np.reshape(speed, len(speed))
min_max_indicator = np.zeros([component*2,2])
print("min_max : " ,min_max_indicator)
for i in range(component):
    ind = np.argwhere(clust_ind == i)
    rspeed = speed[ind]
    min_max_indicator[2*i,0] = np.min(rspeed)
    min_max_indicator[2*i+1,0] = np.max(rspeed)
    min_max_indicator[2*i,1] = i
    min_max_indicator[2*i+1,1] = i
print("min_max : " ,min_max_indicator)
minimum = min_max_indicator[np.argwhere(min_max_indicator[:,0] == np.min(min_max_indicator[:,0])) , 1]
maximum = min_max_indicator[np.argwhere(min_max_indicator[:,0] == np.max(min_max_indicator[:,0])) , 1]
print("minimum ",minimum)
print("maximum", maximum)
min_speed = speed[np.argwhere(clust_ind == minimum)]
print("min_speed : ", min_speed)

speed = np.delete(speed, np.argwhere((clust_ind == minimum) | (clust_ind == maximum)), 0)
rspeed = signal.resample(speed, 10)
print("rspeed : " , rspeed)

component = 2
gmm = mixture.GMM(n_components=component, covariance_type='full')
gmm.fit(min_speed)
clust_ind = gmm.predict(min_speed)
rspeed_temp = signal.resample(min_speed[np.argwhere(clust_ind == 0)], 10)
rspeed = np.append(rspeed, rspeed_temp)
rspeed_temp = signal.resample(min_speed[np.argwhere(clust_ind == 1)], 10)
rspeed = np.append(rspeed, rspeed_temp)





a = np.zeros([len(rspeed),len(inp_data[1,:])])
a[:,2] = -0.6
a[:,1] = 18.2
a[:,0] = rspeed
pred = estimator.predict(a)


sp = np.linspace(6, 14)
popt, pcov = curve_fit(func, rspeed, pred)
plt.figure(1)
plt.plot(sp, popt[0]* np.power(sp,3)+popt[1] ,  color = 'red'), plt.scatter(rspeed, pred)
plt.show()
