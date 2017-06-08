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
from numpy import dot
from numpy.linalg import inv
import csv
from scipy import signal
from numpy import inf as inf
import glob
import os
from matplotlib import pyplot as plt
from sklearn import mixture
#from geopy.distance import vincenty
from vincenty import vincenty
from sklearn.ensemble import GradientBoostingRegressor as gtb
from scipy.optimize import curve_fit
from math import radians

# dtr = pi/ 180
# rtd = 180/pi

# kn2ms = 0.5144
# ms2kn = 1/0.5144


    
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
    
def speed_calculation(pt1, pt2, dt):
    """
    lat1 = radians(pt1[0]/100)
    lon1 = radians(pt1[1]/100)
    lat2 = radians(pt2[0]/100)
    lon2 = radians(pt2[1]/100)    
    
    pt1 = [lat1, lon1]
    pt2 = [lat2, lon2]

    distance = vincenty(pt1, pt2).meters

    speed = distance/dt * 1/0.5144
    
    return speed
    """

    R = 6373.0* 1000

    lat1 = pt1[0]/100
    temp = int(lat1)
    lat1 = radians(temp + (lat1 * 100 - temp * 100)/60)
    
    lon1 = (pt1[1]/100)
    temp = int(lon1)
    lon1 = radians(temp + (lon1 * 100 - temp * 100)/60)

    lat2 = (pt2[0]/100)
    temp = int(lat2)
    lat2 = radians(temp + (lat2 * 100 - temp * 100)/60)

    lon2 = (pt2[1]/100)
    temp = int(lon2)
    lon2 = radians(temp + (lon2 * 100 - temp * 100)/60)
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    distance = R * c
    speed = distance / dt * 1/ 0.5144
    return speed, distance
    

def speed_filter(measure_distance, init_speed, dt):
    A = np.array([[1, dt],
                 [0,  1]])
    H = np.array([[0, 1]])
    Q = np.array([[0.01, 0],
                 [0, 0.01]])
    x = np.array([[0], [init_speed]])
    R = 1000
    P = 100000 * np.eye(2)
    
    filt_distance = np.zeros([len(measure_distance),1])
    filt_speed = np.zeros([len(measure_distance),1])
    k_stack = np.zeros([len(measure_distance),2])
    for i in range(len(filt_distance)):
        x_hat = dot(A,x)
        P_hat = dot(dot(A,P), A.T) + Q
        K = dot(dot(P_hat,H.T),1/(dot(dot(H,P_hat),H.T) + R))
        
        x = x_hat + (K * (measure_distance[i] - dot(H,x_hat)))
        P = P_hat - dot(dot(K,H), P_hat)
        
        filt_distance[i][0] = x[0][0]
        filt_speed[i][0] = x[1][0]
        k_stack[i][:] = K.T
    return filt_distance, filt_speed / 0.5144, k_stack
        
    
# #data_tuple = pd.read_csv('bona.csv')
# inp = np.array(np.zeros([1, 8]))

# file = glob.glob("PAN HOPE.csv")

# data_tuple = pd.read_csv(file[0])

# data_center = data_tuple[['LAT', 'LON']]
# #data_tuple = data_tuple.rolling(window = 1).mean()

# # 결측값 제거
# data_tuple = data_tuple.dropna()#,'propeller_rpm',  'speed_VS_x', 'fp_drft', 'ap_drft','rel_wind_speed_M','rel_wind_direction_M','rudder_angle', 'shaft_Power', 'seawater_Temp_M', 'shaft_torque_KNM (OPTION)', 'propeller_rpm']]
# #date= pd.to_datetime(data_tuple['time_UTC'])
# #data_center = pd.DataFrame(data_center.values, index = date)
# #    data_center = data_center.resample('120S').asfreq()
# #    data_center = data_center.interpolate()
# #data_tuple = data_center.dropna()ABS_WIND_SPEED
# #rpm_std = pd.rolling_std(data_tuple[11],window =5)
# #data_tuple = data_tuple.rolling(window = 5).mean()
# #data_tuple[11] = rpm_std
# #data_tuple = data_tuple.dropna()
# sog = np.array(data_tuple['SPEED_VG'])
# power = np.array(data_tuple['ME1_FOC_HOUR'])

# coordinate = np.array(data_tuple[['LAT', 'LON']])
# raw_speed = np.zeros([len(sog),1])
# raw_distance = np.zeros([len(sog),1])
# raw_speed[0] = sog[0]
# raw_distance[0] = 0

# for i in range(len(coordinate)-1):
#     speed, distance = speed_calculation(coordinate[i], coordinate[i+1],10) 
#     raw_speed[i+1] = speed
#     raw_distance[i+1] = raw_distance[i] + distance

# filt_distance, filt_speed, kstack = speed_filter(sog*0.5144, sog[0], 10)
# plt.plot(sog, color = 'red') ,plt.plot(filt_speed, color = 'blue')
# plt.show()
# """
# sog = np.array(data_tuple['SPEED_VG'])
# sog_u = np.array(data_tuple['SPEED_LG'])
# sog_v = np.array(data_tuple['SPEED_TG'])
# foc = np.array(data_tuple['ME_FOC_HOUR'])
# rpm = np.array(data_tuple['SHAFT_REV'])
# slip = np.array(data_tuple['SLIP'])
# power = np.array(data_tuple['SHAFT_POWER'])
# wind = np.array(data_tuple['WIND'])
# win_dir = np.array(data_tuple['WDIR'])
# resistance = np.array(data_tuple['WIND_RESISTANCE'])

# wave_dir = np.array(data_tuple['DIRPW'])
# wave_val = np.array(data_tuple['HTSGW'])

# cog = np.array(data_tuple['COURSE_OVER_GROUND'])
# current_dir = np.array(data_tuple['CURRENT_DIR'])
# current_rel = np.subtract(current_dir,cog)
# current_speed = ms2kn * np.array(data_tuple['CURRENT_VEL'])
# current_u = current_speed * cos(dtr * current_rel)
# current_v = current_speed * sin(dtr * current_rel)


# sog_u = sog_u - current_u
# sog_v = sog_v - current_v

# speed = np.sqrt(sog_u* sog_u + sog_v * sog_v)

# draft = 0.5 * (np.array(data_tuple['DRAFT_MID_PORT']) + np.array(data_tuple['DRAFT_MID_STBD']))

# trim =  np.array(data_tuple['DRAFT_FORE']) - np.array(data_tuple['DRAFT_AFT'])

# rudder = np.array(data_tuple['RUDDER_ANGLE'])
# """
# """
# inp_data component description
# 0 : speed, 1: draft, 2: trim, 3: wind_speed, 4: wind_direction, 5: wave_height, 6: wave_direction, 
# 7 : slip, 8: rudder

# output: foc

# """
# """
# inp_data = np.append(speed.reshape([len(speed),1]), draft.reshape([len(speed),1]),1)
# inp_data = np.append(inp_data, trim.reshape([len(speed),1]),1)
# inp_data = np.append(inp_data, wind.reshape([len(speed),1]),1)
# inp_data = np.append(inp_data, win_dir.reshape([len(speed),1]),1)
# inp_data = np.append(inp_data, wave_val.reshape([len(speed),1]),1)
# inp_data = np.append(inp_data, wave_dir.reshape([len(speed),1]),1)
# inp_data = np.append(inp_data, slip.reshape([len(speed),1]),1)
# inp_data = np.append(inp_data, rudder.reshape([len(speed),1]),1)

# output = foc

# """
# """
# Data filtering
# 1st: by upper and lower limit 
# 2st: PCA - foc Vs power

# """
# """
# first_filter_index = np.argwhere((np.abs(foc) < 100) | (speed > 14) | (speed < 6) | (foc > 1700)|(np.abs(rudder) > 5))

# inp_data = np.delete(inp_data, first_filter_index, 0)
# speed = np.delete(speed, first_filter_index, 0)
# output = np.delete(output, first_filter_index, 0)


# estimator = gtb(learning_rate = 0.01, max_depth = 4, min_samples_leaf = 2, max_features = 0.5, n_estimators=5000)
# estimator.fit(inp_data, output)
# """
# #test = inp_data
# ##test[:,1] = 7.5
# #test[:,2] = -3.1
# #test[:,3:7] = 0
# #test[:,7] = 0.08
# #test[:,8] = 0
# #prediction = estimator.predict(test)


# """
# clustering operation region by GMM and EM
# min_max_indicator first component : min, second component : max

# Fistly clust 4 component and discard maximum region
# then for minimum region will be reclustered by two region
# """
# """
# component = 4
# gmm = mixture.GMM(n_components=component, covariance_type='full')
# ind = np.argwhere((speed < 14) &(speed > 6))
# speed = speed[ind]

# gmm.fit(speed)
# clust_ind = gmm.predict(speed)

# speed = np.reshape(speed, len(speed))
# min_max_indicator = np.zeros([component*2,2])

# for i in range(component):
#     ind = np.argwhere(clust_ind == i)
#     rspeed = speed[ind]
#     min_max_indicator[2*i,0] = np.min(rspeed)
#     min_max_indicator[2*i+1,0] = np.max(rspeed)
#     min_max_indicator[2*i,1] = i
#     min_max_indicator[2*i+1,1] = i

# minimum = min_max_indicator[np.argwhere(min_max_indicator[:,0] == np.min(min_max_indicator[:,0])) , 1]
# maximum = min_max_indicator[np.argwhere(min_max_indicator[:,0] == np.max(min_max_indicator[:,0])) , 1]

# min_speed = speed[np.argwhere(clust_ind == minimum)]

# speed = np.delete(speed, np.argwhere((clust_ind == minimum) | (clust_ind == maximum)), 0)
# rspeed = signal.resample(speed, 10)

# component = 2
# gmm = mixture.GMM(n_components=component, covariance_type='full')
# gmm.fit(min_speed)
# clust_ind = gmm.predict(min_speed)
# rspeed_temp = signal.resample(min_speed[np.argwhere(clust_ind == 0)], 10)
# rspeed = np.append(rspeed, rspeed_temp)
# rspeed_temp = signal.resample(min_speed[np.argwhere(clust_ind == 1)], 10)
# rspeed = np.append(rspeed, rspeed_temp)



# plt.scatter(test[:,0], prediction)
# """

# #ind = np.argwhere((speed > 11) & (speed < 14))
# #rspeed = signal.resample(speed[ind], 10)
# #ind = np.argwhere((speed > 8) & (speed < 11))
# #rspeed11 = signal.resample(speed[ind],10)
# #rspeed = np.append(rspeed, rspeed11)
# #ind = np.argwhere(speed < 8)
# #rspeed9 = signal.resample(speed[ind],10)
# #rspeed = np.append(rspeed, rspeed9)

# """
# a = np.zeros([len(rspeed),len(inp_data[1,:])])
# a[:,2] = -0.6
# a[:,1] = 18.2
# a[:,0] = rspeed
# pred = estimator.predict(a)


# sp = np.linspace(6, 14)
# popt, pcov = curve_fit(func, rspeed, pred)
# plt.figure(1)
# plt.plot(sp, popt[0]* np.power(sp,3)+popt[1] ,  color = 'red'), plt.scatter(rspeed, pred)
# """
# """
# #speed = np.delete(speed, first_filter_index, 0)
# slip = np.delete(slip, first_filter_index, 0)
# power = np.delete(power, first_filter_index,0)
# foc = np.delete(foc, first_filter_index, 0)

# reference_index = np.argwhere(np.abs(slip) < 0.5)
# reference_slip = slip[reference_index]


# data_labeling = np.append(foc.reshape([len(foc),1]), power.reshape([len(foc),1]),1)
# pca_transform = PCA(n_components=2)
# pca_temp = pca_transform.fit_transform(data_labeling)
# threshold = pca_temp[:,1]

# pca_inv = pca_transform.inverse_transform(pca_temp)

# #deduction = np.abs(pca_inv - data_labeling)
# filter_index = np.argwhere((np.abs(threshold) > 1500) | (np.abs(wind) > 25.5))

# power = np.delete(power, filter_index, 0)
# sog = np.delete(sog, filter_index, 0)
# foc = np.delete(foc, filter_index, 0)
# resistance = np.delete(resistance, filter_index, 0)
# plt.scatter(sog,power)
# net_power = np.subtract(power,resistance)
# plt.figure(2)
# plt.scatter(sog,net_power)
# """

# """
# at_sea_index = np.array(data_center['AT_SEA'])
# SOG = np.array(data_center['SPEED_VG'])

# find_index = np.argwhere(at_sea_index == 0)
# find_index = find_index.reshape(len(find_index))

# index_diff = np.gradient(find_index)

# slicing_index = np.argwhere(index_diff > 100)
# slicing_start = slicing_index[::2]
# slicing_finish = slicing_index[1::2]
# slicing_start = find_index[slicing_start]
# slicing_finish = find_index[slicing_finish]

# for i in range(len(slicing_start)):
#     voyage_data = data_tuple[np.int(slicing_start[i]):np.int(slicing_finish[i])]
#     if len(voyage_data) > 1000 : 
#         #csv_file = open(file[0]+ np.str(i) +".csv","w")
#         #cw = csv.writer(csv_file , delimiter=',', quotechar='|')
#         #cw.writerow(voyage_data)
#         #csv_file.close()
#         voyage_data.to_csv(file[0]+np.str(i))
        
#     if (i == len(slicing_start)-1) & (slicing_finish[i] < len(data_tuple)):
#         voyage_data = data_tuple[np.int(slicing_finish[i]):np.int(len(data_tuple))]
#         voyage_data.to_csv(file[0]+np.str(i+1))                         
                
#     if (i == 0) & (find_index[0]!= 0):
#         voyage_data = data_tuple[0:np.int(find_index[0])]
#         voyage_data.to_csv(file[0]+np.str(i+100))                           

# """
# """
# heading = np.array(data_tuple[0])
# rpm = np.array(data_tuple[1])
# stw = np.array(data_tuple[2])
# dstw = np.gradient(stw,2)
# sog = np.array(data_tuple[2])
# fwd_draft = np.array(data_tuple[3])
# aft_draft = np.array(data_tuple[4])
# draft = 0.5*(fwd_draft + aft_draft)
# trim = fwd_draft - aft_draft
# wind_speed = 0.5144*0.9*np.array(data_tuple[5])
# wind_dir = np.array(data_tuple[6])
# rudder = np.array(data_tuple[7])
# drpm = np.gradient(rpm,2)
# power = np.array(data_tuple[8])
# temprature = np.array(data_tuple[9])
# torque = np.array(data_tuple[10])
# rpm_std = np.array(data_tuple[11])
# #win_x = wind_speed * cos(dtr * wind_dir)
# #win_y = wind_speed * sin(dtr * wind_dir)
# twind_speed, twind_dir, win_x, win_y = twind_cal(sog, heading, wind_speed, wind_dir)
# wave_h, wave_dir, wave_x, wave_y = wave_generator(twind_speed, twind_dir)

# wind_dir = np.mod(wind_dir, 2* pi)

# target = power#np.array(data_tuple['shaft_Power'])
# dtarget = np.gradient(target,2)
# input_data =  np.append(sog.reshape([len(stw),1]), wind_speed.reshape([len(stw),1]),1)
#     #input_data =  np.append(input_data, draft.reshape([len(stw),1]),1) draft.reshape([len(stw),1]), 1 trim.reshape([len(stw),1]),1
# #    input_data =  np.append(input_data, dtarget.reshape([len(stw),1]),1)
# input_data =  np.append(input_data, draft.reshape([len(stw),1]),1)
# input_data =  np.append(input_data, temprature.reshape([len(stw),1]),1)
# input_data =  np.append(input_data, wind_dir.reshape([len(stw),1]),1)
# input_data =  np.append(input_data, wave_h.reshape([len(stw),1]),1)
# input_data =  np.append(input_data, wave_dir.reshape([len(stw),1]),1)
# input_data =  np.append(input_data, rudder.reshape([len(stw),1]),1)

# filter_index = np.argwhere((stw<5) | (target < 1000) | (stw >np.max(stw)-0.5)|(target > 16000))
# target = np.delete(target, filter_index, 0)
# input_data = np.delete(input_data, filter_index, 0)
# stw = np.delete(stw, filter_index, 0)
    
# #inp = np.append(inp, input_data, 0)

# #tar = np.append(tar,target)
# """

# """
# for i in range(1,5):
# #file1 = glob.glob("*29*.csv") 
# #file2 = glob.glob("*37*.csv") 
#     data_tuple = pd.read_csv(file[i])
# #g = pd.read_csv(file1[1])   
# #h = pd.read_csv(file2[1])  
# #data_tuple = f.append(g)
# #data_tuple = data_tuple.append(h)
# #data_tuple = f
#     data_center = data_tuple[['heading_Gyro','propeller_rpm',  'speed_VS_x', 'fp_drft', 'ap_drft','rel_wind_speed_M','rel_wind_direction_M','rudder_angle', 'shaft_Power', 'seawater_Temp_M', 'shaft_torque_KNM (OPTION)', 'propeller_rpm']]
#     date= pd.to_datetime(data_tuple['time_UTC'])
#     data_center = pd.DataFrame(data_center.values, index = date)
# #    data_center = data_center.resample('120').asfreq()
# #    data_center = data_center.interpolate()
#     data_tuple = data_center.dropna()
#     rpm_std = pd.rolling_std(data_tuple[11],window =5)
#     data_tuple = data_tuple.rolling(window = 5).mean()
#     data_tuple[11] = rpm_std
#     data_tuple = data_tuple.dropna()
#     heading = np.array(data_tuple[0])
#     rpm = np.array(data_tuple[1])
#     drpm = np.gradient(rpm,2)
#     stw = np.array(data_tuple[2])
#     dstw = np.gradient(stw,2)
#     sog = np.array(data_tuple[2])
#     fwd_draft = np.array(data_tuple[3])
#     aft_draft = np.array(data_tuple[4])
#     draft = 0.5*(fwd_draft + aft_draft)
#     trim = fwd_draft - aft_draft
#     wind_speed = 0.5144*0.9*np.array(data_tuple[5])
#     wind_dir = np.array(data_tuple[6])
#     rudder = np.array(data_tuple[7])
#     drpm = np.gradient(rpm,2)
#     power = np.array(data_tuple[8])
#     temprature = np.array(data_tuple[9])
#     torque = np.array(data_tuple[10])
#     rpm_std = np.array(data_tuple[11])
#     dtorque = np.gradient(torque,2)
# #win_x = wind_speed * cos(dtr * wind_dir)
# #win_y = wind_speed * sin(dtr * wind_dir)
#     twind_speed, twind_dir, win_x, win_y = twind_cal(sog, heading, wind_speed, wind_dir)
#     wave_h, wave_dir, wave_x, wave_y = wave_generator(twind_speed, twind_dir)

#     wind_dir = np.mod(wind_dir, 2* pi)

#     target = power#np.array(data_tuple['shaft_Power'])
#     dtarget = np.gradient(target,2)
#     ddtarget = np.gradient(dtarget,2)
#     input_data =  np.append(stw.reshape([len(stw),1]), wind_speed.reshape([len(stw),1]),1)
#     #input_data =  np.append(input_data, draft.reshape([len(stw),1]),1) draft.reshape([len(stw),1]), 1 trim.reshape([len(stw),1]),1
# #    input_data =  np.append(input_data, dtarget.reshape([len(stw),1]),1)
#     input_data =  np.append(input_data, draft.reshape([len(stw),1]),1)
#     input_data =  np.append(input_data, rpm.reshape([len(stw),1]),1)
#     input_data =  np.append(input_data, wind_dir.reshape([len(stw),1]),1)
#     input_data =  np.append(input_data, wave_h.reshape([len(stw),1]),1)
#     input_data =  np.append(input_data, wave_dir.reshape([len(stw),1]),1)
#     input_data =  np.append(input_data, rudder.reshape([len(stw),1]),1)
    
#     data_labeling = np.append(rpm.reshape([len(stw),1]), sog.reshape([len(stw),1]),1)
# #    label = np.argwhere(rpm < 60)
# #    data_labeling = np.delete(data_labeling, label, 0)
# #    sog = np.delete(sog, label, 0)
# #    rpm = np.delete(rpm, label, 0)
# #    torque = np.delete(torque, label, 0)
# #    power = np.delete(power, label, 0)
#     #data_labeling = np.append(data_labeling, target.reshape([len(stw),1]),1)
#     #data_labeling = np.append(data_labeling, dtarget.reshape([len(stw),1]),1)
#     scaler = MinMaxScaler(feature_range = (0,1))
#     norm = Normalizer()
    
#     data_norm = norm.fit_transform(data_labeling)
#     #data_labeling = scaler.fit_transform(data_labeling)
#     pca_transform = PCA(n_components=2)
#     data_pca = pca_transform.fit_transform(data_norm)
#     cluster = KMeans(n_clusters= 4)
#     cluster.fit(data_norm)
    
#     label = cluster.labels_
#     filter_index = np.argwhere((stw < 6) |(power < 4000)| (rpm_std > 0.5))
#     target = np.delete(target, filter_index, 0)
#     input_data = np.delete(input_data, filter_index, 0)
#     stw = np.delete(stw, filter_index, 0)
    
#     inp = np.append(inp, input_data, 0)
    
    
#     tor = np.append(tor,  torque)
#     tar = np.append(tar,target)
# #    tar = np.delete(tar,0,0)
#     #plt.figure(i)
#     #plt.scatter(stw,target, color = 'blue', alpha = 0.3), plt.xlabel('speed(knots)'), plt.ylabel('power(kW)'), plt.title(file[i])

# """

# """
# inp = np.delete(inp,0,0)
# #inp_clf = inp[:,0]
# #.reshape([len(inp[:,0]),1])
# #inp_clf = np.append(inp_clf.reshape([len(inp[:,0]),1]), inp[:,3].reshape([len(inp[:,0]),1]),1)

# #clf = EllipticEnvelope(contamination=0.3)
# #clf.fit(inp_clf)
# #out = clf.predict(inp_clf)
# #index = np.argwhere(out == 1)

# #index1 = np.argwhere(out == -1)
# plt.scatter(inp[:,0], tar, color = 'blue', alpha = 0.2)#,plt.scatter(inp[index1,0], tar[index1], color = 'red', alpha = 0.2)
# kw2t = 24*0.185/1000
# foc = np.mean(tar) * kw2t
# """

# """
# inp_train, inp_test, out_train, out_test = cv.train_test_split(inp, tar, test_size = 0.1)
# estimator = gtb(learning_rate = 0.01, max_depth = 4, min_samples_leaf = 2, max_features = 0.5, n_estimators=5000)
# estimator.fit(inp_train, out_train)
# prediction = estimator.predict(inp)

# alpha = 0.5
# plt.scatter(inp[:,0], prediction, alpha = alpha, hold = 'on', c = 'red'), plt.scatter(inp[:,0], tar,alpha = alpha)

# address_dump = 'D:\Pan_bona_data\Dump_after_ballast'
# os.chdir(address_dump)

# input_wo_env = np.zeros((len(inp[:,0]),len(inp[0,:])))
# input_wo_env[:,0] = inp[:,0] 
# input_wo_env[:,2] = inp[:,2] 
# input_wo_env[:,3] = inp[:,3]
# pre_wo_env = estimator.predict(input_wo_env)
# plt.scatter(inp[:,0], pre_wo_env, c = 'green',alpha = alpha)
# plt.xlabel('speed(knots)')
# plt.ylabel('Power(kW)')
# plt.title(file[i-2])
# plt.savefig(file[i-2]+"total.jpg")


# ballast = np.mean(inp[:,2])
# laden = np.mean(inp[:,2])

# test = np.zeros((200,len(input_data[0,:])))

# test[:,0] = np.linspace(6,14, len(test[:,0]))
# #test[:,1] = 7.2
# test[:,2] = 18
# test[:,3] = 28#np.mean(inp[:,3])
# prediction = estimator.predict(test)
# plt.figure(2)


# test_speed = test[:,0]
# kw2t = 24*0.185/1000
# test_out = prediction *kw2t
# plt.scatter(test_speed,test_out, hold = 'on')


# popt, pcov = curve_fit(func, test_speed, test_out)

# plt.plot(test_speed, popt[0]* np.power(test_speed,3)+popt[1] ,  color = 'red')
# plt.legend(['Estimated', 'Fitting'], loc = 'upper left')
# plt.grid('on')
# plt.title(file[i-2])
# plt.xlabel('speed(knots)')
# plt.ylabel('Fuel consumption(t/24h)')
# plt.savefig(file[i-2]+".jpg")#, plt.axis([10, 17, 10, 80])
# #input_data = input_data[1:7000,:]
# #target = target[1:7000]
# #sog = sog[1:7000]


# #from sklearn.externals import joblib
# #import pickle
# #joblib.dump(estimator,file[i-1]+".pkl")

# csv_file = open(file[i-2]+".csv","w")
# cw = csv.writer(csv_file , delimiter=',', quotechar='|')
# cw.writerow([test_speed,test_out])
# csv_file.close()

# csv_file = open(file[i-2]+"coe.csv","w")
# cw = csv.writer(csv_file , delimiter=',', quotechar='|')
# cw.writerow(popt)
# csv_file.close()

# """
# """
# min_speed = 9.5
# max_speed = 12.0
# max_draft = 18.3
# min_draft = 16
# max_rudder = 10
# max_drpm = 3     
# min_rpm = 65
# max_rpm = 70
# max_power = 10000
# min_power = 6000     
# max_wind = 16                           
# max_dtarget = np.mean(np.abs(dtarget))
# filter_index  = np.argwhere((label == 2) | (label ==1))
# #filter_index = np.argwhere((sog < min_speed) | (sog >max_speed) |(np.abs(dtarget) > max_dtarget)|(np.abs(drpm) > max_drpm) | (power < min_power) | (power >max_power) | (np.abs(rudder) > max_rudder)|(wind_speed > max_wind))
# #(np.abs(drpm) > 3) | | (rpm < min_rpm) | (rpm >max_rpm) (target > 15000)| (sog<3)

# target = np.delete(target, filter_index, 0)
# input_data = np.delete(input_data, filter_index, 0)

# estimator = gtb(learning_rate = 0.01, max_depth = 7, min_samples_leaf = 2, max_features = 0.5, n_estimators=5000)
# inp_train, inp_test, out_train, out_test = cv.train_test_split(input_data, target, test_size = 0.05)

# #estimator.fit(inp_train, out_train)
# estimator.fit(inp_train, out_train)
# prediction = estimator.predict(input_data)
# alpha = 0.5
# plt.scatter(input_data[:,0], prediction, alpha = alpha, hold = 'on', c = 'red'), plt.scatter(input_data[:,0], target,alpha = alpha)

# input_wo_env = np.zeros((len(input_data[:,0]),len(input_data[0,:])))
# #복제할 것 , 속도, 건드릴 것 draft,
# input_wo_env[:,0] = input_data[:,0] 
# #input_wo_env[:,1] = 17.2#input_data[:,2]
# #input_wo_env[:,1] = 0.1#input_data[:,1]
# #input_wo_env[:,3] = 0
# #input_wo_env[:,5] = 0
# #input_wo_env[:,6] = pi
# pre_wo_env = estimator.predict(input_wo_env)
# plt.scatter(input_data[:,0], pre_wo_env, c = 'green',alpha = alpha)#, plt.axis([11,13,6000,10000])
# plt.show()



# test = np.zeros((20,len(input_data[0,:])))
# #test[:,0] = np.linspace(np.min(input_data[:,0]),np.max(input_data[:,0]), len(test[:,0]))
# test[:,0] = np.linspace(10,13, len(test[:,0]))
# #test[:,1] = 17.2
# prediction = estimator.predict(test)
# plt.figure(2)
# test_speed = test[:,0]
# test_out = prediction
# plt.scatter(test_speed,test_out)
# """