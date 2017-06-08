#%%
import shipData as sd
import numpy as np
import pandas as pd

def DataQuery(callSign, startDate, endDate, dataCount, QueryData, isShuffle):
    # # 읽어본 데이터 배열 만들기
    _data = sd.shipDataQuery(callSign, startDate, endDate, dataCount, QueryData, isShuffle)
    _data = _data.dropna(axis=0)
    _data = _data[(_data['SPEED_VG'] > 3) & (_data['SPEED_VG'] < 20) \
    & (_data['SPEED_LW'] > 3) & (_data['SPEED_LW'] < 20) & (_data['SHAFT_REV'] > 10) & (_data['SHAFT_REV'] < 100) \
    & (_data['SLIP'] > -50) & (_data['SLIP'] < 50) & (_data['DRAFT_FORE'] > 3) & (_data['DRAFT_FORE'] < 30) \
    & (_data['DRAFT_AFT'] > 3) & (_data['DRAFT_AFT'] < 30) & (_data['REL_WIND_DIR'] >= 0) & (_data['REL_WIND_DIR'] <= 360) \
    & (_data['REL_WIND_SPEED'] > -200) & (_data['REL_WIND_SPEED'] < 200) & (_data['RUDDER_ANGLE'] > -5) & (_data['RUDDER_ANGLE'] < 5) \
    & (_data['BHP_BY_FOC'] > 1000) & (_data['BHP_BY_FOC'] < 30000)]
    # data = data.loc[:,['SPEED_VG', 'SPEED_LW', 'SLIP', 'DRAFT_FORE', 'DRAFT_AFT', 'REL_WIND_DIR', 'REL_WIND_SPEED', 'RUDDER_ANGLE']]
    return _data




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

data = DataQuery(data_opt['callSign'], data_opt['startTrainDate'], data_opt['endTrainDate'], data_opt['trainDataCount'], data_opt['queryData'], data_opt['isTrainDataShuffle'])
# print(data)

print(data.describe())