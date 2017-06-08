import pymssql
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# train 데이터 set 만들기
def shipDataQuery(callSign, beginDate, EndDate, dataSplitCount, features, isShuffle):
    conn = pymssql.connect(server='218.39.195.13:21000', user='sa', password='@120bal@', database='SHIP_DB_EARTH')
    # queryDataSet = conn.cursor()
    stmt = "SELECT" + features + "FROM [SHIP_DB].[dbo].[SAILING_DATA] WHERE CALLSIGN ='"+callSign+"' AND TIME_STAMP > '"  + beginDate + "' AND TIME_STAMP <  '"+ EndDate +"' AND AT_SEA = 1 AND PRIMARY_DATA_CHECK = 1 AND ERROR_MEFOFLOW_DATA = 1 AND ERROR_SHAFTREV_DATA = 1 AND DRAFT_FORE >12 AND SPEED_VG > 6 AND SPEED_VG < 18 AND BHP_BY_FOC > 1000 AND SLIP > -50 AND SLIP < 50 ORDER BY TIME_STAMP"

    df = pd.read_sql(stmt,conn)    
    rawDataCount = df.shape[0]
    dataCount =  dataSplitCount



    if df.shape[0] == 0:
        print("error : " + callSign + " : " + str(beginDate) + " ~ " + str(EndDate))
        return 0

    dataSizeRatio = 1 - ( dataSplitCount /  df.shape[0] )
    if  dataSplitCount == 9999 or dataSizeRatio < 0:
        dataCount =  rawDataCount
        dataSizeRatio = 0

    if isShuffle == 1:
        return df.sample(n=dataCount, random_state=0)

    if isShuffle == 0:
        return df.head(dataCount)
   
