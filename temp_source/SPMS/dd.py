
import pymssql

# train 데이터 set 만들기

conn = pymssql.connect(server='218.39.195.13:21000', user='sa', password='@120bal@', database='SHIP_DB_EARTH')
# queryDataSet = conn.cursor()
callSign = '3ewb4'
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
print(result)
   
