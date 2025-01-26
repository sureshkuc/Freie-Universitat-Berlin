from pandas.io import gbq
from matplotlib import pyplot as plt
import pandas as pd

df_social_behaviors = gbq.read_gbq('select * from Students.social_behaviors',project_id='ipads2020-assignment9')
df_conv_frequency = gbq.read_gbq('select * from Students.convo_frequency',project_id='ipads2020-assignment9')
df_conv_duration = gbq.read_gbq('select * from Students.convo_duration',project_id='ipads2020-assignment9')

df_social_behaviors.isna().any()

df_social_behaviors.convo_num_day_avg.fillna(df_social_behaviors.convo_num_day_avg.mean())

df_social_behaviors

df_social_behaviors['convo_num_day_avg'].plot(kind = 'hist', bins = 40)

df_social_behaviors.replace('NA',373.38, inplace = True)

df_social_behaviors

df_social_behaviors.convo_dur_day_avg = df_social_behaviors.convo_dur_day_avg.astype(float) 

df_social_behaviors['convo_dur_day_avg'].plot(kind = 'hist', bins = 30, color = 'red')

df_social_behaviors.groupby(by = 'female').mean()

df_conv_frequency.columns

df_conv_frequency.isna().any()

df_conv_frequency.columns = ['id','userid','date','sums_24hours']

df_conv_frequency.sums_24hours.replace('NA',29, inplace = True)
df_conv_frequency.dtypes



df_conv_frequency = df_conv_frequency[:-1]
df_conv_frequency.sums_24hours = df_conv_frequency.sums_24hours.astype(float)

df_frequency_count = df_conv_frequency.groupby('userid').sum()
df_frequency_count.sort_values('sums_24hours', ascending=False)

df_conv_duration.columns
df_conv_duration.columns = ['id','userid','date','sum_24hours']

df_conv_duration = df_conv_duration[0:-1]

df_conv_duration.tail()

df_conv_duration.replace('NA',275,inplace=True)

df_conv_duration.tail()

df_conv_duration.dtypes

df_conv_duration.sum_24hours = df_conv_duration.sum_24hours.astype(float)

df_dur_count = df_conv_duration.groupby('userid').sum()
print(df_dur_count)
df_dur_count.sort_values('sum_24hours', ascending=False)



from pandas.io import gbq
from matplotlib import pyplot as plt
import pandas as pd

import sys
print(sys.getrecursionlimit())

df_sms_in_freq = gbq.read_gbq('select userid, sums_24hours as SMS_IN_FREQ from Students.S2_sms_in_freq',project_id='ipads2020-assignment9')
df_sms_in_len = gbq.read_gbq('select  userid, sums_24hours as SMS_IN_LEN from Students.S2_sms_in_len',project_id='ipads2020-assignment9')
df_sms_out_freq = gbq.read_gbq('select  userid, sums_24hours as SMS_OUT_FREQ from Students.S2_sms_out_freq',project_id='ipads2020-assignment9')
df_sms_out_len = gbq.read_gbq('select  userid, sums_24hours as SMS_OUT_LEN from Students.S2_sms_out_len',project_id='ipads2020-assignment9')
df_call_out_freq = gbq.read_gbq('select  userid, sums_24hours as CALL_OUT_FREQ from Students.S2_call_out_freq',project_id='ipads2020-assignment9')
df_call_out_len = gbq.read_gbq('select  userid, sums_24hours as CALL_OUT_LEN from Students.S2_call_out_len',project_id='ipads2020-assignment9')
df_call_in_freq = gbq.read_gbq('select  userid, sums_24hours as CALL_IN_FREQ from Students.S2_call_in_freq',project_id='ipads2020-assignment9')
df_call_in_len = gbq.read_gbq('select  userid, sums_24hours as CALL_IN_LEN from Students.S2_call_in_len',project_id='ipads2020-assignment9')

df_sms_in_freq.head()

df_call_in_freq.info()

set_df=['df_call_in_freq','df_call_out_freq','df_call_in_len','df_call_out_len','df_sms_in_freq','df_sms_out_freq','df_sms_in_len','df_sms_out_len']

df_call_in_freq.replace('NA',  0, inplace=True)
df_call_in_freq['CALL_IN_FREQ']=df_call_in_freq.CALL_IN_FREQ.astype('int64')

df_call_in_len.replace('NA',  0, inplace=True)
df_call_in_len['CALL_IN_LEN']=df_call_in_len.CALL_IN_LEN.astype('int64')

df_call_out_freq.replace('NA',  0, inplace=True)
df_call_out_freq['CALL_OUT_FREQ']=df_call_out_freq.CALL_OUT_FREQ.astype('int64')

df_call_out_len.replace('NA',  0, inplace=True)
df_call_out_len['CALL_OUT_LEN']=df_call_out_len.CALL_OUT_LEN.astype('int64')


df_sms_in_freq.replace('NA',  0, inplace=True)
df_sms_in_freq['SMS_IN_FREQ']=df_sms_in_freq.SMS_IN_FREQ.astype('int64')
   
df_sms_in_len.replace('NA',  0, inplace=True)
df_sms_in_len['SMS_IN_LEN']=df_sms_in_len.SMS_IN_LEN.astype('int64')
   
df_sms_out_freq.replace('NA',  0, inplace=True)
df_sms_out_freq['SMS_OUT_FREQ']=df_sms_out_freq.SMS_OUT_FREQ.astype('int64')
   
df_sms_out_len.replace('NA',  0, inplace=True)
df_sms_out_len['SMS_OUT_LEN']=df_sms_out_len.SMS_OUT_LEN.astype('int64')

df_sms_out_freq

pip install rpy2

import rpy2 as ro
from rpy2.robjects import DataFrame, IntVector,FloatVector, pandas2ri
from rpy2.robjects.packages import importr

import rpy2 as ro

package_name="ICC"
ro.robjects.r(f'install.packages("{package_name}")')
pkg=importr(package_name)

def cal_icc():
  r_icc = importr("ICC")
  #def_res[]
  j=0
  set_df=['df_call_in_freq','df_call_out_freq','df_call_in_len','df_call_out_len','df_sms_in_freq','df_sms_out_freq','df_sms_in_len','df_sms_out_len']
  for i in df_call_in_freq,df_call_out_freq,df_call_in_len,df_call_out_len,df_sms_in_freq,df_sms_out_freq,df_sms_in_len,df_sms_out_len :
    df = i
    groups=df['userid'].to_list()
    values=df.iloc[:,1].to_list()
    #print(type(i))
    r_icc = importr("ICC")
    df = DataFrame({"groups": IntVector(groups),
               "values": FloatVector(values)})
    icc_res = r_icc.ICCbare("groups", "values", data=df)
    icc_val = icc_res[0] 
    icc_res1=r_icc.ICCest("groups", "values", data=df)
    icc_res1
    #print( icc_res1)
    #icc_df = pandas2ri.rpy2py(icc[0])
    print(set_df[j])
    print("ICC:"+ str(icc_res1[0])+"Lower ICC :" + str(icc_res1[1])+ "Upper ICC :"+str(icc_res1[2])  )
    j=j+1
   

cal_icc()

df_s2_call_in_freq = gbq.read_gbq('select * from Students.S2_call_in_freq',project_id='ipads2020-assignment9')



df_s2_call_in_freq.head()

from datetime import datetime

df_s2_call_in_freq['date'] = pd.to_datetime(df_s2_call_in_freq['date'], format='%Y%m%d').dt.strftime("%Y-%m-%d")


df_s2_call_in_freq.head()

df_s2_call_in_freq.info()

df_s2_call_in_freq.sums_24hours.replace('NA',  0, inplace=True)

df_s2_call_in_freq['sums_24hours']=df_s2_call_in_freq['sums_24hours'].astype('int64')

df_s2_call_in_freq['userid'].value_counts()

df_s2_var = df_s2_call_in_freq[['userid','sums_24hours']]

df_s2_var.groupby("userid").var().sort_values(by='sums_24hours', ascending=False)

data_u1=df_s2_call_in_freq.query('userid== 173512')

data_u2=df_s2_call_in_freq.query('userid == 553102')

data_u1.head()

data_u2.head()

#userid 173512
data_u1 = [data_u1["date"], data_u1["sums_24hours"]]
headers = ["ds", "y"]
df_u1 = pd.concat(data_u1, axis=1, keys=headers)


#userid 553102
data_u2 = [data_u2["date"], data_u2["sums_24hours"]]
headers = ["ds", "y"]
df_u2 = pd.concat(data_u2, axis=1, keys=headers)

df_u1.head()

df_u2.head()

from fbprophet import Prophet
m1 = Prophet()
m1.fit(df_u1)

future1 = m1.make_future_dataframe(periods=30)
future1.tail()

forecast1 = m1.predict(future1)
#forecast1[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig1 = m1.plot(forecast1)

fig2 = m1.plot_components(forecast1)



from fbprophet.plot import plot_plotly, plot_components_plotly

plot_plotly(m1, forecast1)

plot_components_plotly(m1, forecast1)

m2 = Prophet()
m2.fit(df_u2)

future2 = m2.make_future_dataframe(periods=30)
future2.tail()

forecast2 = m2.predict(future2)
#forecast2[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig1 = m2.plot(forecast2)

fig2 = m2.plot_components(forecast2)



from fbprophet.plot import plot_plotly, plot_components_plotly

plot_plotly(m2, forecast2)

plot_components_plotly(m2, forecast2)

df_s3 = gbq.read_gbq('select * from Students.S3_social_behaviour',project_id='ipads2020-assignment9')

df_s3.head()

r_icc = importr("ICC")
  #def_res[]
  #set_df=['df_call_in_freq','df_call_out_freq','df_call_in_len','df_call_out_len','df_sms_in_freq','df_sms_out_freq','df_sms_in_len','df_sms_out_len']
  #for i in df_call_in_freq,df_call_out_freq,df_call_in_len,df_call_out_len,df_sms_in_freq,df_sms_out_freq,df_sms_in_len,df_sms_out_len :
   # df = i
groups=df_s3['call_out_dur_avg_daily'].to_list()
values=df_s3['BFSI_E'].to_list()
    #print(type(i))
r_icc = importr("ICC")
df = DataFrame({"groups": IntVector(groups),
               "values": FloatVector(values)})
icc_res = r_icc.ICCbare("groups", "values", data=df)
icc_val = icc_res[0] 
icc_res1=r_icc.ICCest("groups", "values", data=df)
icc_res1
#print( icc_res[0])
    #icc_df = pandas2ri.rpy2py(icc[0])
    #print(set_df[j])
print("Daily CALL OUT DUR Extraversion ICC:"+ str(icc_res1[0])+"Lower ICC :" + str(icc_res1[1])+ "Upper ICC :"+str(icc_res1[2])  )

groups=df_s3['call_out_dur_avg_daily'].to_list()
values=df_s3['BFSI_O'].to_list()
    #print(type(i))
r_icc = importr("ICC")
df = DataFrame({"groups": IntVector(groups),
               "values": FloatVector(values)})
icc_res = r_icc.ICCbare("groups", "values", data=df)
icc_val = icc_res[0] 
icc_res1=r_icc.ICCest("groups", "values", data=df)
icc_res1
#print( icc_res[0])
    #icc_df = pandas2ri.rpy2py(icc[0])
    #print(set_df[j])
print("Daily CALL OUT DUR openness ICC:"+ str(icc_res1[0])+"Lower ICC :" + str(icc_res1[1])+ "Upper ICC :"+str(icc_res1[2])  )

groups=df_s3['call_out_dur_avg_daily'].to_list()
values=df_s3['BFSI_C'].to_list()
    #print(type(i))
r_icc = importr("ICC")
df = DataFrame({"groups": IntVector(groups),
               "values": FloatVector(values)})
icc_res = r_icc.ICCbare("groups", "values", data=df)
icc_val = icc_res[0] 
icc_res1=r_icc.ICCest("groups", "values", data=df)
icc_res1
#print( icc_res[0])
    #icc_df = pandas2ri.rpy2py(icc[0])
    #print(set_df[j])
print("Daily CALL OUT DUR conscientiousness ICC:"+ str(icc_res1[0])+"Lower ICC :" + str(icc_res1[1])+ "Upper ICC :"+str(icc_res1[2])  )

groups=df_s3['call_out_dur_avg_daily'].to_list()
values=df_s3['BFSI_A'].to_list()
    #print(type(i))
r_icc = importr("ICC")
df = DataFrame({"groups": IntVector(groups),
               "values": FloatVector(values)})
icc_res = r_icc.ICCbare("groups", "values", data=df)
icc_val = icc_res[0] 
icc_res1=r_icc.ICCest("groups", "values", data=df)
icc_res1
#print( icc_res[0])
    #icc_df = pandas2ri.rpy2py(icc[0])
    #print(set_df[j])
print("Daily CALL OUT DUR agreeableness ICC:"+ str(icc_res1[0])+"Lower ICC :" + str(icc_res1[1])+ "Upper ICC :"+str(icc_res1[2])  )

groups=df_s3['call_out_dur_avg_daily'].to_list()
values=df_s3['BFSI_N'].to_list()
    #print(type(i))
r_icc = importr("ICC")
df = DataFrame({"groups": IntVector(groups),
               "values": FloatVector(values)})
icc_res = r_icc.ICCbare("groups", "values", data=df)
icc_val = icc_res[0] 
icc_res1=r_icc.ICCest("groups", "values", data=df)
icc_res1
print("Daily CALL OUT DUR  neuroticism ICC:"+ str(icc_res1[0])+"Lower ICC :" + str(icc_res1[1])+ "Upper ICC :"+str(icc_res1[2])  )

r_icc = importr("ICC")
  #def_res[]
  #set_df=['df_call_in_freq','df_call_out_freq','df_call_in_len','df_call_out_len','df_sms_in_freq','df_sms_out_freq','df_sms_in_len','df_sms_out_len']
  #for i in df_call_in_freq,df_call_out_freq,df_call_in_len,df_call_out_len,df_sms_in_freq,df_sms_out_freq,df_sms_in_len,df_sms_out_len :
   # df = i
groups=df_s3['call_in_dur_avg_daily'].to_list()
values=df_s3['BFSI_E'].to_list()
    #print(type(i))
r_icc = importr("ICC")
df = DataFrame({"groups": IntVector(groups),
               "values": FloatVector(values)})
icc_res = r_icc.ICCbare("groups", "values", data=df)
icc_val = icc_res[0] 
icc_res1=r_icc.ICCest("groups", "values", data=df)
icc_res1
#print( icc_res[0])
    #icc_df = pandas2ri.rpy2py(icc[0])
    #print(set_df[j])
print("Daily CALL IN DUR Extraversion ICC:"+ str(icc_res1[0])+"Lower ICC :" + str(icc_res1[1])+ "Upper ICC :"+str(icc_res1[2])  )

groups=df_s3['call_in_dur_avg_daily'].to_list()
values=df_s3['BFSI_O'].to_list()
    #print(type(i))
r_icc = importr("ICC")
df = DataFrame({"groups": IntVector(groups),
               "values": FloatVector(values)})
icc_res = r_icc.ICCbare("groups", "values", data=df)
icc_val = icc_res[0] 
icc_res1=r_icc.ICCest("groups", "values", data=df)
icc_res1
#print( icc_res[0])
    #icc_df = pandas2ri.rpy2py(icc[0])
    #print(set_df[j])
print("Daily CALL IN DUR openness ICC:"+ str(icc_res1[0])+"Lower ICC :" + str(icc_res1[1])+ "Upper ICC :"+str(icc_res1[2])  )

groups=df_s3['call_in_dur_avg_daily'].to_list()
values=df_s3['BFSI_C'].to_list()
    #print(type(i))
r_icc = importr("ICC")
df = DataFrame({"groups": IntVector(groups),
               "values": FloatVector(values)})
icc_res = r_icc.ICCbare("groups", "values", data=df)
icc_val = icc_res[0] 
icc_res1=r_icc.ICCest("groups", "values", data=df)
icc_res1
#print( icc_res[0])
    #icc_df = pandas2ri.rpy2py(icc[0])
    #print(set_df[j])
print("Daily CALL IN DUR conscientiousness ICC:"+ str(icc_res1[0])+"Lower ICC :" + str(icc_res1[1])+ "Upper ICC :"+str(icc_res1[2])  )

groups=df_s3['call_in_dur_avg_daily'].to_list()
values=df_s3['BFSI_A'].to_list()
    #print(type(i))
r_icc = importr("ICC")
df = DataFrame({"groups": IntVector(groups),
               "values": FloatVector(values)})
icc_res = r_icc.ICCbare("groups", "values", data=df)
icc_val = icc_res[0] 
icc_res1=r_icc.ICCest("groups", "values", data=df)
icc_res1
#print( icc_res[0])
    #icc_df = pandas2ri.rpy2py(icc[0])
    #print(set_df[j])
print("Daily CALL IN DUR agreeableness ICC:"+ str(icc_res1[0])+"Lower ICC :" + str(icc_res1[1])+ "Upper ICC :"+str(icc_res1[2])  )

groups=df_s3['call_in_dur_avg_daily'].to_list()
values=df_s3['BFSI_N'].to_list()
    #print(type(i))
r_icc = importr("ICC")
df = DataFrame({"groups": IntVector(groups),
               "values": FloatVector(values)})
icc_res = r_icc.ICCbare("groups", "values", data=df)
icc_val = icc_res[0] 
icc_res1=r_icc.ICCest("groups", "values", data=df)
icc_res1
print("Daily CALL IN DUR  neuroticism ICC:"+ str(icc_res1[0])+"Lower ICC :" + str(icc_res1[1])+ "Upper ICC :"+str(icc_res1[2])  )

r_icc = importr("ICC")
  #def_res[]
  #set_df=['df_call_in_freq','df_call_out_freq','df_call_in_len','df_call_out_len','df_sms_in_freq','df_sms_out_freq','df_sms_in_len','df_sms_out_len']
  #for i in df_call_in_freq,df_call_out_freq,df_call_in_len,df_call_out_len,df_sms_in_freq,df_sms_out_freq,df_sms_in_len,df_sms_out_len :
   # df = i
groups=df_s3['sms_in_num_avg_daily'].to_list()
values=df_s3['BFSI_E'].to_list()
    #print(type(i))
r_icc = importr("ICC")
df = DataFrame({"groups": IntVector(groups),
               "values": FloatVector(values)})
icc_res = r_icc.ICCbare("groups", "values", data=df)
icc_val = icc_res[0] 
icc_res1=r_icc.ICCest("groups", "values", data=df)
icc_res1
#print( icc_res[0])
    #icc_df = pandas2ri.rpy2py(icc[0])
    #print(set_df[j])
print("Daily SMS IN NUM Extraversion ICC:"+ str(icc_res1[0])+"Lower ICC :" + str(icc_res1[1])+ "Upper ICC :"+str(icc_res1[2])  )

groups=df_s3['sms_in_num_avg_daily'].to_list()
values=df_s3['BFSI_O'].to_list()
    #print(type(i))
r_icc = importr("ICC")
df = DataFrame({"groups": IntVector(groups),
               "values": FloatVector(values)})
icc_res = r_icc.ICCbare("groups", "values", data=df)
icc_val = icc_res[0] 
icc_res1=r_icc.ICCest("groups", "values", data=df)
icc_res1
#print( icc_res[0])
    #icc_df = pandas2ri.rpy2py(icc[0])
    #print(set_df[j])
print("Daily SMS IN NUM openness ICC:"+ str(icc_res1[0])+"Lower ICC :" + str(icc_res1[1])+ "Upper ICC :"+str(icc_res1[2])  )

groups=df_s3['sms_in_num_avg_daily'].to_list()
values=df_s3['BFSI_C'].to_list()
    #print(type(i))
r_icc = importr("ICC")
df = DataFrame({"groups": IntVector(groups),
               "values": FloatVector(values)})
icc_res = r_icc.ICCbare("groups", "values", data=df)
icc_val = icc_res[0] 
icc_res1=r_icc.ICCest("groups", "values", data=df)
icc_res1
#print( icc_res[0])
    #icc_df = pandas2ri.rpy2py(icc[0])
    #print(set_df[j])
print("Daily SMS IN NUM conscientiousness ICC:"+ str(icc_res1[0])+"Lower ICC :" + str(icc_res1[1])+ "Upper ICC :"+str(icc_res1[2])  )

groups=df_s3['sms_in_num_avg_daily'].to_list()
values=df_s3['BFSI_A'].to_list()
    #print(type(i))
r_icc = importr("ICC")
df = DataFrame({"groups": IntVector(groups),
               "values": FloatVector(values)})
icc_res = r_icc.ICCbare("groups", "values", data=df)
icc_val = icc_res[0] 
icc_res1=r_icc.ICCest("groups", "values", data=df)
icc_res1
#print( icc_res[0])
    #icc_df = pandas2ri.rpy2py(icc[0])
    #print(set_df[j])
print("Daily SMS IN NUM agreeableness ICC:"+ str(icc_res1[0])+"Lower ICC :" + str(icc_res1[1])+ "Upper ICC :"+str(icc_res1[2])  )

groups=df_s3['sms_in_num_avg_daily'].to_list()
values=df_s3['BFSI_N'].to_list()
    #print(type(i))
r_icc = importr("ICC")
df = DataFrame({"groups": IntVector(groups),
               "values": FloatVector(values)})
icc_res = r_icc.ICCbare("groups", "values", data=df)
icc_val = icc_res[0] 
icc_res1=r_icc.ICCest("groups", "values", data=df)
icc_res1
print("Daily SMS IN NUM  neuroticism ICC:"+ str(icc_res1[0])+"Lower ICC :" + str(icc_res1[1])+ "Upper ICC :"+str(icc_res1[2])  )

df_s3_social_behaviors = gbq.read_gbq('select * from Students.S3_social_behaviour',project_id='ipads2020-assignment9')

df_s3_social_behaviors.shape

df_s3_social_behaviors

df_s3_social_behaviors.info(verbose=True,null_counts=True)

df_s3_social_behaviors['demog_sex'].value_counts()

import numpy as np
pd.set_option('display.max_rows', None)

print(df_s3_social_behaviors[df_s3_social_behaviors.columns[1:]].corr()['BFSI_E'][:].sort_values(ascending=False))

df_s3_social_behaviors

import pandas as pd
import numpy as np

dum_df = pd.get_dummies(df_s3_social_behaviors, columns=["demog_sex"] )
dum_df

dum_df.info(verbose=True,null_counts=True)

X=dum_df.drop(['BFSI_O', 'BFSI_C', 'BFSI_E','BFSI_A','BFSI_N','int64_field_0'], axis=1)

X.head()

y = dum_df[['BFSI_O', 'BFSI_C', 'BFSI_E','BFSI_A','BFSI_N']].copy()

y.head()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X=X.values
y=y.values
xtrain, xtest, ytrain, ytest=train_test_split(X, y, test_size=0.15)
print("xtrain:", xtrain.shape, "ytrian:", ytrain.shape)

print("xtest:", xtest.shape, "ytest:", ytest.shape)

scalar=StandardScaler()
scalar.fit(xtrain)
xtrain=scalar.transform(xtrain)
xtest=scalar.transform(xtest)
 

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
#gradient bossting model
gbr = GradientBoostingRegressor(learning_rate=0.1,n_estimators=100, max_depth=1)
model = MultiOutputRegressor(estimator=gbr)
print(model)

model.fit(xtrain, ytrain)
score = model.score(xtrain, ytrain)
print("Training score:", score)
model.score(xtest, ytest)

from sklearn.metrics import mean_squared_error
ypred = model.predict(xtest)



print("y1 BFSI_O MSE:%.4f" % mean_squared_error(ytest[:,0], ypred[:,0]))
print("y2 BFSI_C MSE:%.4f" % mean_squared_error(ytest[:,1], ypred[:,1]))
print("y3 BFSI_E MSE:%.4f" % mean_squared_error(ytest[:,2], ypred[:,2]))
print("y4 BFSI_A MSE:%.4f" % mean_squared_error(ytest[:,3], ypred[:,3]))
print("y5 BFSI_N MSE:%.4f" % mean_squared_error(ytest[:,4], ypred[:,4]))

pip install autofeat

def test_autofeat(dataset, feateng_steps=2):
    # load data
    X, y=dataset
    units=None
    # split in training and test parts
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
    # run autofeat
    afreg = AutoFeatRegressor(verbose=1, feateng_steps=feateng_steps, units=units)
    # fit autofeat on less data, otherwise ridge reg model with xval will overfit on new features
    X_train_tr = afreg.fit_transform(X_train, y_train)
    X_test_tr = afreg.transform(X_test)
    print("autofeat new features:", len(afreg.new_feat_cols_))
    print("autofeat MSE on training data:", mean_squared_error(y_train, afreg.predict(X_train_tr)))
    print("autofeat MSE on test data:", mean_squared_error(y_test, afreg.predict(X_test_tr)))
    print("autofeat R^2 on training data:", r2_score(y_train, afreg.predict(X_train_tr)))
    print("autofeat R^2 on test data:", r2_score(y_test, afreg.predict(X_test_tr)))
    # train rreg on transformed train split incl cross-validation for parameter selection
    print("# Ridge Regression")
    rreg = Ridge()
    param_grid = {"alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1., 2.5, 5., 10., 25., 50., 100., 250., 500., 1000., 2500., 5000., 10000.]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gsmodel = GridSearchCV(rreg, param_grid, scoring='neg_mean_squared_error', cv=5)
        gsmodel.fit(X_train_tr, y_train)
    print("best params:", gsmodel.best_params_)
    print("best score:", gsmodel.best_score_)
    print("MSE on training data:", mean_squared_error(y_train, gsmodel.predict(X_train_tr)))
    print("MSE on test data:", mean_squared_error(y_test, gsmodel.predict(X_test_tr)))
    print("R^2 on training data:", r2_score(y_train, gsmodel.predict(X_train_tr)))
    print("R^2 on test data:", r2_score(y_test, gsmodel.predict(X_test_tr)))
    print("# Random Forest")
    rforest = RandomForestRegressor(n_estimators=100, random_state=13)
    param_grid = {"min_samples_leaf": [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2]}
    gsmodel = GridSearchCV(rforest, param_grid, scoring='neg_mean_squared_error', cv=5)
    gsmodel.fit(X_train_tr, y_train)
    print("best params:", gsmodel.best_params_)
    print("best score:", gsmodel.best_score_)
    print("MSE on training data:", mean_squared_error(y_train, gsmodel.predict(X_train_tr)))
    print("MSE on test data:", mean_squared_error(y_test, gsmodel.predict(X_test_tr)))
    print("R^2 on training data:", r2_score(y_train, gsmodel.predict(X_train_tr)))
    print("R^2 on test data:", r2_score(y_test, gsmodel.predict(X_test_tr)))


import os
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston, load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from autofeat import AutoFeatRegressor

test_autofeat((xtrain, ytrain[:,0]), feateng_steps=1)

from sklearn import linear_model
clf = linear_model.Lasso(alpha=10000)
clf.fit(xtrain, ytrain[:,2])
ypred=clf.predict(xtest)
clf.score(xtest, ytest[:,2])


from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor


#adaboost model
ada = AdaBoostRegressor()
model1 = MultiOutputRegressor(estimator=ada)

model1.fit(xtrain, ytrain)
score = model1.score(xtrain, ytrain)
print("Training score:", score)

ypred = model1.predict(xtest)
print("y1 BFSI_O MSE:%.4f" % mean_squared_error(ytest[:,0], ypred[:,0]))
print("y2 BFSI_C MSE:%.4f" % mean_squared_error(ytest[:,1], ypred[:,1]))
print("y3 BFSI_E MSE:%.4f" % mean_squared_error(ytest[:,2], ypred[:,2]))
print("y4 BFSI_A MSE:%.4f" % mean_squared_error(ytest[:,3], ypred[:,3]))
print("y5 BFSI_N MSE:%.4f" % mean_squared_error(ytest[:,4], ypred[:,4]))

from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
#randomforest regressor
rfr = RandomForestRegressor(n_estimators=200,max_depth=5 , max_features='log2' )
model2 = MultiOutputRegressor(estimator=rfr)
model2.fit(xtrain, ytrain)
score = model2.score(xtrain, ytrain)
print("Training score:", score,'test score', model2.score(xtest, ytest))

#decision treee
dt=DecisionTreeRegressor(max_depth=1)
model3= MultiOutputRegressor(estimator=dt)
model3.fit(xtrain, ytrain)
score = model3.score(xtrain, ytrain)
print("Training score:", score,'test score', model3.score(xtest, ytest))
#support vector regressor
svr=SVR()
model4= MultiOutputRegressor(estimator=svr)
model4.fit(xtrain, ytrain)
score = model4.score(xtrain, ytrain)
print("Training score:", score,'test score', model4.score(xtest, ytest))

#randomforestmodel prediction
ypred = model2.predict(xtest)
model_list=['BFSI_O ','BFSI_C','BFSI_E ','BFSI_A','BFSI_N']
print("y1 BFSI_O MSE:%.4f" % mean_squared_error(ytest[:,0], ypred[:,0]))
print("y2 BFSI_C MSE:%.4f" % mean_squared_error(ytest[:,1], ypred[:,1]))
print("y3 BFSI_E MSE:%.4f" % mean_squared_error(ytest[:,2], ypred[:,2]))
print("y4 BFSI_A MSE:%.4f" % mean_squared_error(ytest[:,3], ypred[:,3]))
print("y5 BFSI_N MSE:%.4f" % mean_squared_error(ytest[:,4], ypred[:,4]))

from sklearn.metrics import mean_squared_error, r2_score
for i,model in enumerate(model_list):
  print(model,"R2 score : %.2f" % r2_score(ytest[:,i],ypred[:,i]))

x_ax = range(len(xtest))
#plt.plot(x_ax, ytest[:,0], label="y1-test", color='c')
#plt.plot(x_ax, ypred[:,0], label="y1-pred", color='b')
plt.plot(x_ax, ytest[:,2], label="y2-test", color='m')
plt.plot(x_ax, ypred[:,2], label="y2-pred", color='r')
plt.legend()
plt.show()



#Classification Task

ytrain1=np.where(ytrain>0,1,0)
ytest1=np.where(ytest>0,1,0)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
for i, val in enumerate(model_list):
  model5 = LogisticRegression( C=0.01)
  model5.fit(xtrain, ytrain1[:,i])
  y_pred=model5.predict_proba(xtest)
  #cm = confusion_matrix(ytest1[:,2], y_pred)
  pred=np.where(y_pred[:,1]>0.5,1,0)
  #print(classification_report(ytest1[:,i],pred))
  fpr, tpr, thresholds = metrics.roc_curve(ytest1[:,i],z)
  print(val,'auc:',metrics.auc(fpr, tpr))
