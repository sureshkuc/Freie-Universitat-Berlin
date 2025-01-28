from pandas.io import gbq
from matplotlib import pyplot as plt
import pandas as pd

df_activity_u10 = gbq.read_gbq('select * from Students.activity_u10',project_id='ipads2020assignment8')
df_activity_u16 = gbq.read_gbq('select * from Students.activity_u16',project_id='ipads2020assignment8')

df_stressscale = gbq.read_gbq('select string_field_0,string_field_4	 from Students.stressScale', project_id='ipads2020assignment8')

df_stressscale.shape

df_activity_u10.head()

df_activity_u10.shape

df_activity_u16.tail()

df_activity_u16.shape

df_stressscale.columns

df_stressscale.columns = ['UID', 'In the last month, how often have you felt nervous and "stressed"?']

df_stressscale.drop(df_stressscale.tail(1).index,inplace=True)

print('List of frequency of the students who felt stressful in the last one month : ')
df_stressCounts = df_stressscale['In the last month, how often have you felt nervous and "stressed"?'].value_counts()
print(df_stressCounts)

labels = df_stressCounts.index
counts = df_stressCounts.values
fig, ax = plt.subplots(figsize = (5,5))
ax.pie(counts, labels = labels)
plt.show()

gbq.read_gbq('select string_field_4	 from Students.stressScale where string_field_0 = "u10"', project_id='ipads2020assignment8')

gbq.read_gbq('select string_field_4	 from Students.stressScale where string_field_0 = "u16"', project_id='ipads2020assignment8')

df_activity_u10['timestamp']=pd.to_datetime(df_activity_u10['timestamp'],unit='s')
df_activity_u16['timestamp']=pd.to_datetime(df_activity_u16['timestamp'],unit='s')
df_activity_u10.dropna(axis=0,inplace = True)
df_activity_u16.dropna(axis=0,inplace = True)

df_activity_u10_counts = df_activity_u10['_activity_inference'].value_counts()
labels = ['Stationary', 'Walking','Unknown','Running']
counts = df_activity_u10_counts.values
fig, ax = plt.subplots(figsize = (5,5))
ax.pie(counts, labels = labels)
plt.show()

df_activity_u16_counts = df_activity_u16['_activity_inference'].value_counts()
labels = ['Stationary', 'Walking','Unknown','Running']
counts = df_activity_u16_counts.values
fig, ax = plt.subplots(figsize = (5,5))
ax.pie(counts, labels = labels)
plt.show()

U10_Stationary = (df_activity_u10['_activity_inference'].value_counts()[0]/df_activity_u10.shape[0])*100
U16_Stationary = (df_activity_u16['_activity_inference'].value_counts()[0]/df_activity_u16.shape[0])*100
U10_Activness = 100 - U10_Stationary
U16_Activness = 100 - U16_Stationary
from tabulate import tabulate
print(tabulate([['U10', U10_Stationary,U10_Activness], ['U16', U16_Stationary,U16_Activness]], headers=['Name', 'Stationary','Activity']))

df_activity = pd.concat([df_activity_u10, df_activity_u16])
df_activity.shape

df_activity['timestamp']=pd.to_datetime(df_activity['timestamp'],unit='s')
df_activity.head()

print('Students Stationary time (In Percentage) is',(df_activity['_activity_inference'].value_counts()[0]/df_activity.shape[0])*100)



from fbprophet import Prophet

#df_act = df_activity.drop('timestamp', axis=1)
#df_act

df_activity.columns = ['y', 'ds']

df_activity

df_activity.dropna(axis = 0)

#Swapping the Columns 
columns_titles = ["ds","y"] 

df_reorder=df_activity.reindex(columns=columns_titles)
df_reorder

df_reorder.dropna(axis = 0, inplace = True)
df_reorder.isna().any()

m = Prophet()
m.fit(df_reorder)

future = m.make_future_dataframe(periods = 60)
future.tail()

forecast = m.predict(df_reorder)
fig1 = m.plot(forecast)






from pandas.io import gbq
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

df_activity_u10 = gbq.read_gbq('select * from Students.activity_u10',project_id='ipads2020assignment8')
df_activity_u16 = gbq.read_gbq('select * from Students.activity_u16',project_id='ipads2020assignment8')

st_list=['u00','u01','u02','u03','u04','u05','u07.','u08','u09','u10']
stress_val_con={'Fairly often':0,'Sometime':1,'Almost never':2,'Very often':3}
df_stressscale = gbq.read_gbq('select string_field_0,string_field_4	 from Students.stressScale where string_field_1="pre" ', project_id='ipads2020assignment8')
file_list=['conversation_u00.csv',
 'conversation_u01.csv',
 'conversation_u02.csv',
 'conversation_u03.csv',
 'conversation_u04.csv',
 'conversation_u05.csv',
 'conversation_u08.csv',
 'conversation_u09.csv',
 'conversation_u10.csv']
avg_list=[]
for table in file_list:
  query='SELECT avg(start_timestamp-_end_timestamp) FROM Students.'+table.split('.')[0]
  temp=gbq.read_gbq(query, project_id='ipads2020assignment8').head()
  avg_list.append([x[1] for x in temp.itertuples()][0])
avg_list
stress_dict={}
for i in df_stressscale.itertuples():
  if i[1] in st_list:
    stress_dict[i[1]]=i[2]
stress_class=[]
for key,val in stress_dict.items():
   stress_dict[key]=stress_val_con[val]
   stress_class.append(stress_val_con[val])
#correlation between stress and conversation
np.corrcoef(avg_list, stress_class)