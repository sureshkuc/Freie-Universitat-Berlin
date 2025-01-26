from pandas.io import gbq
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
# Importing the required packages 
import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

#task 2 & 3
# Classifer model to predict the pre precieved stress


#student list for pre  preceived stress prediction model
st_list=['u00',
 'u01',
 'u02',
 'u03',
 'u04',
 'u05',
 'u07',
 'u08',
 'u09',
 'u10',
 'u12',
 'u13',
 'u14',
 'u15',
 'u16',
 'u17',
 'u18',
 'u19',
 'u20',
 'u22',
 'u23',
 'u24',
 'u27',
 'u30',
 'u32',
 'u33',
 'u34',
 'u35',
 'u36',
 'u42',
 'u43',
 'u44',
 'u45',
 'u46',
 'u47',
 'u49',
 'u50',
 'u51',
 'u52',
 'u53',
 'u54',
 'u56',
 'u57',
 'u58']
stress_val_con={'Almost never':0,'Sometime':1,'Fairly often':2,'Very often':3}
df_stressscale = gbq.read_gbq('select string_field_0,string_field_4	 from Students.stressScale where string_field_1="pre" ', project_id='ipads2020assignment8')
file_list=['conversation_'+x for x in st_list]
avg_list=[]
for table in file_list:
  try:
    print('try',table)
    query='SELECT avg(start_timestamp-_end_timestamp) FROM Students.'+table.split('.')[0]
    temp=gbq.read_gbq(query, project_id='ipads2020assignment8').head()
    avg_list.append([x[1] for x in temp.itertuples()][0])
  except:
    print('except',table)
    query='SELECT avg(start_timestamp-end_timestamp) FROM Students.'+table.split('.')[0]
    temp=gbq.read_gbq(query, project_id='ipads2020assignment8').head()
    avg_list.append([x[1] for x in temp.itertuples()][0])
    pass
avg_list
stress_dict={}
for i in df_stressscale.itertuples():
  if i[1] in st_list:
    stress_dict[i[1]]=i[2]
stress_class=[]
for key in st_list:
  stress_dict[key]=stress_val_con[stress_dict[key]]
  stress_class.append(stress_dict[key])
#correlation between stress and conversation
np.corrcoef(avg_list, stress_class)

file_list=['phonecharge_'+x for x in st_list]
avg_list1=avg_list.copy()
avg_list=[]
for table in file_list:
  query='SELECT avg(t.start- t.end) FROM Students.'+table.split('.')[0]+' as t'
  temp=gbq.read_gbq(query, project_id='ipads2020assignment8').head()
  avg_list.append([x[1] for x in temp.itertuples()][0])
avg_list

#correlation between stress and phonecharge
np.corrcoef(avg_list, stress_class)

file_list=['phonelock_'+x for x in st_list]
avg_list2=avg_list.copy()
avg_list=[]
for table in file_list:
  query='SELECT avg(t.start- t.end) FROM Students.'+table.split('.')[0]+' as t'
  temp=gbq.read_gbq(query, project_id='ipads2020assignment8').head()
  avg_list.append([x[1] for x in temp.itertuples()][0])
avg_list

#correlation between stress and phone lock
np.corrcoef(avg_list, stress_class)

file_list=['activity_'+x for x in st_list]
avg_list3=avg_list.copy()
avg_list=[]
for table in file_list:
  try: 
    query='SELECT avg(activity_inference) FROM Students.'+table.split('.')[0]
    temp=gbq.read_gbq(query, project_id='ipads2020assignment8').head()
    avg_list.append([x[1] for x in temp.itertuples()][0])
  except:
    #print('except',table)
    query='SELECT avg(_activity_inference) FROM Students.'+table.split('.')[0]
    temp=gbq.read_gbq(query, project_id='ipads2020assignment8').head()
    avg_list.append([x[1] for x in temp.itertuples()][0])
    pass

avg_list

#correlation between stress and activity
np.corrcoef(avg_list, stress_class)

file_list=['Sleep_'+x for x in st_list]
hour=[]
rate=[]
social=[]
for table in file_list:
  query='SELECT avg(t.hour),avg(rate),avg(social) FROM Students.'+table.split('.')[0]+' as t'
  temp=gbq.read_gbq(query, project_id='ipads2020assignment8').head()
  hour.append([x[1] for x in temp.itertuples()][0])
  rate.append([x[2] for x in temp.itertuples()][0])
  social.append([x[3] for x in temp.itertuples()][0])

#correlation between stress and sleep hour, rate and social next day
print(np.corrcoef(hour, stress_class))
print(np.corrcoef(rate, stress_class))
print(np.corrcoef(social, stress_class))

file_list=['Class_'+x for x in st_list]
experience=[]
hours=[]
due=[]
for table in file_list:
  query='SELECT avg(t.experience),avg(t.hours),avg(t.due) FROM Students.'+table.split('.')[0]+' as t'
  try:
    temp=gbq.read_gbq(query, project_id='ipads2020assignment8').head()
    experience.append([x[1] for x in temp.itertuples()][0])
    hours.append([x[2] for x in temp.itertuples()][0])
    due.append([x[3] for x in temp.itertuples()][0])
  except:
    experience.append(0)
    hours.append(0)
    due.append(0)

#correlation between stress and class experience, studied outside class (hours), assignment due
print(np.corrcoef(experience, stress_class))
print(np.corrcoef(hours, stress_class))
print(np.corrcoef(due, stress_class))

file_list=['Lab_'+x for x in st_list]
enjoy=[]
duration=[]
for table in file_list:
  try:
    query='SELECT avg(t.enjoy ),avg(t.duration) FROM Students.'+table.split('.')[0]+' as t'
    temp=gbq.read_gbq(query, project_id='ipads2020assignment8').head()
    enjoy.append([x[1] for x in temp.itertuples()][0])
    duration.append([x[2] for x in temp.itertuples()][0])
  except:
    enjoy.append(0)
    duration.append(0)
    pass

#correlation between stress and class experience, studied outside class (hours), assignment due
print(np.corrcoef(enjoy, stress_class))
print(np.corrcoef(duration, stress_class))

X=[]
for i, j,k,l,m,n,o,p,q,r,s,t in zip(avg_list1,avg_list2,avg_list3,avg_list,hour,rate,social,experience, hours, due, enjoy, duration ):
    X.append([i,j,k,l,m,n,o,p,q,r,s,t])
X=np.array(X)
Y=np.array(stress_class)
X_copy,Y_copy=X.copy(),Y.copy()


X_train, X_test, y_train, y_test = train_test_split( X_copy, Y_copy, test_size = 0.2,random_state=82)
# Creating the classifier object 
clf_gini = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=5, min_samples_leaf=1)
# Performing training 
clf_gini.fit(X_train, y_train) 
y_pred=clf_gini.predict(X_test)
print("Report decision tree(pre): ", 
classification_report(y_test, y_pred)) 

clf = AdaBoostClassifier(n_estimators=1000, random_state=0)
clf.fit(X_train, y_train) 
y_pred=clf.predict(X_test)
print("Report Adaboost(pre): ",
classification_report(y_test, y_pred)) 


#student list for post precieved stress prediction 
st_list=['u00',
 'u01',
 'u02',
 'u03',
 'u04',
 'u05',
 'u07',
 'u09',
 'u10',
 'u14',
 'u15',
 'u16',
 'u17',
 'u18',
 'u19',
 'u20',
 'u23',
 'u24',
 'u27',
 'u30',
 'u31',
 'u32',
 'u33',
 'u34',
 'u35',
 'u36',
 'u42',
 'u43',
 'u44',
 'u45',
 'u46',
 'u47',
 'u49',
 'u51',
 'u52',
 'u53',
 'u54',
 'u56']
stress_val_con={'Fairly often':2,'Sometime':1,'Almost never':0,'Very often':3}
df_stressscale = gbq.read_gbq('select string_field_0,string_field_4	 from Students.stressScale where string_field_1="post" ', project_id='ipads2020assignment8')
file_list=['conversation_'+x for x in st_list]
avg_list=[]
for table in file_list:
  try:
    print('try',table)
    query='SELECT avg(start_timestamp-_end_timestamp) FROM Students.'+table.split('.')[0]
    temp=gbq.read_gbq(query, project_id='ipads2020assignment8').head()
    avg_list.append([x[1] for x in temp.itertuples()][0])
  except:
    print('except',table)
    query='SELECT avg(start_timestamp-end_timestamp) FROM Students.'+table.split('.')[0]
    temp=gbq.read_gbq(query, project_id='ipads2020assignment8').head()
    avg_list.append([x[1] for x in temp.itertuples()][0])
    pass


avg_list
stress_dict={}
for i in df_stressscale.itertuples():
  if i[1] in st_list:
    stress_dict[i[1]]=i[2]
stress_class=[]
for key in st_list:
  stress_dict[key]=stress_val_con[stress_dict[key]]
  stress_class.append(stress_dict[key])
#correlation between stress and conversation
np.corrcoef(avg_list, stress_class)

file_list=['phonecharge_'+x for x in st_list]
avg_list1=avg_list.copy()
avg_list=[]
for table in file_list:
  query='SELECT avg(t.start- t.end) FROM Students.'+table.split('.')[0]+' as t'
  temp=gbq.read_gbq(query, project_id='ipads2020assignment8').head()
  avg_list.append([x[1] for x in temp.itertuples()][0])
avg_list

#correlation between stress and phonecharge
np.corrcoef(avg_list, stress_class)

file_list=['phonelock_'+x for x in st_list]
avg_list2=avg_list.copy()
avg_list=[]
for table in file_list:
  query='SELECT avg(t.start- t.end) FROM Students.'+table.split('.')[0]+' as t'
  temp=gbq.read_gbq(query, project_id='ipads2020assignment8').head()
  avg_list.append([x[1] for x in temp.itertuples()][0])
avg_list

#correlation between stress and phone lock
np.corrcoef(avg_list, stress_class)

file_list=['activity_'+x for x in st_list]
avg_list3=avg_list.copy()
avg_list=[]
for table in file_list:
  try: 
    query='SELECT avg(activity_inference) FROM Students.'+table.split('.')[0]
    temp=gbq.read_gbq(query, project_id='ipads2020assignment8').head()
    avg_list.append([x[1] for x in temp.itertuples()][0])
  except:
    print('except',table)
    query='SELECT avg(_activity_inference) FROM Students.'+table.split('.')[0]
    temp=gbq.read_gbq(query, project_id='ipads2020assignment8').head()
    avg_list.append([x[1] for x in temp.itertuples()][0])
    pass

avg_list

#correlation between stress and activity
np.corrcoef(avg_list, stress_class)

file_list=['Sleep_'+x for x in st_list]
hour=[]
rate=[]
social=[]
for table in file_list:
  query='SELECT avg(t.hour),avg(rate),avg(social) FROM Students.'+table.split('.')[0]+' as t'
  temp=gbq.read_gbq(query, project_id='ipads2020assignment8').head()
  hour.append([x[1] for x in temp.itertuples()][0])
  rate.append([x[2] for x in temp.itertuples()][0])
  social.append([x[3] for x in temp.itertuples()][0])

#correlation between stress and sleep hour, rate and social next day
print(np.corrcoef(hour, stress_class))
print(np.corrcoef(rate, stress_class))
print(np.corrcoef(social, stress_class))

file_list=['Class_'+x for x in st_list]
experience=[]
hours=[]
due=[]
for table in file_list:
  query='SELECT avg(t.experience),avg(t.hours),avg(t.due) FROM Students.'+table.split('.')[0]+' as t'
  try:
    temp=gbq.read_gbq(query, project_id='ipads2020assignment8').head()
    experience.append([x[1] for x in temp.itertuples()][0])
    hours.append([x[2] for x in temp.itertuples()][0])
    due.append([x[3] for x in temp.itertuples()][0])
  except:
    experience.append(0)
    hours.append(0)
    due.append(0)

#correlation between stress and class experience, studied outside class (hours), assignment due
print(np.corrcoef(experience, stress_class))
print(np.corrcoef(hours, stress_class))
print(np.corrcoef(due, stress_class))

file_list=['Lab_'+x for x in st_list]
enjoy=[]
duration=[]
for table in file_list:
  try:
    query='SELECT avg(t.enjoy ),avg(t.duration) FROM Students.'+table.split('.')[0]+' as t'
    temp=gbq.read_gbq(query, project_id='ipads2020assignment8').head()
    enjoy.append([x[1] for x in temp.itertuples()][0])
    duration.append([x[2] for x in temp.itertuples()][0])
  except:
    enjoy.append(0)
    duration.append(0)
    pass

#correlation between stress and class experience, studied outside class (hours), assignment due
print(np.corrcoef(enjoy, stress_class))
print(np.corrcoef(duration, stress_class))

X=[]
for i, j,k,l,m,n,o,p,q,r,s,t in zip(avg_list1,avg_list2,avg_list3,avg_list,hour,rate,social,experience, hours, due, enjoy, duration ):
    X.append([i,j,k,l,m,n,o,p,q,r,s,t])
X=np.array(X)
Y=np.array(stress_class)

# Classifer model to predict the post precieved stress
from sklearn.ensemble import AdaBoostClassifier

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2, stratify=Y, random_state=3)
# Creating the classifier object 
clf_gini = DecisionTreeClassifier(criterion = "entropy",random_state=15, max_depth=10, min_samples_leaf=1)
# Performing training 
clf_gini.fit(X_train, y_train) 
y_pred=clf_gini.predict(X_test)
print("Report decision tree (post): ",
classification_report(y_test, y_pred)) 

clf = AdaBoostClassifier(n_estimators=30, random_state=0)
clf.fit(X_train, y_train) 
y_pred=clf.predict(X_test)
print("Report adaboost(post) : ",
classification_report(y_test, y_pred)) 



q1='''
with
help1 as (
select *, case when stress_level is null then 0 else 1 end ChangeIndicator
from `studentLife.stress`
)
, help2 as (
select *, Sum(ChangeIndicator) over (order by student_id, date) RowGroup from help1
)
select *,
 
 
case when stress_level is not null then stress_level
else first_value(stress_level) over (partition by RowGroup order by student_id, date)
end stress_level_FillDown
from help2
order by student_id, date
'''

bq_assistant.estimate_query_size(q1)
df_stress_students = bq_assistant.query_to_pandas(q1)
df_stress_students



q6='''
with
help1 as (
select *, case when stress_level is null then 0 else 1 end ChangeIndicator
from `studentLife.stress`
)
, help2 as (
select *, Sum(ChangeIndicator) over (order by student_id, date) RowGroup from help1
)
select *,
 
 
case when stress_level is not null then stress_level
else first_value(stress_level) over (partition by RowGroup order by student_id, date)
end stress_level_FillDown
from help2
order by student_id, date
'''

bq_assistant.estimate_query_size(q6)
df_stress_students = bq_assistant.query_to_pandas(q6)
df_stress_students

q2='''
with
help1 as (
select *, case when _activity_inference is null then 0 else 1 end ChangeIndicator
from `studentLife.activity`
)
, help2 as (
select *, Sum(ChangeIndicator) over (order by student_id, date) RowGroup from help1
)
select *,
 
 
case when _activity_inference is not null then _activity_inference
else first_value(_activity_inference) over (partition by RowGroup order by student_id, date)
end activity_FillDown
from help2
order by student_id, date
'''

bq_assistant.estimate_query_size(q2)
df_activity_students = bq_assistant.query_to_pandas(q2)
df_activity_students



u10_stress=df_stress_students.loc[df_stress_students['student_id']=='u10']
u16_stress=df_stress_students.loc[df_stress_students['student_id']=='u16']
u10_activity=df_activity_students.loc[df_activity_students['student_id']=='u10']
u16_activity=df_activity_students.loc[df_activity_students['student_id']=='u16']

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
plt.style.use('fivethirtyeight') # For plots

data = u10_stress
data=data.iloc[:, [8]].astype(float)
data.index=u10_stress.date.dt.tz_convert(None)
fig, axs = plt.subplots(figsize=(15, 5), facecolor='w', edgecolor='k')

axs.plot(u10_stress.date, u10_stress.stress_level_FillDown.astype(float))
name = 'u10_stress'
axs.set_title(f'# of EMA Stress for {name}')
axs.set_ylabel('EMA Stress')
axs.set_xlabel('Date')

plt.tight_layout()
plt.show()

stress_train = data[:105].copy()
stress_test = data[105:].copy()
data.shape

# Setup and train model and fit
model_u10_stress = Prophet()
model_u10_stress.fit(stress_train.reset_index() \
              .rename(columns={'date':'ds',
                               'stress_level_FillDown':'y'}))
# Predict on training set with model
u10_stress_test_fcst = model_u10_stress.predict(df=stress_test.reset_index() \
                                   .rename(columns={'date':'ds'}))
# Plot the components of the model
fig = model_u10_stress.plot_components(u10_stress_test_fcst)

data = u10_activity
data=data.iloc[:, [6]].astype(float)
data.index=u10_activity.date.dt.tz_convert(None)
fig, axs = plt.subplots(figsize=(15, 5), facecolor='w', edgecolor='k')

axs.plot(u10_activity.date, u10_activity.activity_FillDown.astype(float))
name = 'u10_activity'
axs.set_title(f'# of activity for {name}')
axs.set_ylabel('Activity')
axs.set_xlabel('Date')

plt.tight_layout()
plt.show()

 data.shape

activity_train = data[:530000].copy()
activity_test = data[530000:].copy()

# Setup and train model and fit
model_u10_activity = Prophet()
model_u10_activity.fit(activity_train.reset_index() \
              .rename(columns={'date':'ds',
                               'activity_FillDown':'y'}))
# Predict on training set with model
u10_activity_test_fcst = model_u10_activity.predict(df=activity_test.reset_index() \
                                   .rename(columns={'date':'ds'}))
# Plot the components of the model
fig = model_u10_activity.plot_components(u10_activity_test_fcst)

data = u16_stress
data=data.iloc[:, [8]].astype(float)
data.index=u16_stress.date.dt.tz_convert(None)
fig, axs = plt.subplots(figsize=(15, 5), facecolor='w', edgecolor='k')

axs.plot(u16_stress.date, u16_stress.stress_level_FillDown.astype(float))
name = 'u16_stress'
axs.set_title(f'# of EMA Stress for {name}')
axs.set_ylabel('EMA Stress')
axs.set_xlabel('Date')

plt.tight_layout()
plt.show()

stress_train = data[:105].copy()
stress_test = data[105:].copy()

# Setup and train model and fit
model_u16_stress = Prophet()
model_u16_stress.fit(stress_train.reset_index() \
              .rename(columns={'date':'ds',
                               'stress_level_FillDown':'y'}))
# Predict on training set with model
u16_stress_test_fcst = model_u16_stress.predict(df=stress_test.reset_index() \
                                   .rename(columns={'date':'ds'}))
# Plot the components of the model
fig = model_u16_stress.plot_components(u16_stress_test_fcst)

data = u16_activity
data=data.iloc[:, [6]].astype(float)
data.index=u16_activity.date.dt.tz_convert(None)
fig, axs = plt.subplots(figsize=(15, 5), facecolor='w', edgecolor='k')

axs.plot(u16_activity.date, u16_activity.activity_FillDown.astype(float))
name = 'u16_activity'
axs.set_title(f'# of activity for {name}')
axs.set_ylabel('Activity')
axs.set_xlabel('Date')

plt.tight_layout()
plt.show()

data.shape

activity_train = data[:490000].copy()
activity_test = data[490000:].copy()

# Setup and train model and fit
model_u16_activity = Prophet()
model_u16_activity.fit(activity_train.reset_index() \
              .rename(columns={'date':'ds',
                               'activity_FillDown':'y'}))
# Predict on training set with model
u16_activity_test_fcst = model_u16_activity.predict(df=activity_test.reset_index() \
                                   .rename(columns={'date':'ds'}))
# Plot the components of the model
fig = model_u16_activity.plot_components(u16_activity_test_fcst)


u10_stress.date = pd.to_datetime(u10_stress.date, format='%d/%m/%y %H.%M')
u10_activity.date = pd.to_datetime(u10_activity.date, format='%d/%m/%y %H.%M')
u10_stress.set_index('date', inplace=True)
u10_activity.set_index('date', inplace=True)

import scipy.stats as stats

merged = pd.merge_asof(u10_stress[['stress_level_FillDown']], u10_activity[['activity_FillDown']], left_index=True, right_index=True, direction='nearest')
merged['stress_level_FillDown'].astype(float)
#print(merged['stress_level_FillDown'].astype(float))
merged['activity_FillDown'].astype(float)
#print(merged['activity_FillDown'].astype(float))

r, p = stats.pearsonr(merged['stress_level_FillDown'].astype(float), merged['activity_FillDown'].astype(float))
print(f"Scipy computed Pearson r: {r} and p-value: {p}")
merged.astype(float).corr()

merged = pd.merge_asof(u16_stress[['stress_level_FillDown']], u16_activity[['activity_FillDown']], left_index=True, right_index=True, direction='nearest')
merged['stress_level_FillDown'].astype(float)
#print(merged['stress_level_FillDown'].astype(float))
merged['activity_FillDown'].astype(float)
#print(merged['activity_FillDown'].astype(float))
r, p = stats.pearsonr(merged['stress_level_FillDown'].astype(float), merged['activity_FillDown'].astype(float))
print(f"Scipy computed Pearson r: {r} and p-value: {p}")
merged.astype(float).corr()