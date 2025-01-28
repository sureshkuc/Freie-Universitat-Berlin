pip install pandas_gbq

import pandas as pd
import os
import pandas_gbq

PROJECT_ID = 'ipads2020assignment8'
for folder in os.listdir('./sensing/'):
    for file in os.listdir('./sensing/'+folder):
        print('./sensing/'+folder+'/'+file)
        data= pd.read_csv('./sensing/'+folder+'/'+file)
        columns={x:'_'.join(x.split()) for x in data.columns}
        data=data.rename(columns=columns)
        try:
            pandas_gbq.to_gbq(
        data, 'Students.'+file.split('.')[0], project_id=PROJECT_ID, if_exists='fail',
    )
        except:
            print('./sensing/'+folder+'/'+file)
            pass

os.listdir('./sensing')

PROJECT_ID = 'ipads2020assignment8'
d=[]
for folder in os.listdir('./sensing/'):
    a=[]
    for file in os.listdir('./sensing/'+folder):
        a.append(file.split('.')[0].split('_')[-1])
    d.append(a)
        


import json

with open('/home/suresh/Profile-Areas/Project7/studentLife_no-audio__no-wifi/studentLife/EMA/response/Sleep/Sleep_u00.json') as f:
  data = json.load(f)

# Output: {'name': 'Bob', 'languages': ['English', 'Fench']}
print(data)

dataframe=pd.read_json('/home/suresh/Profile-Areas/Project7/studentLife_no-audio__no-wifi/studentLife/EMA/response/Sleep/Sleep_u00.json')

os.listdir('./EMA/response/Sleep')

PROJECT_ID = 'ipads2020assignment8'

for file in os.listdir('./EMA/response/Lab/'):
    print('./EMA/response/Sleep/'+file)
    data= pd.read_json('./EMA/response/Lab/'+file)
    columns={x:'_'.join(x.split()) for x in data.columns}
    data=data.rename(columns=columns)
    try:
        pandas_gbq.to_gbq(
    data, 'Students.'+file.split('.')[0], project_id=PROJECT_ID, if_exists='fail',
)
    except:
        print('except','./EMA/response/Lab/'+file)
        pass

for folder in os.listdir('./EMA/response/'):
    for file in os.listdir('./EMA/response/'+folder):
        print('./EMA/response/'+folder+'/'+file)
        data= pd.read_json('./EMA/response/'+folder+'/'+file)
        columns={x:'_'.join(x.split()) for x in data.columns}
        data=data.rename(columns=columns)
        try:
            pandas_gbq.to_gbq(
        data, 'Students.'+file.split('.')[0], project_id=PROJECT_ID, if_exists='fail',
    )
        except:
            print('except','./EMA/response/'+folder+'/'+file)
            pass

