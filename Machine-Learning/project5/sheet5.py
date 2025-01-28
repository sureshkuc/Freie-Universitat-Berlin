import numpy as np
import pandas as pd
import os

cwd = os.getcwd()
df = pd.read_csv(cwd+'/fifa.csv', index_col = 0)
print(df.shape)
df.head()

df.describe()

pd.set_option('display.max_row',30) #show #nans per column
attributes = len(df) - df.count(axis = 0)
attributes

attributes48 = attributes[attributes == 48].index #print columns where 38 players have nans

missing48 = df[attributes48][df[attributes48].isna().all(axis=1)].index

df.loc[missing48]

pd.set_option('display.max_row',34)
attributes[(attributes!= 48) & (attributes!=0)]

attributes2085 = attributes[attributes == 2085].index #print columns where 38 players have nans
missing2085 = df[attributes2085][df[attributes2085].isna().all(axis=1)].index


df.loc[missing2085] # all the sampe

df[pd.isnull(df.Position)& pd.isnull(df['Jersey Number'])]

X = df['Finishing']
N = df.shape[0]
X_var = X.var()
nsamples = [15000,10000, 5000, 1000,100]
sample_var_SRS = pd.DataFrame(index = nsamples)
sample_mean_SRS = pd.DataFrame(index = nsamples)
for n in nsamples:
    for i in range(100):
        sample = X.sample(n, replace = False, random_state = i)
        sample_var_SRS.at[n,i] = (sample.var()/n)*(1-(n-1)/(N-1))
        sample_mean_SRS.at[n,i] = sample.mean()

sample_mean_SRS.var( axis = 1)

sample_var_SRS

import matplotlib.pyplot as plt
plt.figure(figsize=(16,9))
for sample_size in range(len(sample_var_SRS)):
    plt.subplot(1, 5,sample_size+1)
    plt.scatter(range(1,101), sample_var_SRS.iloc[sample_size].values)
    plt.axhline(sample_var_SRS.iloc[sample_size].mean())
    plt.axhline(sample_mean_SRS.var(axis = 1).iloc[sample_size], color = 'red')
    #plt.ylim(2,50)
    plt.xlabel("Sample #")
    if sample_size == 0:
        plt.ylabel("Sampling Variance")
    plt.title('Variance in '+ str(sample_var_SRS.index[sample_size])+' samples')
# 0.004475<0.0172<0.0554<0.36<3.75  variance of sample mean is growing with the size of sampe decreasing



"""I was made aware that Exercise 1 on the current Assignment sheet contains a term that was not defined in the lecture, 
and, in fact, is used incorrectly here. Instead of "sample variance", it should read "variance of the sample mean". 
The analytic expression for this variance is given on Slide 43 of Monday's lecture. The assignment sheet has been updated.
"""

samplingratio = 1/5

sample_var_Simple = []
sample_mean_Simple = []
for i in range(100):
    sample = df.sample(frac=samplingratio, replace=False, random_state=i)
    sample_var_Simple.append((sample.Finishing.var()/sample.shape[0])*(1-(sample.shape[0]-1)/(N-1)))
    sample_mean_Simple.append(sample.Finishing.mean())

sample_var_Stratified = []
sample_mean_Stratified = []
for i in range(100):
    sample = df.groupby('Position').apply(lambda x: x.sample(frac=samplingratio,random_state=i))
    sample_var_Stratified.append((sample.Finishing.var()/sample.shape[0])*(1-(sample.shape[0]-1)/(N-1)))
    sample_mean_Stratified.append(sample.Finishing.mean())


plt.figure(figsize=(16,5))

plt.subplot(121)
plt.scatter(range(1,101), sample_var_Simple)
plt.axhline(np.mean(sample_var_Simple))
plt.axhline(np.var(sample_mean_Simple), color = 'red')
plt.ylim(0.04, 0.09)
plt.xlabel("Sample #")
plt.ylabel("Variance of Sampling Mean")
plt.title("SRS (without replacement)")

plt.subplot(122)
plt.scatter(range(1,101), sample_var_Stratified)
plt.axhline(np.mean(sample_var_Stratified))
plt.axhline(np.var(sample_mean_Stratified), color = 'red')
plt.ylim(0.04, 0.09)
plt.xlabel("Sample #")
plt.ylabel("Variance of Sampling Mean")
plt.title("Stratified Sampling by Position")

plt.show()

X = df['Overall']
N = df.shape[0]
X_var = X.var()
nsamples = [15000,10000, 5000, 1000,100]
sample_var_SRS_Overall = pd.DataFrame(index = nsamples)
sample_mean_SRS_Overall = pd.DataFrame(index = nsamples)
for n in nsamples:
    for i in range(100):
        sample = X.sample(n, replace = False, random_state = i)
        sample_var_SRS_Overall.at[n,i] = (sample.var()/n)*(1-(n-1)/(N-1))
        sample_mean_SRS_Overall.at[n,i] = sample.mean()
    


import matplotlib.pyplot as plt
plt.figure(figsize=(16,9))
for sample_size in range(len(sample_var_SRS_Overall)):
    plt.subplot(1, 5,sample_size+1)
    plt.scatter(range(1,101), sample_var_SRS_Overall.iloc[sample_size].values)
    plt.axhline(sample_var_SRS_Overall.iloc[sample_size].mean())
    plt.axhline(sample_mean_SRS_Overall.var(axis = 1).iloc[sample_size], color = 'red')
    #plt.ylim(2,50)
    plt.xlabel("Sample #")
    if sample_size == 0:
        plt.ylabel("Variance of Sampling mean")
    plt.title('Variance in '+ str(sample_var_SRS_Overall.index[sample_size])+' samples')
# 0.0005615<0.00215<0.00694<0.045<0.48  variance of sample mean is growing with the size of sampe decreasing


samplingratio = 1/5

sample_var_Simple_overall = []
sample_mean_Simple_overall = []
for i in range(100):
    sample = df.sample(frac=samplingratio, replace=False, random_state=i)
    sample_var_Simple_overall.append((sample.Overall.var()/sample.shape[0])*(1-(sample.shape[0]-1)/(N-1)))
    sample_mean_Simple_overall.append(sample.Overall.mean())

sample_var_Stratified_Overall = []
sample_mean_Stratified_Overall = []
for i in range(100):
    sample = df.groupby('Position').apply(lambda x: x.sample(frac=samplingratio,random_state=i))
    sample_var_Stratified_Overall.append((sample.Overall.var()/sample.shape[0])*(1-(sample.shape[0]-1)/(N-1)))
    sample_mean_Stratified_Overall.append(sample.Overall.mean())


plt.figure(figsize=(16,5))

plt.subplot(121)
plt.scatter(range(1,101), sample_var_Simple_overall)
plt.axhline(np.mean(sample_var_Simple_overall))
plt.axhline(np.var(sample_mean_Simple_overall), color = 'red')
plt.ylim(0.009, 0.012)
plt.xlabel("Sample #")
plt.ylabel("Variance of Sampling Mean")
plt.title("SRS (without replacement)")

plt.subplot(122)
plt.scatter(range(1,101), sample_var_Stratified_Overall)
plt.axhline(np.mean(sample_var_Stratified_Overall))
plt.axhline(np.var(sample_mean_Stratified_Overall), color = 'red')
#plt.ylim(0.009, 0.012)
plt.xlabel("Sample #")
plt.ylabel("Variance of Sampling Mean")
plt.title("Stratified Sampling")

plt.show()

df.Overall.head()

a = df.groupby('Nationality').size()
a[a>10]

samplingratio = 1/5

#first two already exist



sample_var_Stratified_new = []
sample_mean_Stratified_new = []
for i in range(100):
    sample = df[pd.notnull(df['LAM'])].groupby('LAM').apply(lambda x: x.sample(frac=samplingratio,random_state=i))
    sample_var_Stratified_new.append((sample.Finishing.var()/sample.shape[0])*(1-(sample.shape[0]-1)/(N-1)))
    sample_mean_Stratified_new.append(sample.Finishing.mean())




plt.figure(figsize=(16,5))

plt.subplot(131)
plt.scatter(range(1,101), sample_var_Simple)
plt.axhline(np.mean(sample_var_Simple))
plt.axhline(np.var(sample_mean_Simple), color = 'red')
plt.ylim(0.04, 0.09)
plt.xlabel("Sample #")
plt.ylabel("Variance of Sampling Mean")
plt.title("SRS (without replacement)")

plt.subplot(132)
plt.scatter(range(1,101), sample_var_Stratified)
plt.axhline(np.mean(sample_var_Stratified))
plt.axhline(np.var(sample_mean_Stratified), color = 'red')
plt.ylim(0.04, 0.09)
plt.xlabel("Sample #")
plt.ylabel("Variance of Sampling Mean")
plt.title("Stratified sampling py Position")

plt.subplot(133)
plt.scatter(range(1,101), sample_var_Stratified_new)
plt.axhline(np.mean(sample_var_Stratified_new))
plt.axhline(np.var(sample_mean_Stratified_new), color = 'red')
plt.ylim(0.04, 0.09)
plt.xlabel("Sample #")
plt.ylabel("Variance of Sampling Mean")
plt.title("Stratified sampling by LAM")

plt.show()

columns_new = df.dtypes[(df.dtypes!='int64') & (df.dtypes != 'float64')].index.to_list()
to_remove = ['Name','Photo','Flag', 'Club Logo']
for i in to_remove:
    columns_new.remove(i)
columns_new = columns_new+['Age', 'Special']

df[columns_new]

dup = df[['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB','RCB', 'RB']].T.drop_duplicates()

all_positions = ['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB','RCB', 'RB']
unique_positions = dup.index.to_list()
for i in all_positions:
    if i in unique_positions:
        continue
    else:
        columns_new.remove(i)


df[columns_new].columns

df[columns_new]

import math

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error as mse
from sklearn.manifold import TSNE

#import lightgbm as lgb

import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import plotly.express as px

#from iso3166 import countries
import matplotlib.pyplot as plt

missed = pd.DataFrame()
missed['column'] = df.columns
missed['total_missing_values']=[
     df[col].isnull().sum() for col in df.columns
]
missed['percent'] = [
    round(100* df[col].isnull().sum() / len(df), 2) for col in df.columns
]
missed = missed[missed['percent']>0].sort_values('percent')
fig = px.bar(
    missed, 
    x='percent',
    y="column", 
    orientation='h', 
    title='Missed values percent for every column (percent > 0)', 
    height=1300, 
    width=800
)

fig.show()

missed[missed['percent']>0][-5:]

def plot_bar_plot(data, categorical_feature, target_feature, orientation, title, top_records=None, sort=False):
    data = data.groupby(categorical_feature)[target_feature].count().reset_index()
    fig = px.bar(
        data, 
        x=categorical_feature, 
        y=target_feature, 
        orientation=orientation, 
        title=title,
        height=600,
        width=800
    )
    fig.show()
    
def plot_pie_count(data, field="Nationality", percent_limit=0.5, title="Number of players by "):
    
    title += field
    data[field] = data[field].fillna('NA')
    data = data[field].value_counts().to_frame()

    total = data[field].sum()
    data['percentage'] = 100 * data[field]/total    

    percent_limit = percent_limit
    otherdata = data[data['percentage'] < percent_limit] 
    others = otherdata['percentage'].sum()  
    maindata = data[data['percentage'] >= percent_limit]

    data = maindata
    other_label = "Others(<" + str(percent_limit) + "% each)"
    data.loc[other_label] = pd.Series({field:otherdata[field].sum()}) 
    
    labels = data.index.tolist()   
    datavals = data[field].tolist()
    
    trace=go.Pie(
        labels=labels,
        values=datavals
    )

    layout = go.Layout(
        title = title,
        height=500,
        width=800
    )
    
    fig = go.Figure(data=[trace], layout=layout)
    iplot(fig)

plot_bar_plot(
    df, 
    'Position', 
    'Value', 
    'v', 
    'Number of players by position'
)

plot_pie_count(df, 'Nationality')

plot_pie_count(df, 'Preferred Foot')

plot_pie_count(df, 'Work Rate', 0.1)
plot_pie_count(df, 'Body Type', 0.1)