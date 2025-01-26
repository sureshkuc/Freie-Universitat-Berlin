import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost
import csv as csv
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.model_selection import cross_val_score,KFold
from sklearn.model_selection import  train_test_split,GridSearchCV
from sklearn.metrics import mean_absolute_error
from scipy.stats import skew
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
from fbprophet import Prophet
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import register_matplotlib_converters
import torch
import torch.nn as nn
from pandas import DataFrame
import itertools
import os
import sys
import warnings
from sklearn.datasets import load_boston, load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from autofeat import AutoFeatRegressor
from fbprophet.plot import plot_cross_validation_metric
from fbprophet.diagnostics import performance_metrics
from fbprophet.diagnostics import cross_validation
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
%matplotlib inline
%load_ext autoreload
%autoreload 2
register_matplotlib_converters()
# Comment this if the data visualisations doesn't work on your side
%matplotlib inline

data=pd.read_csv('https://raw.githubusercontent.com/sureshkuc/chemovator./main/Stainless-Steel-Prices-Forecasty-Assignment.csv')



data

data.describe()

#Exploratory Data Analysis:
#In Data Analysis We will Analyze To Find out the below stuff
#1. Missing Values if any 
#2. Distribution of the Numerical Variables
#3. Relationship between independent and dependent feature(StainlessSteelPrice)
#4. Relationship in between independent features


data.info()

print(data['StainlessSteelPrice'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(data['StainlessSteelPrice'], color='g', bins=10, hist_kws={'alpha': 0.4})

from datetime import date
data['dd'],data['mm'],data['yyyy']=data['Date'].str.split("/", expand=True)[0],data['Date'].str.split("/", expand=True)[1],data['Date'].str.split("/", expand=True)[2]
today = date.today()
number_of_days=[]
dates=[]
for yy, mm, dd in zip(data['yyyy'],data['mm'],data['dd']):
  d1 = date(int(yy),int(mm),int(dd))
  dates.append(d1)
  delta = today - d1
  number_of_days.append(delta.days)
data['Date']=dates

data['number_of_days_from_today']=number_of_days
data.hist(figsize=(16, 20), bins=10, xlabelsize=8, ylabelsize=8)

correlatedcolumns=list()
for i in [col for col in data.columns if col not in ['Date','dd','mm','yyyy','StainlessSteelPrice']]:
  df_num_corr = data.corr()[i][1:] # 1 because the first row is StainlessSteelPrice
  removable_features_list = df_num_corr[abs(df_num_corr) > 0.9  ].sort_values(ascending=False)
  for key, value in dict(removable_features_list).items():
    if key != i:
      correlatedcolumns.append(key)
  print("There is {} strongly correlated values with {}:\n{}".format(len(removable_features_list), i, removable_features_list))
columns_need_to_remove=list(set(correlatedcolumns))

#heatmap
corr = data.drop('StainlessSteelPrice', axis=1).corr() 
plt.figure(figsize=(12, 10))

sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);

#we now have a list of strongly correlated values but this list is incomplete as we know that correlation is affected by outliers. So we could proceed as follow:

#Plot the numerical features and see which ones have very few or explainable outliers
#Remove the outliers from these features and see which one can have a good correlation without their outliers
#Btw, correlation by itself does not always explain the relationship between data so ploting them could even lead us to new insights and in the same manner, check that our correlated values have a linear relationship to the StainlessSteelPrice.

#For example, relationships such as curvilinear relationship cannot be guessed just by looking at the correlation value so lets take the features we excluded from our correlation table and plot them to see if they show some kind of pattern.

for i in range(1, len(data.columns), 5):
    sns.pairplot(data=data,
                x_vars=data.columns[i:i+5],
                y_vars=['StainlessSteelPrice'])

data.groupby(['yyyy'])['StainlessSteelPrice'].median().plot()
plt.xlabel('Year')
plt.ylabel('Median  Price')
plt.title("Price vs Year")

every_column_except_y= [col for col in data.columns if col not in ['Date','dd','mm','yyyy','StainlessSteelPrice']+columns_need_to_remove]

#Mean Absolute Percentage Error
def mean_absolute_percentage_error(y_true, y_pred):
  y_true, y_pred = np.array(y_true), np.array(y_pred)
  return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#Directional Symmetry Statistic
def directional_symmetry(y_true, y_pred):
  y_true, y_pred = np.array(y_true), np.array(y_pred)
  return (1/(len(y_true)-1))*100*np.sum(np.where((y_true[1:]-y_true[:-1])*(y_pred[1:]-y_pred[:-1])>0,1,0))


df_tmp=data.copy()
# split into train and test
# Split the data into X & y
X = df_tmp.drop(columns=['dd','mm','yyyy']+columns_need_to_remove,axis=1)
data_train=X
data_train.rename(columns={"Date": "ds", "StainlessSteelPrice": "y"}, inplace=True)
#data_test.rename(columns={"Date": "ds", "StainlessSteelPrice": "y"}, inplace=True)
#y_pred = m.predict(data_test)
#y_pred
#y_hat=np.array(y_pred[ ['yhat']]).reshape(2)
#print("MAPE on test data:", mean_absolute_percentage_error(data_test['y'], y_hat))
#print("Directional Symmetry Statistic on test data:", directional_symmetry(data_test['y'], y_hat))


param_grid = {  
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
}

# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
mape = []  # Store the RMSEs for each params here

# Use cross validation to evaluate all parameters
for params in all_params:
    m = Prophet(**params) # Fit model with given params
    m = Prophet()
    for i in data_train.columns:
      if i not in ['ds','y']:
        m.add_regressor(i)
    m.fit(data_train)
    #df_cv = cross_validation(m, cutoffs=cutoffs, horizon='30 days', )
    cutoffs = pd.to_datetime(['2013-01-10', '2014-01-09', '2015-01-09','2016-01-09','2017-01-09','2018-01-09','2019-01-09'])
    df_cv = cross_validation(m, cutoffs=cutoffs, horizon='2 days',parallel="processes")
    #df_p = performance_metrics(df_cv)
    df_p = performance_metrics(df_cv, rolling_window=1)
    mape.append(df_p['mape'].values[0])

# Find the best parameters
tuning_results = pd.DataFrame(all_params)
tuning_results['mape'] = mape
print(tuning_results)


tuning_results['mape'] = mape
print(tuning_results)
best_params = all_params[np.argmin(mape)]
print(best_params)


m = Prophet(changepoint_prior_scale=0.001, seasonality_prior_scale=0.01)
for i in data_train.columns:
  if i not in ['ds','y']:
    m.add_regressor(i)
m.fit(data_train)


cutoffs = pd.to_datetime(['2013-01-10', '2014-01-09', '2015-01-09','2016-01-09','2017-01-09','2018-01-09','2019-01-09'])
df_cv = cross_validation(m, cutoffs=cutoffs, horizon='2 days')
df_p = performance_metrics(df_cv)
df_p.head()
fig = plot_cross_validation_metric(df_cv, metric='mape')


y_pred = m.predict(data_train)
print("MAPE Mean Absolute Percentage Error:", mean_absolute_percentage_error(data_train['y'], np.array(y_pred[['yhat']]).reshape(y_pred[['yhat']].shape[0])))
print("Directional Symmetry Statistic:", directional_symmetry(data_train['y'], np.array(y_pred[['yhat']]).reshape(y_pred[['yhat']].shape[0])))
fig2 = m.plot_components(y_pred)

fig_test = m.plot(y_pred)

m = Prophet(changepoint_prior_scale=0.001, seasonality_prior_scale=0.01)
m.fit(data_train)

#next 6 month predictioins
future = m.make_future_dataframe(periods=180)
y_pred = m.predict(future)
fig_test = m.plot(y_pred)

#next 3 month predictioins
future = m.make_future_dataframe(periods=90)
y_pred = m.predict(future)
fig_test = m.plot(y_pred)

split_date = date(2020,1,1)
split_date = pd.to_datetime(split_date,format='%Y-%m-%d',utc=True)
#X['Date']= pd.to_datetime(X['Date'],format='%Y-%m-%d')
data_train = X.loc[X['ds'] <= split_date].copy()
data_test = X.loc[X['ds'] > split_date].copy()

#setting 'date' column as index columna as forecasting will be done for this column
#making 'TT' as float for statistical calculations
train_df =data_train
train_df = train_df.set_index('ds')
train_df['y'] = train_df['y'].astype(float)

train_df.head()

#Decomposing data to observe if there exists a sesional trend


result = seasonal_decompose(train_df['y'].values, model='additive',freq=12)

fig = plt.figure()  
fig = result.plot()  
fig.set_size_inches(20, 12)


 #adfuller stands for Augmented Dickey-Fuller unit root test.

#The function find mean and standard deviation of the series and and performs augmented dickey fuller test.
#returns pvale .. The samaller the pvalue more stationary is the series.

def test_stationarity(timeseries, window = 15, cutoff = 0.01):
  rolmean = timeseries.rolling(window).mean()
  rolstd = timeseries.rolling(window).std()
  fig = plt.figure(figsize=(12, 8))
  orig = plt.plot(timeseries, color='blue',label='Original')
  mean = plt.plot(rolmean, color='red', label='Rolling Mean')
  std = plt.plot(rolstd, color='black', label = 'Rolling Std')
  plt.legend(loc='best')
  plt.title('Rolling Mean & Standard Deviation')
  plt.show()

  print('Results of Dickey-Fuller Test:')
  dftest = adfuller(timeseries, autolag='AIC',)
  dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
  for key,value in dftest[4].items():
      dfoutput['Critical Value (%s)'%key] = value
  pvalue = dftest[1]
  if pvalue < cutoff:
      print('p-value = %.4f. The series is likely stationary.' % pvalue)
  else:
      print('p-value = %.4f. The series is likely non-stationary.' % pvalue)
  
  print(dfoutput)

test_stationarity(train_df['y'],window = 11)
#calling the function gives below result , where we can observe the huge gap between orignal data and mean,std
#also the pvalue is 0.178976 which is not so good and hence , the output says "The series is likely non-stationary."

import statsmodels.api as sm
def acf_pacf(train_df):
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(train_df.y, ax=ax1, ) # using default value of lag
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(train_df.y, ax=ax2) # using default value of lag

df_log = np.log(train_df.y )
plt.plot(df_log)

rolling_mean = df_log.rolling(window=12).mean()
df_log_minus_mean = df_log - rolling_mean
df_log_minus_mean.dropna(inplace=True)

test_stationarity(df_log_minus_mean, window = 12)

df_log_minus_mean = pd.DataFrame({'ds':df_log_minus_mean.index, 'y':df_log_minus_mean.values})


acf_pacf(df_log_minus_mean)

sarimax_mod = sm.tsa.statespace.SARIMAX(train_df.y, order=(11, 1, 0), seasonal_order=(0, 0, 0, 0), trend='ct').fit()
print(sarimax_mod.summary())

sarimax_mod.plot_diagnostics(figsize=(20, 14))
plt.show()

import scipy.stats as stats
import seaborn as sns # informative statistical graphics.
import statsmodels.api as sm #for ARIMA and SARIMAX
resid = sarimax_mod.resid #gives residual degree of freedom (mu, sigma, pvalue ... )

fig = plt.figure(figsize=(12,8))
ax0 = fig.add_subplot(111)

sns.distplot(resid ,fit = stats.norm, ax = ax0) # need to import scipy.stats

# Get the fitted parameters used by the function
(mu, sigma) = stats.norm.fit(resid)

#Now plot the distribution using 
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('Residual distribution')


# ACF and PACF
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(sarimax_mod.resid, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(sarimax_mod.resid, ax=ax2)

future_predict=sarimax_mod.forecast(steps=5)
future_predict=pd.DataFrame(future_predict)

data_test

#next 6 month prediction
sarimax_mod.predict(3,180)

future_predict['ds']=data_test.reset_index()['ds'].values
future_predict = future_predict.set_index('ds')
data_test = data_test.set_index('ds')

data_test.reset_index()['ds'].values



print("MAPE Mean Absolute Percentage Error:", mean_absolute_percentage_error(data_test['y'], future_predict[0]))
print("Directional Symmetry Statistic:", directional_symmetry(data_test['y'], future_predict[0]))

#6 month forcasting
figg = plt.figure(figsize=(12, 8))
orig = plt.plot(train_df[['y']], color='blue',label='Train Data')
pred_train=plt.plot(sarimax_mod.predict(3,16),color='purple',label='ARIMA model Prediction on Train Data')
fore = plt.plot(future_predict, color='green', label='ARIMA Model Prediction on test data')
fes = plt.plot(data_test[['y']], color='red', label='Test Data')
plt.legend(loc='best')
plt.title('Forecast of Price')
plt.show()

# Stacked LSTM for price forcasting with memory
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back),1:]
		dataX.append(a)
		dataY.append(dataset[i + look_back,0])
	return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility
numpy.random.seed(7)

X

dataset=X.copy()

dataset = dataset.set_index('ds')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset.drop(['y'],axis=1))
y=np.array(X['y'])
dataset=np.hstack((y.reshape(y.shape[0],1),dataset))
#split into train and test sets
train_size = int(len(dataset) * 0.90)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# reshape into X=t and Y=t+1
look_back = 1
X_train, y_train= create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)

X_train=X_train.reshape(X_train.shape[0],X_train.shape[2],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[2],1)
X_tr_t=X_train.copy()
X_tst_t=X_test.copy()

batch_size = 1
look_back=13
#model = Sequential()
#model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
#model.add(Dense(1))

#Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (X_tr_t.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(loss='mean_absolute_percentage_error', optimizer='adam')
model.fit(X_tr_t , y_train, epochs=1000, batch_size=batch_size, verbose=2, shuffle=False)



trainPredict = model.predict(X_tr_t, batch_size=batch_size)
model.reset_states()

testPredict = model.predict(X_tst_t, batch_size=batch_size)


look_back=1
trainPredictPlot = np.empty_like(dataset[:,0] )
trainPredictPlot[:] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back] = trainPredict.reshape(trainPredict.shape[0])
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset[:,0] )
testPredictPlot[:] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset )-1] = testPredict.reshape(testPredict.shape[0])
# plot baseline and predictions
plt.figure(figsize=(20,10))
plt.plot(dataset[:,0], label='Original Data')
plt.plot(trainPredictPlot, label='Train Data Prediction by LSTM')
plt.plot(testPredictPlot,label='Test Data Prediction by LSTM')
plt.ylabel('number')
plt.legend()
plt.show()

print("MAPE Mean Absolute Percentage Error:", mean_absolute_percentage_error(y_test, testPredict.reshape(testPredict.shape[0])))
print("Directional Symmetry Statistic:", directional_symmetry(y_test, testPredict.reshape(testPredict.shape[0])))

X


# Split the data into X & y
df_tmp=data.copy()
X = df_tmp.drop(columns=['Date','dd','mm','yyyy','StainlessSteelPrice']+columns_need_to_remove,axis=1)
y = df_tmp["StainlessSteelPrice"]
#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
X_train,X_test,y_train,y_test=X[:78],X[78:],y[:78], y[78:]
print(X_train.shape)
X_test=np.array(X_test)
y_test=np.array(y_test)
col_index=X_test[:,-1].argsort()[::-1]
X_test=X_test[col_index]
y_test=y_test[col_index]
scaler = StandardScaler()
scaler.fit(X_train)
print('Feature,Mean, Variance:')
for i, j ,k in zip(every_column_except_y,scaler.mean_,scaler.var_):
  print(i,j,k)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
print(len(X_train))
print(len(X_test))

#########################xgboost model with grid search to find hyperparameters######################
n_estimators = [10,20,30,40,50,60,70,80,100, 500, 900, 1100, 1500]
max_depth = [2, 3, 5, 10, 15]
booster = ['gbtree', 'gblinear']
base_score = [0.25, 0.5, 0.75, 1]
learning_rate = [0.001,0.01,0.05, 0.1, 0.15, 0.20,1]
min_child_weight = [1, 2, 3, 4]
# Define the grid of hyperparameters to search
hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth': max_depth,
    'learning_rate' : learning_rate,
    'min_child_weight' : min_child_weight,
    'booster' : booster,
    'base_score' : base_score
    }
mape=make_scorer(mean_absolute_percentage_error,greater_is_better=False)
regressor = xgboost.XGBRegressor()

random_cv = RandomizedSearchCV(estimator=regressor, param_distributions=hyperparameter_grid, cv=5, 
                              n_iter=50, scoring = mape,n_jobs = 4, 
                              verbose = 5, return_train_score = True, random_state=42)
random_cv.fit(X_train,y_train)
print(random_cv.best_params_)
regressor = xgboost.XGBRegressor(base_score= 0.5,
 booster= 'gbtree',
 learning_rate= 0.2,
 max_depth= 5,
 min_child_weight=1,
 n_estimators= 80)
y_preds = regressor.fit(X_train,y_train).predict(X_test)
mae_rf = mean_absolute_percentage_error(y_test,y_preds)
ds= directional_symmetry(y_test, y_preds)
print('MAPE',mae_rf, 'DS',ds)
#############################Random Forest Regressor#####################
model = RandomForestRegressor(n_jobs=-1)
print(model)
model.fit(X_train,y_train)
# Evaluate model using mean absolute error
from sklearn.metrics import mean_absolute_error
y_preds = model.predict(X_test)
mae_rf = mean_absolute_percentage_error(y_test,y_preds)
ds= directional_symmetry(y_test, y_preds)
print('MAPE',mae_rf, 'DS',ds)

###########################linear Regression model##########################

model=LinearRegression().fit(X_train, y_train)
y_preds = model.predict(X_test)
mae_rf = mean_absolute_percentage_error(y_test,y_preds)
ds= directional_symmetry(y_test, y_preds)
print('MAPE',mae_rf, 'DS',ds)

!pip install autofeat

def test_autofeat( feateng_steps=2,units=None):
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
    # run autofeat
    afreg = AutoFeatRegressor(verbose=1, feateng_steps=feateng_steps, units=units)
    # fit autofeat on less data, otherwise ridge reg model with xval will overfit on new features
    X_train_tr = afreg.fit_transform(X_train, y_train)
    X_test_tr = afreg.transform(X_test)
    print("autofeat new features:", len(afreg.new_feat_cols_))
    print("MAPE on training data:", mean_absolute_percentage_error(y_train, afreg.predict(X_train_tr)))
    print("MAPE on test data:", mean_absolute_percentage_error(y_test, afreg.predict(X_test_tr)))
    print("Directional Symmetry Statistic on test data:", directional_symmetry(y_test, afreg.predict(X_test_tr)))

    # train rreg on transformed train split incl cross-validation for parameter selection
    print("# Ridge Regression")
    rreg = Ridge()
    param_grid = {"alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1., 2.5, 5., 10., 25., 50., 100., 250., 500., 1000., 2500., 5000., 10000.]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gsmodel = GridSearchCV(rreg, param_grid, scoring = mape, cv=5)
        gsmodel.fit(X_train_tr, y_train)
    print("best params:", gsmodel.best_params_)
    print("best score:", gsmodel.best_score_)
    print("MAPE on training data:", mean_absolute_percentage_error(y_train, gsmodel.predict(X_train_tr)))
    print("MAPE on test data:", mean_absolute_percentage_error(y_test, gsmodel.predict(X_test_tr)))
    print("Directional Symmetry Statistic on test data:", directional_symmetry(y_test,  gsmodel.predict(X_test_tr)))
    print('Lasso')
    lasso=Lasso()
    param_grid = {"alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1., 2.5, 5., 10., 25., 50., 100., 250., 500., 1000., 2500., 5000., 10000.]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gsmodel = GridSearchCV(lasso, param_grid, scoring = mape, cv=5)
        gsmodel.fit(X_train_tr, y_train)
    print("best params:", gsmodel.best_params_)
    print("best score:", gsmodel.best_score_)
    print("MAPE on training data:", mean_absolute_percentage_error(y_train, gsmodel.predict(X_train_tr)))
    print("MAPE on test data:", mean_absolute_percentage_error(y_test, gsmodel.predict(X_test_tr)))
    print("Directional Symmetry Statistic on test data:", directional_symmetry(y_test,  gsmodel.predict(X_test_tr)))
    
    print("# Random Forest")
    rforest = RandomForestRegressor(n_estimators=100, random_state=13)
    param_grid = {"min_samples_leaf": [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2]}
    gsmodel = GridSearchCV(rforest, param_grid, scoring = mape, cv=5)
    gsmodel.fit(X_train_tr, y_train)
    print("best params:", gsmodel.best_params_)
    print("best score:", gsmodel.best_score_)
    print("MAPE on training data:", mean_absolute_percentage_error(y_train, gsmodel.predict(X_train_tr)))
    print("MAPE on test data:", mean_absolute_percentage_error(y_test, gsmodel.predict(X_test_tr)))
    print("Directional Symmetry Statistic on test data:", directional_symmetry(y_test,  gsmodel.predict(X_test_tr)))
for i in range(1):
  print('##################feateng_steps=',i+1,' ######################################')
  test_autofeat( feateng_steps=i+1)

X_train = torch.from_numpy(np.array(X_train)).float()
y_train = torch.from_numpy(np.array(y_train)).float()
X_test = torch.from_numpy(np.array(X_test)).float()
y_train=y_train.reshape(y_train.shape[0],1)
class Net(nn.Module):
    def __init__(self, D_in, H1, H2, H3, D_out):
        super(Net, self).__init__()
        
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, H3)
        self.linear4 = nn.Linear(H3, D_out)
        
    def forward(self, x):
        y_pred = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(y_pred).clamp(min=0)
        y_pred = self.linear3(y_pred).clamp(min=0)
        y_pred = self.linear4(y_pred)
        return y_pred
H1, H2, H3 = 500, 1000, 200
D_in, D_out = X_train.shape[1], y_train.shape[1]
model1 = Net(D_in, H1, H2, H3, D_out)
criterion = nn.MSELoss(reduction='sum')
#optimizer = torch.optim.SGD(model1.parameters(), lr=1e-4 * 2)
optimizer = torch.optim.Adam(model1.parameters(), lr=1e-4 )
losses1 = []

for t in range(500):
    y_pred = model1(X_train)
    
    loss = criterion(y_train, y_pred)
    print(t, loss.item())
    losses1.append(loss.item())
    
    if torch.isnan(loss):
        break
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
plt.figure(figsize=(12, 10))
plt.plot(range(len(losses1)), losses1)
#plt.plot(range(len(losses3)), losses3)
plt.show()
y_pred = model1(X_test)
#print("MAPE on training data:", mean_absolute_percentage_error(y_train, 
print("MAPE on test data:", mean_absolute_percentage_error(y_test, y_pred.detach().numpy().reshape(y_pred.detach().numpy().shape[0])))
print("Directional Symmetry Statistic on test data:", directional_symmetry(y_test, y_pred.detach().numpy().reshape(y_pred.detach().numpy().shape[0])))


