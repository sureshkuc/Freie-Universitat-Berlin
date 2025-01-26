import pandas as pd
from fbprophet import Prophet

df1 = pd.read_csv('data.csv')
df1.head()

df=df1[['time_iso8601','sum_cases']].copy()


 #df3 = df1[['ticker_symbol']].copy()
  #  df3['ASX code'] = df2[['ASX code']].copy()

df.head()

df.rename(columns={"time_iso8601": "ds", "sum_cases": "y"}, inplace=True)

df.info()

df.columns

df['ds']= pd.to_datetime(df['ds'],format='%Y-%m-%d',utc=True)

df['ds']=df['ds'].astype('datetime64[ns]')

df.info()

df.head()

from datetime import datetime as dt


df['ds'] = pd.to_datetime(df['ds']).dt.date

df['ds']

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=365)
future.tail()

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


fig1 = m.plot(forecast)


fig2 = m.plot_components(forecast)

from fbprophet.plot import plot_plotly, plot_components_plotly

plot_plotly(m, forecast)

plot_components_plotly(m, forecast)


from fbprophet.plot import add_changepoints_to_plot
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)


m = Prophet(changepoint_prior_scale=0.5)
forecast = m.fit(df).predict(future)
fig = m.plot(forecast)

m = Prophet(changepoint_prior_scale=0.001)
forecast = m.fit(df).predict(future)
fig = m.plot(forecast)

m = Prophet(changepoints=['2020-04-12','2020-10-27']) # specifying change points
forecast = m.fit(df).predict(future)
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)

split_date = '2020-11-27'
split_date = pd.to_datetime(split_date,format='%Y-%m-%d',utc=True)
data_train = df.loc[df['ds'] <= split_date].copy()
data_test = df.loc[df['ds'] > split_date].copy()

m1 = Prophet()
m1.fit(data_train)

data_test.head()

y_pred = m1.predict(data_test)
y_pred[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


data_test['y']

y_pred['yhat']

fig_test = m1.plot(y_pred)

import math 
mse = sm.mean_squared_error(data_test['y'], y_pred['yhat'])

rmse = math.sqrt(mse)

print("RMSE:",rmse)

import numpy as np
np.mean(data_test['y'])

data_train['cap'] = 83906168
data_test['cap'] = 83906168 #1.025643e+06

m1_logistic = Prophet(growth='logistic')
m1_logistic.fit(data_train)

y_pred_log = m1_logistic.predict(data_test)
y_pred_log[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig = m1_logistic.plot(y_pred_log)

import math 
mse = sm.mean_squared_error(data_test['y'], y_pred_log['yhat'])

rmse = math.sqrt(mse)

print(rmse)

import sklearn.metrics as sm

print("Explain variance score =", round(sm.explained_variance_score(data_test['y'], y_pred['yhat']), 2)) 
print("R2 score =", round(sm.r2_score(data_test['y'], y_pred['yhat']), 2))



import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#setting 'date' column as index columna as forecasting will be done for this column
#making 'TT' as float for statistical calculations
train_df =data_train
train_df = train_df.set_index('ds')
train_df['y'] = train_df['y'].astype(float)

train_df.head()

#Decomposing data to observe if there exists a sesional trend

from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(train_df['y'], model='additive',freq=7)

fig = plt.figure()  
fig = result.plot()  
fig.set_size_inches(20, 12)


from statsmodels.tsa.stattools import adfuller #adfuller stands for Augmented Dickey-Fuller unit root test.

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
#also the pvalue is 0.9778 which is not so good and hence , the output says "The series is likely non-stationary."

#here are various methods for making series stationary like log, differencing and so on..
#here we are using differencing , shift operator shifts the 'TT' cloumn of df by 4 places and difference is taken.

#plotting the data after differencing we see the pvalue is reduced to 0.3427 which is quite good as compared to our previous value 0.9778
#you can try different values in shift to reduce the pvalue (if possible , #try to choose one where number of observations used is MAX abd pval is MIN)
first_diff = train_df.y - train_df.y.shift(4)
first_diff = first_diff.dropna(inplace = False)
test_stationarity(first_diff, window = 11)


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

# we can see a recurring correlation exists in both ACF and PACF hece we should choose SARIMAX model which also deals with seasonality

#RULE : A model with no orders of differencing assumes that the original series is stationary (mean-reverting). A model with one order of differencing assumes that 
      #the original series has a constant average trend (e.g. a random walk or SES-type model, with or without growth). A model with two orders of total differencing assumes that 
      #the original series has a time-varying trend 

#Since our series has a contant average trend ( with growth ) we would take I = 1 and MA = 0 ( I-1 ).


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

future_predict=sarimax_mod.forecast(steps=16)
future_predict=pd.DataFrame(future_predict)


sarimax_mod.predict(3,130)

future_predict['ds']=data_test['ds']

future_predict['ds']

future_predict = future_predict.set_index('ds')
data_test = data_test.set_index('ds')

future_predict = future_predict.iloc[:-2]

df_dropped_last_n.iloc[:]

future_predict

data_test.iloc[:,-1]

np.mean(data_test) #1152499

import math 
import sklearn.metrics as sm

mse = sm.mean_squared_error(data_test, future_predict)

rmse = math.sqrt(mse)

print("RMSE:",rmse)

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
figg = plt.figure(figsize=(12, 8))
orig = plt.plot(train_df, color='blue',label='Train Data')
pred_train=plt.plot(sarimax_mod.predict(3,140),color='purple',label='ARIMA model Prediction on Train Data')
fore = plt.plot(future_predict, color='green', label='ARIMA Model Prediction on test data')
fes = plt.plot(data_test, color='red', label='Test Data')
plt.legend(loc='best')
plt.title('Forecast of upcomming Covid-19 Cases')
plt.show()

# Stacked LSTM for covid-19 problem with memory
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
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset


#dataset = dataset.astype('float32')

df

dataset=df

dataset = dataset.set_index('ds')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.80)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1
look_back = 3
X_train, y_train= create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)

new_data=pd.DataFrame(X_train)
new_data

# reshape input to be [samples, time steps, features]
X_tr_t = numpy.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_tst_t = numpy.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


def adj_r2_score(r2, n, k):
    return 1-((1-r2)*((n-1)/(n-k-1)))


batch_size = 1
model = Sequential()
model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_tr_t , y_train, epochs=100, batch_size=batch_size, verbose=2, shuffle=False)


import math
from sklearn.metrics import mean_squared_error

trainPredict = model.predict(X_tr_t, batch_size=batch_size)
model.reset_states()

testPredict = model.predict(X_tst_t, batch_size=batch_size)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
y_train = scaler.inverse_transform([y_train])
testPredict = scaler.inverse_transform(testPredict)
y_test = scaler.inverse_transform([y_test])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(y_train[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

trainPredictPlot = np.empty_like(dataset )
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset )
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset )-1, :] = testPredict
# plot baseline and predictions
plt.figure(figsize=(20,10))
plt.plot(scaler.inverse_transform(dataset ), label='Original Data')
plt.plot(trainPredictPlot, label='Train Data Prediction by LSTM')
plt.plot(testPredictPlot,label='Test Data Prediction by LSTM')
plt.ylabel('number of Covid-19 Cases')
plt.legend()
plt.show()

def GridSearch(dataset):

    from math import sqrt
    from multiprocessing import cpu_count
    from joblib import Parallel
    from joblib import delayed
    from warnings import catch_warnings
    from warnings import filterwarnings
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from sklearn.metrics import mean_squared_error

    # one-step sarima forecast
    def sarima_forecast(history, config):
        order, sorder, trend = config
        # define model
        model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
        # fit model
        model_fit = model.fit(disp=False)
        # make one step forecast
        yhat = model_fit.predict(len(history), len(history))
        return yhat[0]

    # root mean squared error or rmse
    def measure_rmse(actual, predicted):
        return sqrt(mean_squared_error(actual, predicted))

    # split a univariate dataset into train/test sets
    def train_test_split(data, n_test):
        return data[:-n_test], data[-n_test:]

    # walk-forward validation for univariate data
    def walk_forward_validation(data, n_test, cfg):
        predictions = list()
        # split dataset
        train, test = train_test_split(data, n_test)
        # seed history with training dataset
        history = [x for x in train]
        # step over each time-step in the test set
        for i in range(len(test)):
            # fit model and make forecast for history
            yhat = sarima_forecast(history, cfg)
            # store forecast in list of predictions
            predictions.append(yhat)
            # add actual observation to history for the next loop
            history.append(test[i])
        # estimate prediction error
        error = measure_rmse(test, predictions)
        return error

    # score a model, return None on failure
    def score_model(data, n_test, cfg, debug=False):
        result = None
        # convert config to a key
        key = str(cfg)
        # show all warnings and fail on exception if debugging
        if debug:
            result = walk_forward_validation(data, n_test, cfg)
        else:
            # one failure during model validation suggests an unstable config
            try:
                # never show warnings when grid searching, too noisy
                with catch_warnings():
                    filterwarnings("ignore")
                    result = walk_forward_validation(data, n_test, cfg)
            except:
                error = None
        # check for an interesting result
        if result is not None:
            print(' > Model[%s] %.3f' % (key, result))
        return (key, result)

    # grid search configs
    def grid_search(data, cfg_list, n_test, parallel=True):
        scores = None
        if parallel:
            # execute configs in parallel
            executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
            tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
            scores = executor(tasks)
        else:
            scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
        # remove empty results
        scores = [r for r in scores if r[1] != None]
        # sort configs by error, asc
        scores.sort(key=lambda tup: tup[1])
        return scores

    # create a set of sarima configs to try
    def sarima_configs(seasonal=[0]):
        models = list()
        # define config lists
        p_params = [x for x in range(15)]
        d_params = [0, 1]
        q_params = [0, 1, 2]
        t_params = ['n','c','t','ct']
        P_params = [0, 1, 2]
        D_params = [0, 1]
        Q_params = [0, 1, 2]
        m_params = seasonal
        # create config instances
        for p in p_params:
            for d in d_params:
                for q in q_params:
                    for t in t_params:
                        for P in P_params:
                            for D in D_params:
                                for Q in Q_params:
                                    for m in m_params:
                                        cfg = [(p,d,q), (P,D,Q,m), t]
                                        models.append(cfg)
        return models

    
    # define dataset
    data = dataset
    print(data)
    # data split
    n_test = 14
    # model configs
    cfg_list = sarima_configs()
    # grid search
    scores = grid_search(data, cfg_list, n_test)
    print('done')
        # list top 3 configs
    for cfg, error in scores[:3]:
      print(cfg, error)
