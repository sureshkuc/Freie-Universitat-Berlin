# chemovator.
Tasks:
1. Conduct exploratory data analysis and report the insights 
  Answer:Exploratory Data Analysis has been done. Following steps
performed for this purpose #1. Missing Values if any #2. Distribution of the Numerical Variables #3. Relationship between
independent and dependent feature(StainlessSteelPrice) #4. Relationship in between independent features #5. Removed high
correlated independent features #6. Analysed and displayed the trend, seasonality etc.

2. Forecast the prices of stainless-steel for the given time period(s) using the following methods
  a). Statistical Models (at least one type): 
   Answer: Following methods has been used: #1. Prophet Statistical method #2. SARIMA
  b). Machine Learning (at least one type) 
  Answer: Linear Regresion, LASSO, Ridge Regression, Gradient Boosting Method(xgboost),
Neural Network,
  c).Deep Learning (at least one type) 
  Answer: LSTM and deep neural network have been used for this purpose
  
Select the features - Column C (Steel_Inventory_M_USD) through column T (Copper_Global_USD) – that help to improve the accuracy
Answer: I have used some features for price forcasting

3. Provide insights on how your model made the prediction 
   There are two ways to learn the time searies data:
   
  a). Use the time or date with other independent fearture. For this purpose the previous prices as well as independent features have
been taken into account to learn this time series data. Here I have used the prophet, SARIMA, LSTM approaches to forcast the
stainless steel prices.

  b). Take into account the time or effect as a new variable and build the machine learning model. Here I have created a new variable
number of days from today to that date, it can be taken as future date as well to take the effect of future as well). Neural Network,
Gradient boosting and Random forest approach have been used to learn these data.

4. Use the following evaluation criteria for model selection Mean Absolute Percentage Error (MAPE) Directional Symmetry 

Both methods have been implemented an have been used to evaluate the models

Conclusion (Summary): Among all models prophet model is perfoming best with Mean Absolute Percentage Error(MAPE) 4.25.
Crossvalidation has been used to evaluate this model. For each year 2 data points has been used to test the model with total 13
independent features
