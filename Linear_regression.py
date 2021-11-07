import pandas as pd
import numpy as np
import quandl, math, datetime
from sklearn import preprocessing, svm
# import sklearn.cross_validation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import os

#Getting the Dataframe
df1 = quandl.get("WIKI/GOOGL")

#Having only the required columns and modifying the dataframe
df1 = df1[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
df1['HL_PCT'] = ((df1['Adj. High'] - df1['Adj. Low'])/ df1['Adj. Close']) * 100
df1['PCT_CHANGE'] = ((df1['Adj. Close'] - df1['Adj. Open'])/df1['Adj. Open']) * 100
df1 = df1[['Adj. Close', 'HL_PCT', 'PCT_CHANGE', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df1.fillna(-9999, inplace = True)

forecast_out = int(math.ceil(0.01 * len(df1)))
df1['Label'] = df1[forecast_col].shift(-forecast_out)

#Deciding features and Labels
X = np.array(df1.drop(['Label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df1.dropna(inplace=True)
y = np.array(df1['Label'])

#Training, testing and predicting the data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence  = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)

print (forecast_set, confidence, forecast_out)
df1['Forecast'] = np.nan
last_date = df1.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    df1.loc[next_date] = [np.nan for _ in range (len(df1.columns)-1)] + [i]
    next_unix += 86400

print df1.tail()
df1['Adj. Close'].plot()
df1['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
