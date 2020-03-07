import pandas as pd
import statsmodels.api as stats
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


#%%
data = pd.read_excel("USEMP.xlsx")
data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)
y = data['EMP']
y = y.iloc[0:656]
X = data.iloc[0:656, 1:]
x_pred = data.iloc[656, 1:]

def WeightedForecast(y, X, x_pred, y_pred):
    predictions = np.zeros(np.shape(X)[1])
    for i in range(np.shape(X)[1]):
        xx = X.iloc[:, i]
        xx = stats.add_constant(xx)
        xx_pred = np.hstack((1, x_pred.iloc[i]))
        model = stats.regression.linear_model.OLS(y, xx)
        results = model.fit()
        predictions[i] = results.predict(xx_pred)
    y_hat = np.mean(predictions)
    error = y_pred - y_hat
    return error

def AR1(y, X, x_pred, y_pred):
    xx = X.iloc[:, 0]
    xx = stats.add_constant(xx)
    xx_pred = np.hstack((1, x_pred.iloc[0]))
    model = stats.regression.linear_model.OLS(y, xx)
    results = model.fit()
    y_hat = results.predict(xx_pred)
    error = y_pred - y_hat
    return error