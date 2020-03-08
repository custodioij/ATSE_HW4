import pandas as pd
import statsmodels.api as stats
import numpy as np
import OLS_func as ols
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import itertools as iter


data = pd.read_excel("USEMP.xlsx")
dt = data.Month
mData = np.array(data.iloc[:,1:])
mX = mData[:, 1:]
mX = stats.add_constant(mX)
vY = mData[:, 0]  # Careful with sahpe here

# cf: https://stackoverflow.com/questions/22898824/filtering-pandas-dataframes-on-dates
# df[(df['date'] > '2013-01-01') & (df['date'] < '2013-02-01')]
# dates[(dates >= '1980-01-01')]  #& (dates['Month'] < '2013-02-01')
# dates[(dates >= '1980-01-01')].index


def window_fct(dates=dt, first_forecast="1980-01-01", window_length=None):
    # Overly complicated for such a simple task
    # last on will be 655, so that index 657 can be forecast and tested
    if window_length is None:
        window_length = dates[(dates == first_forecast)].index[0] - 1
    last_forecast = dates.index[-1]
    forecast_periods = last_forecast - window_length
    windows = []
    for i in range(forecast_periods):
        windows += [(last_forecast - forecast_periods - window_length + i,
                     last_forecast - forecast_periods + i + 1)]
    return windows


def kitchen_sink(vY, mX, vX_new, y_new):
    """
    Run OLS with all predictors and predict with a new period.
    Return the predicted Y only.
    """
    betas = ols.EstimateMM(vY, mX)
    yFit = ols.OLS_predict(vX_new, betas)
    return yFit - y_new


def WeightedForecast(y, X, x_pred, y_pred):
    predictions = np.zeros(np.shape(X)[1])
    for i in range(np.shape(X)[1]-1):
        xx = X[:, [0, i+1]]
        xx_pred = x_pred[[0, i+1]]
        model = stats.regression.linear_model.OLS(y, xx)
        results = model.fit()
        predictions[i] = results.predict(xx_pred)
    y_hat = np.mean(predictions)
    error = y_pred - y_hat
    return error


def AR1(y, X, x_pred, y_pred):
    xx = X[:, 0:2]
    xx_pred = x_pred[0:2]
    model = stats.regression.linear_model.OLS(y, xx)
    results = model.fit()
    y_hat = results.predict(xx_pred)
    error = y_pred - y_hat
    return error


def factor_augmented(vY, mX, vX_new, y_new):
    pca = PCA(n_components=3)

    xx = mX[:, 2:].copy()  # Exclude constant and lagged y
    xx_new = vX_new[2:].copy()
    xx_new -= np.mean(xx)
    xx -= np.mean(xx)
    xx_new /= np.std(xx)
    xx /= np.std(xx)

    pca_fit = pca.fit(xx)
    PC = np.hstack((pca_fit.transform(xx), mX[:, 1:2]))
    PC_new = np.hstack((pca_fit.transform(xx_new.reshape(1, -1))[0], vX_new[1:2]))

    betas = ols.EstimateMM(vY, PC)
    yFit = ols.OLS_predict(PC_new, betas)
    error = yFit - y_new
    return error


def csr(vY, mX, vX_new, y_new, lag=False, k=None):
    mX = mX[:, 1:]  # Remove constant
    vX_new = vX_new[1:]
    if lag:
        current = mX[1:, :]
        lagged = mX[:-1, :]
        mX = np.hstack((current, lagged))
    if k is None:
        k = mX.shape[1] # Account for the constant!
    # Create combinations
    l_subset = []
    for i in range(k):
        l_subset += list(iter.combinations([j for j in range(k)], i+1))
    l_yfit = []
    for t_subset in l_subset:
        xx = stats.add_constant(mX[:, t_subset])
        xnew = np.hstack((1, vX_new[list(t_subset)]))
        betas = ols.EstimateMM(vY, xx)
        l_yfit += [ols.OLS_predict(xnew, betas)]
    error = y_new - np.mean(l_yfit)
    return error


""" Big loop to get predictions """

windows = window_fct()
e_KS = []
e_WF = []
e_AR = []
e_FA = []
e_mean = []
e_csr = []
for window in windows:
    X = mX[window[0]:window[1], :]
    X_new = mX[window[1], :]
    Y = vY[window[0]:window[1]]
    y_new = vY[window[1]]
    e_KS += [kitchen_sink(Y, X, X_new, y_new)]
    e_WF += [WeightedForecast(Y, X, X_new, y_new)]
    e_AR += [AR1(Y, X, X_new, y_new)]
    e_FA += [factor_augmented(Y, X, X_new, y_new)]
    e_mean += [np.mean(Y) - y_new]
    e_csr += [csr(Y, X, X_new, y_new)]

""" Comparison """
# RMSE:
rmse = lambda xx: np.sqrt(np.mean([x**2 for x in xx]))
mafe = lambda xx: np.mean(np.abs([x for x in xx]))


df_rmse = np.vstack([rmse(e_mean), rmse(e_AR), rmse(e_KS), rmse(e_WF), rmse(e_FA), rmse(e_csr)]).T
df_mafe = np.vstack([mafe(e_mean), mafe(e_AR), mafe(e_KS), mafe(e_WF), mafe(e_FA), mafe(e_csr)]).T
mOutM= np.vstack([df_rmse, df_mafe])
dfOut1 = pd.DataFrame(mOutM, columns=['$Mean model$', '$AR(1)$', '$Kitchen-sink$', '$Weighted forecast$', '$FAVAR$', '$CSR$'],
                      index=['MSFE','MAFE'])
print(dfOut1.to_latex(escape=False))
print(rmse(e_mean))
print(rmse(e_AR))

print(rmse(e_KS))
print(rmse(e_WF))
print(rmse(e_FA))
print(rmse(e_csr))

""" Investigate instability """
# Calculate 2-year moving average of forecast errors and plot

errors = [e_AR, e_mean, e_KS, e_WF, e_FA, e_csr]
moving_avg = []

for error in errors:
    local_mean = []
    for t in range(len(error) - 24):
        local_mean += [rmse(error[t:(t+24)])]
    moving_avg += [local_mean]

to_plot = pd.DataFrame(dt[-len(moving_avg[1]):])
to_plot = pd.concat([to_plot, pd.DataFrame(np.array(moving_avg).T, index=to_plot.index)], axis=1)
to_plot.columns = ['Month', 'AR', 'Mean', 'KS', 'WF', 'FA', 'CSR']

# plt.close('all')
# fig = plt.figure()
to_plot.plot(x='Month')
plt.show()
