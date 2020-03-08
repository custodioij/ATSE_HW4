import pandas as pd
import statsmodels.api as stats
import numpy as np
import OLS_func as ols
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import itertools as iter
from scipy.stats import norm


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
    return [yFit - y_new, yFit]


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
    return [error, y_hat]


def AR1(y, X, x_pred, y_pred):
    xx = X[:, 0:2]
    xx_pred = x_pred[0:2]
    model = stats.regression.linear_model.OLS(y, xx)
    results = model.fit()
    y_hat = results.predict(xx_pred)
    error = y_pred - y_hat
    return [error[0], y_hat[0]]


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
    return [error, yFit]


def csr(vY, mX, vX_new, y_new, lag=False, k=None):
    mX = mX[:, 1:]  # Remove constant
    vX_new = vX_new[1:]
    if lag:
        current = mX[1:, :]
        lagged = mX[:-1, :]
        mX = np.hstack((current, lagged))
        # current = mX[1:, :]
        # lagged = mX[:-1, :]
        # mX = np.hstack((current, lagged))
        # Keep lagged y
        lagy = mX[:, [0]]
        lagy_new = vX_new[0]
        # Now remove it
        mX = mX[:, 1:]
        vX_new = vX_new[1:]
    if k is None:
        k = mX.shape[1]  # Account for the constant!
        k = mX.shape[1]  # Account for the constant!
    # Create combinations
    l_subset = []
    for i in range(k):
        l_subset += list(iter.combinations([j for j in range(k)], i + 1))
    l_yfit = []
    for t_subset in l_subset:
        xx = stats.add_constant(mX[:, t_subset])
        xnew = np.hstack((1, vX_new[list(t_subset)]))
        if lag:
            xx = np.hstack((lagy, xx))
            xnew = np.hstack((lagy_new, xnew))
        betas = ols.EstimateMM(vY, xx)
        l_yfit += [ols.OLS_predict(xnew, betas)]
    yfit = np.mean(l_yfit)
    error = y_new - yfit
    return [error, yfit]


""" Big loop to get predictions """

windows = window_fct()
e_KS, yFit_KS = [], []
e_WF, yFit_WF = [], []
e_AR, yFit_AR = [], []
e_FA, yFit_FA = [], []
e_mean = []
e_csr, yFit_CSR = [], []
e_csr_lag, yFit_CSR_lag = [], []
for window in windows:
    X = mX[window[0]:window[1], :]
    X_new = mX[window[1], :]
    Y = vY[window[0]:window[1]]
    y_new = vY[window[1]]
    e_KS += [kitchen_sink(Y, X, X_new, y_new)[0]]
    e_WF += [WeightedForecast(Y, X, X_new, y_new)[0]]
    e_AR += [AR1(Y, X, X_new, y_new)[0]]
    e_FA += [factor_augmented(Y, X, X_new, y_new)[0]]
    csr_temp = csr(Y, X, X_new, y_new)
    csr_lag_temp = csr(Y, X, X_new, y_new, lag=True)
    e_csr += [csr_temp[0]]
    e_csr_lag += [csr_lag_temp[0]]

    yFit_KS += [kitchen_sink(Y, X, X_new, y_new)[1]]
    yFit_WF += [WeightedForecast(Y, X, X_new, y_new)[1]]
    yFit_AR += [AR1(Y, X, X_new, y_new)[1]]
    yFit_FA += [factor_augmented(Y, X, X_new, y_new)[1]]
    yFit_CSR += [csr_temp[1]]
    yFit_CSR_lag += [csr_lag_temp[1]]
    e_mean += [np.mean(Y) - y_new]

""" Comparison """
# RMSE:
rmse = lambda xx: np.sqrt(np.mean([x**2 for x in xx]))
mafe = lambda xx: np.mean(np.abs([x for x in xx]))


df_rmse = np.vstack([rmse(e_mean), rmse(e_AR), rmse(e_KS), rmse(e_WF), rmse(e_FA), rmse(e_csr), rmse(e_csr_lag)]).T
df_mafe = np.vstack([mafe(e_mean), mafe(e_AR), mafe(e_KS), mafe(e_WF), mafe(e_FA), mafe(e_csr), mafe(e_csr_lag)]).T
mOutM= np.vstack([df_rmse, df_mafe])
dfOut1 = pd.DataFrame(mOutM, columns=['$Mean model$', '$AR(1)$', '$Kitchen-sink$', '$Weighted forecast$', '$FAVAR$',
                                      '$CSR$', '$CSRL$'], index=['MSFE','MAFE'])
print(dfOut1.to_latex(escape=False))
print(rmse(e_mean))
print(rmse(e_AR))

print(rmse(e_KS))
print(rmse(e_WF))
print(rmse(e_FA))
print(rmse(e_csr))


def DieboldMarianoTest(mE, Name, loss):
    '''
    Diebold-Mariano test
    '''
    iT = np.shape(mE)[1]
    mRes = np.zeros((iT, iT))
    for i in range(iT):
        for j in range(i + 1):
            if loss == 'RMSE':
                d = mE[:, i] ** 2 - mE[:, j] ** 2
            if loss == 'MAFE':
                d = np.abs(mE[:, i]) - np.abs(mE[:, j])
            dmean = np.mean(d)
            dstd = np.std(d)
            test = np.sqrt(np.shape(d)[0]) * dmean / dstd
            mRes[j, i] = (1 - norm.cdf(np.abs(test))) * 2
    dfDM = pd.DataFrame(np.round(mRes, 4), index=Name, columns=Name)
    print(dfDM.to_latex(escape=False))

mE = np.vstack(np.array([e_mean, e_AR, e_KS, e_WF, e_FA])).T
Name = np.array(('$Mean model$', '$AR(1)$', '$Kitchen-sink$', '$Weighted forecast$', '$FAVAR$'))
DieboldMarianoTest(mE, Name, 'RMSE')
DieboldMarianoTest(mE, Name, 'MAFE')


""" Investigate instability """
# Calculate 2-year moving average of forecast errors and plot

errors = [e_AR, e_mean, e_KS, e_WF, e_FA, e_csr, e_csr_lag]
moving_avg = []

for error in errors:
    local_mean = []
    for t in range(len(error) - 24):
        local_mean += [rmse(error[t:(t+24)])]
    moving_avg += [local_mean]

to_plot = pd.DataFrame(dt[-len(moving_avg[1]):])
to_plot = pd.concat([to_plot, pd.DataFrame(np.array(moving_avg).T, index=to_plot.index)], axis=1)
to_plot.columns = ['Month', 'AR', 'Mean', 'KS', 'WF', 'FA', 'CSR', 'CSRL']

# plt.close('all')
# fig = plt.figure()
to_plot.plot(x='Month')
plt.savefig('errors.png')
plt.show()



moving_avg_abs = []
for error in errors:
    local_mean_abs = []
    for t in range(len(error) - 24):
     local_mean_abs += [mafe(error[t:(t+24)])]
    moving_avg_abs += [local_mean_abs]

to_plot2 = pd.DataFrame(dt[-len(moving_avg_abs[1]):])
to_plot2 = pd.concat([to_plot2, pd.DataFrame(np.array(moving_avg_abs).T, index=to_plot2.index)], axis=1)
to_plot2.columns = ['Month', 'AR', 'Mean', 'KS', 'WF', 'FA', 'CSR', 'CSRL']
to_plot2.plot(x='Month')
plt.show()

Columns = np.array(('AR(1)', 'Kitchen-sink', 'FAVAR'))
yFit = np.array((yFit_AR, yFit_KS, yFit_FA)).T
ii = dt[(dt == "1980-01-01")].index[0]
for i in range(3):
    to_plot3 = pd.DataFrame(dt[ii:])
    to_plot3 = pd.concat([to_plot3, pd.DataFrame(np.vstack((data.iloc[ii:, 1], yFit[:,i])).T, index=to_plot3.index)], axis=1)
    to_plot3 = to_plot3.set_index('Month')
    to_plot3.columns = ['true', Columns[i]]
    to_plot3.plot()
    plt.savefig(Columns[i] + ' forecast.png')
    plt.show()
#check